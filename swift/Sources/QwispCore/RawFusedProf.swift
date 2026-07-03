import Foundation
import MLX
import Metal

/// D1 残 upside: raw fused engine の M=1(decode)/M>1(verify)step 実測プロファイラ。
/// 目的: ~22ms/tok が GPU 実処理か CPU/commit overhead かを分離し最適化方向を決める。
/// wall(呼出全体)vs GPU-exec(cb.gpuEnd-gpuStart)。QWISP_RUN=raw-fused-prof。
public enum RawFusedProf {
    public static func run(modelDir: String, refPath: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        let engine = RawEngine.build(store: store)
        guard let (fwd, fnBuf) = engine.makeFused(maxM: 32, maxSeqLen: 2048) else { return "[raw-fused-prof] makeFused 失敗" }
        _ = fnBuf
        guard fwd.head != nil else { return "[raw-fused-prof] head 無し(stepArgmax 1-CB 経路が必要)" }

        // prefill: ref prompt を M=1 で順に食わせて cache を現実的長さに(profile は decode 定常を測る)
        guard let r = try? loadArrays(url: URL(fileURLWithPath: refPath)),
              let pa = r["spec_prompt"] else { return "[raw-fused-prof] ref 無し" }
        let prompt = pa.asType(.int32).asArray(Int32.self)
        let warmPrompt = Array(prompt.prefix(64))
        for t in warmPrompt { _ = fwd.stepArgmax([t]) }
        var cur = warmPrompt.last ?? 0

        func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        var out = "[raw-fused-prof] resident raw fused engine (prompt=\(prompt.count), warm=\(warmPrompt.count))"

        // M=1 decode 定常: wall と GPU-exec を分離
        for M in [1, 8, 17] {
            let toks = (0 ..< M).map { _ -> Int32 in let v = cur; return v }   // 同一 token M 個(profile 用, cache は伸びる)
            for _ in 0 ..< 5 { _ = fwd.stepArgmax(toks) }                        // warm
            let reps = M == 1 ? 60 : 20
            var wall = 0.0, gpu = 0.0
            for _ in 0 ..< reps {
                let s = nowMs()
                guard let o = fwd.stepArgmax(toks) else { return out + "\n  step 失敗 M=\(M)" }
                wall += nowMs() - s
                gpu += RawFusedVerify.RawFusedForward.profLastGPUMs
                cur = Int32(o[0])
            }
            let w = wall / Double(reps), g = gpu / Double(reps)
            out += String(format: "\n  M=%2d: wall=%.2fms  GPU-exec=%.2fms  CPU-overhead=%.2fms  (%.1f tok/s @M=1相当は wall/M)",
                          M, w, g, w - g, 1000.0 / (w / Double(M)))
        }
        // segment 分解(M=1): forwardRows(40層+norm)GPU vs stepArgmax(全体)GPU の差 = lm_head+head
        let fnBuf2 = fnBuf
        let xEmb = engine.embed(tokens: [cur])
        for _ in 0 ..< 5 { _ = fwd.forwardRows(xEmb, M: 1, finalNormW: fnBuf2) }
        var layersG = 0.0, fullG = 0.0; let sreps = 40
        for _ in 0 ..< sreps {
            _ = fwd.forwardRows(xEmb, M: 1, finalNormW: fnBuf2)
            layersG += RawFusedVerify.RawFusedForward.profLastGPUMs
            _ = fwd.stepArgmax([cur])
            fullG += RawFusedVerify.RawFusedForward.profLastGPUMs
        }
        let lg = layersG / Double(sreps), fg = fullG / Double(sreps)
        out += String(format: "\n  [segment M=1] 40層+norm GPU=%.2fms  full(=+embed+lm_head+argmax) GPU=%.2fms  → lm_head+head≈%.2fms",
                      lg, fg, fg - lg)
        out += "\n  読み: GPU-exec 支配→kernel 最適化。固定 per-forward コスト大→ M=1 で weight-read 非償却。lm_head+head が大なら qmm4_tiled(M=1)→qmv 切替候補"

        // component 内訳(M=1, wall): 40層 12ms が MoE / attn / GDN のどこか。fused component 関数を単体計測。
        // 実モデル層構成: 40層中 attn=10(index%4==3), gdn=30。全層に MoE。
        let x1 = MLXRandom.normal([1, RawEngine.H]).asType(.float16); x1.eval()
        func timeIt(_ n: Int, _ f: () -> Void) -> Double { for _ in 0..<3 { f() }; let s = nowMs(); for _ in 0..<n { f() }; return (nowMs()-s)/Double(n) }
        // 代表 attn 層 / gdn 層 / MoE ブロックを 1 つずつ
        var attnMs = 0.0, gdnMs = 0.0, moeMs = 0.0
        if let al = engine.layers.first(where: { !$0.isLinear }), let aw = al.attn {
            var kc = MLXRandom.normal([2, 16, 256]).asType(.float16), vc = MLXRandom.normal([2, 16, 256]).asType(.float16)
            MLX.eval([kc, vc])
            attnMs = timeIt(30) { var k = kc, v = vc; _ = RawVerifyForward.attnLayerRows(x1, aw, kCache: &k, vCache: &v, M: 1) }
            _ = (kc, vc)
        }
        if let gl = engine.layers.first(where: { $0.isLinear }), let gw = gl.gdn {
            let cs = MLXRandom.normal([3, 8192]).asType(.float16), rs = MLXRandom.normal([1, 32, 128, 128]).asType(.float32)
            MLX.eval([cs, rs])
            gdnMs = timeIt(30) { var c = cs, r = rs; _ = RawVerifyForward.gdnLayerRows(x1, gw, convState: &c, recState: &r, M: 1) }
        }
        if let ml = engine.layers.first {
            moeMs = timeIt(30) { _ = RawVerifyForward.moeBlockRows(x1, ml.moe, M: 1, E: 256, I: engine.moeI, Ktop: 8, metalRoute: true) }
        }
        out += String(format: "\n  [component M=1 wall/回] attn層=%.2fms gdn層=%.2fms MoEブロック=%.2fms → forward概算=attn×10+gdn×30+MoE×40=%.1fms",
                      attnMs, gdnMs, moeMs, attnMs*10 + gdnMs*30 + moeMs*40)
        out += "\n  (注: これは per-op CB 版の wall で 1-CB より遅い。相対比で律速 component を特定する用)"
        return out
    }
}
