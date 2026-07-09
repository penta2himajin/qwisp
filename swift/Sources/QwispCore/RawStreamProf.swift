import Foundation
import MLX
import MLXRandom

/// streaming-tier D1 feasibility probe (notes/17 扉1).
///
/// 目的: strict streaming(C<256)で MTP-D1 投機を配線する前に、利得を左右する唯一の
/// 未知数 = **M=2 gate**(verify forward M=2 / M=1 の wall 比、expert union 拡大に伴う
/// ensure/IO 込み)を実測する。resident の gate は 1.38(chain に敗北)だったが、streaming は
/// per-step floor(route+ensure+dispatch)が大きく shared されるため gate が 1 に近づく仮説。
///
/// 実装しない理由: MTP head を streaming に配線せずとも、実連続トークンを M=2 で流せば
/// verify forward の compute+IO は忠実に再現できる(draft=真の次トークンが最良の draft)。
/// c_head(draft CB)は resident 実測 ~3.5ms の固定 1-CB として投影に流用する。
///   // ponytail: c_head は resident 実測の固定 ms 代用。扉が GO なら実 head で再測。
///
/// 環境変数: QWISP_RAW_C(既定 64), QWISP_STREAMPROF_WARM(既定 96), QWISP_STREAMPROF_STEPS(既定 48)
public enum RawStreamProf {
    public static func run(modelDir: String, refPath: String) throws -> String {
        let env = ProcessInfo.processInfo.environment
        let C = Int(env["QWISP_RAW_C"] ?? "") ?? 64
        let warmN = Int(env["QWISP_STREAMPROF_WARM"] ?? "") ?? 96
        let nSteps = Int(env["QWISP_STREAMPROF_STEPS"] ?? "") ?? 48
        let cHeadMs = Double(env["QWISP_STREAMPROF_CHEAD_MS"] ?? "") ?? 3.5   // resident 実測代用

        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()                              // gate/shared/attn/gdn 常駐, experts stream
        let engine = RawEngine.build(store: store)
        guard let (fwd, _, _) = engine.makeFusedStreaming(
                modelDir: modelDir, maxM: 4, maxSeqLen: 2048, C: C) else {
            return "[raw-stream-prof] streaming forward 構築失敗"
        }
        guard fwd.head != nil else { return "[raw-stream-prof] head 無し" }

        guard let r = try? loadArrays(url: URL(fileURLWithPath: refPath)),
              let pa = r["spec_prompt"] else { return "[raw-stream-prof] ref 無し" }
        let prompt = pa.asType(.int32).asArray(Int32.self)
        guard prompt.count >= warmN + 2 * nSteps + 2 else {
            return "[raw-stream-prof] prompt 短すぎ(\(prompt.count) < \(warmN + 2 * nSteps + 2))"
        }

        func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }
        var out = "[raw-stream-prof] streaming C=\(C) warm=\(warmN) steps=\(nSteps) (prompt=\(prompt.count))"

        // 共通 warm(M=1 で現実的 cache 長 + arena LRU を steady state に)
        func warm() {
            for t in prompt.prefix(warmN) { _ = fwd.stepArgmax([t]) }
        }
        // stream 計測: window の実トークンを group ずつ食わせ、per-step wall と ensure/IO を集計。
        // snapshot/rollback は使わない(実 decode の miss/IO を忠実に測るため cache を前進させる)。
        func measure(group M: Int) -> (wallPerStep: Double, ensureMsPerStep: Double, preadMsPerStep: Double, chunksPerStep: Double) {
            warm()
            LayerExpertCache.ensureNanos = 0; LayerExpertCache.preadNanos = 0; LayerExpertCache.chunkTotal = 0
            var wall = 0.0
            var steps = 0
            var idx = warmN
            for _ in 0 ..< nSteps {
                let toks = Array(prompt[idx ..< idx + M])
                let s = nowMs()
                guard fwd.stepArgmax(toks) != nil else { break }
                wall += nowMs() - s
                steps += 1
                idx += M
            }
            let sd = Double(max(steps, 1))
            return (wall / sd,
                    Double(LayerExpertCache.ensureNanos) / 1e6 / sd,
                    Double(LayerExpertCache.preadNanos) / 1e6 / sd,
                    Double(LayerExpertCache.chunkTotal) / sd)
        }

        let m1 = measure(group: 1)
        let m2 = measure(group: 2)
        let gate = m2.wallPerStep / max(m1.wallPerStep, 1e-6)

        // ── 扉2: 実 streaming MTP head の c_head 実測(draft CB + accept-pair feed CB) ──
        // MTP head は自前 experts を init で全 MTLBuffer 常駐 → per-draft SSD IO 無し(resident と同型)。
        // draftArgmax=読み取り専用 draft、feedPairs=accept pair の k/v commit。step6 fold(commitLastDraft)で
        // accept 時の feed は無料化されうるので c_draft+c_feed は c_head の上限。
        var measuredCHeadMs = cHeadMs   // 失敗時は estimate にフォールバック
        var headNote = "(estimate)"
        if let spec = try? RawMTPValidate.loadSpec(modelDir: modelDir, store: store),
           let head = RawFusedVerify.RawMTPHead(spec: spec) {
            let hArr = (MLXRandom.normal([1, 2048]) * 0.5).asType(.float16)
            MLX.eval([hArr])
            if let (device, _) = RawMetalForward.ensure(),
               let hBuf = RawMetalForward.mtlBuf(hArr, device) {
                // warm head cache to a realistic len
                for _ in 0 ..< min(warmN, 200) { _ = head.feedPairs(hBuf: hBuf, rowRange: 0 ..< 1, toks: [Int32(1)]) }
                let reps = 40
                var draftWall = 0.0, feedWall = 0.0
                for _ in 0 ..< reps {
                    let s0 = nowMs(); _ = head.draftArgmax(hPrevBuf: hBuf, hPrevRow: 0, tok: Int32(1)); draftWall += nowMs() - s0
                    let s1 = nowMs(); _ = head.feedPairs(hBuf: hBuf, rowRange: 0 ..< 1, toks: [Int32(1)]); feedWall += nowMs() - s1
                }
                let cDraft = draftWall / Double(reps), cFeed = feedWall / Double(reps)
                measuredCHeadMs = cDraft + cFeed
                headNote = String(format: "(実測: draft=%.2fms + feed=%.2fms、fold で accept 時 feed は無料化=下限 draft のみ)", cDraft, cFeed)
            }
        }
        let cHead = measuredCHeadMs

        out += String(format: "\n  M=1 step: wall=%.2fms  ensure=%.2fms  pread=%.2fms  chunks=%.1f  (%.1f tok/s)",
                      m1.wallPerStep, m1.ensureMsPerStep, m1.preadMsPerStep, m1.chunksPerStep, 1000.0 / m1.wallPerStep)
        out += String(format: "\n  M=2 step: wall=%.2fms  ensure=%.2fms  pread=%.2fms  chunks=%.1f",
                      m2.wallPerStep, m2.ensureMsPerStep, m2.preadMsPerStep, m2.chunksPerStep)
        out += String(format: "\n  ★M=2 gate = %.3f  (resident は 1.38 / gate→1 ほど D1 有利)", gate)

        // 投影速度: notes/17 モデル per-token = (c_head + gate + (1-a)) / (1+a)、baseline=1(M=1 forward)。
        // c_head は W1 に対する比。speedup = 1/per-token。break-even は speedup=1。
        let cHeadFrac = cHead / m1.wallPerStep
        out += "\n  c_head " + headNote
        out += String(format: "\n  c_head=%.2fms → W1 比 %.3f。投影 speedup = (1+a)/(c_head+gate+(1-a)):", cHead, cHeadFrac)
        out += "\n    a(accept) :  0.49(shortnl)   0.60         0.74(code)    0.85"
        var row = "\n    speedup   :"
        for a in [0.49, 0.60, 0.74, 0.85] {
            let perTok = (cHeadFrac + gate + (1.0 - a)) / (1.0 + a)
            row += String(format: "  %6.3fx     ", 1.0 / perTok)
        }
        out += row
        // break-even gate at each a (gate below which D1 wins, given this c_head): 1+a = c_head+gate+(1-a) → gate = 2a - c_head
        out += "\n  読み: gate < 2a − c_head で D1 が勝つ。streaming gate が resident 1.38 を大きく割れば shortnl でも黒字圏。"
        return out
    }
}
