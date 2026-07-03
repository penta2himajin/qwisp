import Foundation
import MLX
import MLXFast

/// U2a: raw-smoke — build LayerSpec[40] from real model weights (resident) and compare
/// raw verify forward (path A per-token + path B batched) vs MLX on real weights.
/// Hard gate: rawA-vs-rawB hidden must be bit-exact.  MLX comparison is informational.
///
/// QWISP_RUN=raw-smoke  QWISP_GEN=8  (default T=8, cap 16)
public enum RawSmokeRunner {

    static let eps: Float    = 1e-6
    static let H:   Int      = 2048
    static let numLayers: Int = 40
    static let fullAttnInterval: Int = 4

    // ── Helpers ───────────────────────────────────────────────────────────────

    static func isLinear(_ i: Int) -> Bool { (i + 1) % fullAttnInterval != 0 }

    /// Bit-exact comparison at float32 precision (same logic as RawVerifyTests).
    static func bitEqual(_ a: MLXArray, _ b: MLXArray) -> (Bool, String) {
        let af = a.reshaped([-1]).asType(.float32)
        let bf = b.reshaped([-1]).asType(.float32)
        MLX.eval([af, bf])
        let na = af.size, nb = bf.size
        guard na == nb else { return (false, "size \(a.shape) vs \(b.shape)") }
        let aArr = af.asArray(Float.self), bArr = bf.asArray(Float.self)
        var maxDiff: Float = 0; var firstIdx = -1
        for i in 0 ..< na {
            let d = abs(aArr[i] - bArr[i])
            if d > maxDiff { maxDiff = d; if firstIdx < 0 { firstIdx = i } }
        }
        if maxDiff == 0 { return (true, "ok") }
        return (false,
                "max|Δ|=\(maxDiff) first@idx=\(firstIdx) got=\(aArr[firstIdx]) ref=\(bArr[firstIdx])")
    }

    // ── Layer-spec construction ───────────────────────────────────────────────

    static func buildLayerSpec(_ i: Int, store: WeightStore, moeI: Int) -> RawVerifyForward.LayerSpec {
        let p  = "language_model.model.layers.\(i)"
        let mp = "\(p).mlp"
        let lin = isLinear(i)

        let inputLN = store.req("\(p).input_layernorm.weight")
        let postLN  = store.req("\(p).post_attention_layernorm.weight")

        var gdnW:  RawVerifyForward.GDNLayerW? = nil
        var attnW: RawVerifyForward.AttnLayerW? = nil

        if lin {
            let la = "\(p).linear_attn"
            gdnW = RawVerifyForward.GDNLayerW(
                qkvWq: store.req("\(la).in_proj_qkv.weight"),
                qkvSc: store.req("\(la).in_proj_qkv.scales"),
                qkvBi: store.req("\(la).in_proj_qkv.biases"),
                zWq:   store.req("\(la).in_proj_z.weight"),
                zSc:   store.req("\(la).in_proj_z.scales"),
                zBi:   store.req("\(la).in_proj_z.biases"),
                bWq:   store.req("\(la).in_proj_b.weight"),
                bSc:   store.req("\(la).in_proj_b.scales"),
                bBi:   store.req("\(la).in_proj_b.biases"),
                aWq:   store.req("\(la).in_proj_a.weight"),
                aSc:   store.req("\(la).in_proj_a.scales"),
                aBi:   store.req("\(la).in_proj_a.biases"),
                outWq: store.req("\(la).out_proj.weight"),
                outSc: store.req("\(la).out_proj.scales"),
                outBi: store.req("\(la).out_proj.biases"),
                conv1dW:    store.req("\(la).conv1d.weight"),
                normWeight: store.req("\(la).norm.weight"),
                aLog:   store.req("\(la).A_log"),
                dtBias: store.req("\(la).dt_bias"))
        } else {
            let sa = "\(p).self_attn"
            attnW = RawVerifyForward.AttnLayerW(
                qWq: store.req("\(sa).q_proj.weight"),
                qSc: store.req("\(sa).q_proj.scales"),
                qBi: store.req("\(sa).q_proj.biases"),
                kWq: store.req("\(sa).k_proj.weight"),
                kSc: store.req("\(sa).k_proj.scales"),
                kBi: store.req("\(sa).k_proj.biases"),
                vWq: store.req("\(sa).v_proj.weight"),
                vSc: store.req("\(sa).v_proj.scales"),
                vBi: store.req("\(sa).v_proj.biases"),
                oWq: store.req("\(sa).o_proj.weight"),
                oSc: store.req("\(sa).o_proj.scales"),
                oBi: store.req("\(sa).o_proj.biases"),
                qNorm: store.req("\(sa).q_norm.weight"),
                kNorm: store.req("\(sa).k_norm.weight"))
        }

        // Shared gate is [1, H] 8-bit; qmm8 requires N%8==0.
        // Pad to [8, ...] by repeating the single row — moeBlockRows uses col 0 only.
        let sgW0 = store.req("\(mp).shared_expert_gate.weight")
        let sgS0 = store.req("\(mp).shared_expert_gate.scales")
        let sgB0 = store.req("\(mp).shared_expert_gate.biases")
        let sgWPad = MLX.concatenated(Array(repeating: sgW0, count: 8), axis: 0)
        let sgSPad = MLX.concatenated(Array(repeating: sgS0, count: 8), axis: 0)
        let sgBPad = MLX.concatenated(Array(repeating: sgB0, count: 8), axis: 0)
        MLX.eval([sgWPad, sgSPad, sgBPad])

        let moeW = RawVerifyForward.MoEBlockW(
            gateWq: store.req("\(mp).gate.weight"),
            gateSc: store.req("\(mp).gate.scales"),
            gateBi: store.req("\(mp).gate.biases"),
            swGWq:  store.req("\(mp).switch_mlp.gate_proj.weight"),
            swGSc:  store.req("\(mp).switch_mlp.gate_proj.scales"),
            swGBi:  store.req("\(mp).switch_mlp.gate_proj.biases"),
            swUWq:  store.req("\(mp).switch_mlp.up_proj.weight"),
            swUSc:  store.req("\(mp).switch_mlp.up_proj.scales"),
            swUBi:  store.req("\(mp).switch_mlp.up_proj.biases"),
            swDWq:  store.req("\(mp).switch_mlp.down_proj.weight"),
            swDSc:  store.req("\(mp).switch_mlp.down_proj.scales"),
            swDBi:  store.req("\(mp).switch_mlp.down_proj.biases"),
            shGWq:  store.req("\(mp).shared_expert.gate_proj.weight"),
            shGSc:  store.req("\(mp).shared_expert.gate_proj.scales"),
            shGBi:  store.req("\(mp).shared_expert.gate_proj.biases"),
            shUWq:  store.req("\(mp).shared_expert.up_proj.weight"),
            shUSc:  store.req("\(mp).shared_expert.up_proj.scales"),
            shUBi:  store.req("\(mp).shared_expert.up_proj.biases"),
            shDWq:  store.req("\(mp).shared_expert.down_proj.weight"),
            shDSc:  store.req("\(mp).shared_expert.down_proj.scales"),
            shDBi:  store.req("\(mp).shared_expert.down_proj.biases"),
            sharedGateWq:  sgWPad,
            sharedGateSc:  sgSPad,
            sharedGateBi:  sgBPad)

        return RawVerifyForward.LayerSpec(
            isLinear: lin, inputLN: inputLN, postLN: postLN,
            gdn: gdnW, attn: attnW, moe: moeW, moeE: 256, moeI: moeI)
    }

    // ── Cache initialisation ──────────────────────────────────────────────────

    /// Fresh cold caches: GDN convState [3,8192] f16 / recState [1,32,128,128] f32;
    /// attn kCache/vCache [2,0,256] f16 (zero-length sequence axis).
    static func makeFreshCaches() -> [RawVerifyForward.LayerCaches] {
        (0 ..< numLayers).map { i in
            if isLinear(i) {
                return RawVerifyForward.LayerCaches(
                    convState: MLX.zeros([3, 8192],         dtype: .float16),
                    recState:  MLX.zeros([1, 32, 128, 128], dtype: .float32))
            } else {
                return RawVerifyForward.LayerCaches(
                    kCache: MLX.zeros([2, 0, 256], dtype: .float16),
                    vCache: MLX.zeros([2, 0, 256], dtype: .float16))
            }
        }
    }

    // ── Main runner ───────────────────────────────────────────────────────────

    public static func run(modelDir: String, refPath: String) throws -> String {
        print("[raw-smoke] loading model from \(modelDir) ...")
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        print("[raw-smoke] residentAll complete")

        // Infer expert inner dim I from switch_mlp.gate_proj.weight [E, I, packedK].
        let gateW0 = store.req("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight")
        let moeI   = gateW0.shape[1]
        print("[raw-smoke] moeI=\(moeI) (inferred from gate_proj.weight shape \(gateW0.shape))")

        // Build 40 LayerSpecs.
        print("[raw-smoke] building LayerSpecs[40] ...")
        let layers = (0 ..< numLayers).map { i -> RawVerifyForward.LayerSpec in
            buildLayerSpec(i, store: store, moeI: moeI)
        }
        print("[raw-smoke] LayerSpecs ready")

        // Load prompt from refPath (spec_prompt: 1-D int32 [P]).
        let refArrays = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let promptRaw = refArrays["spec_prompt"] else {
            return "[raw-smoke] ERROR: no 'spec_prompt' in \(refPath)"
        }
        let promptIds: [Int32] = promptRaw.asType(.int32).asArray(Int32.self)
        let promptLen = promptIds.count
        let T = Swift.min(Tell.envInt("QWISP_GEN", 8), Swift.min(16, promptLen))
        print("[raw-smoke] promptLen=\(promptLen) T=\(T)")

        // Shared weight handles.
        let embedW  = store.req("language_model.model.embed_tokens.weight")
        let embedS  = store.req("language_model.model.embed_tokens.scales")
        let embedB  = store.req("language_model.model.embed_tokens.biases")
        let fnW     = store.req("language_model.model.norm.weight")      // final rmsNorm
        let lmW     = store.req("language_model.lm_head.weight")
        let lmS     = store.req("language_model.lm_head.scales")
        let lmB     = store.req("language_model.lm_head.biases")
        let vocab   = lmW.dim(0)

        // ── MLX reference: QwispModel, cached teacher-forced per-token ─────
        print("[raw-smoke] MLX reference path (\(T) tokens) ...")
        let qm       = QwispModel(store: store)
        let mlxCaches = qm.makeCaches()
        var mlxArgmax: [Int] = []
        var mlxHiddens: [MLXArray] = []
        for t in 0 ..< T {
            let tok = MLXArray([promptIds[t]]).reshaped([1, 1])   // [1,1] int32
            let (hidden, logits) = qm.forwardHidden(tok, caches: mlxCaches)
            MLX.eval([hidden, logits])
            // hidden [1,1,H], logits [1,1,vocab]
            mlxArgmax.append(MLX.argMax(logits[0, 0], axis: -1).item(Int.self))
            mlxHiddens.append(hidden[0, 0])   // [H]
        }
        print("[raw-smoke] MLX done")

        // ── RAW path A: per-token (M=1), sequential ───────────────────────
        print("[raw-smoke] RAW path A (per-token, M=1) ...")
        let cachesA = makeFreshCaches()
        var rawAArgmax: [Int] = []
        var rawAHiddens: [MLXArray] = []
        for t in 0 ..< T {
            let tok = MLXArray([promptIds[t]]).reshaped([1, 1])   // [1,1]
            let emb = ModelHead.embed(ids: tok, weight: embedW, scales: embedS, biases: embedB, bits: 4)
            let x   = emb.reshaped([1, H])                        // [1,H]
            guard let h = RawVerifyForward.verifyForwardRows(x, layers: layers, caches: cachesA, M: 1)
            else { return "[raw-smoke] ERROR: verifyForwardRows nil t=\(t) (A)" }
            guard let normed = RawMetalForward.rmsNormRows(h, fnW, M: 1, eps: eps, D: H)
            else { return "[raw-smoke] ERROR: rmsNormRows nil t=\(t) (A)" }
            guard let logits = RawMetalForward.qmmTiled(normed, lmW, scales: lmS, biases: lmB, M: 1, K: H, N: vocab)
            else { return "[raw-smoke] ERROR: qmmTiled nil t=\(t) (A)" }
            MLX.eval([logits])
            rawAArgmax.append(MLX.argMax(logits[0], axis: -1).item(Int.self))
            rawAHiddens.append(normed[0])   // [H]
        }
        print("[raw-smoke] RAW-A done")

        // ── RAW path B: batched (M=T) ─────────────────────────────────────
        print("[raw-smoke] RAW path B (batched M=\(T)) ...")
        let cachesB  = makeFreshCaches()
        let batchIds = MLXArray(Array(promptIds.prefix(T))).reshaped([1, T])
        let emBatch  = ModelHead.embed(ids: batchIds, weight: embedW, scales: embedS, biases: embedB, bits: 4)
        let xBatch   = emBatch.reshaped([T, H])   // [T,H]
        guard let hBatch = RawVerifyForward.verifyForwardRows(xBatch, layers: layers, caches: cachesB, M: T)
        else { return "[raw-smoke] ERROR: batched verifyForwardRows returned nil" }
        guard let normedB = RawMetalForward.rmsNormRows(hBatch, fnW, M: T, eps: eps, D: H)
        else { return "[raw-smoke] ERROR: batched rmsNormRows returned nil" }
        guard let logitsB = RawMetalForward.qmmTiled(normedB, lmW, scales: lmS, biases: lmB, M: T, K: H, N: vocab)
        else { return "[raw-smoke] ERROR: batched qmmTiled returned nil" }
        MLX.eval([logitsB])
        var rawBArgmax: [Int] = []
        var rawBHiddens: [MLXArray] = []
        for t in 0 ..< T {
            rawBArgmax.append(MLX.argMax(logitsB[t], axis: -1).item(Int.self))
            rawBHiddens.append(normedB[t])   // [H]
        }
        print("[raw-smoke] RAW-B done")

        // ── Comparisons ───────────────────────────────────────────────────

        var lines: [String] = []

        // (1) A vs B hidden — HARD gate (must be bit-exact).
        let hidA = MLX.stacked(rawAHiddens, axis: 0)   // [T, H]
        let hidB = MLX.stacked(rawBHiddens, axis: 0)
        MLX.eval([hidA, hidB])
        let (abBitEq, abDetail) = bitEqual(hidA, hidB)
        lines.append("[raw-smoke] rawA-vs-rawB hidden: \(abBitEq ? "bit-equal" : "NOT bit-equal — \(abDetail)")")

        let abArgMatch = zip(rawAArgmax, rawBArgmax).filter { $0 == $1 }.count
        lines.append("[raw-smoke] rawA-vs-rawB argmax: \(abArgMatch)/\(T) equal")

        // (2) A vs MLX hidden rel (informational).
        let hidAF  = hidA.asType(.float32)
        let hidMLX = MLX.stacked(mlxHiddens, axis: 0).asType(.float32)
        MLX.eval([hidAF, hidMLX])
        let absErr = MLX.max(MLX.abs(hidAF - hidMLX)).item(Float.self)
        let absMax = Swift.max(MLX.max(MLX.abs(hidMLX)).item(Float.self), Float(1e-9))
        let relErr = absErr / absMax
        lines.append(String(format: "[raw-smoke] rawA-vs-MLX hidden rel: %.3e", relErr))

        // (3) A vs MLX argmax (informational — lm_head tiled vs MLX qmv may near-tie flip).
        var mismatchPos: [Int] = []
        for t in 0 ..< T { if rawAArgmax[t] != mlxArgmax[t] { mismatchPos.append(t) } }
        let amMatch = T - mismatchPos.count
        let mismatchStr = mismatchPos.isEmpty ? "" : " mismatch@\(mismatchPos)"
        lines.append("[raw-smoke] rawA-vs-MLX argmax: \(amMatch)/\(T) match\(mismatchStr)")

        // Summary.
        lines.append("RAWSMOKE \(abBitEq ? "PASS" : "FAIL")")
        return lines.joined(separator: "\n")
    }
}
