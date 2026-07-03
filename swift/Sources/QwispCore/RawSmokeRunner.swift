import Foundation
import MLX
import MLXFast

/// U2a: raw-smoke — build LayerSpec[40] from real model weights (resident) and compare
/// raw verify forward (path A per-token + path B batched) vs MLX on real weights.
/// Hard gate: rawA-vs-rawB hidden must be bit-exact.  MLX comparison is informational.
///
/// Refactored to use RawEngine for model-building (output format unchanged).
/// QWISP_RUN=raw-smoke  QWISP_GEN=8  (default T=8, cap 16)
public enum RawSmokeRunner {

    static let eps: Float    = RawEngine.eps
    static let H:   Int      = RawEngine.H

    // ── Helpers ───────────────────────────────────────────────────────────────

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

    // ── Main runner ───────────────────────────────────────────────────────────

    public static func run(modelDir: String, refPath: String) throws -> String {
        print("[raw-smoke] loading model from \(modelDir) ...")
        let store = try WeightStore(modelDir: modelDir)
        store.residentAll()
        print("[raw-smoke] residentAll complete")

        print("[raw-smoke] building RawEngine (LayerSpecs[40]) ...")
        let engine = RawEngine.build(store: store)
        let layers = engine.layers
        print("[raw-smoke] moeI=\(engine.moeI) (inferred from gate_proj.weight shape)")
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

        let vocab = engine.vocab

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
        let cachesA = engine.freshCaches()
        var rawAArgmax: [Int] = []
        var rawAHiddens: [MLXArray] = []
        for t in 0 ..< T {
            let x = engine.embed(tokens: [promptIds[t]])   // [1, H]
            guard let normed = engine.forwardRows(x, caches: cachesA, M: 1)
            else { return "[raw-smoke] ERROR: verifyForwardRows nil t=\(t) (A)" }
            guard let logits = engine.logits(normed, M: 1)
            else { return "[raw-smoke] ERROR: qmmTiled nil t=\(t) (A)" }
            MLX.eval([logits])
            rawAArgmax.append(MLX.argMax(logits[0], axis: -1).item(Int.self))
            rawAHiddens.append(normed[0])   // [H]
        }
        print("[raw-smoke] RAW-A done")

        // ── RAW path B: batched (M=T) ─────────────────────────────────────
        print("[raw-smoke] RAW path B (batched M=\(T)) ...")
        let cachesB  = engine.freshCaches()
        let xBatch   = engine.embed(tokens: Array(promptIds.prefix(T)))   // [T, H]
        guard let normedB = engine.forwardRows(xBatch, caches: cachesB, M: T)
        else { return "[raw-smoke] ERROR: batched verifyForwardRows returned nil" }
        guard let logitsB = engine.logits(normedB, M: T)
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
