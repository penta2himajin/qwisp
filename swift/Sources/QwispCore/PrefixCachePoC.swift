import Foundation
import MLX

// Feasibility PoC for cross-request prefix caching (issue: agentic TTFT dominated by re-prefilling
// the whole ~8K context every request). Proves — using ONLY shipped primitives (prefill / forward /
// snapshot / rollback / stepArgmax), no forward-path change — that the full cross-request protocol
//
//     prefill(A) → snapshot S → prefill(tail we discard) → rollback(S) → prefill(B) → decode
//
// yields a BYTE-IDENTICAL token stream to the baseline prefill(A+B) → decode, while paying only the
// prefill of the suffix B instead of the whole A+B. A=stable content prefix (reused across requests),
// tail = the generation prompt + generated tokens that the next request rewinds past, B = the new
// content appended in the next request.
extension Tell {
    // Prefill throughput probe: measures tok/s prefilling a fixed prompt at several chunk sizes.
    // If larger chunks are much faster → per-forward overhead dominates (raise the chunk).
    // If flat → per-token compute is the limit (kernel-level work). Measure-before-implement.
    public static func prefillThroughputProbe(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[prefill-probe] load fail" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let promptLen = 4096
        let prompt = (0..<promptLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }
        var lines = ["[prefill-probe] promptLen=\(promptLen), resident"]
        for chunk in [64, 128, 256, 512] {
            guard let b = Tell.fusedBackend(engine: engine, maxM: chunk + 8, maxSeqLen: promptLen + 128) else {
                lines.append("  chunk=\(chunk): backend nil"); continue
            }
            let t0 = Date()
            var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + chunk, promptLen)
                _ = b.forward(Array(prompt[pos ..< end]))
                pos = end
            }
            let dt = Date().timeIntervalSince(t0)
            lines.append(String(format: "  chunk=%4d: %.3fs  %.0f tok/s", chunk, dt, Double(promptLen) / dt))
        }
        return lines.joined(separator: "\n") + "\nPREFILLPROBE done"
    }

    public static func prefixCachePoC(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[prefix-poc] load fail\nPREFIXPOC FAIL" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)

        let aLen = 1024, tailLen = 128, bLen = 256, decodeN = 32
        func toks(_ n: Int, _ salt: Int) -> [Int32] { (0..<n).map { Int32((($0 &* 7 &+ salt) % 5000) + 100) } }
        let A = toks(aLen, 13), tail = toks(tailLen, 999), B = toks(bLen, 5)
        let full = A + B
        let maxSeqLen = A.count + tailLen + B.count + decodeN + 128

        func greedy(_ backend: Tell.SpecBackend, firstNormed: MLXArray) -> [Int] {
            guard let l0 = engine.logits(firstNormed, M: 1) else { return [] }
            MLX.eval([l0])
            var tok = MLX.argMax(l0[0], axis: -1).item(Int.self)
            var out = [tok]
            for _ in 1..<decodeN {
                guard let nx = backend.stepArgmax([Int32(tok)])?.first else { break }
                out.append(nx); tok = nx
            }
            return out
        }

        // Run the reuse protocol against a backend and compare to its own baseline. `full` builds a
        // fresh backend each call, so the two paths share no mutable state.
        func trial(_ label: String, _ mk: () -> Tell.SpecBackend?, timed: Bool) -> String {
            guard let bBase = mk(), let bReuse = mk() else { return "  \(label): backend nil" }
            let t0 = Date()
            guard let nBase = Tell.prefill(promptIds: full, backend: bBase) else { return "  \(label): prefill base nil" }
            let tFull = Date().timeIntervalSince(t0)
            let seqBase = greedy(bBase, firstNormed: nBase)

            _ = Tell.prefill(promptIds: A, backend: bReuse)   // cache the stable content prefix
            // Full-state snapshot for arbitrary rewind; falls back to snapshot (composed's is already full).
            let snap = (bReuse.fullSnapshot ?? bReuse.snapshot)()          // content-boundary snapshot
            _ = Tell.prefill(promptIds: tail, backend: bReuse) // gen prompt + generated (request-specific)
            (bReuse.fullRestore ?? bReuse.rollback)(snap)      // next request rewinds past them
            let t1 = Date()
            guard let nReuse = Tell.prefill(promptIds: B, backend: bReuse) else { return "  \(label): prefill reuse nil" }
            let tSuffix = Date().timeIntervalSince(t1)
            let seqReuse = greedy(bReuse, firstNormed: nReuse)

            let ok = seqBase == seqReuse
            let sp = tSuffix > 0 ? String(format: " full=%.2fs suffix=%.2fs speedup=%.1fx", tFull, tSuffix, tFull / tSuffix) : ""
            let detail = ok ? "" : "  base=\(Array(seqBase.prefix(6))) reuse=\(Array(seqReuse.prefix(6)))"
            return "  \(label): byte-identical=\(ok ? "YES" : "NO")\(timed ? sp : "")\(detail)"
        }

        let composed = trial("composed(full copyState)", { Tell.composedBackend(engine: engine) }, timed: false)
        let fused = trial("fused(full snapshot)", { Tell.fusedBackend(engine: engine, maxM: 96, maxSeqLen: maxSeqLen) }, timed: true)
        let pass = composed.contains("byte-identical=YES") && fused.contains("byte-identical=YES")
        return """
        [prefix-poc] A=\(aLen) tail=\(tailLen) B=\(bLen) decodeN=\(decodeN)
        \(composed)
        \(fused)
        PREFIXPOC \(pass ? "PASS" : "FAIL")   (both paths must be byte-identical to full prefill)
        """
    }
}
