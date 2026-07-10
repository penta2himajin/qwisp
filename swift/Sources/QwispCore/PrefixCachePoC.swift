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

    // Prefill component breakdown: differential timing via profSkip flags on the raw forward.
    // Runs a full prefill, then repeats with each component skipped; cost(X) ≈ t_full - t_skipX.
    // Timing-only (skipped output is garbage) — attributes the cold-prefill wall to GDN-recur /
    // GDN-matmul / attention / MoE-experts / routing / floor, so we know which lever (if any) pays.
    // Env: QWISP_PREFILL_LEN (default 8192), QWISP_PREFILL_CHUNK (default 64).
    public static func prefillBreakdownProbe(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[prefill-bd] load fail\nPREFILLBD done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let promptLen = Tell.envInt("QWISP_PREFILL_LEN", 8192)
        let chunk = Tell.envInt("QWISP_PREFILL_CHUNK", 64)
        let prompt = (0..<promptLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }

        // reset all skip flags to a known-off baseline
        func allOff() {
            SeedlessMetalForward.profSkipGDNMatmul = false
            SeedlessMetalForward.profSkipGDNRecur = false
            SeedlessMetalForward.profSkipMixer = false
            SeedlessMetalForward.profSkipMoEExperts = false
            SeedlessMetalForward.profSkipMoERouted = false
            SeedlessMetalForward.profSkipMoEShared = false
            SeedlessMetalForward.profSkipSingleThread = false
        }
        // one full prefill of `prompt` at `chunk`, returns wall seconds. Fresh backend each run so
        // KV starts empty (identical work every config).
        func runPrefill() -> Double {
            guard let b = Tell.fusedBackend(engine: engine, maxM: chunk + 8, maxSeqLen: promptLen + 128) else { return -1 }
            let t0 = Date()
            var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + chunk, promptLen)
                _ = b.forward(Array(prompt[pos ..< end]))
                pos = end
            }
            return Date().timeIntervalSince(t0)
        }

        // min-of-2 to damp run-to-run variance (single-run differentials were non-additive/noisy).
        func best(_ set: () -> Void) -> Double {
            allOff(); set(); let a = runPrefill(); let b = runPrefill(); allOff(); return Swift.min(a, b)
        }
        func floorSet() {   // skip ALL heavy GPU compute → what remains is the irreducible floor
            SeedlessMetalForward.profSkipMixer = true          // GDN body + attention body
            SeedlessMetalForward.profSkipMoEExperts = true     // routed+shared+combine gather
            SeedlessMetalForward.profSkipSingleThread = true   // route_top8/shared_gate8
        }
        allOff(); _ = runPrefill()                             // warmup: compile all pipelines

        let tFull = best {}
        // Floor swept over chunk size: if floor DROPS with bigger chunk → per-chunk dispatch/encode
        // overhead (amortizable by larger chunk). If FLAT → per-token ops (norms/gate/KV, not amortizable).
        var floorLines: [String] = []
        for fc in [chunk, 256, 1024] {
            let savedChunk = chunk
            _ = savedChunk
            // re-run floor at this chunk by temporarily overriding the closure's chunk via a local loop
            func runAt(_ c: Int) -> Double {
                guard let bk = Tell.fusedBackend(engine: engine, maxM: c + 8, maxSeqLen: promptLen + 128) else { return -1 }
                let t0 = Date(); var pos = 0
                while pos < promptLen { let e = Swift.min(pos + c, promptLen); _ = bk.forward(Array(prompt[pos ..< e])); pos = e }
                return Date().timeIntervalSince(t0)
            }
            allOff(); floorSet(); let a = runAt(fc); let b = runAt(fc); allOff()
            let f = Swift.min(a, b)
            floorLines.append(String(format: "  floor @chunk=%4d        %6.1fs  %5.1f%% of full   (%d chunks)",
                                     fc, f, 100 * f / tFull, (promptLen + fc - 1) / fc))
        }
        // rough single-component split at the production chunk (noisy — min-of-2, may not be additive).
        let tMixer = best { SeedlessMetalForward.profSkipMixer = true }
        let tMoE   = best { SeedlessMetalForward.profSkipMoEExperts = true }
        let tRoute = best { SeedlessMetalForward.profSkipSingleThread = true }
        func pct(_ c: Double) -> String { String(format: "%6.1fs  %5.1f%%", c, 100 * c / tFull) }
        var out = ["[prefill-bd] promptLen=\(promptLen) chunk=\(chunk) resident  (min-of-2)",
                   String(format: "  FULL prefill            %6.1fs  (%.0f tok/s)", tFull, Double(promptLen) / tFull),
                   "  ── rough component (t_full − t_skip, noisy) ──",
                   "  all mixers (GDN+attn)   \(pct(tFull - tMixer))",
                   "  MoE experts (all)       \(pct(tFull - tMoE))",
                   "  routing single-thread   \(pct(tFull - tRoute))",
                   "  ── FLOOR (all heavy compute skipped), swept over chunk ──"]
        out += floorLines
        out.append("PREFILLBD done")
        return out.joined(separator: "\n")
    }

    // End-to-end lossless gate for the Design-B + multi-slot prefix cache: drives SeedlessBackend.generate
    // through a scripted request sequence with the cache ON (warm: reuse/extend/cross-branch/reset) and
    // compares each token stream to the same request generated with the cache OFF (segmented cold path).
    // Byte-identical everywhere ⇒ the multi-slot restore + stride re-prefill + arena growth are lossless.
    public static func prefixCacheE2E(modelDir: String) -> String {
        guard let backend = try? SeedlessBackend(modelDir: modelDir) else { return "[prefix-e2e] load fail\nPREFIXE2E FAIL" }
        // stride small so a shared 256-token prefix produces reusable sub-content boundaries.
        setenv("QWISP_PREFIX_SNAP_STRIDE", "64", 1)

        func toks(_ n: Int, _ salt: Int) -> [Int] { (0..<n).map { (($0 &* 7 &+ salt) % 5000) + 100 } }
        let A = toks(256, 13)                       // shared "system+tools" prefix
        // (content, contentLen) per request; full prompt = content + 8-token gen-prompt suffix.
        func req(_ content: [Int]) -> (prompt: [Int], cl: Int) { (content + toks(8, 777), content.count) }
        let c1 = A + toks(32, 1)                     // R1: A + userX
        let c2 = c1 + toks(40, 2)                    // R2: extends R1 (intra-conversation)
        let c3 = A + toks(32, 3)                     // R3: A + userZ (cross-conversation: shares A)
        let c4 = toks(200, 999)                      // R4: shares nothing → cold reset (restore empty)
        let reqs = [req(c1), req(c2), req(c3), req(c4)]
        let maxTok = 24

        final class Box: @unchecked Sendable { var v: [Int] = [] }
        func gen(_ p: [Int], _ cl: Int) -> [Int] {
            let box = Box()
            let sem = DispatchSemaphore(value: 0)
            let stream = backend.generate(p, options: GenerateOptions(maxTokens: maxTok, promptContentLen: cl))
            Task { for await t in stream { box.v.append(t) }; sem.signal() }
            sem.wait()
            return box.v
        }

        // Reference: cache OFF (segmented cold path) — each request independent.
        backend.prefixCacheForced = false
        let ref = reqs.map { gen($0.prompt, $0.cl) }
        // Cached: cache ON, cold start, requests in sequence so the cache warms/reuses across them.
        backend.prefixCacheForced = true
        backend.resetPrefixCache()
        let got = reqs.map { gen($0.prompt, $0.cl) }

        var lines = ["[prefix-e2e] 4 requests (reuse/extend/cross-branch/reset), maxTok=\(maxTok)"]
        var pass = true
        let labels = ["R1 cold      ", "R2 extend    ", "R3 cross-conv", "R4 reset     "]
        for i in 0..<reqs.count {
            let ok = ref[i] == got[i]
            pass = pass && ok
            lines.append("  \(labels[i])  ref=\(ref[i].count)tok  cached=\(got[i].count)tok  \(ok ? "IDENTICAL ✅" : "DIVERGE ❌")")
            if !ok { lines.append("    ref   : \(ref[i].prefix(12))"); lines.append("    cached: \(got[i].prefix(12))") }
        }
        lines.append("PREFIXE2E \(pass ? "PASS" : "FAIL")")
        return lines.joined(separator: "\n")
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
