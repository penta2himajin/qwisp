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

        // v2: split WALL vs GPU-busy (profLastGPUMs per CB) per config. The decisive question:
        // is prefill time GPU execution (kernel-level, capped ~22% by the skip test) or CPU-side
        // gap (MLX embed eval + hBuf upload + normed readback + commit/wait per chunk) — the
        // latter is a scheduling prize, lossless by construction (no math change).
        func runAt(_ c: Int) -> (wall: Double, gpu: Double) {
            guard let bk = Tell.fusedBackend(engine: engine, maxM: c + 8, maxSeqLen: promptLen + 128) else { return (-1, 0) }
            var gpu = 0.0
            let t0 = Date(); var pos = 0
            while pos < promptLen {
                let e = Swift.min(pos + c, promptLen)
                _ = bk.forward(Array(prompt[pos ..< e]))
                gpu += SeedlessFusedVerify.SeedlessFusedForward.profLastGPUMs / 1000.0
                pos = e
            }
            return (Date().timeIntervalSince(t0), gpu)
        }
        func floorSet() {   // skip ALL heavy GPU compute → what remains is the irreducible floor
            SeedlessMetalForward.profSkipMixer = true          // GDN body + attention body
            SeedlessMetalForward.profSkipMoEExperts = true     // routed+shared+combine gather
            SeedlessMetalForward.profSkipSingleThread = true   // route_top8/shared_gate8
        }
        allOff(); _ = runAt(chunk)                             // warmup: compile all pipelines
        _ = runPrefill                                          // (kept for API symmetry)

        var lines = ["[prefill-bd] promptLen=\(promptLen) resident  — wall vs GPU-busy per config"]
        func row(_ label: String, _ c: Int, floored: Bool) -> Double {
            allOff(); if floored { floorSet() }
            let a = runAt(c), b = runAt(c)
            allOff()
            let r = a.wall < b.wall ? a : b
            lines.append(String(format: "  %@ chunk=%4d   wall %6.1fs   gpu %6.1fs   cpu-gap %6.1fs (%4.1f%%)   %.0f tok/s",
                                label, c, r.wall, r.gpu, r.wall - r.gpu, 100 * (r.wall - r.gpu) / r.wall,
                                Double(promptLen) / r.wall))
            return r.wall
        }
        _ = row("FULL ", chunk, floored: false)
        _ = row("FULL ", 256, floored: false)
        _ = row("floor", chunk, floored: true)
        _ = row("floor", 256, floored: true)
        _ = row("floor", 1024, floored: true)
        lines.append("PREFILLBD done")
        return lines.joined(separator: "\n")
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

    // Speed probe for the multi-slot cache: TTFT (time-to-first-token) for a cold request vs a
    // cross-conversation request that shares a large prefix vs an intra-conversation extend.
    // Shows the multi-slot win = skipping re-prefill of the shared system+tools prefix on a NEW convo.
    // Env: QWISP_PREFIX_SHARED (shared prefix length, default 4096).
    public static func prefixCacheSpeedProbe(modelDir: String) -> String {
        guard let backend = try? SeedlessBackend(modelDir: modelDir) else { return "[prefix-speed] load fail\nPREFIXSPEED done" }
        setenv("QWISP_PREFIX_SNAP_STRIDE", "2048", 1)
        let P = Tell.envInt("QWISP_PREFIX_SHARED", 4096)
        func toks(_ n: Int, _ salt: Int) -> [Int] { (0..<n).map { (($0 &* 7 &+ salt) % 5000) + 100 } }
        let shared = toks(P, 13)
        func req(_ content: [Int]) -> (p: [Int], cl: Int) { (content + toks(8, 777), content.count) }
        let r1 = req(shared + toks(64, 1))                       // cold: cache empty
        let r2 = req(shared + toks(64, 2))                       // cross-conversation: shares `shared`
        let r3 = req(shared + toks(64, 2) + toks(80, 3))         // intra-conversation: extends r2

        final class Box: @unchecked Sendable { var v: [Int] = []; var ttft = 0.0 }
        func ttft(_ p: [Int], _ cl: Int) -> Double {
            let box = Box(); let sem = DispatchSemaphore(value: 0); let t0 = Date()
            let stream = backend.generate(p, options: GenerateOptions(maxTokens: 4, promptContentLen: cl))
            Task { for await t in stream { if box.v.isEmpty { box.ttft = Date().timeIntervalSince(t0) }; box.v.append(t) }; sem.signal() }
            sem.wait(); return box.ttft
        }

        backend.prefixCacheForced = true
        backend.resetPrefixCache()
        let cold  = ttft(r1.p, r1.cl)     // cache empty → full prefill of `shared`+tail
        let cross = ttft(r2.p, r2.cl)     // multi-slot restores a boundary inside `shared`
        let intra = ttft(r3.p, r3.cl)     // extends r2 → restores the top boundary
        func fmt(_ s: Double) -> String { String(format: "%.2fs", s) }
        return ["[prefix-speed] shared prefix=\(P) tok, resident/fused, greedy",
                "  R1 cold (cache empty)      TTFT \(fmt(cold))",
                String(format: "  R2 cross-conversation      TTFT %@   (%.1fx vs cold)", fmt(cross), cold / max(cross, 1e-3)),
                String(format: "  R3 intra-conversation      TTFT %@   (%.1fx vs cold)", fmt(intra), cold / max(intra, 1e-3)),
                "PREFIXSPEED done"].joined(separator: "\n")
    }

    // Prefill stage profile (kernel-speedup recon): ① raw path — per-stage GPU time via
    // forwardRowsProfiled (CB-split, exact math) bucketed into GDN/attn/MoE, swept over chunk size.
    // ② MLX path — same prompt chunk-prefilled through QwispModel (MLX gather_qmm uses matrix-unit
    // kernels at M>=14), as free evidence of what a GEMM-structured MoE kernel buys at prefill M.
    // Env: QWISP_PREFILL_LEN (default 2048).
    public static func prefillStageProfile(modelDir: String) -> String {
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[stage-prof] load fail\nSTAGEPROF done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let promptLen = Tell.envInt("QWISP_PREFILL_LEN", 2048)
        let promptI32 = (0..<promptLen).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }
        var lines = ["[stage-prof] promptLen=\(promptLen) resident"]

        // ① raw per-stage GPU time, chunk sweep
        lines.append("  ── raw path: per-stage GPU ms (CB-split, exact) ──")
        for chunk in [64, 256, 1024] {
            guard let (fwd, _) = engine.makeFused(maxM: chunk + 8, maxSeqLen: promptLen + 128) else {
                lines.append("  chunk=\(chunk): makeFused nil"); continue
            }
            var g = 0.0, a = 0.0, m = 0.0
            let t0 = Date(); var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + chunk, promptLen)
                let x = engine.embed(tokens: Array(promptI32[pos ..< end]))
                guard let t = fwd.forwardRowsProfiled(x, M: end - pos) else { break }
                g += t.gdn; a += t.attn; m += t.moe
                pos = end
            }
            let wall = Date().timeIntervalSince(t0)
            let tot = g + a + m
            lines.append(String(format: "  chunk=%4d  GDN %6.2fs (%4.1f%%)  attn %6.2fs (%4.1f%%)  MoE %6.2fs (%4.1f%%)  | stageSum %5.1fs  wall %5.1fs  %.0f tok/s",
                                chunk, g/1000, 100*g/tot, a/1000, 100*a/tot, m/1000, 100*m/tot,
                                tot/1000, wall, Double(promptLen)/wall))
        }

        // ② MLX path (matrix-unit gather_qmm at M>=14): same prompt, chunked cached prefill
        lines.append("  ── MLX path (QwispModel, gather_qmm matrix-unit) ──")
        let model = QwispModel(store: store)
        let ids = promptI32.map { Int($0) }
        for chunk in [64, 256] {
            let caches = model.makeCaches()
            let t0 = Date(); var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + chunk, promptLen)
                let (hidden, _) = model.forwardHidden(MLXArray(ids[pos ..< end].map { Int32($0) }).reshaped([1, end - pos]),
                                                      caches: caches)
                MLX.eval(hidden)
                pos = end
            }
            let wall = Date().timeIntervalSince(t0)
            lines.append(String(format: "  chunk=%4d  wall %5.1fs  %.0f tok/s", chunk, wall, Double(promptLen)/wall))
        }
        // MLX GDN stage attribution (GatedDeltaNetLayer prof counters; eval barriers inflate wall,
        // stage sums still attribute where MLX spends its GDN time vs our sequential T-loop kernel).
        StreamingMoEBlock.profileLayers = true
        StreamingMoEBlock.tGdnInproj = 0; StreamingMoEBlock.tGdnConv = 0
        StreamingMoEBlock.tGdnKernel = 0; StreamingMoEBlock.tGdnOut = 0
        do {
            let caches = model.makeCaches()
            let t0 = Date(); var pos = 0
            while pos < promptLen {
                let end = Swift.min(pos + 64, promptLen)
                let (hidden, _) = model.forwardHidden(MLXArray(ids[pos ..< end].map { Int32($0) }).reshaped([1, end - pos]),
                                                      caches: caches)
                MLX.eval(hidden)
                pos = end
            }
            let wall = Date().timeIntervalSince(t0)
            func s(_ ns: UInt64) -> Double { Double(ns) / 1e9 }
            lines.append(String(format: "  MLX GDN stages (c64, barriered wall %.1fs): inproj %.2fs  conv %.2fs  recur-kernel %.2fs  out %.2fs  | GDN total %.2fs",
                                wall, s(StreamingMoEBlock.tGdnInproj), s(StreamingMoEBlock.tGdnConv),
                                s(StreamingMoEBlock.tGdnKernel), s(StreamingMoEBlock.tGdnOut),
                                s(StreamingMoEBlock.tGdnInproj + StreamingMoEBlock.tGdnConv + StreamingMoEBlock.tGdnKernel + StreamingMoEBlock.tGdnOut)))
        }
        StreamingMoEBlock.profileLayers = false
        lines.append("STAGEPROF done")
        return lines.joined(separator: "\n")
    }

    // Verification ① for the MLX-dense-prefill route: per-element M-invariance of MLX
    // quantizedMatmul. Row r's output must be bit-identical no matter how many other rows share
    // the call (and under zero-padding), or split-point-invariant prefill via MLX dense is dead.
    // MLX dispatches qmv (M<14, scalar) vs steel qmm (M>=14, matrix units) — the padding variant
    // tests whether pinning ALL prefill matmuls to the steel class (pad M up to >=14) is coherent.
    public static func mlxQmmInvariance(modelDir: String) -> String {
        var lines = ["[mlx-qmm-minv] MLX quantizedMatmul per-element M-invariance (value-equality per row)"]
        for (label, K, N) in [("2048→512", 2048, 512), ("2048→12352", 2048, 12352)] {
            let wf = MLXRandom.normal([N, K]).asType(.float16)
            let (wq, sc, biOpt) = MLX.quantized(wf, groupSize: 64, bits: 4, mode: .affine)
            guard let bi = biOpt else { return "[mlx-qmm-minv] biases nil" }
            let xAll = MLXRandom.normal([1024, K]).asType(.float16)
            MLX.eval([wq, sc, bi, xAll])
            func qmm(_ x: MLXArray) -> MLXArray {
                let y = MLX.quantizedMatmul(x, wq, scales: sc, biases: bi, transpose: true, groupSize: 64, bits: 4)
                y.eval(); return y
            }
            let full = qmm(xAll)                                   // M=1024 reference
            func rowsEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
                let d = MLX.all(a .== b)
                d.eval(); return d.item(Bool.self)
            }
            lines.append("  ── \(label) ──")
            for M in [1, 8, 13, 14, 16, 64, 256] {
                let out = qmm(xAll[0..<M])
                let ok = rowsEqual(out, full[0..<M])
                lines.append("    M=\(M)\tvs M=1024 rows[0..<\(M)]: \(ok ? "IDENTICAL ✅" : "DIFFER ❌")")
            }
            // zero-pad variant: 3 real rows + 13 zero rows = M=16 call; real rows vs full
            let pad = MLX.concatenated([xAll[0..<3], MLXArray.zeros([13, K]).asType(.float16)], axis: 0)
            let outPad = qmm(pad)
            let okPad = rowsEqual(outPad[0..<3], full[0..<3])
            lines.append("    M=3+13pad(=16)\tvs M=1024 rows[0..<3]: \(okPad ? "IDENTICAL ✅" : "DIFFER ❌")")
        }
        lines.append("MLXQMMMINV done")
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
