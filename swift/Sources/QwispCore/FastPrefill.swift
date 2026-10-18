import Foundation
import Metal
import MLX
import MLXFast
import Cmlx

/// Non-strict fast prefill (QWISP_FAST_PREFILL=1, resident tier).
///
/// The strict hybrid prefill (forwardRowsHybrid) syncs CPU↔GPU 4-5 times per layer and routes the
/// MoE through R=16-padded steel tiles; on an M3 Pro it runs at ~1/4 of mlx-lm's prefill rate on
/// the same checkpoint. This path instead evaluates each chunk as ONE lazy MLX graph over all 40
/// layers (mlx-lm structure: SDPA attention, GatedDelta Metal kernel, expert-sorted gather_qmm),
/// then hands the resulting KV rows / GDN states to the Seedless buffers so decode continues on
/// the raw engine unchanged.
///
/// NOT lossless: prefill numerics differ from the canonical strict stream (different kernels,
/// accumulation order). Decode itself is the same Seedless path. Off by default.
public final class FastPrefill {
    public static let enabled = ProcessInfo.processInfo.environment["QWISP_FAST_PREFILL"] == "1"
    /// Tokens per MLX graph (mlx-lm default prefill_step_size = 2048).
    public static let chunk = Swift.max(64, Tell.envInt("QWISP_FAST_PREFILL_CHUNK", 2048))
    /// QWISP_FAST_PREFILL_CHECK=1: per chunk, also run the strict hybrid forward from the same
    /// state and print the hidden / cache deviation (measurement only; ~2x prefill cost).
    static let check = ProcessInfo.processInfo.environment["QWISP_FAST_PREFILL_CHECK"] == "1"

    /// Wire MLX allocations so macOS cannot compress/evict the resident weights. Measured on a
    /// 36 GB M3 Pro under ordinary desktop load: unwired, 17 of the process's 20 GB sat in the
    /// memory compressor and every GPU touch paid decompression (prefill 2-5x slower, erratic).
    /// mlx-lm does the same per generate() (maybe_set_recommended_wired_limit), but at the full
    /// recommended working set; here the limit is sized to weights + margin so wiring does not
    /// starve the rest of the system (a 28 GB wire collapsed long-context prefill).
    ///   QWISP_MLX_WIRED  0 → off; 1 → auto (weights + QWISP_MLX_WIRED_MARGIN_MB, default 4096,
    ///   capped at the recommended working set); N > 1 → N MB. Memory policy only; numerics untouched.
    nonisolated(unsafe) static var wiredLimit = 0
    /// Resolves the limit once at load; it is applied only around prefill (wiredBegin/End):
    /// measured on the same machine, keeping it wired through decode made the raw-Metal decode
    /// ~20% slower (21.9 vs 17.9 ms/token), while prefill gained 20-35%.
    public static func applyWiredLimit(store: WeightStore) {
        let mode = Tell.envInt("QWISP_MLX_WIRED", 0)
        guard mode != 0, let dev = MTLCreateSystemDefaultDevice() else { return }
        let rec = Int(dev.recommendedMaxWorkingSetSize)
        let weights = store.arrays.values.reduce(0) { $0 + $1.nbytes }
        let margin = Tell.envInt("QWISP_MLX_WIRED_MARGIN_MB", 4096) * 1_048_576
        wiredLimit = mode > 1 ? mode * 1_048_576 : Swift.min(rec, weights + margin)
        FileHandle.standardError.write(Data(String(format: "[qwisp] MLX wired limit %.1f GB during prefill (weights %.1f GB, recommended %.1f GB)\n",
            Double(wiredLimit) / 1073741824, Double(weights) / 1073741824, Double(rec) / 1073741824).utf8))
    }
    static func wiredBegin() {
        guard wiredLimit > 0 else { return }
        var prev = 0; mlx_set_wired_limit(&prev, wiredLimit)
    }
    static func wiredEnd() {
        guard wiredLimit > 0 else { return }
        var prev = 0; mlx_set_wired_limit(&prev, 0)
    }

    nonisolated(unsafe) static var model: QwispModel?
    nonisolated(unsafe) static var normW: MLXArray?

    /// Called once by the backend after the WeightStore is resident.
    static let cachePolicyMB = Tell.envInt("QWISP_FAST_PREFILL_CACHE_MB", 2048)

    public static func install(store: WeightStore) {
        guard enabled else { return }
        if cachePolicyMB > 0 { MLX.Memory.cacheLimit = cachePolicyMB * 1_048_576 }
        model = QwispModel(store: store)
        normW = store.req("language_model.model.norm.weight")
        MoEBlock.sortedGather = true
        GatedDelta.usePacked = Tell.envInt("QWISP_FAST_GDN_PACKED", 1) != 0
        GatedDeltaNetLayer.fuseRMSGated = Tell.envInt("QWISP_FAST_RMSGATED", 1) != 0
        AttentionLayer.prefillQueryBlock = Swift.max(0, Tell.envInt("QWISP_FAST_ATTN_QBLOCK", 512))
        FileHandle.standardError.write(Data(
            "[qwisp] fast prefill ON (non-strict: MLX graph prefill, chunk \(chunk)) — prefill numerics differ from strict\n".utf8))
    }

    let model: QwispModel
    let normW: MLXArray
    let fwd: SeedlessFusedVerify.SeedlessFusedForward
    let reference: (([Int32]) -> MLXArray?)?

    init?(fwd: SeedlessFusedVerify.SeedlessFusedForward, reference: (([Int32]) -> MLXArray?)?) {
        guard let m = FastPrefill.model, let n = FastPrefill.normW else { return nil }
        self.model = m; self.normW = n; self.fwd = fwd; self.reference = reference
    }

    /// Same contract as forwardRowsHybrid: append `tokens` to the Seedless caches and return the
    /// post-final-norm hidden rows [M, H].
    func forward(_ tokens: [Int32]) -> MLXArray? {
        if FastPrefill.check, let ref = reference, tokens.count <= fwd.maxM { return checked(tokens, ref) }
        return run(tokens)
    }

    // MLX-side caches kept across the chunks of one prefill, so each chunk does not re-import
    // the whole KV. Reused only while the Seedless state is still exactly what this object last
    // exported (stamp: attention length + sample bytes of the state buffers); anything that
    // rewound or restored the engine in between (prefix cache, decode) forces a re-import.
    private var mirror: [LayerCache]? = nil
    private var mirrorStamp: [UInt8] = []

    private func stamp() -> [UInt8] {
        var out: [UInt8] = []
        for L in fwd.layers {
            if let kv = L.kvCache {
                withUnsafeBytes(of: kv.len) { out += $0 }
                if kv.len > 0 {
                    let row = UnsafeRawBufferPointer(start: kv.kCache.contents() + (kv.len - 1) * kv.D * 2, count: 64)
                    out += row
                }
                break
            }
        }
        if let g = fwd.layers.first(where: { $0.gdnCache != nil })?.gdnCache {
            out += UnsafeRawBufferPointer(start: g.state.contents(), count: 256)
            out += UnsafeRawBufferPointer(start: g.convHist.contents(), count: 64)
        }
        return out
    }

    /// End of a Tell.prefill call: drop the MLX-side mirror so decode does not carry it.
    func prefillEnded() {
        mirror = nil; mirrorStamp = []
        MLX.Memory.clearCache()
    }

    private func run(_ tokens: [Int32]) -> MLXArray? {
        fwd.cancelPendingChain()
        let M = tokens.count
        let t0 = DispatchTime.now().uptimeNanoseconds
        let caches: [LayerCache]
        let reused: Bool
        if let m = mirror, mirrorStamp == stamp() { caches = m; reused = true } else { caches = importState(); reused = false }
        mirror = nil
        let t1 = DispatchTime.now().uptimeNanoseconds
        var h = model.embed(MLXArray(tokens).reshaped([1, M]))
        for (i, layer) in model.layers.enumerated() { h = layer(h, cache: caches[i]) }
        let hidden = MLXFast.rmsNorm(h, weight: normW, eps: model.eps).reshaped([M, -1])
        MLX.eval([hidden] + caches.flatMap { $0.stateArrays })
        let t2 = DispatchTime.now().uptimeNanoseconds
        guard exportState(caches, M: M) else { return nil }
        if SeedlessFusedVerify.SeedlessFusedForward.prefillProf {
            let t3 = DispatchTime.now().uptimeNanoseconds
            FileHandle.standardError.write(Data(String(format: "[fast-prefill-prof] M=%d import %.0fms%@ graph %.0fms export %.0fms (%.0f tok/s graph-only)\n",
                M, Double(t1 - t0) / 1e6, reused ? "(mirror)" : "", Double(t2 - t1) / 1e6, Double(t3 - t2) / 1e6,
                Double(M) * 1e9 / Double(t2 - t1)).utf8))
        }
        mirror = caches; mirrorStamp = stamp()
        // Left alone the MLX free-buffer pool grew ~8 GB here and pinned the process at the
        // device's recommended working set (mlx-lm clears it every chunk). Policy:
        // QWISP_FAST_PREFILL_CACHE_MB  <0 → clearCache() per chunk; 0 → leave; >0 → cap (default 2048).
        if FastPrefill.cachePolicyMB < 0 { MLX.Memory.clearCache() }
        return hidden
    }

    // ── Seedless → MLX ──────────────────────────────────────────────────────────────────────
    private func importState() -> [LayerCache] {
        fwd.layers.map { L in
            let c = LayerCache()
            if let kv = L.kvCache, kv.len > 0 {
                _ = c.kv.update(FastPrefill.readRows(kv.kCache, kv), FastPrefill.readRows(kv.vCache, kv))
            }
            if let g = L.gdnCache {
                let (conv, rec) = SeedlessFusedVerify.readGdnCache(g)      // [K-1, C], [1, Hv, Dv, Dk]
                c.gdn.convState = conv.reshaped([1, g.K - 1, g.C])
                c.gdn.recState = rec
            }
            return c
        }
    }

    /// First `kv.len` rows of a [KV, maxLen, D] f16 Seedless cache → MLX [1, KV, len, D].
    static func readRows(_ buf: MTLBuffer, _ kv: SeedlessFusedVerify.KVCacheBufs) -> MLXArray {
        let (KV, len, D, maxLen) = (kv.KV, kv.len, kv.D, kv.maxLen)
        var a = [Float16](repeating: 0, count: KV * len * D)
        a.withUnsafeMutableBytes { dst in
            for h in 0 ..< KV {
                memcpy(dst.baseAddress! + h * len * D * 2, buf.contents() + h * maxLen * D * 2, len * D * 2)
            }
        }
        return MLXArray(a, [1, KV, len, D])
    }

    // ── MLX → Seedless ──────────────────────────────────────────────────────────────────────
    private func exportState(_ caches: [LayerCache], M: Int) -> Bool {
        for (i, L) in fwd.layers.enumerated() {
            let c = caches[i]
            if let kv = L.kvCache {
                guard let k = c.kv.keys, let v = c.kv.values, kv.len + M <= kv.maxLen else { return false }
                FastPrefill.writeRows(k, kv.kCache, kv, from: kv.len, count: M)
                FastPrefill.writeRows(v, kv.vCache, kv, from: kv.len, count: M)
                kv.len += M
            }
            if let g = L.gdnCache, let conv = c.gdn.convState, let rec = c.gdn.recState {
                SeedlessFusedVerify.writeGdnCache(g, conv: conv.reshaped([g.K - 1, g.C]), rec: rec)
            }
        }
        return true
    }

    /// Rows [from, from+count) of an MLX [1, KV, from+count, D] cache → Seedless [KV, maxLen, D].
    static func writeRows(_ a: MLXArray, _ buf: MTLBuffer, _ kv: SeedlessFusedVerify.KVCacheBufs,
                          from: Int, count: Int) {
        let (KV, D, maxLen) = (kv.KV, kv.D, kv.maxLen)
        let rows = a[0, 0..., from ..< (from + count), 0...].asType(.float16)   // [KV, count, D]
        let arr = rows.asArray(Float16.self)
        arr.withUnsafeBytes { src in
            for h in 0 ..< KV {
                memcpy(buf.contents() + (h * maxLen + from) * D * 2, src.baseAddress! + h * count * D * 2, count * D * 2)
            }
        }
    }

    // ── measurement: deviation vs the strict hybrid forward from the same starting state ────
    private func checked(_ tokens: [Int32], _ ref: ([Int32]) -> MLXArray?) -> MLXArray? {
        let snap = fwd.fullSnapshot()
        let t0 = Date()
        guard let hRef = ref(tokens) else { return nil }
        let tRef = Date().timeIntervalSince(t0)
        let refState = fwd.layers.map { L -> (MLXArray?, MLXArray?) in
            if let kv = L.kvCache { let (k, v) = SeedlessFusedVerify.readKVCache(kv); return (k, v) }
            if let g = L.gdnCache { let (c, r) = SeedlessFusedVerify.readGdnCache(g); return (c, r) }
            return (nil, nil)
        }
        fwd.fullRestore(snap)
        let t1 = Date()
        guard let hFast = run(tokens) else { return nil }
        let tFast = Date().timeIntervalSince(t1)
        // ‖a−b‖ / ‖b‖ (Frobenius): robust to the few fp16 outlier channels a max-abs ratio amplifies
        func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
            let d = (a.asType(.float32) - b.asType(.float32)).square().sum().sqrt().item(Float.self)
            return d / (b.asType(.float32).square().sum().sqrt().item(Float.self) + 1e-9)
        }
        var kMax: Float = 0, vMax: Float = 0, convMax: Float = 0, recMax: Float = 0
        for (i, L) in fwd.layers.enumerated() {
            if let kv = L.kvCache, let rk = refState[i].0, let rv = refState[i].1 {
                let (k, v) = SeedlessFusedVerify.readKVCache(kv)
                kMax = Swift.max(kMax, rel(k, rk)); vMax = Swift.max(vMax, rel(v, rv))
            }
            if let g = L.gdnCache, let rc = refState[i].0, let rr = refState[i].1 {
                let (cc, rs) = SeedlessFusedVerify.readGdnCache(g)
                convMax = Swift.max(convMax, rel(cc, rc)); recMax = Swift.max(recMax, rel(rs, rr))
            }
        }
        let M = tokens.count
        let lastRel = rel(hFast[M - 1], hRef[M - 1])
        if let dump = ProcessInfo.processInfo.environment["QWISP_FAST_PREFILL_DUMP"], M <= 64,
           !FileManager.default.fileExists(atPath: dump) {
            try? MLX.save(arrays: ["tokens": MLXArray(tokens), "strict": hRef.asType(.float32),
                                   "fast": hFast.asType(.float32)], url: URL(fileURLWithPath: dump))
        }
        FileHandle.standardError.write(Data(String(format:
            "[fast-prefill-check] M=%d max-layer rel-L2: hidden(last) %.2e  k %.2e  v %.2e  conv %.2e  rec %.2e  | strict %.0f tok/s  fast %.0f tok/s\n",
            M, lastRel, kMax, vMax, convMax, recMax, Double(M) / tRef, Double(M) / tFast).utf8))
        return hFast
    }
}
