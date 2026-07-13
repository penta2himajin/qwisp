import Foundation
import MLX

// Product facade (productization step 2, strict-first).
//
// LLMBackend is the coarse MLX-compat surface the server swaps backends over.
// It operates on token IDs `[Int]`; the tokenizer + chat template live ABOVE it
// in the server layer. SeedlessBackend wraps the EXISTING shipped strict decode
// (Tell.runSpecLoop + engine/backend builders) — it is the keep-set
// anchor for delete-all. Bolt (runBoltMode) is entangled with measurement code
// and stays an explicit keep-list entry until steps 3-4 thin it (see AGENTS.md /
// HANDOFF.md).

/// RAM/quality tier. `.auto` resolves C from device RAM (DeviceCalibration.defaultC()).
public enum SeedlessTier {
    case auto
    case resident            // strict, all experts resident (≥32GB grain: C≥256)
    case streaming(c: Int)   // strict streaming, C-slot expert arena (<32GB grain: 0<C<256)
}

public struct GenerateOptions {
    public var maxTokens: Int
    /// Token ids that halt decoding (EOS / chat turn-end). Empty = run to maxTokens.
    /// A stop token is NOT emitted; the runtime stops before yielding it.
    public var stopTokens: [Int]
    /// Sampling (Option B prototype). temperature 0 → greedy/lossless (default); >0 → sample.
    /// topP < 1 → nucleus truncation. Penalties + logitBias reshape the logits before sampling.
    /// `sampling` is true when any of these shape the distribution (so it leaves the greedy path).
    public var temperature: Double
    public var topP: Double
    public var seed: UInt64
    public var frequencyPenalty: Double
    public var presencePenalty: Double
    public var logitBias: [Int: Double]
    /// Prefix-cache boundary: number of leading prompt tokens that are stable "content" (i.e. the
    /// prompt WITHOUT the trailing generation-prompt suffix). The cross-request cache snapshots at
    /// this position so the next request re-prefills only the new content. nil → no caching.
    public var promptContentLen: Int? = nil
    public var sampling: Bool {
        temperature > 0 || topP < 1.0 || frequencyPenalty != 0 || presencePenalty != 0 || !logitBias.isEmpty
    }
    public init(maxTokens: Int = 128, stopTokens: [Int] = [],
                temperature: Double = 0, topP: Double = 1.0, seed: UInt64 = 0,
                frequencyPenalty: Double = 0, presencePenalty: Double = 0, logitBias: [Int: Double] = [:],
                promptContentLen: Int? = nil) {
        self.maxTokens = maxTokens
        self.stopTokens = stopTokens
        self.temperature = temperature
        self.topP = topP
        self.seed = seed
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.logitBias = logitBias
        self.promptContentLen = promptContentLen
    }
}

/// Coarse backend protocol: load + generate + tier. Both a future MLXBackend and
/// SeedlessBackend conform, so the server picks a backend in one line.
public protocol LLMBackend {
    init(modelDir: String, tier: SeedlessTier) throws
    func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int>
}

// ponytail: SeedlessBackend is a thin LLMBackend conformer that holds the built
// engine and delegates decode to the `Tell` runtime (Tell.runSpecLoop etc.).
// The two layers (this + Tell) could collapse into one `final class Tell: LLMBackend`
// (net −1 type), but only once a forcing function appears — step-5 server wiring
// making the indirection bite, or per-instance state the static runtime can't thread.
// Until then the split works and is gate-tested; don't restructure for tidiness.
public final class SeedlessBackend: LLMBackend, @unchecked Sendable {

    /// Pure, GPU-free sizing seam. Mirrors Tell.run()'s tier arithmetic so
    /// the facade sizes the backend identically to the shipped strict path.
    struct Config: Equatable {
        var isStreaming: Bool
        var c: Int
        var maxK: Int
        var maxM: Int
        var maxSeqLen: Int
    }

    static func config(tier: SeedlessTier, promptLen: Int, maxTokens: Int) -> Config {
        let c: Int
        switch tier {
        case .auto:            c = DeviceCalibration.defaultC()
        case .resident:        c = 256                    // resident grain (C≥256)
        case .streaming(let x): c = x
        }
        let isStreaming = c > 0 && c < 256
        let maxK = isStreaming ? Swift.max(4, c * 3 / 8) : 96
        let pendingCap = 24
        let maxM = Swift.max(pendingCap + maxK + 1, 64)
        let maxSeqLen = promptLen + maxTokens + maxK + 64
        return Config(isStreaming: isStreaming, c: c, maxK: maxK, maxM: maxM, maxSeqLen: maxSeqLen)
    }

    let store: WeightStore
    let engine: SeedlessEngine
    /// --lossless / QWISP_LOSSLESS: force strict (bit-exact) on every tier. Streaming
    /// tiers (<32GB) otherwise default to bolt (near-lossless L3) — productization
    /// tier→mode decision. Set by the CLI after init (resolved flag > env > config).
    public var losslessForced: Bool? = nil
    public var lossless: Bool { losslessForced ?? (ProcessInfo.processInfo.environment["QWISP_LOSSLESS"] == "1") }
    private var boltServe: BoltServe? = nil
    let modelDir: String
    let tier: SeedlessTier
    let contextLen: Int      // model context window (max_position_embeddings); caps unbounded generation.

    // ── Cross-request prefix cache (default ON; QWISP_PREFIX_CACHE=0 opts out) ──────────
    // Persistent per-instance fused backend + a content-boundary fullSnapshot, so consecutive
    // requests (agentic loops) re-prefill only the new suffix. Serialized by the server lock.
    // Design B: the arena grows monotonically to fit each request (prompt+generation) instead of a
    // fixed generation cap, so cached mode never truncates a long answer the segmented path would allow.
    // Multi-slot: several stride-aligned restore points along the current path let a NEW conversation
    // reuse a shared prefix (system+tools) instead of re-prefilling it. Lossless — SuffixSpec verifies
    // every drafted token; snapshot/restore is byte-identical (PrefixCachePoC). ~60MB / slot (GDN state).
    public var prefixCacheForced: Bool? = nil      // test/override hook; nil → env flag
    // Default ON (lossless: PREFIXE2E gate; growth removed the truncation risk). QWISP_PREFIX_CACHE=0 opts out.
    var prefixCacheEnabled: Bool { prefixCacheForced ?? (ProcessInfo.processInfo.environment["QWISP_PREFIX_CACHE"] != "0") }
    var prefixArenaMax: Int { Swift.min(contextLen, Swift.max(4096, Tell.envInt("QWISP_PREFIX_MAX", 65536))) }
    var prefixSnapStride: Int { Swift.max(512, Tell.envInt("QWISP_PREFIX_SNAP_STRIDE", 2048)) }
    var prefixMaxSlots: Int { Swift.max(2, Tell.envInt("QWISP_PREFIX_MAX_SLOTS", 6)) }
    private var prefixBackend: Tell.SpecBackend?
    private var prefixArenaLen = 0                          // current backend's maxSeqLen (0 = not built)
    private var prefixEmptySnap: Any?                       // fullSnapshot of the fresh arena, for reset
    private var arenaContent: [Int32] = []                  // tokens currently prefilled into the arena (one path)
    private var prefixSlots: [(len: Int, snap: Any)] = []   // restore points along arenaContent, sorted by len

    /// Read `text_config.max_position_embeddings` from the checkpoint's config.json.
    /// Falls back to 32768 if absent — a sane bound that still lets any real chat reach EOS.
    static func readContextLen(_ modelDir: String) -> Int {
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let top = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return 32768 }
        let tc = (top["text_config"] as? [String: Any]) ?? top
        return (tc["max_position_embeddings"] as? Int) ?? 32768
    }

    public init(modelDir: String, tier: SeedlessTier = .auto) throws {
        self.modelDir = modelDir
        self.tier = tier
        self.contextLen = SeedlessBackend.readContextLen(modelDir)
        let store = try WeightStore(modelDir: modelDir)
        // residency by tier (mirrors run(): streaming keeps only non-experts resident).
        if SeedlessBackend.config(tier: tier, promptLen: 0, maxTokens: 0).isStreaming {
            store.residentNonExperts()
        } else {
            store.residentAll()
        }
        self.store = store
        self.engine = SeedlessEngine.build(store: store)
    }

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int> {
        // c / maxK / maxM / isStreaming are prompt-length independent; only maxSeqLen (per
        // segment, below) tracks the growing sequence. Resolve those once.
        let cfg = SeedlessBackend.config(tier: tier, promptLen: 0, maxTokens: 0)
        let promptIds0 = prompt.map { Int32($0) }
        // Hard ceiling on GENERATED tokens: <0 (unset / `--max-tokens -1`) → fill to the model
        // context; else the caller's cap. Both clamped to the context headroom.
        let headroom = Swift.max(0, contextLen - prompt.count)
        let ceiling = options.maxTokens < 0 ? headroom : Swift.min(options.maxTokens, headroom)
        // Prefix cache path: resident/fused, greedy, gen-prompt present, prompt fits the arena with
        // room to decode. Everything else falls through to the segmented-growth path below.
        if prefixCacheEnabled, !cfg.isStreaming, !options.sampling,
           let cl = options.promptContentLen, cl < prompt.count, prompt.count + 64 <= prefixArenaMax {
            return generateCached(prompt: prompt, contentLen: cl, ceiling: ceiling, cfg: cfg)
        }
        // First KV arena budget; grows geometrically to `ceiling` only if generation needs it.
        let baseline = Swift.max(1, Tell.envInt("QWISP_CTX_BASELINE", 8192))
        let cancel = CancelFlag()
        return AsyncStream { continuation in
            // Consumer dropped the stream (EOS at the consumer / client disconnect) → cancel the
            // detached decode so it stops at the next spec-step boundary instead of running to the
            // ceiling on the (exclusive) GPU and overlapping the next request's decode.
            continuation.onTermination = { _ in cancel.cancel() }
            // Decode is synchronous + GPU-bound → run off the caller's task, yielding each token as
            // it lands. The spec backend is built INSIDE the thread so no non-Sendable state is
            // captured; generation is serialized upstream (server AsyncLock).
            //
            // Segmented growth: start with an 8K-token KV arena and only enlarge it (rebuild +
            // re-prefill prompt+generated-so-far) if a segment fills without stopping. The common
            // case (answer < 8K, ends at EOS) never grows and pays nothing. ponytail: re-prefill on
            // grow costs ~2× prefill on the tail; growing the KV buffer in place would need engine
            // changes (frozen forward path). The spec backend is also rebuilt per segment — fine for
            // a single-user server; a persistent per-instance backend is the throughput lever.
            // QWISP_FORCE_SAMPLE=1 forces the sampling loop even at temperature 0 (T=0-equivalence gate).
            let forceSample = Tell.envFlag("QWISP_FORCE_SAMPLE")
            let wantSample = options.sampling || forceSample
            // Productization tier→mode: streaming tiers (<32GB) decode with bolt (near-lossless
            // L3, buddy remap) by default; --lossless / QWISP_LOSSLESS forces strict. Sampling
            // requests fall back to strict streaming (the bolt loop is greedy-deterministic; v1).
            let useBolt = cfg.isStreaming && !wantSample && !self.lossless
            Thread.detachNewThread {
                var seq = promptIds0          // prompt + all accepted tokens so far
                var produced = 0
                var budget = Swift.min(baseline, Swift.max(1, ceiling))
                while produced < ceiling && !cancel.isCancelled {
                    let segN = Swift.min(budget, ceiling - produced)
                    let maxSeqLen = seq.count + segN + cfg.maxK + 64
                    let onTok: (Int) -> Void = { continuation.yield($0) }
                    if useBolt {
                        if self.boltServe == nil {
                            self.boltServe = BoltServe(engine: self.engine, modelDir: self.modelDir,
                                                       C: cfg.c, maxK: cfg.maxK, maxM: cfg.maxM)
                        }
                        guard let seg = self.boltServe?.runSegment(
                            promptIds: seq, N: segN, maxSeqLen: maxSeqLen,
                            isCancelled: { cancel.isCancelled }, onToken: onTok),
                            !seg.isEmpty else { break }
                        seq += seg.map { Int32($0) }
                        produced += seg.count
                        if seg.count < segN { break }
                        budget = Swift.min(budget * 2, 65536)
                        continue
                    }
                    let sb: Tell.SpecBackend? = cfg.isStreaming
                        ? Tell.streamingBackend(engine: self.engine, modelDir: self.modelDir,
                                                maxM: cfg.maxM, maxSeqLen: maxSeqLen, C: cfg.c).map { $0.0 }
                        : Tell.fusedBackend(engine: self.engine,
                                            maxM: ProcessInfo.processInfo.environment["QWISP_HYBRID_PREFILL"] != "0" ? Swift.max(cfg.maxM, 1032) : cfg.maxM,
                                            maxSeqLen: maxSeqLen)
                    guard let backend = sb else { break }
                    // Each segment dispatches to greedy or sampling. Sampling keeps its own state
                    // per segment (RNG restarts, so vary the seed by `produced` to avoid reusing the
                    // same stream across segment boundaries). ponytail: penalty counts reset per
                    // segment — only bites penalized generation past the 8K baseline, which is rare.
                    let seg: [Int]?
                    let gpuSample = wantSample && backend.stepSampleRows != nil && !Tell.envFlag("QWISP_SAMPLE_CPU")
                    if gpuSample {
                        // GPU sampler: temperature/top_p/penalties/logit_bias on-GPU, tiny readback.
                        // top_p lowers acceptance → narrower draft cuts forward + per-row work.
                        let sampMaxK = options.topP < 1.0 ? Swift.min(cfg.maxK, 24) : cfg.maxK
                        seg = Tell.runSpecSampleLoopGPU(promptIds: seq, backend: backend, engine: self.engine,
                                                        N: segN, maxK: sampMaxK, temperature: options.temperature,
                                                        topP: options.topP, seed: options.seed &+ UInt64(produced),
                                                        frequencyPenalty: options.frequencyPenalty,
                                                        presencePenalty: options.presencePenalty, logitBias: options.logitBias,
                                                        isCancelled: { cancel.isCancelled }, onToken: onTok)
                    } else if wantSample {
                        // CPU sampling fallback (QWISP_SAMPLE_CPU / no head): logits readback, maxK ≤ 8.
                        seg = Tell.runSpecSampleLoop(promptIds: seq, backend: backend, engine: self.engine,
                                                     N: segN, maxK: Swift.min(cfg.maxK, 8), temperature: options.temperature,
                                                     topP: options.topP, seed: options.seed &+ UInt64(produced),
                                                     frequencyPenalty: options.frequencyPenalty,
                                                     presencePenalty: options.presencePenalty, logitBias: options.logitBias,
                                                     isCancelled: { cancel.isCancelled }, onToken: onTok)
                    } else {
                        seg = Tell.runSpecLoop(promptIds: seq, backend: backend, engine: self.engine,
                                               N: segN, maxK: cfg.maxK, isCancelled: { cancel.isCancelled }, onToken: onTok)
                    }
                    guard let seg, !seg.isEmpty else { break }
                    seq += seg.map { Int32($0) }
                    produced += seg.count
                    if seg.count < segN { break }   // stopped early (consumer cancel / EOS) → done
                    budget = Swift.min(budget * 2, 65536)
                }
                continuation.finish()
            }
        }
    }

    // Prefix-cache generation (Design B + multi-slot): a persistent fused backend whose KV arena grows
    // to fit each request, plus stride-aligned restore points along the current path. Restores the
    // longest cached prefix of the new content and re-prefills only the delta. Greedy only; provably
    // lossless — SuffixSpec verifies every drafted token (see prefix-cache-poc).
    private func generateCached(prompt: [Int], contentLen: Int, ceiling: Int, cfg: Config) -> AsyncStream<Int> {
        let cancel = CancelFlag()
        return AsyncStream { continuation in
            continuation.onTermination = { _ in cancel.cancel() }
            Thread.detachNewThread {
                // Design B: ensure the arena fits prompt + this request's generation; grow (rebuild,
                // geometric, monotonic) if not. ponytail: unbounded/huge generations cap at
                // prefixArenaMax (fits ≤64K total by default); >that falls through upstream. Mid-decode
                // arena growth is the upgrade path if a real workload needs >prefixArenaMax generation.
                let genBudget = Swift.max(1, Swift.min(ceiling, self.prefixArenaMax - prompt.count))
                let needed = prompt.count + genBudget
                if self.prefixBackend == nil || self.prefixArenaLen < needed {
                    var newLen = Swift.max(self.prefixArenaLen, 16384)
                    while newLen < needed { newLen *= 2 }
                    newLen = Swift.min(newLen, self.prefixArenaMax)
                    if newLen < needed { newLen = needed }
                    // Steel-prefill hybrid wants chunk=1024 → bump maxM (scratch ~200MB, resident tier).
                    let mm = ProcessInfo.processInfo.environment["QWISP_HYBRID_PREFILL"] != "0" ? Swift.max(cfg.maxM, 1032) : cfg.maxM
                    guard let b = Tell.fusedBackend(engine: self.engine, maxM: mm, maxSeqLen: newLen) else {
                        continuation.finish(); return
                    }
                    self.prefixBackend = b; self.prefixArenaLen = newLen
                    self.prefixEmptySnap = b.fullSnapshot?()
                    self.prefixSlots = []; self.arenaContent = []   // new arena → old snapshots invalid
                }
                let backend = self.prefixBackend!
                let full = prompt.map { Int32($0) }
                let content = Array(full[0 ..< contentLen])

                // Multi-slot restore: the longest cached restore point that is a prefix of the new
                // content (positions [0..len] still hold that prefix — append-only KV). A NEW
                // conversation sharing the system+tools prefix restores a sub-content boundary instead
                // of re-prefilling it. Boundaries past the divergence point are stale → drop them.
                let lcp = SeedlessBackend.commonPrefixLen(content, self.arenaContent)
                let restoreLen = self.prefixSlots.last(where: { $0.len <= lcp })?.len ?? 0
                if restoreLen > 0, let s = self.prefixSlots.last(where: { $0.len == restoreLen })?.snap {
                    backend.fullRestore?(s)
                } else if let e = self.prefixEmptySnap {
                    backend.fullRestore?(e)
                }
                self.prefixSlots.removeAll { $0.len > lcp }

                // Re-prefill the delta [restoreLen ..< contentLen], snapshotting at stride-aligned
                // boundaries (so different conversations that share a prefix snapshot at the SAME
                // positions → cross-conversation reuse), plus one at the content boundary.
                var pos = restoreLen
                while pos < contentLen {
                    let end = Swift.min(pos - (pos % self.prefixSnapStride) + self.prefixSnapStride, contentLen)
                    _ = Tell.prefill(promptIds: Array(content[pos ..< end]), backend: backend)
                    if let snap = backend.fullSnapshot?() { self.prefixSlots.append((len: end, snap: snap)) }
                    pos = end
                }
                if self.prefixSlots.last?.len != contentLen, let snap = backend.fullSnapshot?() {
                    self.prefixSlots.append((len: contentLen, snap: snap))
                }
                self.prefixSlots.sort { $0.len < $1.len }
                self.evictSlots()
                self.arenaContent = content

                // Prefill the generation-prompt suffix + decode. prefillTokens = suffix (arena already
                // holds content); hist = the full prompt so SuffixSpec drafting is unchanged.
                let genSuffix = Array(full[contentLen...])
                _ = Tell.runSpecLoop(promptIds: full, backend: backend, engine: self.engine,
                                     N: genBudget, maxK: cfg.maxK, prefillTokens: genSuffix,
                                     isCancelled: { cancel.isCancelled },
                                     onToken: { continuation.yield($0) })
                continuation.finish()
            }
        }
    }

    /// Test/override hook: drop the persistent arena + all restore points (start cold).
    public func resetPrefixCache() {
        prefixBackend = nil; prefixArenaLen = 0; prefixEmptySnap = nil
        arenaContent = []; prefixSlots = []
    }

    // Cap resident snapshots (~60MB each) at prefixMaxSlots, evicting the densest neighbour so a coarse
    // spread of low boundaries (cross-conversation reuse) and the top boundary (next turn) both survive.
    private func evictSlots() {
        while prefixSlots.count > prefixMaxSlots {
            var mi = 1, mgap = Int.max
            for i in 1 ..< prefixSlots.count {
                let g = prefixSlots[i].len - prefixSlots[i - 1].len
                if g < mgap { mgap = g; mi = i }
            }
            prefixSlots.remove(at: mi)
        }
    }

    static func commonPrefixLen(_ a: [Int32], _ b: [Int32]) -> Int {
        let n = Swift.min(a.count, b.count)
        var i = 0
        while i < n && a[i] == b[i] { i += 1 }
        return i
    }
}

/// Thread-safe one-shot cancellation flag: set from the AsyncStream consumer side
/// (onTermination), polled from the detached decode thread. NSLock keeps it boring.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    func cancel() { lock.lock(); flag = true; lock.unlock() }
}
