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
    private var samplingFallbackNoted = false
    // Segment gate (issue #47): a new request's decode thread must wait for the previous
    // decode thread's FULL exit. Stream teardown (consumer break / client disconnect) only
    // requests cooperative cancellation — the old thread keeps running until its next
    // spec-step boundary and then BoltServe's deferred async-refresh drain, and the server's
    // AsyncLock is released at teardown, i.e. EARLIER than that exit. Unserialized, the
    // drain's staged swaps overlap the next request's prefill/ensure() on the same providers
    // and shared staging arenas (measured: run-to-run nondeterminism, stale-slice arena
    // corruption, buildBuddyTable force-unwrap crash).
    private let segGate = DispatchSemaphore(value: 1)

    /// Join the in-flight decode thread, if any. A consumer that breaks out of the stream
    /// (EOS is detected consumer-side only) leaves the detached thread running — possibly a
    /// whole segment rebuild + re-prefill with no cancellation checks — and a CLI that then
    /// returns from main races C++ static teardown of the MLX scheduler singleton
    /// (rc 133/139 SIGTRAP/SIGSEGV in get_default_stream, #47 handoff). Call before exit.
    public func drain() {
        segGate.wait()
        segGate.signal()
        // Graceful-shutdown write-back of the calib warm-start artifact (issue #73):
        // the basis tracked the latest recalib window — persist last-known-good.
        boltServe?.saveArtifact()
    }
    let modelDir: String
    let tier: SeedlessTier
    let contextLen: Int      // model context window (max_position_embeddings); caps unbounded generation.

    // ── Cross-request prefix cache (default ON; QWISP_PREFIX_CACHE=0 opts out) ──────────
    // Persistent per-instance backend (fused on resident; strict streaming since #76) + a
    // content-boundary fullSnapshot, so consecutive requests (agentic loops) re-prefill only
    // the new suffix. Serialized by the server lock.
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
    // Streaming tier (#76): the C-slot expert arena persists across cached-backend rebuilds
    // (only the KV/GDN arena is rebuilt on growth) — rebuilding providers would re-stream GBs.
    private var prefixProviders: [ArenaExpertProvider]? = nil
    private var prefixArenaLen = 0                          // current backend's maxSeqLen (0 = not built)
    private var prefixEmptySnap: Any?                       // fullSnapshot of the fresh arena, for reset
    private var arenaContent: [Int32] = []                  // tokens currently prefilled into the arena (one path)
    private var prefixSlots: [(len: Int, snap: Any)] = []   // restore points along arenaContent, sorted by len
    // #117 RAM tier: whole-conversation states (persistentState blobs, path-independent) so
    // interleaved conversations don't thrash the single-arena path above. LRU by byte budget;
    // default 2GB resident / OFF streaming (wired-memory pressure — see PR #70).
    private var prefixRAM = PrefixRAMStore()
    public private(set) var prefixRAMHits = 0               // gate observability, mirrors PrefixPersist.restoreHits
    func prefixRAMBudget(isStreaming: Bool) -> Int {
        Swift.max(0, Tell.envInt("QWISP_PREFIX_RAM_MB", isStreaming ? 0 : 2048)) * 1_048_576
    }

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
        // Prefix cache path: greedy, gen-prompt present, prompt fits the arena with room to
        // decode. Tiers (#76): resident, or strict streaming (--lossless / sampling-free) —
        // the snapshot is attention KV + GDN state, independent of expert residency. Bolt
        // keeps the segmented path: its per-segment strict-prefill→freeze order is
        // load-bearing (buddy tables map experts to arena SLOTS), so restore-skipping-prefill
        // is a separate follow-up. QWISP_PREFIX_STREAMING=0 opts the extension out.
        // Everything else falls through to the segmented-growth path below.
        let streamingCacheOK = !cfg.isStreaming
            || (lossless && Tell.envInt("QWISP_PREFIX_STREAMING", 1) != 0)
        if prefixCacheEnabled, streamingCacheOK, !options.sampling,
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
            // Strict/sampling streaming uses a budget-fit C (issue #69): the tier C's wired
            // footprint starves a real small-RAM Mac and collapses strict ~10x. bolt keeps
            // cfg.c — its frozen residency needs the bigger arena and tolerates the wired set.
            let strictC = cfg.isStreaming ? DeviceCalibration.strictStreamingC(tierC: cfg.c) : cfg.c
            let strictMaxK = cfg.isStreaming ? Swift.max(4, strictC * 3 / 8) : cfg.maxK
            if strictC != cfg.c && !useBolt && !samplingFallbackNoted {
                FileHandle.standardError.write(Data(
                    "[qwisp] strict streaming: C \(cfg.c)→\(strictC) to fit this machine's memory budget (override: QWISP_STRICT_C)\n".utf8))
            }
            // The fallback is silent otherwise and reads as a mystery slowdown (field report,
            // issue #45): sampling skips bolt calibration (instant start) and decodes at strict
            // speed. Say so once per process.
            if cfg.isStreaming && wantSample && !self.lossless && !samplingFallbackNoted {
                samplingFallbackNoted = true
                FileHandle.standardError.write(Data(
                    "[qwisp] sampling (temperature/top_p/…) on a streaming tier decodes via strict streaming — bolt is greedy-only, expect strict speed; temperature 0 restores bolt\n".utf8))
            }
            // Loop guard (#47 Part A, DEFAULT ON; QWISP_BOLT_STABILITY_GUARD=0 disables):
            // only the bolt greedy path can enter the free-run repetition attractor. When it
            // does, rewind the degenerate tail to its clean onset and re-decode on strict; on
            // the re-arm policy (QWISP_LOOP_REARM=M > 0) return to bolt after M strict tokens
            // (0 = stay strict for the rest of the request). All greedy tokens flow through
            // the guard so seq/produced derive from it and rewind auto-reconciles.
            // Belt-and-braces with mixed cov128 (lever A): the soak showed cov128 makes loops
            // rare and late (3/10 incl. a 3000-tok run) but not impossible — the guard heals
            // period ≤64 invisibly (64-token hold-back); long-period semantic loops remain
            // the documented residual.
            let guardOn = useBolt && Tell.envInt("QWISP_BOLT_STABILITY_GUARD", 1) != 0
            // Re-arm default 128: "cliff passed" is NOT detectable (p14: ignition clusters
            // recover mid-run; rho hazard detectors FP-ridden; agreement rate discriminates
            // nothing) — so return to bolt mechanically after M strict tokens and let the
            // still-armed guard bound the cost of a wrong re-entry (≤ detection lag, unsent).
            // Each re-trip doubles the window (backoff) so pathological prompts converge to
            // strict. QWISP_LOOP_REARM=0 = stay strict after the first trip.
            let rearmM = Tell.envInt("QWISP_LOOP_REARM", 128)
            let guardDbg = Tell.envFlag("QWISP_LOOP_DEBUG")
            Thread.detachNewThread {
                self.segGate.wait()           // join the previous segment's decode thread (issue #47)
                defer { self.segGate.signal() }
                let lguard = guardOn ? LoopGuard() : nil
                var seq = promptIds0          // prompt + all accepted tokens so far
                var produced = 0
                var budget = Swift.min(baseline, Swift.max(1, ceiling))
                var boltActive = useBolt
                var strictSinceTrip = 0
                var rearmCur = rearmM          // per-request re-arm window; doubles on each re-trip
                var trips = 0
                // Guarded greedy tokens flow through the rollback buffer; everything else
                // (sampling) streams directly. reconcile() re-derives seq/produced from the
                // guard's authoritative token list (auto-truncated on rewind).
                let onTok: (Int) -> Void = { tok in
                    if let g = lguard { for t in g.push(tok) { continuation.yield(t) } }
                    else { continuation.yield(tok) }
                }
                func reconcile() {
                    guard let g = lguard else { return }
                    seq = promptIds0 + g.gen.map { Int32($0) }
                    produced = g.gen.count
                }
                while produced < ceiling && !cancel.isCancelled {
                    // In a strict re-arm window, cap the segment at M so we can return to bolt.
                    let inRearmWindow = lguard != nil && !boltActive && rearmCur > 0
                    let segN = inRearmWindow ? Swift.min(rearmCur, ceiling - produced)
                                             : Swift.min(budget, ceiling - produced)
                    let maxSeqLen = seq.count + segN + cfg.maxK + 64
                    if boltActive {
                        if self.boltServe == nil {
                            // ★ W3b mixed residency (notes/18): bolt tier swaps the C-slot 4-bit arena
                            //   for K4 4-bit core + M2 2-bit tail (coverage 128 = the measured healthy
                            //   line, in the same bytes — lever A, 4/4 loop-free battery). DEFAULT ON
                            //   when a 2-bit experts checkpoint is present (cal128 preferred; QWISP_MIXED=0
                            //   disables). No checkpoint → generic bolt (warn only on explicit opt-in —
                            //   never silently change the model).
                            var tailDir: String? = nil
                            if Tell.envInt("QWISP_MIXED", 1) != 0 {
                                let home = FileManager.default.homeDirectoryForCurrentUser.path
                                let candidates = ProcessInfo.processInfo.environment["QWISP_EXPERTS_2BIT"].map { [$0] }
                                    ?? ["\(home)/.mtplx/models/qwisp-experts-2bit-cal128",
                                        "\(home)/.mtplx/models/qwisp-experts-2bit-cal",
                                        "\(home)/.mtplx/models/qwisp-experts-2bit"]
                                tailDir = candidates.first {
                                    FileManager.default.fileExists(atPath: $0 + "/model.safetensors.index.json")
                                }
                                if tailDir == nil, ProcessInfo.processInfo.environment["QWISP_MIXED"] != nil {
                                    FileHandle.standardError.write(Data(
                                        "[qwisp] QWISP_MIXED=1 but no 2-bit experts checkpoint found (tried: \(candidates.joined(separator: ", "))) — using generic bolt\n".utf8))
                                }
                            }
                            self.boltServe = BoltServe(engine: self.engine, modelDir: self.modelDir,
                                                       C: cfg.c, maxK: cfg.maxK, maxM: cfg.maxM,
                                                       mixedTailDir: tailDir)
                        }
                        guard let seg = self.boltServe?.runSegment(
                            promptIds: seq, N: segN, maxSeqLen: maxSeqLen,
                            contentLen: options.promptContentLen,   // #76: bolt prefix-reuse boundary
                            isCancelled: { cancel.isCancelled || lguard?.trip != nil }, onToken: onTok),
                            !seg.isEmpty else { break }
                        if let g = lguard, let tr = g.trip {
                            _ = g.rewind()          // truncate tail to onset (unsent tokens only)
                            reconcile()
                            boltActive = false
                            strictSinceTrip = 0
                            trips += 1
                            if trips > 1 { rearmCur = Swift.min(rearmCur * 2, 65536) }   // backoff
                            if guardDbg { FileHandle.standardError.write(Data(
                                "[qwisp] loop guard: rewind to onset \(tr.onset) (period \(tr.period), span \(tr.span)) → strict\n".utf8)) }
                            continue
                        }
                        reconcile()
                        if seg.count < segN { break }
                        budget = Swift.min(budget * 2, 65536)
                        continue
                    }
                    let sb: Tell.SpecBackend? = cfg.isStreaming
                        ? Tell.streamingBackend(engine: self.engine, modelDir: self.modelDir,
                                                maxM: cfg.maxM, maxSeqLen: maxSeqLen, C: strictC).map { $0.0 }
                        : Tell.fusedBackend(
                            engine: self.engine,
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
                        let sampMaxK = options.topP < 1.0 ? Swift.min(strictMaxK, 24) : strictMaxK
                        seg = Tell.runSpecSampleLoopGPU(promptIds: seq, backend: backend, engine: self.engine,
                                                        N: segN, maxK: sampMaxK, temperature: options.temperature,
                                                        topP: options.topP, seed: options.seed &+ UInt64(produced),
                                                        frequencyPenalty: options.frequencyPenalty,
                                                        presencePenalty: options.presencePenalty, logitBias: options.logitBias,
                                                        isCancelled: { cancel.isCancelled }, onToken: onTok)
                    } else if wantSample {
                        // CPU sampling fallback (QWISP_SAMPLE_CPU / no head): logits readback, maxK ≤ 8.
                        seg = Tell.runSpecSampleLoop(promptIds: seq, backend: backend, engine: self.engine,
                                                     N: segN, maxK: Swift.min(strictMaxK, 8), temperature: options.temperature,
                                                     topP: options.topP, seed: options.seed &+ UInt64(produced),
                                                     frequencyPenalty: options.frequencyPenalty,
                                                     presencePenalty: options.presencePenalty, logitBias: options.logitBias,
                                                     isCancelled: { cancel.isCancelled }, onToken: onTok)
                    } else if let rp = ProcessInfo.processInfo.environment["QWISP_TF_REPLAY"],
                              let tokStr = try? String(contentsOfFile: rp, encoding: .utf8) {
                        // #47 probe 14: teacher-forced strict replay of a bolt token stream
                        // (realized-flip ground truth). Diagnostic-only; yields no output tokens.
                        let toks = tokStr.split(whereSeparator: \.isNewline).compactMap { Int($0) }
                        let outP = ProcessInfo.processInfo.environment["QWISP_TF_OUT"] ?? (rp + ".tf.tsv")
                        seg = Tell.runTFReplay(promptIds: seq, backend: backend, engine: self.engine,
                                               toks: toks, outPath: outP)
                    } else {
                        seg = Tell.runSpecLoop(promptIds: seq, backend: backend, engine: self.engine,
                                               N: segN, maxK: strictMaxK, isCancelled: { cancel.isCancelled }, onToken: onTok)
                    }
                    guard let seg, !seg.isEmpty else { break }
                    // Greedy strict flows through the guard (unified seq/produced + re-arm);
                    // sampling streams directly and keeps its own accounting.
                    if lguard != nil && !wantSample {
                        reconcile()
                        strictSinceTrip += seg.count
                        if seg.count < segN { break }        // EOS → done
                        if rearmCur > 0 && strictSinceTrip >= rearmCur {
                            boltActive = true                // re-arm: return to bolt
                            if guardDbg { FileHandle.standardError.write(Data(
                                "[qwisp] loop guard: re-arm → bolt after \(strictSinceTrip) strict tok\n".utf8)) }
                        }
                    } else {
                        seq += seg.map { Int32($0) }
                        produced += seg.count
                        if seg.count < segN { break }   // stopped early (consumer cancel / EOS) → done
                    }
                    budget = Swift.min(budget * 2, 65536)
                }
                if let g = lguard { for t in g.flush() { continuation.yield(t) } }
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
                self.segGate.wait()           // same gate as generate() — one decode thread at a time
                defer { self.segGate.signal() }
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
                    let built: Tell.SpecBackend?
                    if cfg.isStreaming {
                        // #76 strict streaming: budget-fit C (#69) + persistent expert arena —
                        // existingProviders survive KV-arena growth rebuilds.
                        let strictC = DeviceCalibration.strictStreamingC(tierC: cfg.c)
                        if let (sb, _, provs) = Tell.streamingBackend(
                            engine: self.engine, modelDir: self.modelDir,
                            maxM: cfg.maxM, maxSeqLen: newLen, C: strictC,
                            existingProviders: self.prefixProviders) {
                            self.prefixProviders = provs
                            built = sb
                        } else { built = nil }
                    } else {
                        // Steel-prefill hybrid wants chunk=1024 → bump maxM (scratch ~200MB, resident tier).
                        let mm = ProcessInfo.processInfo.environment["QWISP_HYBRID_PREFILL"] != "0" ? Swift.max(cfg.maxM, 1032) : cfg.maxM
                        built = Tell.fusedBackend(engine: self.engine, maxM: mm, maxSeqLen: newLen)
                    }
                    guard let b = built else {
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
                var restoreLen = self.prefixSlots.last(where: { $0.len <= lcp })?.len ?? 0
                // #117 RAM tier: another conversation's stored state whose prefix beats the
                // in-path restore point by more than a restore is worth (~512 tok of prefill).
                self.prefixRAM.budget = self.prefixRAMBudget(isStreaming: cfg.isStreaming)
                if let hit = self.prefixRAM.bestMatch(content: content), hit.tokens.count > restoreLen + 512 {
                    if backend.restorePersistentState?(hit.state) == true {
                        self.arenaContent = hit.tokens
                        self.prefixSlots = []
                        if let snap = backend.fullSnapshot?() {
                            self.prefixSlots = [(len: hit.tokens.count, snap: snap)]
                        }
                        restoreLen = hit.tokens.count
                        self.prefixRAMHits += 1
                    } else {
                        // A failed restore may have half-written the arena — the in-path
                        // state is dead; fall back to a cold prefill from the empty state.
                        self.arenaContent = []; self.prefixSlots = []; restoreLen = 0
                        if let e = self.prefixEmptySnap { backend.fullRestore?(e) }
                    }
                } else if restoreLen > 0, let s = self.prefixSlots.last(where: { $0.len == restoreLen })?.snap {
                    backend.fullRestore?(s)
                } else if PrefixPersist.enabled,
                          let hit = PrefixPersist.bestMatch(modelDir: self.modelDir, content: content),
                          backend.restorePersistentState?(hit.state) == true {
                    // #89 (opt-in): no in-memory restore point — cross-process warm start from
                    // disk. The restore wrote the KV bytes + GDN state into THIS arena, so the
                    // arena path IS hit.tokens now; seed one in-memory slot at that boundary.
                    self.arenaContent = Array(hit.tokens)
                    self.prefixSlots = []
                    if let snap = backend.fullSnapshot?() {
                        self.prefixSlots = [(len: hit.tokens.count, snap: snap)]
                    }
                    restoreLen = hit.tokens.count
                    PrefixPersist.restoreHits += 1
                } else if let e = self.prefixEmptySnap {
                    backend.fullRestore?(e)
                }
                self.prefixSlots.removeAll { $0.len > Swift.max(lcp, restoreLen) }

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
                // #89 (opt-in): persist the content-boundary state for cross-process reuse.
                // The state is exactly at the boundary here (gen-suffix prefill comes next);
                // persistentState() is a CPU copy of shared buffers, the file write is async.
                if self.prefixRAM.budget > 0 || PrefixPersist.enabled,
                   let blob = backend.persistentState?() {
                    // #117: RAM tier keeps this conversation warm across arena-path switches.
                    if self.prefixRAM.budget > 0 { self.prefixRAM.save(tokens: content, state: blob) }
                    if PrefixPersist.enabled {
                        let model = self.modelDir
                        DispatchQueue.global(qos: .utility).async {
                            PrefixPersist.save(modelDir: model, tokens: content, state: blob)
                        }
                    }
                }

                // Prefill the generation-prompt suffix + decode. prefillTokens = suffix (arena already
                // holds content); hist = the full prompt so SuffixSpec drafting is unchanged.
                let genSuffix = Array(full[contentLen...])
                // Streaming (#76): maxK follows the budget-fit C actually driving the backend,
                // not the tier C (mirrors generate()'s strictMaxK).
                let maxK = cfg.isStreaming
                    ? Swift.max(4, DeviceCalibration.strictStreamingC(tierC: cfg.c) * 3 / 8)
                    : cfg.maxK
                _ = Tell.runSpecLoop(promptIds: full, backend: backend, engine: self.engine,
                                     N: genBudget, maxK: maxK, prefillTokens: genSuffix,
                                     isCancelled: { cancel.isCancelled },
                                     onToken: { continuation.yield($0) })
                continuation.finish()
            }
        }
    }

    /// Test/override hook: drop the persistent arena + all restore points (start cold).
    public func resetPrefixCache() {
        prefixBackend = nil; prefixProviders = nil; prefixArenaLen = 0; prefixEmptySnap = nil
        arenaContent = []; prefixSlots = []
        prefixRAM.removeAll()   // #117: "restart" semantics — only the DISK store (#89) survives
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
