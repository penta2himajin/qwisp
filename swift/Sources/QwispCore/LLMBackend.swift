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
    public init(maxTokens: Int = 128, stopTokens: [Int] = []) {
        self.maxTokens = maxTokens
        self.stopTokens = stopTokens
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
    let modelDir: String
    let tier: SeedlessTier

    public init(modelDir: String, tier: SeedlessTier = .auto) throws {
        self.modelDir = modelDir
        self.tier = tier
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
        let cfg = SeedlessBackend.config(tier: tier, promptLen: prompt.count, maxTokens: options.maxTokens)
        let promptIds = prompt.map { Int32($0) }
        let cancel = CancelFlag()
        return AsyncStream { continuation in
            // Consumer dropped the stream (early stop token / maxTokens / client disconnect)
            // → cancel the detached decode so it stops at the next spec-step boundary instead
            // of running to maxTokens on the GPU. Without this the orphaned thread keeps the
            // (exclusive) GPU busy and the next request's decode overlaps it.
            continuation.onTermination = { _ in cancel.cancel() }
            // The decode loop is synchronous + GPU-bound. Run it off the caller's task and
            // yield each accepted token as it lands (true incremental streaming). The spec
            // backend is built INSIDE the thread so no non-Sendable state is captured;
            // generation is serialized upstream (server AsyncLock) so touching engine is safe.
            // ponytail: the spec backend is rebuilt per request — fine for a single-user
            // server; make it a persistent per-instance backend if throughput matters.
            Thread.detachNewThread {
                let specBackend: Tell.SpecBackend? = cfg.isStreaming
                    ? Tell.streamingBackend(engine: self.engine, modelDir: self.modelDir,
                                            maxM: cfg.maxM, maxSeqLen: cfg.maxSeqLen, C: cfg.c).map { $0.0 }
                    : Tell.fusedBackend(engine: self.engine, maxM: cfg.maxM, maxSeqLen: cfg.maxSeqLen)
                guard let sb = specBackend else { continuation.finish(); return }
                _ = Tell.runSpecLoop(promptIds: promptIds, backend: sb, engine: self.engine,
                                     N: options.maxTokens, maxK: cfg.maxK,
                                     isCancelled: { cancel.isCancelled }) { tok in
                    continuation.yield(tok)
                }
                continuation.finish()
            }
        }
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
