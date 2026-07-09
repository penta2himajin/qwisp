import Foundation
import MLX

// Product facade (productization step 2, strict-first).
//
// LLMBackend is the coarse MLX-compat surface the server swaps backends over.
// It operates on token IDs `[Int]`; the tokenizer + chat template live ABOVE it
// in the server layer. SeedlessBackend wraps the EXISTING shipped strict decode
// (RawSpecRunner.runSpecLoop + engine/backend builders) — it is the keep-set
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
    public init(maxTokens: Int = 128) { self.maxTokens = maxTokens }
}

/// Coarse backend protocol: load + generate + tier. Both a future MLXBackend and
/// SeedlessBackend conform, so the server picks a backend in one line.
public protocol LLMBackend {
    init(modelDir: String, tier: SeedlessTier) throws
    func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int>
}

public final class SeedlessBackend: LLMBackend {

    /// Pure, GPU-free sizing seam. Mirrors RawSpecRunner.run()'s tier arithmetic so
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
    let engine: RawEngine
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
        self.engine = RawEngine.build(store: store)
    }

    public func generate(_ prompt: [Int], options: GenerateOptions) -> AsyncStream<Int> {
        let cfg = SeedlessBackend.config(tier: tier, promptLen: prompt.count, maxTokens: options.maxTokens)
        let promptIds = prompt.map { Int32($0) }
        let backend: RawSpecRunner.SpecBackend? = cfg.isStreaming
            ? RawSpecRunner.streamingBackend(engine: engine, modelDir: modelDir,
                                             maxM: cfg.maxM, maxSeqLen: cfg.maxSeqLen, C: cfg.c).map { $0.0 }
            : RawSpecRunner.fusedBackend(engine: engine, maxM: cfg.maxM, maxSeqLen: cfg.maxSeqLen)
        // ponytail: batch-decode then replay as a stream. True incremental SSE streaming
        // (yield per accepted token) is a follow-up for when the server needs token latency
        // — runSpecLoop currently returns the full [Int], and the GPU is exclusive anyway.
        let out: [Int] = backend.flatMap {
            RawSpecRunner.runSpecLoop(promptIds: promptIds, backend: $0, engine: engine,
                                      N: options.maxTokens, maxK: cfg.maxK)
        } ?? []
        return AsyncStream { cont in
            for t in out { cont.yield(t) }
            cont.finish()
        }
    }
}
