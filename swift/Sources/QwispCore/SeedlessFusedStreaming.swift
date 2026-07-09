import Foundation
import MLX
import Metal

/// Per-layer expert provider for the streaming fused engine.
/// gatherBuffers: 9 persistent device-visible buffers [C, ...] in order
///   gW,gS,gB,uW,uS,uB,dW,dS,dB
///   (weight uint32-packed 4bit, scales/biases f16 — same layouts as resident tensors sliced per expert).
/// ensure(experts): map each expert id to a slot in [0,C), pread/copying misses into the arena
///   SYNCHRONOUSLY. Caller guarantees GPU has finished all previously-encoded gathers that read
///   these buffers before calling ensure (command buffer waited).
public protocol SeedlessFusedExpertProvider: AnyObject {
    var C: Int { get }
    func gatherBuffers(device: MTLDevice) -> [MTLBuffer]?
    func ensure(_ experts: [Int]) -> [Int: Int]
}

/// 本番 provider: LayerExpertCache(per-layer LRU + ExpertSource pread)を fused engine に接続。
/// gatherBuffers: arena の 9 MLXArray を MTLBuffer に wrap し、キャッシュして返す。
/// ensure: cache.ensure(experts) をそのまま委譲。
/// noCopy buffer の寿命規約: arena arrays は ExpertArena の persistent slot であり class member だが、
/// 安全のため retainedArrays にも保持する(noCopy lifetime は retain に依存)。
public final class ArenaExpertProvider: SeedlessFusedExpertProvider {
    public let cache: LayerExpertCache
    public var C: Int { cache.C }

    // gatherBuffers のキャッシュ(初回のみ構築)
    private var cachedBuffers: [MTLBuffer]? = nil
    // noCopy lifetime 保持
    private var retainedArrays: [MLXArray] = []

    public init(cache: LayerExpertCache) {
        self.cache = cache
    }

    public func gatherBuffers(device: MTLDevice) -> [MTLBuffer]? {
        if let cached = cachedBuffers { return cached }

        // 順序: gW,gS,gB, uW,uS,uB, dW,dS,dB
        // ExpertSource.projs = ["gate_proj","up_proj","down_proj"]
        // ExpertSource.parts = ["weight","scales","biases"]
        var bufs: [MTLBuffer] = []
        var arrays: [MLXArray] = []
        for proj in ExpertSource.projs {
            for part in ExpertSource.parts {
                let arr = cache.arena.arr(proj, part)
                // サニティチェック: scales/biases は f16 でなければならない(raw kernels は half を読む)
                if part != "weight" && bufs.isEmpty && arrays.isEmpty {
                    // 初回 scales チェック(最初の scales は gate_proj.scales)
                }
                arrays.append(arr)
                guard let buf = SeedlessMetalForward.mtlBuf(arr, device) else {
                    print("[ArenaExpertProvider] ERROR: mtlBuf failed for \(proj).\(part)")
                    return nil
                }
                bufs.append(buf)
            }
        }

        // scales dtype サニティチェック(gate_proj.scales = bufs[1], dtype of arrays[1])
        let scalesArr = arrays[1]   // gate_proj.scales
        if scalesArr.dtype != .float16 {
            print("[ArenaExpertProvider] ERROR: scales dtype=\(scalesArr.dtype), expected .float16 — raw kernels will read garbage")
            return nil
        }

        retainedArrays = arrays
        cachedBuffers = bufs
        return bufs
    }

    public func ensure(_ experts: [Int]) -> [Int: Int] {
        cache.ensure(experts)
    }
}

/// テスト用 synthetic provider。
/// 全エキスパートの量子化重みを保持し、arena([C, ...])へ CPU memcpy でロード。
/// LRU eviction。noCopy MTLBuffer の寿命規約を retained で担保。
final class TestExpertProvider: SeedlessFusedExpertProvider {
    let C: Int
    let E: Int, I: Int, H: Int

    // 全エキスパート重み buffer(noCopy → retained が寿命を保持)
    private let fullGW: MTLBuffer, fullGS: MTLBuffer, fullGB: MTLBuffer
    private let fullUW: MTLBuffer, fullUS: MTLBuffer, fullUB: MTLBuffer
    private let fullDW: MTLBuffer, fullDS: MTLBuffer, fullDB: MTLBuffer
    private let retained: [MLXArray]

    // arena [C, ...] buffer
    private let arenaGW: MTLBuffer, arenaGS: MTLBuffer, arenaGB: MTLBuffer
    private let arenaUW: MTLBuffer, arenaUS: MTLBuffer, arenaUB: MTLBuffer
    private let arenaDW: MTLBuffer, arenaDS: MTLBuffer, arenaDB: MTLBuffer

    // LRU 状態
    private var slotOf: [Int: Int] = [:]   // expert id → slot
    private var lastUsed: [Int]            // per slot
    private var tick = 0

    // per-expert byte サイズ
    private let gwExpBytes: Int    // I * (H/8) * 4 (gate/up weight uint32)
    private let gsExpBytes: Int    // I * (H/64) * 2 (gate/up scale/bias f16)
    private let dwExpBytes: Int    // H * (I/8) * 4 (down weight uint32)
    private let dsExpBytes: Int    // H * (I/64) * 2 (down scale/bias f16)

    /// gW,gS,gB: [E,I,H/8] uint32 + [E,I,H/64] f16 (gate)
    /// uW,uS,uB: same layout (up)
    /// dW,dS,dB: [E,H,I/8] uint32 + [E,H,I/64] f16 (down)
    /// scales/biases はすでに f16 変換済みを渡すこと(MLX.quantized の出力を asType(.float16) 後に渡す)。
    init?(E: Int, I: Int, H: Int, C: Int,
          gW: MLXArray, gSf16: MLXArray, gBf16: MLXArray,
          uW: MLXArray, uSf16: MLXArray, uBf16: MLXArray,
          dW: MLXArray, dSf16: MLXArray, dBf16: MLXArray,
          device: MTLDevice) {
        self.C = C; self.E = E; self.I = I; self.H = H
        self.lastUsed = [Int](repeating: 0, count: C)
        gwExpBytes = I * (H / 8) * 4
        gsExpBytes = I * (H / 64) * 2
        dwExpBytes = H * (I / 8) * 4
        dsExpBytes = H * (I / 64) * 2

        MLX.eval([gW, gSf16, gBf16, uW, uSf16, uBf16, dW, dSf16, dBf16])

        guard let fgW = SeedlessMetalForward.mtlBuf(gW, device),
              let fgS = SeedlessMetalForward.mtlBuf(gSf16, device),
              let fgB = SeedlessMetalForward.mtlBuf(gBf16, device),
              let fuW = SeedlessMetalForward.mtlBuf(uW, device),
              let fuS = SeedlessMetalForward.mtlBuf(uSf16, device),
              let fuB = SeedlessMetalForward.mtlBuf(uBf16, device),
              let fdW = SeedlessMetalForward.mtlBuf(dW, device),
              let fdS = SeedlessMetalForward.mtlBuf(dSf16, device),
              let fdB = SeedlessMetalForward.mtlBuf(dBf16, device) else { return nil }
        fullGW = fgW; fullGS = fgS; fullGB = fgB
        fullUW = fuW; fullUS = fuS; fullUB = fuB
        fullDW = fdW; fullDS = fdS; fullDB = fdB
        retained = [gW, gSf16, gBf16, uW, uSf16, uBf16, dW, dSf16, dBf16]

        guard let agW = device.makeBuffer(length: C * gwExpBytes, options: .storageModeShared),
              let agS = device.makeBuffer(length: C * gsExpBytes, options: .storageModeShared),
              let agB = device.makeBuffer(length: C * gsExpBytes, options: .storageModeShared),
              let auW = device.makeBuffer(length: C * gwExpBytes, options: .storageModeShared),
              let auS = device.makeBuffer(length: C * gsExpBytes, options: .storageModeShared),
              let auB = device.makeBuffer(length: C * gsExpBytes, options: .storageModeShared),
              let adW = device.makeBuffer(length: C * dwExpBytes, options: .storageModeShared),
              let adS = device.makeBuffer(length: C * dsExpBytes, options: .storageModeShared),
              let adB = device.makeBuffer(length: C * dsExpBytes, options: .storageModeShared) else { return nil }
        arenaGW = agW; arenaGS = agS; arenaGB = agB
        arenaUW = auW; arenaUS = auS; arenaUB = auB
        arenaDW = adW; arenaDS = adS; arenaDB = adB
    }

    func gatherBuffers(device: MTLDevice) -> [MTLBuffer]? {
        [arenaGW, arenaGS, arenaGB,
         arenaUW, arenaUS, arenaUB,
         arenaDW, arenaDS, arenaDB]
    }

    func ensure(_ experts: [Int]) -> [Int: Int] {
        tick += 1
        var result: [Int: Int] = [:]
        var locked = Set<Int>()

        // パス1: キャッシュ済み
        for e in experts {
            if let slot = slotOf[e] {
                result[e] = slot
                lastUsed[slot] = tick
                locked.insert(slot)
            }
        }

        // パス2: miss をロード
        for e in experts {
            guard result[e] == nil else { continue }
            var lruSlot = -1, lruTime = Int.max
            for s in 0..<C {
                if locked.contains(s) { continue }
                if lastUsed[s] < lruTime { lruTime = lastUsed[s]; lruSlot = s }
            }
            precondition(lruSlot >= 0, "TestExpertProvider: no free slot (C=\(C) too small)")
            // 退去
            if let old = slotOf.first(where: { $0.value == lruSlot })?.key {
                slotOf.removeValue(forKey: old)
            }
            slotOf[e] = lruSlot
            lastUsed[lruSlot] = tick
            locked.insert(lruSlot)
            copyExpert(e, slot: lruSlot)
            result[e] = lruSlot
        }
        return result
    }

    private func copyExpert(_ e: Int, slot: Int) {
        func cp(_ dst: MTLBuffer, _ src: MTLBuffer, expBytes: Int) {
            memcpy(dst.contents().advanced(by: slot * expBytes),
                   src.contents().advanced(by: e * expBytes),
                   expBytes)
        }
        cp(arenaGW, fullGW, expBytes: gwExpBytes)
        cp(arenaGS, fullGS, expBytes: gsExpBytes)
        cp(arenaGB, fullGB, expBytes: gsExpBytes)
        cp(arenaUW, fullUW, expBytes: gwExpBytes)
        cp(arenaUS, fullUS, expBytes: gsExpBytes)
        cp(arenaUB, fullUB, expBytes: gsExpBytes)
        cp(arenaDW, fullDW, expBytes: dwExpBytes)
        cp(arenaDS, fullDS, expBytes: dsExpBytes)
        cp(arenaDB, fullDB, expBytes: dsExpBytes)
    }
}
