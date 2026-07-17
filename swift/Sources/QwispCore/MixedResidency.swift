import Foundation
import MLX
import Metal

/// MixedResidency — W2 (notes/18): residency layer for the mixed 4-bit core + 2-bit tail
/// MoE gather kernels shipped in W1b (gqmmMixRows / gqmmMixSwigluRows).
///
/// Slot geometry (GLOBAL slot space):
///   - core slots 0..<K4         : 4-bit experts, static-pinned (assigned once, never evicted)
///   - tail slots K4..<K4+M2     : 2-bit experts, LRU
///   4-bit weight buffer w4 is indexed by the GLOBAL slot (0..<K4); 2-bit weight buffer w2 is
///   indexed by the LOCAL tail index (slot − K4); scales/biases live in ONE uniform buffer of
///   K4+M2 rows indexed by the GLOBAL slot.
///
/// This wave (W2) delivers the types + locked tests only — NO engine wiring (that is W3).
public struct MixedCacheState {
    public let K4: Int
    public let M2: Int
    public var coreOf: [Int: Int] = [:]        // expert → core slot 0..<K4 (static after pinCore)
    public var tailSlotOf: [Int: Int] = [:]    // expert → GLOBAL tail slot K4..<K4+M2
    public var tailExpertAt: [Int]             // size M2, local index → expert (-1 = empty)
    public var tailTick: [Int]                 // size M2, local index → last-use tick (LRU)
    public var clock: Int = 0
    public var hits: Int = 0
    public var misses: Int = 0

    public init(K4: Int, M2: Int) {
        self.K4 = K4
        self.M2 = M2
        self.tailExpertAt = [Int](repeating: -1, count: M2)
        self.tailTick = [Int](repeating: 0, count: M2)
    }

    /// Assign core slots 0..<K4 to `experts` sorted ASC (deterministic). Fails on wrong count / dups.
    public mutating func pinCore(_ experts: [Int]) -> Bool {
        guard experts.count == K4 else { return false }
        guard Set(experts).count == experts.count else { return false }
        let sorted = experts.sorted()
        var m: [Int: Int] = [:]
        for (i, e) in sorted.enumerated() { m[e] = i }
        coreOf = m
        return true
    }

    /// Resolve `experts` to GLOBAL slots. Core → coreOf slot. Tail hit → touch. Tail miss → fill
    /// empty else evict LRU (excluding slots touched THIS call), append pread job with GLOBAL slot.
    /// Returns nil (overflow, state unchanged) when distinct tail experts this call > M2.
    public mutating func ensure(_ experts: [Int]) -> (slots: [Int: Int], missJobs: [(e: Int, slot: Int)])? {
        var distinctTail = Set<Int>()
        for e in experts where coreOf[e] == nil { distinctTail.insert(e) }
        if distinctTail.count > M2 { return nil }   // overflow — bail before any mutation

        var slots: [Int: Int] = [:]
        var missJobs: [(e: Int, slot: Int)] = []
        var touchedThisCall = Set<Int>()   // local tail indices touched (hit or newly assigned) this call

        for e in experts {
            if let cs = coreOf[e] { slots[e] = cs; continue }
            clock += 1
            if let gs = tailSlotOf[e] {
                let local = gs - K4
                tailTick[local] = clock
                touchedThisCall.insert(local)
                hits += 1
                slots[e] = gs
                continue
            }
            misses += 1
            var local = -1
            for i in 0 ..< M2 where tailExpertAt[i] == -1 { local = i; break }
            if local == -1 {
                // ★ same-call invariant (A5 LFU disaster precedent, ExpertArena.ensure): a slot
                //   touched (hit or newly assigned) earlier in THIS call must never be evicted
                //   later in the same call. Exclude touchedThisCall from the LRU scan.
                var oldest = -1
                for i in 0 ..< M2 where !touchedThisCall.contains(i) {
                    if oldest == -1 || tailTick[i] < tailTick[oldest] { oldest = i }
                }
                local = oldest
                let evicted = tailExpertAt[local]
                if evicted != -1 { tailSlotOf.removeValue(forKey: evicted) }
            }
            tailExpertAt[local] = e
            tailSlotOf[e] = K4 + local
            tailTick[local] = clock
            touchedThisCall.insert(local)
            slots[e] = K4 + local
            missJobs.append((e: e, slot: K4 + local))
        }
        return (slots, missJobs)
    }

    /// Buddy table over the GLOBAL slot space. Resident e → its current global slot (buddyExpert=e).
    /// Cold e → rotation-tie-break coact argmax among ALL residents (core+tail, sorted ASC) mapped to
    /// that resident's current global slot; no-coact → fallbackSlot with buddyExpert=-1.
    public func buddyTable(coact: [[Int]], nE: Int, fallbackSlot: Int) -> (table: [Int32], buddyExpert: [Int32]) {
        var residentSlot: [Int: Int] = [:]
        for (e, s) in coreOf { residentSlot[e] = s }
        for (e, s) in tailSlotOf { residentSlot[e] = s }
        let hot = residentSlot.keys.sorted()
        let n = hot.count
        var table = [Int32](repeating: 0, count: nE)
        var buddy = [Int32](repeating: -1, count: nE)
        for e in 0 ..< nE {
            if let s = residentSlot[e] { table[e] = Int32(s); buddy[e] = Int32(e); continue }
            var bestH = -1, bestC = -1
            if n > 0 {
                // 決定化: ExpertArena.buildBuddyTable と同じ rotation tie-break（(i+e)%n）。
                for i in 0 ..< n {
                    let h = hot[(i + e) % n]
                    let cc = coact[e][h]
                    if cc > bestC { bestC = cc; bestH = h }
                }
            }
            if bestH >= 0 && bestC > 0 { table[e] = Int32(residentSlot[bestH]!); buddy[e] = Int32(bestH) }
            else { table[e] = Int32(fallbackSlot) }
        }
        return (table, buddy)
    }
}

/// IO shell (ExpertArena pattern): mixed-slot arena — split w4[K4]/w2[M2] weight buffers + uniform
/// scales/biases[K4+M2] per proj, filled by pread from a 4-bit source (core) and 2-bit source (tail).
public final class MixedExpertArena {
    struct Slot { let arr: MLXArray; let buf: MTLBuffer; let ptr: UnsafeMutableRawPointer; let sliceBytes: Int }
    let device: MTLDevice
    let source4: ExpertSource
    let source2: ExpertSource
    public let K4: Int
    public let M2: Int
    var w4Slots: [String: Slot] = [:]          // proj -> Slot (weight, K4 rows, GLOBAL-slot indexed)
    var w2Slots: [String: Slot] = [:]          // proj -> Slot (weight, M2 rows, LOCAL-tail indexed)
    var uniformSlots: [String: Slot] = [:]     // "proj.scales"/"proj.biases" -> Slot (K4+M2 rows, GLOBAL-slot indexed)
    var cachedBuffers: [MTLBuffer]?

    public init(device: MTLDevice, source4: ExpertSource, source2: ExpertSource,
                K4: Int, M2: Int, refLayer: Int = 0) throws {
        self.device = device; self.source4 = source4; self.source2 = source2
        self.K4 = K4; self.M2 = M2
        // ★ pre-populate header/fd caches sequentially (ExpertSource.warm precedent, SeedlessEngine
        //   call site) — loadCore/loadTailMany pread concurrently via DispatchQueue.concurrentPerform,
        //   and ExpertSource.fd()/header() lazily mutate internal dictionaries on first access; doing
        //   that lazily from concurrent threads races (concurrent Dictionary write = crash).
        try source4.warm(numLayers: refLayer + 1)
        try source2.warm(numLayers: refLayer + 1)
        for proj in ExpertSource.projs {
            let restW4 = try source4.restShape(refLayer, proj, "weight")
            let dtW4 = try source4.partDType(refLayer, proj, "weight")
            let sbW4 = try source4.sliceBytes(refLayer, proj, "weight")
            // K4=0 (all-2-bit, cov 115): zero-row MLXArray crashes inside mlx-swift's
            // asMTLBuffer (force-unwrapped nil data ptr) — allocate 1 dummy row; core
            // slots are 0..<K4 so it is never addressed.
            let arrW4 = MLXArray.zeros([Swift.max(K4, 1)] + restW4, dtype: dtW4)
            arrW4.eval()
            guard let bufW4 = arrW4.asMTLBuffer(device: device, noCopy: true) else {
                throw NSError(domain: "MixedExpertArena", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "asMTLBuffer failed: \(proj).weight (w4)"])
            }
            w4Slots[proj] = Slot(arr: arrW4, buf: bufW4, ptr: bufW4.contents(), sliceBytes: sbW4)

            let restW2 = try source2.restShape(refLayer, proj, "weight")
            let dtW2 = try source2.partDType(refLayer, proj, "weight")
            let sbW2 = try source2.sliceBytes(refLayer, proj, "weight")
            let arrW2 = MLXArray.zeros([Swift.max(M2, 1)] + restW2, dtype: dtW2)
            arrW2.eval()
            guard let bufW2 = arrW2.asMTLBuffer(device: device, noCopy: true) else {
                throw NSError(domain: "MixedExpertArena", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "asMTLBuffer failed: \(proj).weight (w2)"])
            }
            w2Slots[proj] = Slot(arr: arrW2, buf: bufW2, ptr: bufW2.contents(), sliceBytes: sbW2)

            for part in ["scales", "biases"] {
                let rest4 = try source4.restShape(refLayer, proj, part)
                let rest2 = try source2.restShape(refLayer, proj, part)
                let sb4 = try source4.sliceBytes(refLayer, proj, part)
                let sb2 = try source2.sliceBytes(refLayer, proj, part)
                guard rest4 == rest2, sb4 == sb2 else {
                    throw NSError(domain: "MixedExpertArena", code: 2, userInfo: [
                        NSLocalizedDescriptionKey:
                            "\(proj).\(part): source4/source2 shape mismatch (\(rest4)/\(sb4) vs \(rest2)/\(sb2))",
                    ])
                }
                let dt = try source4.partDType(refLayer, proj, part)
                let arr = MLXArray.zeros([K4 + M2] + rest4, dtype: dt)
                arr.eval()
                guard let buf = arr.asMTLBuffer(device: device, noCopy: true) else {
                    throw NSError(domain: "MixedExpertArena", code: 1,
                                  userInfo: [NSLocalizedDescriptionKey: "asMTLBuffer failed: \(proj).\(part)"])
                }
                uniformSlots["\(proj).\(part)"] = Slot(arr: arr, buf: buf, ptr: buf.contents(), sliceBytes: sb4)
            }
        }
    }

    /// Load 4-bit expert `e` into core `slot` (< K4): weight → w4[slot], scales/biases → uniform[slot].
    public func loadCore(_ layer: Int, _ e: Int, slot: Int) {
        DispatchQueue.concurrentPerform(iterations: ExpertSource.projs.count) { idx in
            let proj = ExpertSource.projs[idx]
            let w4 = w4Slots[proj]!
            try? source4.preadInto(w4.ptr + slot * w4.sliceBytes, layer, proj, "weight", e)
            for part in ["scales", "biases"] {
                let u = uniformSlots["\(proj).\(part)"]!
                try? source4.preadInto(u.ptr + slot * u.sliceBytes, layer, proj, part, e)
            }
        }
    }

    /// Load 2-bit experts into GLOBAL tail slots (≥ K4): weight → w2[slot−K4], scales/biases → uniform[slot].
    public func loadTailMany(_ layer: Int, _ jobs: [(e: Int, slot: Int)]) {
        guard !jobs.isEmpty else { return }
        let np = ExpertSource.projs.count
        DispatchQueue.concurrentPerform(iterations: jobs.count * np) { k in
            let (e, slot) = jobs[k / np]
            let proj = ExpertSource.projs[k % np]
            let local = slot - K4
            let w2 = w2Slots[proj]!
            try? source2.preadInto(w2.ptr + local * w2.sliceBytes, layer, proj, "weight", e)
            for part in ["scales", "biases"] {
                let u = uniformSlots["\(proj).\(part)"]!
                try? source2.preadInto(u.ptr + slot * u.sliceBytes, layer, proj, part, e)
            }
        }
    }

    /// Buffers in gqmmMixSwigluRows order: [gW4,gW2,gS,gB, uW4,uW2,uS,uB, dW4,dW2,dS,dB].
    public func gatherBuffers12(device: MTLDevice) -> [MTLBuffer]? {
        if let c = cachedBuffers { return c }
        var out: [MTLBuffer] = []
        for proj in ExpertSource.projs {
            guard let w4 = w4Slots[proj], let w2 = w2Slots[proj],
                  let sc = uniformSlots["\(proj).scales"], let bi = uniformSlots["\(proj).biases"] else {
                return nil
            }
            // サニティチェック: scales/biases は f16 でなければならない(raw kernels は half を読む) — ArenaExpertProvider precedent.
            if sc.arr.dtype != .float16 {
                print("[MixedExpertArena] ERROR: \(proj).scales dtype=\(sc.arr.dtype), expected .float16")
                return nil
            }
            out.append(w4.buf); out.append(w2.buf); out.append(sc.buf); out.append(bi.buf)
        }
        cachedBuffers = out
        return out
    }

    /// Per-slot byte stride for (proj, cls) where cls ∈ {"w4","w2","scales","biases"}.
    public func sliceBytes(_ proj: String, _ cls: String) -> Int {
        switch cls {
        case "w4": return w4Slots[proj]?.sliceBytes ?? 0
        case "w2": return w2Slots[proj]?.sliceBytes ?? 0
        case "scales": return uniformSlots["\(proj).scales"]?.sliceBytes ?? 0
        case "biases": return uniformSlots["\(proj).biases"]?.sliceBytes ?? 0
        default: return 0
        }
    }
}

/// Thin per-layer cache: owns arena + state; delegates ensure/pinCore and drives the arena pread.
/// NO protocol conformance in this wave (W3 decides the provider seam).
public final class MixedLayerExpertCache {
    public let arena: MixedExpertArena
    let layer: Int
    var state: MixedCacheState
    public var hits: Int { state.hits }
    public var misses: Int { state.misses }

    public init(device: MTLDevice, source4: ExpertSource, source2: ExpertSource,
                K4: Int, M2: Int, layer: Int) throws {
        self.arena = try MixedExpertArena(device: device, source4: source4, source2: source2,
                                          K4: K4, M2: M2, refLayer: layer)
        self.layer = layer
        self.state = MixedCacheState(K4: K4, M2: M2)
    }

    public func ensure(_ experts: [Int]) -> [Int: Int]? {
        guard let (slots, missJobs) = state.ensure(experts) else { return nil }
        if !missJobs.isEmpty { arena.loadTailMany(layer, missJobs) }
        return slots
    }

    public func pinCore(_ experts: [Int]) -> Bool {
        guard state.pinCore(experts) else { return false }
        for e in experts {
            let s = state.coreOf[e]!
            arena.loadCore(layer, e, slot: s)
        }
        return true
    }

    // ── W3b accessors (BoltServe mixed freeze) ──────────────────────────────
    public var K4: Int { arena.K4 }
    public var M2: Int { arena.M2 }
    public var corePinned: Bool { !state.coreOf.isEmpty }
    public var coreOf: [Int: Int] { state.coreOf }
    /// Current GLOBAL slot of expert e (core or tail), nil if not resident.
    public func slotOfExpert(_ e: Int) -> Int? { state.coreOf[e] ?? state.tailSlotOf[e] }
    public func buddyTable(coact: [[Int]], nE: Int, fallbackSlot: Int) -> (table: [Int32], buddyExpert: [Int32]) {
        state.buddyTable(coact: coact, nE: nE, fallbackSlot: fallbackSlot)
    }
}

/// W3b: mixed provider for the fused engine. Conforms to SeedlessFusedExpertProvider so the
/// forward init consumes it like ArenaExpertProvider — gatherBuffers returns the TWELVE-buffer
/// mixed layout (prepareMoEBlockBufs branches on count>=12) and mixK4 the core width.
/// `C` reports M2 (NOT K4+M2): the protocol's C is the ensure-capacity contract used by the
/// strict-prefill chunk partitioner, and the worst case (zero core hits in a chunk) must still
/// fit the tail LRU — core hits only reduce demand. Global slot ids still span 0..<K4+M2.
public final class MixedArenaExpertProvider: SeedlessFusedExpertProvider {
    public let cache: MixedLayerExpertCache
    public var C: Int { cache.M2 }
    public var mixK4: Int { cache.K4 }
    private var cachedBuffers: [MTLBuffer]? = nil

    public init(cache: MixedLayerExpertCache) { self.cache = cache }

    public func gatherBuffers(device: MTLDevice) -> [MTLBuffer]? {
        if let c = cachedBuffers { return c }
        let bufs = cache.arena.gatherBuffers12(device: device)
        cachedBuffers = bufs
        return bufs
    }

    public func ensure(_ experts: [Int]) -> [Int: Int] {
        if let m = cache.ensure(experts) { return m }
        // Overflow (distinct tail > M2) cannot happen through the chunk partitioner (C = M2
        // contract above); direct callers exceeding it get a loud clamp, never silent aliasing.
        print("[MixedArenaExpertProvider] ensure overflow: \(experts.count) experts > tail capacity \(cache.M2) — clamping")
        var seen = Set<Int>(), kept: [Int] = []
        for e in experts where seen.insert(e).inserted {
            if kept.count >= cache.M2 { break }
            kept.append(e)
        }
        return cache.ensure(kept) ?? [:]
    }
}
