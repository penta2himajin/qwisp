import Foundation

/// Pure-CPU types for the bolt async refresh plan and chunk-swap (notes/14).
/// makePlan is a pure read; CacheState mirrors LayerExpertCache CPU bookkeeping for unit tests.
///
/// Design:
///   makePlan   — pure function: snapshot counts/coact/slot state → Plan (no mutation)
///   CacheState — value type mirroring LayerExpertCache's CPU bookkeeping for unit tests
///   applyChunkCPU  — swap one chunk into CacheState (no pread, no Metal)
///   rebuildBuddyCPU — same co-activation buddy algorithm as buildBuddyTable
///   syncEnsure — plain LRU ensure on CacheState (reference path for chunk_swap test)
public enum BoltAsyncRefresh {

    // ── Plan types ─────────────────────────────────────────────────────────

    public struct Job: Equatable {
        public let expert: Int
        public let victimSlot: Int
        public init(expert: Int, victimSlot: Int) {
            self.expert = expert; self.victimSlot = victimSlot
        }
    }

    /// Immutable plan produced at an R-boundary (pure read, no mutation).
    public struct Plan {
        /// Top-C expert ids from observation window (sorted for determinism).
        public let newTop: [Int]
        /// Experts to load: newTop ∖ currently resident.
        public let diff: [Int]
        /// Each diff expert paired with its pre-assigned victim slot (LRU order,
        /// excluding pinnedSlots and slots holding newTop-resident experts).
        public let jobs: [Job]
        /// jobs split into ≤B groups; background pread processes chunk j while
        /// the decoder emits tokens; chunk j is swapped at out.count ≥ boundary+(j+1)*S.
        public let chunks: [[Job]]
    }

    /// STUB — implementation pending
    /// Build a single-layer async refresh plan from the given observation window and LRU state.
    ///
    /// Algorithm (must match test hand-calc):
    ///   1. newTop = top-C experts sorted by counts desc, tie-break by lower expert id asc.
    ///   2. diff   = newTop ∖ Set(slotOf.keys).
    ///   3. For each expert e in diff (same sort order as newTop):
    ///        victim = LRU slot excluding pinnedSlots ∪ {slots of newTop-resident experts}.
    ///        Assign victim; mark as "taken" so the next expert doesn't reuse it.
    ///   4. chunks = diff jobs partitioned into ⌈len/B⌉ groups of ≤B.
    ///
    public static func makePlan(
        counts: [Int],
        coact: [[Int]],
        slotOf: [Int: Int],
        expertAt: [Int],
        tick: [Int],
        pinnedSlots: Set<Int>,
        C: Int,
        nE: Int,
        B: Int
    ) -> Plan? {
        // 1. newTop = top-C experts sorted by counts DESC, tie-break lower expert-id ASC.
        let newTop = counts.enumerated()
            .sorted { e0, e1 in e0.element != e1.element ? e0.element > e1.element : e0.offset < e1.offset }
            .prefix(C)
            .map { $0.offset }

        // 2. diff = newTop elements NOT in Set(slotOf.keys), preserving newTop order.
        let resident = Set(slotOf.keys)
        let diff = newTop.filter { !resident.contains($0) }

        // 3. excluded = pinnedSlots ∪ {slotOf[e]! for e in newTop if slotOf[e] != nil}
        var excluded = pinnedSlots
        for e in newTop { if let s = slotOf[e] { excluded.insert(s) } }

        // 4. Assign victim slots (LRU tick, excluding `excluded`, grows as victims taken).
        var takenSlots = excluded
        var jobs: [Job] = []
        for e in diff {
            var victim = -1
            for s in 0 ..< expertAt.count where !takenSlots.contains(s) {
                if victim == -1 || tick[s] < tick[victim] { victim = s }
            }
            if victim == -1 { return nil }
            takenSlots.insert(victim)
            jobs.append(Job(expert: e, victimSlot: victim))
        }

        // 5. chunks: stride through jobs by B.
        var chunks: [[Job]] = []
        var i = 0
        while i < jobs.count {
            let end = Swift.min(i + B, jobs.count)
            chunks.append(Array(jobs[i ..< end]))
            i += B
        }

        // 6. return Plan.
        return Plan(newTop: newTop, diff: diff, jobs: jobs, chunks: chunks)
    }

    // ── CPU-only cache state ───────────────────────────────────────────────

    /// Mirrors the CPU bookkeeping of LayerExpertCache for pure-CPU unit tests.
    /// No Metal device or ExpertSource required.
    public struct CacheState {
        public var slotOf: [Int: Int]        // expert → slot
        public var expertAt: [Int]           // slot → expert (-1 = empty)
        public var tick: [Int]              // slot → last-use LRU tick
        public var clock: Int
        public var buddyTableCPU: [Int32]   // expert → slot (after rebuildBuddyCPU)
        public var buddyExpertCPU: [Int32]  // expert → buddy expert id (-1 = none)

        public init(slotOf: [Int: Int], expertAt: [Int], tick: [Int], clock: Int,
                    buddyTableCPU: [Int32], buddyExpertCPU: [Int32]) {
            self.slotOf = slotOf; self.expertAt = expertAt
            self.tick = tick; self.clock = clock
            self.buddyTableCPU = buddyTableCPU; self.buddyExpertCPU = buddyExpertCPU
        }

        /// Apply one chunk of (expert, victimSlot) assignments:
        ///   for each job: evict expertAt[victimSlot] from slotOf, assign expert → victimSlot,
        ///   update expertAt and tick. Does NOT call ensure() or pread.
        /// Slot-consistent on return: no double-mapping, expertAt[slotOf[e]] == e for all e.
        public mutating func applyChunkCPU(jobs: [Job]) -> Bool {
            for job in jobs {
                let v = job.victimSlot
                let cur = expertAt[v]
                if cur >= 0 && cur != job.expert {
                    slotOf.removeValue(forKey: cur)
                }
                slotOf[job.expert] = v
                expertAt[v] = job.expert
                clock += 1
                tick[v] = clock
            }
            return true
        }

        /// Rebuild buddyTableCPU and buddyExpertCPU from current slotOf, using the same
        /// co-activation algorithm as LayerExpertCache.buildBuddyTable (sorted hot set,
        /// rotation tie-break per expert id, slot-0 fallback on zero coact).
        public mutating func rebuildBuddyCPU(coact: [[Int]], nE: Int) -> Bool {
            // Mirrors LayerExpertCache.buildBuddyTable (ExpertArena.swift):
            // sorted hot set, rotation tie-break (i+e)%n, slot-0 fallback on zero coact.
            let hot = slotOf.keys.sorted()
            var bmap = [Int32](repeating: 0, count: nE)
            var bexp = [Int32](repeating: -1, count: nE)
            let n = hot.count
            for e in 0 ..< nE {
                if let s = slotOf[e] { bmap[e] = Int32(s); bexp[e] = Int32(e); continue }
                var bestH = -1, bestC = -1
                for i in 0 ..< n {
                    let h = hot[(i + e) % n]
                    let cc = coact[e][h]
                    if cc > bestC { bestC = cc; bestH = h }
                }
                if bestH >= 0 && bestC > 0 {
                    bmap[e] = Int32(slotOf[bestH]!)
                    bexp[e] = Int32(bestH)
                } else {
                    bmap[e] = 0
                }
            }
            buddyTableCPU = bmap
            buddyExpertCPU = bexp
            return true
        }

        /// Reference LRU ensure: apply each expert in order, evicting the LRU slot
        /// excluding pinnedSlots (same algorithm as LayerExpertCache.ensure(), but on
        /// pure CPU state without pread). Used as the sync reference in chunk_swap tests.
        public mutating func syncEnsure(experts: [Int], pinnedSlots: Set<Int>) -> Bool {
            // Mirrors LayerExpertCache.ensure(): plain LRU on pure CPU state (no pread).
            let C = expertAt.count
            for e in experts {
                clock += 1
                if let s = slotOf[e] { tick[s] = clock; continue }   // hit
                // miss: find victim slot
                var slot = -1
                for s in 0 ..< C where expertAt[s] == -1 { slot = s; break }   // empty slot first
                if slot == -1 {
                    // LRU excluding pinnedSlots
                    var oldest = -1
                    for s in 0 ..< C where !pinnedSlots.contains(s) {
                        if oldest == -1 || tick[s] < tick[oldest] { oldest = s }
                    }
                    if oldest == -1 { return false }
                    slot = oldest
                    slotOf.removeValue(forKey: expertAt[slot])
                }
                expertAt[slot] = e
                slotOf[e] = slot
                tick[slot] = clock
            }
            return true
        }
    }
}
