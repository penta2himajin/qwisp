import Foundation
import MLX

// QWISP_RUN=spec-replay-check (needs QWISP_SPEC_REPLAY=1): byte-compare the decode state after a
// partial-reject verify committed by replay against rollback + rebuild forward, for every accepted
// row count 1...D+1 of a D-draft verify, then compare the next greedy tokens from both states.
extension Tell {
    public static func specReplayCheck(modelDir: String) -> String {
        guard SeedlessFusedVerify.SeedlessFusedForward.specReplay else {
            return "[spec-replay-check] set QWISP_SPEC_REPLAY=1\nSPECREPLAY done"
        }
        guard let store = try? WeightStore(modelDir: modelDir) else { return "[spec-replay-check] load fail\nSPECREPLAY done" }
        store.residentAll()
        let engine = SeedlessEngine.build(store: store)
        let D = Swift.min(15, Swift.max(1, Tell.envInt("QWISP_SPEC_REPLAY_D", 8)))
        let ctx = Tell.envInt("QWISP_SPEC_REPLAY_CTX", 300)
        guard let (b, fwd) = Tell.fusedBackendWithFwd(engine: engine, maxM: 32, maxSeqLen: ctx + 64) else {
            return "[spec-replay-check] backend fail\nSPECREPLAY done"
        }
        fwd.prepareSpecReplay()
        let prompt = (0 ..< ctx).map { Int32((($0 &* 7 &+ 13) % 5000) + 100) }
        guard let lastNormed = Tell.prefill(promptIds: prompt, backend: b),
              let lg0 = engine.logits(lastNormed, M: 1) else { return "[spec-replay-check] prefill fail\nSPECREPLAY done" }
        MLX.eval([lg0])
        let u = MLX.argMax(lg0[0], axis: -1).item(Int.self)
        let base = fwd.persistentStateData()
        func restore() -> Bool { fwd.restorePersistentState(base) }
        // True greedy continuation from u (to build drafts that are right for exactly p tokens).
        guard restore() else { return "[spec-replay-check] restore fail\nSPECREPLAY done" }
        var truth: [Int] = []
        var cur = u
        for _ in 0 ..< D + 1 {
            guard let e = b.stepArgmax([Int32(cur)]) else { return "[spec-replay-check] step fail\nSPECREPLAY done" }
            truth.append(e[0]); cur = e[0]
        }
        func next(_ from: Int, _ n: Int) -> [Int]? {
            var t = from, out: [Int] = []
            for _ in 0 ..< n { guard let e = b.stepArgmax([Int32(t)]) else { return nil }; out.append(e[0]); t = e[0] }
            return out
        }
        var lines = ["[spec-replay-check] ctx=\(ctx) D=\(D) u=\(u)"]
        var fails = 0
        for p in 0 ... D {
            // drafts: the first p are the true continuation, the rest are wrong.
            let drafts = (0 ..< D).map { i in i < p ? truth[i] : (truth[i] + 1) % 248000 }
            let verify = [Int32(u)] + drafts.map { Int32($0) }
            // A: replay
            guard restore() else { return "restore fail" }
            let sA = fwd.snapshot()
            guard let eA = b.stepArgmax(verify) else { return "verify A fail" }
            var pa = 0; while pa < D && drafts[pa] == eA[pa] { pa += 1 }
            // QWISP_SPEC_REPLAY_FAULT=1 replays one row too many: the check must then report FAIL.
            let fault = Tell.envInt("QWISP_SPEC_REPLAY_FAULT", 0)
            let replayed = pa < D ? fwd.replayAccepted(sA, rows: Swift.min(pa + 1 + fault, D + 1)) : true
            fwd.syncReplay()
            let stateA = fwd.persistentStateData()
            let nextA = next(eA[pa], 4) ?? []
            // B: rollback + rebuild
            guard restore() else { return "restore fail" }
            let sB = fwd.snapshot()
            guard let eB = b.stepArgmax(verify) else { return "verify B fail" }
            var pb = 0; while pb < D && drafts[pb] == eB[pb] { pb += 1 }
            if pb < D {
                fwd.rollbackOneStep(sB)
                guard b.forward([Int32(u)] + drafts.prefix(pb).map { Int32($0) }) != nil else { return "rebuild fail" }
            }
            let stateB = fwd.persistentStateData()
            let nextB = next(eB[pb], 4) ?? []
            let ok = replayed && pa == pb && stateA == stateB && nextA == nextB && !nextA.isEmpty
            if !ok { fails += 1 }
            lines.append("  p=\(p) accepted=\(pa)/\(pb) replayed=\(replayed) stateEqual=\(stateA == stateB) (\(stateA.count) B) next \(nextA) vs \(nextB) \(ok ? "OK" : "FAIL")")
        }
        lines.append(fails == 0 ? "SPECREPLAY PASS (\(D + 1)/\(D + 1))" : "SPECREPLAY FAIL (\(fails) of \(D + 1))")
        return lines.joined(separator: "\n") + "\nSPECREPLAY done"
    }
}
