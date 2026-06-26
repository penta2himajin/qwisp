import Foundation
import MLX
import MLXRandom
import Metal

/// 持続 arena: N slot ぶんの switch_mlp expert を native MLXArray で常駐保持し、
/// ExpertSource から in-place pread で各 slot を上書き（concat 無し）。全 layer で形状共通なので使い回す。
/// swift-persistent-arena の核: native 配列を asMTLBuffer(noCopy).contents() に直接書込→gather_qmm に反映。
public final class ExpertArena {
    struct Slot { let arr: MLXArray; let buf: MTLBuffer; let ptr: UnsafeMutableRawPointer; let sliceBytes: Int }
    public let N: Int
    let device: MTLDevice
    let source: ExpertSource
    var slots: [String: Slot] = [:]   // "proj.part" -> Slot

    public init(device: MTLDevice, source: ExpertSource, N: Int = 64, refLayer: Int = 0) throws {
        self.device = device; self.source = source; self.N = N
        for proj in ExpertSource.projs {
            for part in ExpertSource.parts {
                let rest = try source.restShape(refLayer, proj, part)
                let dt = try source.partDType(refLayer, proj, part)
                let sb = try source.sliceBytes(refLayer, proj, part)
                let arr = MLXArray.zeros([N] + rest, dtype: dt)
                arr.eval()
                guard let buf = arr.asMTLBuffer(device: device, noCopy: true) else {
                    throw NSError(domain: "ExpertArena", code: 1)
                }
                slots["\(proj).\(part)"] = Slot(arr: arr, buf: buf, ptr: buf.contents(), sliceBytes: sb)
            }
        }
    }

    /// experts[i] を slot i に in-place pread（9 テンソル）。concat/再確保なし。
    public func load(_ layer: Int, _ experts: [Int]) throws {
        precondition(experts.count <= N, "arena slot 不足: \(experts.count) > \(N)")
        for (i, e) in experts.enumerated() {
            for proj in ExpertSource.projs {
                for part in ExpertSource.parts {
                    let s = slots["\(proj).\(part)"]!
                    try source.preadInto(s.ptr + i * s.sliceBytes, layer, proj, part, e)
                }
            }
        }
    }

    /// expert e を指定 slot に in-place pread（cache の miss ロード用, 9 テンソル並列）。
    public func loadOne(_ layer: Int, _ e: Int, slot: Int) {
        DispatchQueue.concurrentPerform(iterations: ExpertSource.projs.count * ExpertSource.parts.count) { idx in
            let proj = ExpertSource.projs[idx / ExpertSource.parts.count]
            let part = ExpertSource.parts[idx % ExpertSource.parts.count]
            let s = slots["\(proj).\(part)"]!
            try? source.preadInto(s.ptr + slot * s.sliceBytes, layer, proj, part, e)
        }
    }

    /// 複数 (expert, slot) の全 9 テンソルを一括並列 pread（層内 miss をまとめる）。
    public func loadMany(_ layer: Int, _ jobs: [(e: Int, slot: Int)]) {
        let np = ExpertSource.projs.count * ExpertSource.parts.count   // 9
        DispatchQueue.concurrentPerform(iterations: jobs.count * np) { k in
            let (e, slot) = jobs[k / np]
            let idx = k % np
            let proj = ExpertSource.projs[idx / ExpertSource.parts.count]
            let part = ExpertSource.parts[idx % ExpertSource.parts.count]
            let s = slots["\(proj).\(part)"]!
            try? source.preadInto(s.ptr + slot * s.sliceBytes, layer, proj, part, e)
        }
    }

    public func arr(_ proj: String, _ part: String) -> MLXArray { slots["\(proj).\(part)"]!.arr }
}

/// 1 層ぶんの LRU expert キャッシュ。C slot の native arena を token を跨いで持続させ、
/// hit は pread を省く（8GB 予算を expert cache に使う）。miss は LRU evict→slot へ pread。
public final class LayerExpertCache {
    let arena: ExpertArena         // C slots, この層専用
    let layer: Int
    let C: Int
    var slotOf: [Int: Int] = [:]   // expert id -> slot
    var expertAt: [Int]            // slot -> expert id (-1 = 空)
    var tick: [Int]                // slot -> 最終使用 tick（LRU）
    var clock = 0
    public private(set) var hits = 0
    public private(set) var misses = 0
    nonisolated(unsafe) public static var ensureNanos: UInt64 = 0   // ensure(CPU+IO) 累積時間（全層）

    // adaptive fast: 直近 fast forward の inds（miss 検出用、eval 済を読む）
    var lastInds: MLXArray?
    /// lastInds のうち cache 未収容（fast で wrong-slot になった）expert 数。
    public func missCount() -> Int {
        guard let li = lastInds else { return 0 }
        var m = 0
        for e in li.asArray(Int32.self) where slotOf[Int(e)] == nil { m += 1 }
        return m
    }

    // GPU-side slot table（expert id -> slot, 未cache=0）。sync 無し remap 用。
    var slotTableDirty = true
    var slotTableGPU: MLXArray?
    public func gpuSlotTable(numExperts: Int) -> MLXArray {
        if slotTableDirty || slotTableGPU == nil {
            var t = [Int32](repeating: 0, count: numExperts)
            for (e, s) in slotOf { t[e] = Int32(s) }
            let arr = MLXArray(t, [numExperts]); arr.eval()
            slotTableGPU = arr; slotTableDirty = false
        }
        return slotTableGPU!
    }

    public init(device: MTLDevice, source: ExpertSource, layer: Int, C: Int) throws {
        self.arena = try ExpertArena(device: device, source: source, N: C, refLayer: layer)
        self.layer = layer; self.C = C
        expertAt = [Int](repeating: -1, count: C)
        tick = [Int](repeating: 0, count: C)
    }

    /// experts(U) を cache に確保（miss は pread）し、各 U[i] の slot を返す。
    /// miss の slot 割当を先に済ませ、全 miss×9 テンソルの pread を一括並列化。
    public func ensure(_ experts: [Int]) -> [Int: Int] {
        let t0 = DispatchTime.now().uptimeNanoseconds
        defer { LayerExpertCache.ensureNanos += DispatchTime.now().uptimeNanoseconds - t0 }
        var result: [Int: Int] = [:]
        var missList: [(e: Int, slot: Int)] = []
        for e in experts {
            clock += 1
            if let s = slotOf[e] { tick[s] = clock; hits += 1; result[e] = s; continue }
            misses += 1
            var slot = -1
            for s in 0 ..< C where expertAt[s] == -1 { slot = s; break }
            if slot == -1 {
                var oldest = 0
                for s in 1 ..< C where tick[s] < tick[oldest] { oldest = s }
                slot = oldest
                slotOf.removeValue(forKey: expertAt[slot])
            }
            expertAt[slot] = e; slotOf[e] = slot; tick[slot] = clock
            result[e] = slot; missList.append((e, slot)); slotTableDirty = true
        }
        // 全 miss × 9 テンソルを一括並列 pread（層内 miss をまとめてレイテンシ重畳）
        if !missList.isEmpty { arena.loadMany(layer, missList) }
        return result
    }
}

/// 持続 arena 経由で switch_mlp を回す streaming MoE（gate/shared は resident）。
/// cache!=nil で per-layer LRU キャッシュ、nil なら毎回 arena に全ロード。
public final class StreamingMoEBlock {
    let topK: Int, numExperts: Int, normTopk: Bool, expertBits: Int
    let gate: Proj, shGate: Proj, shUp: Proj, shDown: Proj, sharedGate: Proj
    let arena: ExpertArena
    let cache: LayerExpertCache?
    let layer: Int
    nonisolated(unsafe) public static var syncNanos: UInt64 = 0   // inds.asArray(GPU→CPU drain) 累積
    nonisolated(unsafe) public static var probeNoSync = false      // 天井計測: GPU remap, 毎層 sync 無し

    public init(topK: Int, numExperts: Int, normTopk: Bool, expertBits: Int, layer: Int,
                gate: Proj, shGate: Proj, shUp: Proj, shDown: Proj, sharedGate: Proj,
                arena: ExpertArena, cache: LayerExpertCache? = nil) {
        self.topK = topK; self.numExperts = numExperts; self.normTopk = normTopk
        self.expertBits = expertBits; self.layer = layer
        self.gate = gate; self.shGate = shGate; self.shUp = shUp; self.shDown = shDown
        self.sharedGate = sharedGate; self.arena = arena; self.cache = cache
    }

    private func gatherQmm(_ x: MLXArray, _ store: ExpertArena, _ proj: String, _ remap: MLXArray) -> MLXArray {
        gatherQuantizedMatmul(x, store.arr(proj, "weight"), scales: store.arr(proj, "scales"),
                              biases: store.arr(proj, "biases"), rhsIndices: remap,
                              transpose: true, groupSize: 64, bits: expertBits, mode: .affine,
                              sortedIndices: false)
    }

    public func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        let gates = MLX.softmax(gate.apply(x), axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: numExperts - topK, axis: -1)
        let inds = order[0..., (numExperts - topK)...]                 // [T,K]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopk { scores = scores / scores.sum(axis: -1, keepDims: true) }

        // 天井計測: GPU-side slot table で remap、per-layer sync/ensure を完全に省く（miss は近似）
        if StreamingMoEBlock.probeNoSync, let c = cache {
            c.lastInds = inds.asType(.int32)                     // adaptive miss 検出用
            let table = c.gpuSlotTable(numExperts: numExperts)
            let remap = MLX.take(table, inds.asType(.int32), axis: 0).asType(.uint32)
            let xe = x.expandedDimensions(axes: [-2, -3])
            let g = gatherQmm(xe, c.arena, "gate_proj", remap)
            let u = gatherQmm(xe, c.arena, "up_proj", remap)
            let h = (g * MLX.sigmoid(g)) * u
            let d = gatherQmm(h, c.arena, "down_proj", remap).squeezed(axis: -2)
            let y = (d * scores.expandedDimensions(axis: -1)).sum(axis: -2)
            let sg = shGate.apply(x), su = shUp.apply(x)
            let sharedY = shDown.apply((sg * MLX.sigmoid(sg)) * su)
            return y + MLX.sigmoid(sharedGate.apply(x)) * sharedY
        }

        // distinct experts U（CPU 同期）
        let ts = DispatchTime.now().uptimeNanoseconds
        let flat = inds.asType(.int32).asArray(Int32.self)
        StreamingMoEBlock.syncNanos += DispatchTime.now().uptimeNanoseconds - ts
        var seen = Set<Int>(); var U: [Int] = []
        for e32 in flat { let e = Int(e32); if seen.insert(e).inserted { U.append(e) } }

        let store: ExpertArena
        var remapVals = [Int32](repeating: 0, count: flat.count)
        if let c = cache {
            let slotOf = c.ensure(U)                                   // hit は pread 省略
            for (j, e32) in flat.enumerated() { remapVals[j] = Int32(slotOf[Int(e32)]!) }
            store = c.arena
        } else {
            var slot: [Int: Int] = [:]
            for (i, e) in U.enumerated() { slot[e] = i }
            try arena.load(layer, U)                                   // in-place pread（concat 無し）
            for (j, e32) in flat.enumerated() { remapVals[j] = Int32(slot[Int(e32)]!) }
            store = arena
        }
        let remap = MLXArray(remapVals, inds.shape).asType(.uint32)

        let xe = x.expandedDimensions(axes: [-2, -3])
        let g = gatherQmm(xe, store, "gate_proj", remap)
        let u = gatherQmm(xe, store, "up_proj", remap)
        let h = (g * MLX.sigmoid(g)) * u
        let d = gatherQmm(h, store, "down_proj", remap).squeezed(axis: -2)    // [T,K,H]
        let y = (d * scores.expandedDimensions(axis: -1)).sum(axis: -2)

        let sg = shGate.apply(x), su = shUp.apply(x)
        let sharedY = shDown.apply((sg * MLX.sigmoid(sg)) * su)
        let gateScale = MLX.sigmoid(sharedGate.apply(x))
        return y + gateScale * sharedY
    }
}

public enum StreamingMoEValidation {
    public static func run(modelDir: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let model = QwispModel(store: store)   // resident MoEBlock(layer0) を基準に
        let source = try ExpertSource(modelDir: modelDir)
        let arena = try ExpertArena(device: device, source: source, N: 64)

        let p = "language_model.model.layers.0.mlp"
        func q8(_ n: String) -> Proj {
            .quantized(store.req("\(n).weight"), store.req("\(n).scales"), store.req("\(n).biases"), 8)
        }
        func q4(_ n: String) -> Proj {
            .quantized(store.req("\(n).weight"), store.req("\(n).scales"), store.req("\(n).biases"), 4)
        }
        let stream = StreamingMoEBlock(
            topK: 8, numExperts: 256, normTopk: true, expertBits: 4, layer: 0,
            gate: q8("\(p).gate"), shGate: q4("\(p).shared_expert.gate_proj"),
            shUp: q4("\(p).shared_expert.up_proj"), shDown: q4("\(p).shared_expert.down_proj"),
            sharedGate: q8("\(p).shared_expert_gate"), arena: arena)
        let resident = model.buildMoE(p)

        // 同じ x で resident MoE と streaming MoE を比較（T=4 decode 規模）
        let x = MLXRandom.normal([4, 2048]).asType(.float16)
        let yR = resident(x); let yS = try stream(x)
        yR.eval(); yS.eval()
        let d = MLX.max(MLX.abs(yR.asType(.float32) - yS.asType(.float32))).item(Float.self)
            / (MLX.max(MLX.abs(yR.asType(.float32))).item(Float.self) + 1e-9)
        let ok = d < 1e-4
        return String(format: "[S2] streaming arena MoE vs resident: y_rel=%.2e  %@",
                      d, ok ? "OK ✅ in-place arena 正しい(concat無)" : "MISMATCH ❌")
    }
}
