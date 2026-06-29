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
    nonisolated(unsafe) public static var preadNanos: UInt64 = 0    // loadMany(pread IO) のみ
    nonisolated(unsafe) public static var missTotal: Int = 0        // 累積 miss 数
    // ★ issue#7 Step 0: per-layer all-resident 計測（ensure 前=no-sync が exact になる層か）。
    nonisolated(unsafe) public static var measureResident = false
    nonisolated(unsafe) public static var residAllHit: [Int: Int] = [:]   // 層→(top-8 全常駐だった token 数)
    nonisolated(unsafe) public static var residTotal: [Int: Int] = [:]    // 層→(計測 token 数)
    nonisolated(unsafe) public static var residMissSum: [Int: Int] = [:]  // 層→(miss expert 数の累積)

    // adaptive fast: 直近 fast forward の inds（miss 検出用、eval 済を読む）
    var lastInds: MLXArray?
    public var lastGateInput: MLXArray?   // Tell M2: この層の MoE 入力(=真の gate 入力)を capture
    public var preAttnInput: MLXArray?    // 予測器 calib: この層の pre-attention 入力（層入力）を capture
    // 選択的マージン prefetch（M0）: 広い top-marginK と確信度(top-K mass)を別捕捉。
    // 不確実層だけ marginK を prefetch するため、CPU 側で τ 判定に使う。
    public var lastMarginInds: MLXArray?  // 広い top-marginK 候補
    public var lastConf: MLXArray?        // 各 row の top-K softmax mass（[T]）
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
    var slotVersion = 0                 // slotOf 変更ごとに bump（GPU 配列の再構築判定）
    public var pinnedSlots: Set<Int> = [] // hot pin: LRU 退避から保護する slot
    var hotMaskArr: MLXArray?          // GPU hot/cached マスク [numExperts]（1=cached）
    var hotMaskVer = -1
    public var buddyTable: MLXArray?   // BuddyMoE: cold expert → 最類似 hot expert の slot（slot-0 garbage 回避）
    public var slotMap: [Int: Int] { slotOf }   // output-sim buddy 構築用（expert→slot）
    public func gpuSlotTable(numExperts: Int) -> MLXArray {
        if slotTableDirty || slotTableGPU == nil {
            var t = [Int32](repeating: 0, count: numExperts)
            for (e, s) in slotOf { t[e] = Int32(s) }
            let arr = MLXArray(t, [numExperts]); arr.eval()
            slotTableGPU = arr; slotTableDirty = false
        }
        return slotTableGPU!
    }

    /// 現在 cache に居る expert を 1、未cache を 0 とする GPU マスク [numExperts]。
    /// hybrid の per-token hot-miss 計数（routed が cache 内か）に使う。slotVersion で再構築判定。
    public func hotMask(numExperts: Int) -> MLXArray {
        if hotMaskArr == nil || hotMaskVer != slotVersion {
            var m = [Int32](repeating: 0, count: numExperts)
            for (e, _) in slotOf { m[e] = 1 }
            let arr = MLXArray(m, [numExperts]); arr.eval()
            hotMaskArr = arr; hotMaskVer = slotVersion
        }
        return hotMaskArr!
    }

    /// 直近 draft(lastInds) の routed top-K が全て cache 内か（partial-resume の first-miss 判定）。
    /// lastInds は batched eval 済前提（materialized なら asArray は再計算無し）。
    public func indsHot() -> Bool {
        guard let li = lastInds else { return true }
        for e in li.asArray(Int32.self) where slotOf[Int(e)] == nil { return false }
        return true
    }

    /// BuddyMoE: cold expert を「最も共活性化する hot expert」の slot に remap する table を構築。
    /// hot は現在 slotOf にいる expert。coact[e][h] = calib で e と h が同 token で共 routed した回数。
    /// 各 cold e → argmax_h(coact[e][h]) の slot（co-activation 無ければ slot-0 fallback）。
    public func buildBuddyTable(coact: [[Int]], numExperts: Int) {
        let hot = Array(slotOf.keys)
        var bmap = [Int32](repeating: 0, count: numExperts)
        for e in 0 ..< numExperts {
            if let s = slotOf[e] { bmap[e] = Int32(s); continue }    // hot: 自身
            var bestH = -1, bestC = -1
            for h in hot { let cc = coact[e][h]; if cc > bestC { bestC = cc; bestH = h } }
            bmap[e] = (bestH >= 0 && bestC > 0) ? Int32(slotOf[bestH]!) : 0   // cold: buddy slot
        }
        let arr = MLXArray(bmap, [numExperts]); arr.eval()
        buddyTable = arr
    }

    /// experts を常駐ロードし、その slot を pinned に登録（以後 LRU 退避されない）。
    public func pin(_ experts: [Int]) {
        _ = ensure(experts)
        for e in experts { if let s = slotOf[e] { pinnedSlots.insert(s) } }
    }

    public init(device: MTLDevice, source: ExpertSource, layer: Int, C: Int) throws {
        self.arena = try ExpertArena(device: device, source: source, N: C, refLayer: layer)
        self.layer = layer; self.C = C
        expertAt = [Int](repeating: -1, count: C)
        tick = [Int](repeating: 0, count: C)
    }

    /// lastInds(直近 fast forward の routing)の distinct expert を prefetch（cross-layer 予測の駆動）。
    public func prefetchLastInds() {
        guard let li = lastInds else { return }
        var seen = Set<Int>(); var U: [Int] = []
        for e in li.asArray(Int32.self) { let i = Int(e); if seen.insert(i).inserted { U.append(i) } }
        _ = ensure(U)
    }

    /// experts(U) を cache に確保（miss は pread）し、各 U[i] の slot を返す。
    /// miss の slot 割当を先に済ませ、全 miss×9 テンソルの pread を一括並列化。
    public func ensure(_ experts: [Int]) -> [Int: Int] {
        let t0 = DispatchTime.now().uptimeNanoseconds
        defer { LayerExpertCache.ensureNanos += DispatchTime.now().uptimeNanoseconds - t0 }
        var result: [Int: Int] = [:]
        var missList: [(e: Int, slot: Int)] = []
        // ★ issue#7 Step 0: ensure 前(=この token のロード前)の per-layer 残留を計測。
        //   全 distinct expert が既に常駐なら、この層は no-sync gather が exact になる。cold-start で自然蓄積。
        if LayerExpertCache.measureResident {
            let missCnt = experts.reduce(0) { $0 + (slotOf[$1] == nil ? 1 : 0) }
            LayerExpertCache.residTotal[layer, default: 0] += 1
            LayerExpertCache.residMissSum[layer, default: 0] += missCnt
            if missCnt == 0 { LayerExpertCache.residAllHit[layer, default: 0] += 1 }
        }
        for e in experts {
            clock += 1
            if let s = slotOf[e] { tick[s] = clock; hits += 1; result[e] = s; continue }
            misses += 1
            var slot = -1
            for s in 0 ..< C where expertAt[s] == -1 { slot = s; break }
            if slot == -1 {
                // LRU 退避: pinned slot は対象外（hot を保護）
                var oldest = -1
                for s in 0 ..< C where !pinnedSlots.contains(s) {
                    if oldest == -1 || tick[s] < tick[oldest] { oldest = s }
                }
                precondition(oldest != -1, "全 slot が pinned: cold をロードできない（C > pin 数に）")
                slot = oldest
                slotOf.removeValue(forKey: expertAt[slot])
            }
            expertAt[slot] = e; slotOf[e] = slot; tick[slot] = clock
            result[e] = slot; missList.append((e, slot)); slotTableDirty = true; slotVersion += 1
        }
        // 全 miss × 9 テンソルを一括並列 pread（層内 miss をまとめてレイテンシ重畳）
        if !missList.isEmpty {
            let pt = DispatchTime.now().uptimeNanoseconds
            arena.loadMany(layer, missList)
            LayerExpertCache.preadNanos += DispatchTime.now().uptimeNanoseconds - pt
            LayerExpertCache.missTotal += missList.count
        }
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
    nonisolated(unsafe) public static var predictOnly = false      // 軽量予測 pass: routed gather 省略、inds だけ捕捉
    nonisolated(unsafe) public static var captureGateInput = false // Tell M2: 各層の gate 入力を capture
    nonisolated(unsafe) public static var captureInds = false      // calib: 全 mode で routing inds を記録
    nonisolated(unsafe) public static var syncLayers: Set<Int>? = nil  // 適応 sync: この層集合は exact(no-sync 無効)
    nonisolated(unsafe) public static var captureLayerInput = false // 予測器 calib: 層の pre-attention 入力を記録
    nonisolated(unsafe) public static var captureK = 0              // >topK で lastInds に top-K を捕捉（M0 prefetch margin）
    nonisolated(unsafe) public static var marginK = 0               // >topK で lastMarginInds/lastConf を捕捉（M0 選択的マージン）
    nonisolated(unsafe) public static var countHotMiss = false      // hybrid: no-sync 中、routed が cache 外の数を GPU 累積
    nonisolated(unsafe) public static var skipMode = 0              // no-sync 近似改善: 1=cold寄与を0(no renorm), 2=0にして hot再正規化, 3=buddy代替
    nonisolated(unsafe) public static var hotMissAccum: MLXArray? = nil  // 全 MoE 層の hot-miss 累積（token 毎に reset）
    // 層内分解プロファイル（barrier 計測）
    nonisolated(unsafe) public static var profileLayers = false
    nonisolated(unsafe) public static var tGDN: UInt64 = 0
    nonisolated(unsafe) public static var tAttn: UInt64 = 0
    nonisolated(unsafe) public static var tMoEgather: UInt64 = 0
    nonisolated(unsafe) public static var tMoEshared: UInt64 = 0
    nonisolated(unsafe) public static var tNorm: UInt64 = 0
    // GDN 内訳
    nonisolated(unsafe) public static var tGdnInproj: UInt64 = 0
    nonisolated(unsafe) public static var tGdnConv: UInt64 = 0
    nonisolated(unsafe) public static var tGdnKernel: UInt64 = 0
    nonisolated(unsafe) public static var tGdnOut: UInt64 = 0

    public init(topK: Int, numExperts: Int, normTopk: Bool, expertBits: Int, layer: Int,
                gate: Proj, shGate: Proj, shUp: Proj, shDown: Proj, sharedGate: Proj,
                arena: ExpertArena, cache: LayerExpertCache? = nil) {
        self.topK = topK; self.numExperts = numExperts; self.normTopk = normTopk
        self.expertBits = expertBits; self.layer = layer
        self.gate = gate; self.shGate = shGate; self.shUp = shUp; self.shDown = shDown
        self.sharedGate = sharedGate; self.arena = arena; self.cache = cache
    }

    /// 任意 hidden h からこの層の top-k expert inds を予測（cross-layer 予測用、gather しない）。
    public func predictInds(_ h: MLXArray) -> MLXArray {
        let gates = MLX.softmax(gate.apply(h), axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: numExperts - topK, axis: -1)
        return order[0..., (numExperts - topK)...].asType(.int32)
    }

    /// predictInds の幅可変版: 任意 hidden から top-k(k>=topK)を返す（exact-pipeline の prefetch 幅振り用）。
    public func predictIndsK(_ h: MLXArray, _ k: Int) -> MLXArray {
        let kk = Swift.min(Swift.max(k, topK), numExperts)
        let gates = MLX.softmax(gate.apply(h), axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: numExperts - kk, axis: -1)
        return order[0..., (numExperts - kk)...].asType(.int32)
    }

    private func gatherQmm(_ x: MLXArray, _ store: ExpertArena, _ proj: String, _ remap: MLXArray) -> MLXArray {
        gatherQuantizedMatmul(x, store.arr(proj, "weight"), scales: store.arr(proj, "scales"),
                              biases: store.arr(proj, "biases"), rhsIndices: remap,
                              transpose: true, groupSize: 64, bits: expertBits, mode: .affine,
                              sortedIndices: false)
    }

    public func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        if StreamingMoEBlock.captureGateInput { cache?.lastGateInput = x }   // M2: 真の gate 入力を保存
        let gates = MLX.softmax(gate.apply(x), axis: -1, precise: true)
        let order = MLX.argPartition(gates, kth: numExperts - topK, axis: -1)
        let inds = order[0..., (numExperts - topK)...]                 // [T,K]
        if StreamingMoEBlock.captureInds { cache?.lastInds = inds.asType(.int32) }  // calib/計測: 全 mode で routing 記録
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopk { scores = scores / scores.sum(axis: -1, keepDims: true) }

        // 軽量予測 pass: routed gather を省き、inds だけ捕捉して shared expert のみ返す。
        if StreamingMoEBlock.predictOnly, let c = cache {
            c.lastInds = inds.asType(.int32)
            let sg = shGate.apply(x), su = shUp.apply(x)
            let sharedY = shDown.apply((sg * MLX.sigmoid(sg)) * su)
            return MLX.sigmoid(sharedGate.apply(x)) * sharedY
        }
        // 適応 sync: syncLayers に含まれる層は no-sync を無効化し exact 経路へ（hard 層だけ正確化）。
        let noSync = StreamingMoEBlock.probeNoSync
            && !(StreamingMoEBlock.syncLayers?.contains(layer) ?? false)
        // 天井計測 / 適応 no-sync: GPU-side slot table で remap、per-layer sync/ensure を省く（miss は近似）
        if noSync, let c = cache {
            // prefetch margin: captureK>topK なら top-captureK を lastInds に（gather は inds=top8 のまま）
            if StreamingMoEBlock.captureK > topK {
                let ck = StreamingMoEBlock.captureK
                let ordK = MLX.argPartition(gates, kth: numExperts - ck, axis: -1)
                c.lastInds = ordK[0..., (numExperts - ck)...].asType(.int32)
            } else {
                c.lastInds = inds.asType(.int32)                 // adaptive miss 検出用
            }
            // 選択的マージン: top-8 とは別に、広い top-marginK と確信度(top-K mass)を捕捉。
            // CPU 側で層ごとに τ 判定し、不確実層だけ marginK を prefetch する（追加 sync 無し）。
            if StreamingMoEBlock.marginK > topK {
                let mk = StreamingMoEBlock.marginK
                let ordM = MLX.argPartition(gates, kth: numExperts - mk, axis: -1)
                c.lastMarginInds = ordM[0..., (numExperts - mk)...].asType(.int32)
                c.lastConf = MLX.takeAlong(gates, inds, axis: -1).sum(axis: -1)   // [T] 各 row の top-K mass
            }
            // hybrid: この層で routed が cache 外（slot 0 alias になる）数を GPU 累積。
            // token 全層で 0 なら no-sync gather は exact 経路と bit 一致＝採用しても lossless。
            if StreamingMoEBlock.countHotMiss {
                let mask = c.hotMask(numExperts: numExperts)
                let hits = MLX.take(mask, inds.asType(.int32).reshaped([-1]), axis: 0).sum()
                let miss = MLXArray(Int32(inds.shape.reduce(1, *))) - hits
                StreamingMoEBlock.hotMissAccum = StreamingMoEBlock.hotMissAccum.map { $0 + miss } ?? miss
            }
            // skip: cold(slot-0 alias になる)expert の gate 重みを 0 にし slot-0 garbage 混入を防ぐ。
            // mode1=寄与0のみ(scale 保持), mode2=hot で再正規化(amplify)。GPU 完結, 追加 sync 無し。
            if StreamingMoEBlock.skipMode == 1 || StreamingMoEBlock.skipMode == 2 {
                let mask = c.hotMask(numExperts: numExperts)
                let hotness = MLX.take(mask, inds.asType(.int32), axis: 0).asType(scores.dtype)  // [T,K]
                let ms = scores * hotness
                if StreamingMoEBlock.skipMode == 2 {
                    let denom = ms.sum(axis: -1, keepDims: true)   // hot で再正規化（全 cold 行は元へ）
                    scores = MLX.where(denom .> 1e-6, ms / MLX.maximum(denom, MLXArray(Float(1e-6)).asType(ms.dtype)), scores)
                } else {
                    scores = ms                                    // 寄与0のみ（scale は下がるが amplify 無し）
                }
            }
            // buddy(mode3): cold を slot-0 でなく buddy slot へ remap（scores は元のまま=cold の重みで buddy 出力）
            let table = (StreamingMoEBlock.skipMode == 3 && c.buddyTable != nil)
                ? c.buddyTable! : c.gpuSlotTable(numExperts: numExperts)
            let remap = MLX.take(table, inds.asType(.int32), axis: 0).asType(.uint32)
            let xe = x.expandedDimensions(axes: [-2, -3])
            let prof = StreamingMoEBlock.profileLayers
            var t0 = DispatchTime.now().uptimeNanoseconds
            let g = gatherQmm(xe, c.arena, "gate_proj", remap)
            let u = gatherQmm(xe, c.arena, "up_proj", remap)
            let h = (g * MLX.sigmoid(g)) * u
            let d = gatherQmm(h, c.arena, "down_proj", remap).squeezed(axis: -2)
            let y = (d * scores.expandedDimensions(axis: -1)).sum(axis: -2)
            if prof { y.eval(); StreamingMoEBlock.tMoEgather += DispatchTime.now().uptimeNanoseconds - t0; t0 = DispatchTime.now().uptimeNanoseconds }
            let sg = shGate.apply(x), su = shUp.apply(x)
            let sharedY = shDown.apply((sg * MLX.sigmoid(sg)) * su)
            let out = y + MLX.sigmoid(sharedGate.apply(x)) * sharedY
            if prof { out.eval(); StreamingMoEBlock.tMoEshared += DispatchTime.now().uptimeNanoseconds - t0 }
            return out
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
