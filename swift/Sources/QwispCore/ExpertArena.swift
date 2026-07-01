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
    // ★ union-overflow 検出(strict-lossless guard): batched verify で 1 層の routed distinct expert が C を
    //   超えると sync ensure が evict しきれず wrong-slot=silent garbage で誤受理する。ensure に渡る CPU 側
    //   [Int] の distinct 数で検出(GPU sync 不要=安価)。overflowCheck=true の間だけ判定。
    nonisolated(unsafe) public static var overflowCheck = false
    nonisolated(unsafe) public static var overflowMaxUnion = 0   // overflowCheck 中の per-layer distinct routed の最大
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
        // ★ union-overflow guard: distinct routed > C なら C slot に同時常駐できず gather が garbage。
        if LayerExpertCache.overflowCheck {
            let u = Set(experts).count
            if u > LayerExpertCache.overflowMaxUnion { LayerExpertCache.overflowMaxUnion = u }
        }
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

    /// ★ issue#7 style A milestone A1: raw streaming gather(arena cache buffer を slot-remap で読む)が
    /// MLX gather と bit-exact か検証。kernel は buffer 非依存ゆえほぼ既存資産(#5)。streaming 固有=slot binds。
    /// env QWISP_RUN=raw-stream-gather。
    public static func runRawStreamGather(modelDir: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let arena = try ExpertArena(device: device, source: source, N: 64)
        try arena.load(0, Array(0 ..< 16))                      // expert e → slot e（layer 0）
        let w = arena.arr("gate_proj", "weight"), s = arena.arr("gate_proj", "scales"), b = arena.arr("gate_proj", "biases")
        let K = w.dim(-1) * 8, N = w.dim(-2)                    // K=Hin(2048), N=I(512)
        let x = MLXRandom.normal([1, K]).asType(.float16); x.eval()
        let slots = [0, 1, 2, 3, 4, 5, 6, 7]                    // slot 0-7（=expert 0-7）
        // MLX gather（no-sync 経路と同形）
        let remap = MLXArray(slots.map { Int32($0) }, [1, 8]).asType(.uint32)
        let xe = x.expandedDimensions(axes: [-2, -3])
        let mlxG = MLX.gatherQuantizedMatmul(xe, w, scales: s, biases: b, rhsIndices: remap,
                                             transpose: true, groupSize: 64, bits: 4).reshaped([8, N]); mlxG.eval()
        // raw gather（同 arena buffer + slot binds）
        guard let rawG = RawMetalForward.gatherQmm(x, w, scales: s, biases: b,
                                                   inds: MLXArray(slots.map { Int32($0) }), Ktop: 8, K: K, N: N) else {
            return "[A1] raw gatherQmm 失敗(非fast?)"
        }
        rawG.eval()
        let rel = MLX.max(MLX.abs(mlxG.asType(.float32) - rawG.asType(.float32))).item(Float.self)
            / (MLX.max(MLX.abs(mlxG.asType(.float32))).item(Float.self) + 1e-9)
        return String(format: """
            [A1 raw-stream-gather] raw gather(arena cache buffer[C=%d slots] + slot binds)vs MLX gather
              K=%d N=%d, slots=0-7  rel=%.3e  %@
              → bit/near-tie 一致なら style A の gather は既存 raw kernel で streaming 動作(A1 de-risk 完了)。
                残=A3/A4(GPU miss判定+segment+CPU handshake)が真の新規。
            """, 64, K, N, rel, rel < 5e-3 ? "✅ 一致(A1 OK)" : "❌乖離")
    }

    /// ★ task#4 Step B(de-risk): batched MoE gather(M=B verify)の compute vs memory bound を clean microbench で確定。
    /// SUBPROF は barrier 計時で絶対値 inflate ゆえ、ここでは単一 eval で実 ms→achieved GFLOP/s・GB/s を測る。
    /// small union(全 M 行→同 8 expert=最大 compute-bound)と large union(distinct=memory寄り)を M=1/8/24/48 で比較。
    /// 判定: small union M=48 が matrix-unit peak(~10 TFLOP/s FP32 系)に近ければ LUT-GEMM は超えられない=NO-GO。
    ///   memory/overhead-bound(低 GFLOP/s)なら MAC 削減は無効=NO-GO。実 layer-0 expert 重み使用。env QWISP_RUN=verify-gather-bench。
    public static func runVerifyGatherBench(modelDir: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let E = 256, topK = 8
        let arena = try ExpertArena(device: device, source: source, N: E)
        try arena.load(0, Array(0 ..< E))                          // layer 0 全 expert を arena へ
        let wg = arena.arr("gate_proj", "weight"), sg = arena.arr("gate_proj", "scales"), bg = arena.arr("gate_proj", "biases")
        let wu = arena.arr("up_proj", "weight"),   su = arena.arr("up_proj", "scales"),   bu = arena.arr("up_proj", "biases")
        let wd = arena.arr("down_proj", "weight"), sd = arena.arr("down_proj", "scales"), bd = arena.arr("down_proj", "biases")
        let H = wg.dim(-1) * 8, I = wg.dim(-2)                     // H=2048(hidden), I=512(expert intermediate)
        let reps = Int(ProcessInfo.processInfo.environment["QWISP_FC_REPS"] ?? "30") ?? 30
        func gqmm(_ x: MLXArray, _ w: MLXArray, _ s: MLXArray, _ b: MLXArray, _ remap: MLXArray) -> MLXArray {
            MLX.gatherQuantizedMatmul(x, w, scales: s, biases: b, rhsIndices: remap,
                                      transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: false)
        }
        // 1 layer 分の MoE expert FFN(gate+up+swiglu+down)を M トークン分 gather。
        func moeExpert(_ x: MLXArray, _ remap: MLXArray) -> MLXArray {
            let xe = x.expandedDimensions(axes: [-2, -3])
            let g = gqmm(xe, wg, sg, bg, remap), u = gqmm(xe, wu, su, bu, remap)
            let h = (g * MLX.sigmoid(g)) * u
            return gqmm(h, wd, sd, bd, remap).squeezed(axis: -2)
        }
        let Ms = [1, 8, 24, 48]
        var rows: [String] = []
        for unionSmall in [true, false] {
            for M in Ms {
                let x = (MLXRandom.normal([M, H]) * 0.1).asType(.float16)
                // remap[M, topK]: small=全行同一[0..7](union=8), large=行毎に distinct(union≈min(M*8,256))
                var idx = [Int32](repeating: 0, count: M * topK)
                for m in 0 ..< M { for k in 0 ..< topK { idx[m * topK + k] = Int32(unionSmall ? k : (m * topK + k) % E) } }
                let remap = MLXArray(idx, [M, topK]).asType(.uint32)
                MLX.eval(x, remap)
                for _ in 0 ..< 5 { moeExpert(x, remap).eval() }                 // warmup
                let t0 = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< reps { moeExpert(x, remap).eval() }
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6 / Double(reps)
                // FLOP = 3 proj × M × topK × (I×H) MAC × 2。bytes = union × 3 × (I×H × 0.5byte[4bit])。
                let flop = 3.0 * Double(M) * Double(topK) * Double(I) * Double(H) * 2.0
                let union = unionSmall ? topK : Swift.min(M * topK, E)
                let bytes = Double(union) * 3.0 * Double(I) * Double(H) * 0.5
                let gflops = flop / (ms / 1e3) / 1e9
                let gbps = bytes / (ms / 1e3) / 1e9
                rows.append(String(format: "  union=%@ M=%2d: %6.3f ms  %7.1f GFLOP/s  %6.1f GB/s  (union=%d expert)",
                                   unionSmall ? "small" : "large", M, ms, gflops, gbps, union))
            }
        }
        // ★ dispatch 集約 de-risk: gate gather を P=M*topK 行の grouped 形で sorted(=expert 連続) vs
        //   unsorted で比較。sorted で matrix-units が token をグループ GEMM 化できれば GFLOP/s 急増の見込み。
        var sortRows: [String] = []
        for M in [24, 48] {
            let P = M * topK
            let xg = (MLXRandom.normal([P, 1, H]) * 0.1).asType(.float16)
            // unsorted: 行 p → expert (p%E) を散在。sorted: 同じ multiset を昇順に並べ替え(expert 連続)。
            let unsortedE = (0 ..< P).map { Int32(($0 * 7) % E) }
            let sortedE = unsortedE.sorted()
            let idxU = MLXArray(unsortedE, [P, 1]).asType(.uint32)
            let idxS = MLXArray(sortedE, [P, 1]).asType(.uint32)
            MLX.eval(xg, idxU, idxS)
            func benchGate(_ idx: MLXArray, _ sorted: Bool) -> Double {
                func g() -> MLXArray {
                    MLX.gatherQuantizedMatmul(xg, wg, scales: sg, biases: bg, rhsIndices: idx,
                                              transpose: true, groupSize: 64, bits: 4, mode: .affine, sortedIndices: sorted)
                }
                for _ in 0 ..< 5 { g().eval() }
                let t = DispatchTime.now().uptimeNanoseconds
                for _ in 0 ..< reps { g().eval() }
                return Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6 / Double(reps)
            }
            let flopG = Double(P) * Double(I) * Double(H) * 2.0
            for (lbl, idx, sorted) in [("unsorted", idxU, false), ("sorted  ", idxS, true)] {
                let ms = benchGate(idx, sorted)
                sortRows.append(String(format: "  gate gather P=%4d(M=%d) %@: %6.3f ms  %7.1f GFLOP/s",
                                       P, M, lbl, ms, flopG / (ms / 1e3) / 1e9))
            }
        }
        // ★ matrix-unit ceiling: dense quantizedMatmul(gather 無し、単一 expert を P 行へ)。
        //   gather の GFLOP/s がこれに大きく劣るなら、token を expert-group 化→dense GEMM で勝てる(GO)。
        //   同程度なら INT4 matmul 自体が天井=grouping 無益(NO-GO)。
        var denseRows: [String] = []
        for P in [192, 384] {
            let xd = (MLXRandom.normal([P, H]) * 0.1).asType(.float16)
            let w0 = wg[0], s0 = sg[0], b0 = bg[0]                  // expert 0 の gate 重み[I, H]
            MLX.eval(xd, w0, s0, b0)
            func dense() -> MLXArray {
                MLX.quantizedMatmul(xd, w0, scales: s0, biases: b0, transpose: true, groupSize: 64, bits: 4)
            }
            for _ in 0 ..< 5 { dense().eval() }
            let t = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< reps { dense().eval() }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6 / Double(reps)
            let flop = Double(P) * Double(I) * Double(H) * 2.0
            denseRows.append(String(format: "  dense qmm [%4d×%d]×[%d,%d]: %6.3f ms  %7.1f GFLOP/s",
                                    P, H, I, H, ms, flop / (ms / 1e3) / 1e9))
        }
        return "[task#4 StepB verify-gather-bench] MLX gatherQuantizedMatmul(M=B verify gather, 実 layer0, H=\(H) I=\(I))\n"
            + rows.joined(separator: "\n")
            + "\n  M1 Max 概算 peak: ~10 TFLOP/s(FP32系) / ~200-400 GB/s。"
            + "\n  → small union M=48 が高 GFLOP/s(peak 近傍)=matrix-units 飽和→LUT-GEMM 超えられず NO-GO。"
            + "\n     低 GFLOP/s かつ低 GB/s=overhead 律速→MAC 削減無効。large union=memory-bound→LUT 無効。"
            + "\n[dispatch 集約 de-risk] sorted(grouped GEMM) vs unsorted gather:\n"
            + sortRows.joined(separator: "\n")
            + "\n[matrix-unit ceiling] dense quantizedMatmul(gather 無し=純 INT4 GEMM):\n"
            + denseRows.joined(separator: "\n")
            + "\n  ★罠: dense は P=192/384 とも ~0.4ms=固定 dispatch overhead 床(compute はその下)。実 MoE は"
            + "\n     expert 毎に重み別=1個の巨大 dense にできず、per-expert 分割は union×0.4ms 床が乗る。MLX 自身の"
            + "\n     grouped(sortedIndices)も 350 止まり。∴ dense 1968 は単一 expert 限定の蜃気楼、[M,8] gather(443)が実用天井=grouping NO-GO。"
    }

    /// ★ issue#7 style A milestone A2: raw streaming 1層 forward（mixer raw + MoE が arena cache 経由）。
    /// layer 0 を resident(expert 重み[E=256]) と streaming(arena[C=64] + GPU slot-remap binds) の2経路で走らせ、
    /// MoE 出力(sc.combined) が bit-exact か照合。mixer は両経路同一(resident)、差は MoE gather の slot index のみ。
    /// route_top8→slot_remap→arena gather の chain 全体を検証(A1 は単一 gather のみ)。env QWISP_RUN=raw-stream-layer。
    public static func runRawStreamLayer(modelDir: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        guard let (_, queue) = RawMetalForward.ensure(), RawMetalForward.compileSlotRemap() else {
            return "[A2] Metal/slot_remap init 失敗"
        }
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let model = QwispModel(store: store)
        let ids = MLXArray([Int32(1)], [1, 1])
        let H = model.embed(ids).dim(-1)
        guard let layers = model.buildGPULayers(ids, H) else { return "[A2] buildGPULayers 失敗" }
        guard let sc = RawMetalForward.makeGPUScratch(H: H, E: 256, K: 8) else { return "[A2] scratch 失敗" }
        guard let hb = RawMetalForward.makeResidentBuffer(H * 2) else { return "[A2] hBuf 失敗" }
        let L0 = layers[0]

        // 単一層 forward(mixer + MoE)を encode。slotTable!=nil で streaming(arena gather)。
        func runLayer(_ moe: RawMetalForward.MoEBuffers, slotTable: MTLBuffer?, x: MLXArray) -> MLXArray {
            model.resetGPUState()                                  // GDN conv/recurrent state を 0 に(同一入力で同一 postNorm)
            RawMetalForward.writeBuffer(hb, x, H)
            let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
            RawMetalForward.encodeMixerHalf(enc, hBuf: hb, nw: L0.nw, postNormBuf: sc.postNorm,
                                            gdn: L0.gdn, attn: L0.attn, H: H, eps: model.eps, pendingResid: nil)
            RawMetalForward.encodeMoEGPU(enc, postNorm: sc.postNorm, gate: L0.gate, moe: moe, sc: sc,
                                         sharedGateW: L0.sharedGate, H: H, E: 256, K: 8, slotTable: slotTable)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return RawMetalForward.readBuffer(sc.combined, H)
        }

        let x = MLXRandom.normal([1, 1, H]).asType(.float16); x.eval()
        // ① resident 経路(基準)。route_top8 が選んだ expert id を binds から回収。
        let yRef = runLayer(L0.moe, slotTable: nil, x: x)
        let bp = L0.moe.binds.contents().bindMemory(to: Int32.self, capacity: 8)
        let routed = Array(UnsafeBufferPointer(start: bp, count: 8))                  // 選択 expert id(8 distinct)

        // ② arena(C=64) に routed expert を slot i へロードし slotTable(expert→slot)を構築。
        let arena = try ExpertArena(device: device, source: source, N: 64, refLayer: 0)
        try arena.load(0, routed.map { Int($0) })                                     // expert routed[i] → slot i
        var st = [Int32](repeating: 0, count: 256)
        for (i, e) in routed.enumerated() { st[Int(e)] = Int32(i) }
        let stBuf = device.makeBuffer(bytes: &st, length: 256 * 4, options: .storageModeShared)!
        guard let streamMoE = RawMetalForward.prepareStreamingMoEBuffers(arena: arena, resident: L0.moe) else {
            return "[A2] streaming MoEBuffers 構築失敗"
        }
        // ③ streaming 経路: 同一 x、arena gather + GPU slot-remap。
        let yStream = runLayer(streamMoE, slotTable: stBuf, x: x)

        let rel = MLX.max(MLX.abs(yRef.asType(.float32) - yStream.asType(.float32))).item(Float.self)
            / (MLX.max(MLX.abs(yRef.asType(.float32))).item(Float.self) + 1e-9)
        return String(format: """
            [A2 raw-stream-layer] layer0 forward(mixer raw + MoE) resident vs streaming(arena C=64 + GPU slot-remap)
              routed experts=%@  H=%d
              MoE out rel=%.3e  %@
              → 一致なら route_top8→slot_remap→arena gather の全 chain が streaming 動作(A2 OK)。
                残=A3(GPU miss判定+segment境界), A4(CPU miss-service+CB再開, MTLSharedEvent)が真の新規。
            """, routed.map { String($0) }.joined(separator: ","), H,
            rel, rel < 5e-3 ? "✅ bit/near-tie 一致(A2 OK)" : "❌乖離")
    }

    /// ★ issue#7 style A milestone A3a: GPU residency 判定 + miss-list emit の検証。
    /// layer0 の routed 8 expert のうち **一部だけ arena に cache** し、residency_check kernel が
    /// 未収容 expert を正しく検出・emit するか(missCount/missExperts vs CPU ground-truth)を照合。
    /// 出力(MoE)は miss→slot0 garbage ゆえ検査しない。A3b の checkpoint-resume がこの emit を消費。
    /// env QWISP_RUN=raw-stream-miss。QWISP_MISS_CACHED=<n>(cache する routed 数, 既定 5)。
    public static func runRawStreamMissDetect(modelDir: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        guard let (_, queue) = RawMetalForward.ensure(),
              RawMetalForward.compileSlotRemap(), RawMetalForward.compileResidencyCheck() else {
            return "[A3a] Metal/kernel init 失敗"
        }
        let nCached = Int(ProcessInfo.processInfo.environment["QWISP_MISS_CACHED"] ?? "5") ?? 5
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let model = QwispModel(store: store)
        let ids = MLXArray([Int32(1)], [1, 1])
        let H = model.embed(ids).dim(-1)
        guard let layers = model.buildGPULayers(ids, H) else { return "[A3a] buildGPULayers 失敗" }
        guard let sc = RawMetalForward.makeGPUScratch(H: H, E: 256, K: 8) else { return "[A3a] scratch 失敗" }
        guard let hb = RawMetalForward.makeResidentBuffer(H * 2) else { return "[A3a] hBuf 失敗" }
        let L0 = layers[0]
        let x = MLXRandom.normal([1, 1, H]).asType(.float16); x.eval()

        // ① resident 経路で layer0 の routed expert(ground-truth)を回収。
        model.resetGPUState(); RawMetalForward.writeBuffer(hb, x, H)
        do {
            let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
            RawMetalForward.encodeMixerHalf(enc, hBuf: hb, nw: L0.nw, postNormBuf: sc.postNorm,
                                            gdn: L0.gdn, attn: L0.attn, H: H, eps: model.eps, pendingResid: nil)
            RawMetalForward.encodeMoEGPU(enc, postNorm: sc.postNorm, gate: L0.gate, moe: L0.moe, sc: sc,
                                         sharedGateW: L0.sharedGate, H: H, E: 256, K: 8)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        }
        let bp = L0.moe.binds.contents().bindMemory(to: Int32.self, capacity: 8)
        let routed = Array(UnsafeBufferPointer(start: bp, count: 8))                   // inds 順(logit 降順)

        // ② arena に routed の先頭 nCached だけ cache（残り 8-nCached は miss になるはず）。
        let cached = Array(routed.prefix(nCached))
        let expectedMiss = Array(routed.suffix(8 - nCached))                           // inds 順で末尾が miss
        let arena = try ExpertArena(device: device, source: source, N: 64, refLayer: 0)
        try arena.load(0, cached.map { Int($0) })
        var st = [Int32](repeating: 0, count: 256)                                     // slotTable(uncached=0)
        var hot = [Int32](repeating: 0, count: 256)                                    // hotMask
        for (i, e) in cached.enumerated() { st[Int(e)] = Int32(i); hot[Int(e)] = 1 }
        let stBuf = device.makeBuffer(bytes: &st, length: 256 * 4, options: .storageModeShared)!
        let hotBuf = device.makeBuffer(bytes: &hot, length: 256 * 4, options: .storageModeShared)!
        let missCount = device.makeBuffer(length: 40 * 4, options: .storageModeShared)!
        let missExperts = device.makeBuffer(length: 40 * 8 * 4, options: .storageModeShared)!
        memset(missCount.contents(), 0, 40 * 4); memset(missExperts.contents(), 0, 40 * 8 * 4)
        guard let streamMoE = RawMetalForward.prepareStreamingMoEBuffers(arena: arena, resident: L0.moe) else {
            return "[A3a] streaming MoEBuffers 構築失敗"
        }

        // ③ streaming 経路 + residency_check（layerIdx=0）。
        model.resetGPUState(); RawMetalForward.writeBuffer(hb, x, H)
        do {
            let cb = queue.makeCommandBuffer()!; let enc = cb.makeComputeCommandEncoder()!
            RawMetalForward.encodeMixerHalf(enc, hBuf: hb, nw: L0.nw, postNormBuf: sc.postNorm,
                                            gdn: L0.gdn, attn: L0.attn, H: H, eps: model.eps, pendingResid: nil)
            RawMetalForward.encodeMoEGPU(enc, postNorm: sc.postNorm, gate: L0.gate, moe: streamMoE, sc: sc,
                                         sharedGateW: L0.sharedGate, H: H, E: 256, K: 8, slotTable: stBuf,
                                         hotMask: hotBuf, missCount: missCount, missExperts: missExperts, layerIdx: 0)
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        }
        let mc = Int(missCount.contents().bindMemory(to: Int32.self, capacity: 40)[0])
        let mep = missExperts.contents().bindMemory(to: Int32.self, capacity: 40 * 8)
        let emitted = (0 ..< mc).map { mep[$0] }
        let ok = mc == expectedMiss.count && emitted == expectedMiss
        return String(format: """
            [A3a raw-stream-miss] GPU residency 判定 + miss-list emit 検証(layer0, cache %d/8)
              routed(inds順)=%@
              cached=%@
              expected miss=%@
              GPU emit: count=%d  experts=%@
              %@
              → A3a OK なら fused 40層 + checkpoint-resume(A3b)へ。
            """, nCached,
            routed.map { String($0) }.joined(separator: ","),
            cached.map { String($0) }.joined(separator: ","),
            expectedMiss.map { String($0) }.joined(separator: ","),
            mc, emitted.map { String($0) }.joined(separator: ","),
            ok ? "✅ miss 検出・emit 正しい(A3a OK)" : "❌ miss 検出不一致")
    }

    /// ★ issue#7 style A milestone A3b(naive): fused 40層 streaming forward + miss-service resume ループ。
    /// 全40層を arena(per-layer C slot)gather + slot_remap + residency_check で1 CB 楽観実行→CPU が
    /// firstMissLayer m を検出→layer m の miss expert を pread→**token 先頭から再実行**(cold T=1 ゆえ
    /// resetGPUState で deterministic)。miss が無くなれば収束＝全層 cache 内＝no-sync exact forward。
    /// 収束 logits を resident fusedRawForwardGPU と照合(bit-exact なら A3b 成立)。
    /// env QWISP_RUN=raw-stream-fused。QWISP_STREAM_C=<C>(per-layer slot 数, 既定 64)。
    public static func runRawStreamFused(modelDir: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        guard let (_, _) = RawMetalForward.ensure(),
              RawMetalForward.compileSlotRemap(), RawMetalForward.compileResidencyCheck() else {
            return "[A3b] Metal/kernel init 失敗"
        }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_STREAM_C"] ?? "64") ?? 64
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let model = QwispModel(store: store)
        let ids = MLXArray([Int32(1)], [1, 1])
        let H = model.embed(ids).dim(-1)
        let nLayers = model.numLayers

        // ① resident 基準 logits（全 expert 常駐の raw fused forward）。
        guard let refLogits = model.fusedRawForwardGPU(ids) else { return "[A3b] resident forward 失敗" }
        refLogits.eval()
        let refTok = MLX.argMax(refLogits.reshaped([-1]), axis: 0).item(Int.self)

        // ② per-layer streaming 構成（arena cache + streaming MoEBuffers）。resident layers は ① で構築済。
        guard let residentLayers = model.gpuLayers else { return "[A3b] gpuLayers 未構築" }
        var caches: [LayerExpertCache] = []
        var streamLayers: [RawMetalForward.GPULayer] = []
        for i in 0 ..< nLayers {
            let cache = try LayerExpertCache(device: device, source: source, layer: i, C: C)
            guard let sMoE = RawMetalForward.prepareStreamingMoEBuffers(arena: cache.arena, resident: residentLayers[i].moe) else {
                return "[A3b] layer \(i) streaming MoEBuffers 失敗"
            }
            caches.append(cache)
            let R = residentLayers[i]
            streamLayers.append(RawMetalForward.GPULayer(nw: R.nw, gdn: R.gdn, attn: R.attn,
                                                         moe: sMoE, gate: R.gate, sharedGate: R.sharedGate))
        }
        guard let sc = RawMetalForward.makeGPUScratch(H: H, E: 256, K: 8),
              let hb = RawMetalForward.makeResidentBuffer(H * 2) else { return "[A3b] scratch/hBuf 失敗" }
        let missCount = device.makeBuffer(length: nLayers * 4, options: .storageModeShared)!
        let missExperts = device.makeBuffer(length: nLayers * 8 * 4, options: .storageModeShared)!
        let embedX = model.embed(ids); embedX.eval()

        // ③ resume ループ（naive: miss 検出毎に token 先頭から再実行）。
        let maxPasses = nLayers + 20
        var pass = 0, totalServiced = 0
        var serviceLog: [(layer: Int, n: Int)] = []
        while pass < maxPasses {
            pass += 1
            // pass 毎に slotTable/hotMask を現 cache 状態から再構築。
            var slotTables: [MTLBuffer] = [], hotMasks: [MTLBuffer] = []
            for c in caches {
                guard let st = c.gpuSlotTable(numExperts: 256).asMTLBuffer(device: device, noCopy: false),
                      let hm = c.hotMask(numExperts: 256).asMTLBuffer(device: device, noCopy: false) else {
                    return "[A3b] slotTable/hotMask buffer 失敗"
                }
                slotTables.append(st); hotMasks.append(hm)
            }
            model.resetGPUState()
            RawMetalForward.writeBuffer(hb, embedX, H)
            memset(missCount.contents(), 0, nLayers * 4); memset(missExperts.contents(), 0, nLayers * 8 * 4)
            RawMetalForward.fusedForwardGPUStreaming(
                hBuf: hb, layers: streamLayers, scratch: sc, slotTables: slotTables, hotMasks: hotMasks,
                missCount: missCount, missExperts: missExperts, H: H, E: 256, K: 8, eps: model.eps,
                finalNormW: model.ensureFinalNorm())
            // firstMissLayer 検出。
            let mcp = missCount.contents().bindMemory(to: Int32.self, capacity: nLayers)
            var m = -1
            for l in 0 ..< nLayers where mcp[l] > 0 { m = l; break }
            if m < 0 { break }   // miss 無し＝収束
            // layer m の miss expert を pread（cache に確保）。
            let mep = missExperts.contents().bindMemory(to: Int32.self, capacity: nLayers * 8)
            let n = Int(mcp[m])
            let missing = (0 ..< n).map { Int(mep[m * 8 + $0]) }
            _ = caches[m].ensure(missing)
            totalServiced += n; serviceLog.append((m, n))
        }
        let converged = pass < maxPasses

        // ④ 収束 logits を resident と照合。
        let fn = RawMetalForward.readBuffer(sc.normed, H)
        let streamLogits = model.headProj().apply(fn.reshaped([1, 1, H])); streamLogits.eval()
        let streamTok = MLX.argMax(streamLogits.reshaped([-1]), axis: 0).item(Int.self)
        let rel = MLX.max(MLX.abs(refLogits.asType(.float32) - streamLogits.asType(.float32))).item(Float.self)
            / (MLX.max(MLX.abs(refLogits.asType(.float32))).item(Float.self) + 1e-9)
        let firstSvc = serviceLog.prefix(8).map { "L\($0.layer):\($0.n)" }.joined(separator: " ")
        let ok = converged && streamTok == refTok && rel < 5e-3
        return String(format: """
            [A3b raw-stream-fused] fused 40層 streaming(arena C=%d + slot_remap + residency_check)+ resume ループ
              収束=%@  passes=%d  serviced miss 層=%d(計 %d expert)  例: %@ ...
              logits rel=%.3e  argmax stream=%d resident=%d
              %@
              → bit-exact なら全層 cache 内で no-sync exact forward 成立(A3b naive OK)。残=A4(CPU service と pread を非同期重畳)+ checkpoint-resume 最適化。
            """, C, converged ? "YES" : "NO(maxPasses 到達)", pass, serviceLog.count, totalServiced, firstSvc,
            rel, streamTok, refTok,
            ok ? "✅ resident と一致(A3b naive OK)" : "❌ 不一致 or 未収束")
    }

    /// ★ issue#7 style A milestone A3b-opt: checkpoint-resume 版。naive(miss 毎に token 先頭再実行 O(層^2))を
    /// **layer m から再開**(per-layer ckptH + token 開始 GDN state snapshot を layer≥m に restore)に置換。
    /// 正しさは naive と同じ bit-exact。利得=正しい前置層[0..m-1]を再計算しない(GPU work 削減)＋decode の
    /// state 継続に必須の正しいアーキテクチャ。#sync(=passes)削減は naive で既に達成済(segment CB)。
    /// 追加で warm 再実行(全層 cache 内)=1 pass を実証し steady-state token=1 CB(per-layer drain 40 の置換)を示す。
    /// env QWISP_RUN=raw-stream-resume。QWISP_STREAM_C=<C>。
    public static func runRawStreamResume(modelDir: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        guard let (_, _) = RawMetalForward.ensure(), RawMetalForward.compileSlotRemap(),
              RawMetalForward.compileResidencyCheck(), RawMetalForward.compileVecCopy() else {
            return "[A3b-opt] Metal/kernel init 失敗"
        }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_STREAM_C"] ?? "64") ?? 64
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let model = QwispModel(store: store)
        let ids = MLXArray([Int32(1)], [1, 1])
        let H = model.embed(ids).dim(-1)
        let nLayers = model.numLayers

        guard let refLogits = model.fusedRawForwardGPU(ids) else { return "[A3b-opt] resident forward 失敗" }
        refLogits.eval()
        let refTok = MLX.argMax(refLogits.reshaped([-1]), axis: 0).item(Int.self)

        guard let residentLayers = model.gpuLayers else { return "[A3b-opt] gpuLayers 未構築" }
        var caches: [LayerExpertCache] = []
        var streamLayers: [RawMetalForward.GPULayer] = []
        for i in 0 ..< nLayers {
            let cache = try LayerExpertCache(device: device, source: source, layer: i, C: C)
            guard let sMoE = RawMetalForward.prepareStreamingMoEBuffers(arena: cache.arena, resident: residentLayers[i].moe) else {
                return "[A3b-opt] layer \(i) streaming MoEBuffers 失敗"
            }
            caches.append(cache)
            let R = residentLayers[i]
            streamLayers.append(RawMetalForward.GPULayer(nw: R.nw, gdn: R.gdn, attn: R.attn,
                                                         moe: sMoE, gate: R.gate, sharedGate: R.sharedGate))
        }
        guard let sc = RawMetalForward.makeGPUScratch(H: H, E: 256, K: 8),
              let hb = RawMetalForward.makeResidentBuffer(H * 2) else { return "[A3b-opt] scratch/hBuf 失敗" }
        var ckptH: [MTLBuffer] = []
        for _ in 0 ..< nLayers { guard let b = RawMetalForward.makeResidentBuffer(H * 2) else { return "[A3b-opt] ckptH 失敗" }; ckptH.append(b) }
        let missCount = device.makeBuffer(length: nLayers * 4, options: .storageModeShared)!
        let missExperts = device.makeBuffer(length: nLayers * 8 * 4, options: .storageModeShared)!
        let embedX = model.embed(ids); embedX.eval()

        // GDN state(token 開始)snapshot / restore(layer≥m)。resume の layer≥m mixer 再実行を deterministic に。
        func snapshotGDN() -> [Int: (Data, Data)] {
            var snap: [Int: (Data, Data)] = [:]
            for i in 0 ..< nLayers {
                guard let g = residentLayers[i].gdn else { continue }
                let sLen = g.Hv * g.Dv * g.Dk * 4, cLen = g.convKernel * g.convDim * 2
                snap[i] = (Data(bytes: g.stateBuf.contents(), count: sLen),
                           Data(bytes: g.convInput.contents(), count: cLen))
            }
            return snap
        }
        func restoreGDN(_ snap: [Int: (Data, Data)], from m: Int) {
            for i in m ..< nLayers {
                guard let g = residentLayers[i].gdn, let (sd, cd) = snap[i] else { continue }
                sd.withUnsafeBytes { g.stateBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: sd.count) }
                cd.withUnsafeBytes { g.convInput.contents().copyMemory(from: $0.baseAddress!, byteCount: cd.count) }
            }
        }
        func buildTables() -> ([MTLBuffer], [MTLBuffer])? {
            var st: [MTLBuffer] = [], hm: [MTLBuffer] = []
            for c in caches {
                guard let s = c.gpuSlotTable(numExperts: 256).asMTLBuffer(device: device, noCopy: false),
                      let h = c.hotMask(numExperts: 256).asMTLBuffer(device: device, noCopy: false) else { return nil }
                st.append(s); hm.append(h)
            }
            return (st, hm)
        }

        // ===== cold 収束(checkpoint-resume) =====
        model.resetGPUState()
        let snap = snapshotGDN()                                   // cold token 開始 state(=0)を保存
        let maxPasses = nLayers + 20
        var pass = 0, prevStart = 0, totalServiced = 0, layerExecs = 0
        var serviceLog: [Int] = []
        while pass < maxPasses {
            pass += 1
            guard let (slotTables, hotMasks) = buildTables() else { return "[A3b-opt] tables 失敗" }
            restoreGDN(snap, from: prevStart)                      // layer≥prevStart の state を token 開始へ
            if prevStart == 0 { RawMetalForward.writeBuffer(hb, embedX, H) }
            else { memcpy(hb.contents(), ckptH[prevStart].contents(), H * 2) }   // layer m 入口 hidden を復元
            memset(missCount.contents(), 0, nLayers * 4); memset(missExperts.contents(), 0, nLayers * 8 * 4)
            RawMetalForward.fusedForwardGPUStreamingResume(
                hBuf: hb, layers: streamLayers, scratch: sc, slotTables: slotTables, hotMasks: hotMasks,
                missCount: missCount, missExperts: missExperts, ckptH: ckptH, startLayer: prevStart,
                H: H, E: 256, K: 8, eps: model.eps, finalNormW: model.ensureFinalNorm())
            layerExecs += nLayers - prevStart
            let mcp = missCount.contents().bindMemory(to: Int32.self, capacity: nLayers)
            var m = -1
            for l in prevStart ..< nLayers where mcp[l] > 0 { m = l; break }
            if m < 0 { break }
            let mep = missExperts.contents().bindMemory(to: Int32.self, capacity: nLayers * 8)
            let n = Int(mcp[m]); let missing = (0 ..< n).map { Int(mep[m * 8 + $0]) }
            _ = caches[m].ensure(missing); totalServiced += n; serviceLog.append(m)
            prevStart = m                                          // 次 pass は miss 層 m から再開
        }
        let converged = pass < maxPasses
        let fn = RawMetalForward.readBuffer(sc.normed, H)
        let streamLogits = model.headProj().apply(fn.reshaped([1, 1, H])); streamLogits.eval()
        let streamTok = MLX.argMax(streamLogits.reshaped([-1]), axis: 0).item(Int.self)
        let rel = MLX.max(MLX.abs(refLogits.asType(.float32) - streamLogits.asType(.float32))).item(Float.self)
            / (MLX.max(MLX.abs(refLogits.asType(.float32))).item(Float.self) + 1e-9)

        // ===== warm 再実行(全層 cache 内, 同 token)= 1 pass を実証 =====
        model.resetGPUState()
        guard let (wst, whm) = buildTables() else { return "[A3b-opt] warm tables 失敗" }
        RawMetalForward.writeBuffer(hb, embedX, H)
        memset(missCount.contents(), 0, nLayers * 4)
        RawMetalForward.fusedForwardGPUStreamingResume(
            hBuf: hb, layers: streamLayers, scratch: sc, slotTables: wst, hotMasks: whm,
            missCount: missCount, missExperts: missExperts, ckptH: ckptH, startLayer: 0,
            H: H, E: 256, K: 8, eps: model.eps, finalNormW: model.ensureFinalNorm())
        let wmcp = missCount.contents().bindMemory(to: Int32.self, capacity: nLayers)
        var warmMiss = 0; for l in 0 ..< nLayers { warmMiss += Int(wmcp[l]) }
        let wfn = RawMetalForward.readBuffer(sc.normed, H)
        let warmLogits = model.headProj().apply(wfn.reshaped([1, 1, H])); warmLogits.eval()
        let warmRel = MLX.max(MLX.abs(refLogits.asType(.float32) - warmLogits.asType(.float32))).item(Float.self)
            / (MLX.max(MLX.abs(refLogits.asType(.float32))).item(Float.self) + 1e-9)

        let naiveExecs = pass * nLayers
        let ok = converged && streamTok == refTok && rel < 5e-3 && warmMiss == 0 && warmRel < 5e-3
        return String(format: """
            [A3b-opt raw-stream-resume] checkpoint-resume(layer m から再開)+ warm 実証
              cold 収束=%@ passes=%d  serviced miss 層=%d(計 %d expert)
              GPU layer-exec: resume版 %d vs naive(token先頭再実行) %d  (%.2fx 削減)
              cold logits rel=%.3e  argmax stream=%d resident=%d
              warm 再実行(全層cache): miss=%d  rel=%.3e  → 1 pass(1 sync)=steady-state token, vs per-layer drain 40 sync
              %@
              → checkpoint-resume が bit-exact かつ前置層再計算を削減。残=A4(pread を非同期重畳)+decode(KV/pos)。
            """, converged ? "YES" : "NO", pass, serviceLog.count, totalServiced,
            layerExecs, naiveExecs, Double(naiveExecs) / Double(max(1, layerExecs)),
            rel, streamTok, refTok, warmMiss, warmRel,
            ok ? "✅ resume bit-exact + warm 1-pass(A3b-opt OK)" : "❌ 不一致/未収束/warm miss")
    }

    /// ★ issue#7 style A milestone A5: decode regime での真の payoff 計測。連続 token を checkpoint-resume で
    /// decode(decode=true, pos 前進, KV/GDN state 持続, per-layer arena cache 持続)し、**token あたり #sync(=passes)
    /// vs per-layer drain 40** を実測。各 token の argmax を resident greedy decode と照合(lossless)。
    /// cold first token は #miss=40 だが cache 持続で warm token は #miss(=新規 expert)へ収束＝payoff。
    /// env QWISP_RUN=raw-stream-decode。QWISP_GEN=<N gen tokens, 既定16>。QWISP_STREAM_C=<C>。
    public static func runRawStreamDecode(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        guard let (_, _) = RawMetalForward.ensure(), RawMetalForward.compileSlotRemap(),
              RawMetalForward.compileResidencyCheck(), RawMetalForward.compileVecCopy() else {
            return "[A5] Metal/kernel init 失敗"
        }
        let C = Int(ProcessInfo.processInfo.environment["QWISP_STREAM_C"] ?? "64") ?? 64
        let N = Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "16") ?? 16
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let model = QwispModel(store: store)
        let H = model.embed(MLXArray([Int32(0)], [1, 1])).dim(-1)
        let nLayers = model.numLayers
        guard model.buildGPULayers(MLXArray([Int32(1)], [1, 1]), H) != nil else { return "[A5] build 失敗" }
        guard let residentLayers = model.gpuLayers else { return "[A5] gpuLayers 未構築" }

        // prompt: refPath の spec_prompt があれば使用、無ければ synthetic。
        var prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        if let r = try? loadArrays(url: URL(fileURLWithPath: refPath)), let pa = r["spec_prompt"] {
            prompt = pa.asType(.int32).asArray(Int32.self)
        }

        // ===== ① resident greedy decode 基準 =====
        model.resetGPUState(); var pos = 0; var last: MLXArray? = nil
        for t in prompt { last = model.fusedDecodeStepGPU(t, pos: pos, H: H); pos += 1 }
        last?.eval()
        var cur = Int32(MLX.argMax(last!.reshaped([last!.size])).item(Int.self))
        var refGen: [Int32] = []
        let tRef0 = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< N {
            refGen.append(cur)
            guard let lg = model.fusedDecodeStepGPU(cur, pos: pos, H: H) else { break }
            lg.eval(); pos += 1; cur = Int32(MLX.argMax(lg.reshaped([lg.size])).item(Int.self))
        }
        let refSecs = Double(DispatchTime.now().uptimeNanoseconds - tRef0) / 1e9
        let refTokps = Double(N) / refSecs   // resident(全 expert 常駐, 32GB+ tier)decode tok/s

        // ===== ② streaming 構成 =====
        var caches: [LayerExpertCache] = []
        var streamLayers: [RawMetalForward.GPULayer] = []
        for i in 0 ..< nLayers {
            let cache = try LayerExpertCache(device: device, source: source, layer: i, C: C)
            guard let sMoE = RawMetalForward.prepareStreamingMoEBuffers(arena: cache.arena, resident: residentLayers[i].moe) else {
                return "[A5] layer \(i) streaming MoEBuffers 失敗"
            }
            caches.append(cache)
            let R = residentLayers[i]
            streamLayers.append(RawMetalForward.GPULayer(nw: R.nw, gdn: R.gdn, attn: R.attn,
                                                         moe: sMoE, gate: R.gate, sharedGate: R.sharedGate))
        }
        guard let sc = RawMetalForward.makeGPUScratch(H: H, E: 256, K: 8),
              let hb = RawMetalForward.makeResidentBuffer(H * 2) else { return "[A5] scratch/hBuf 失敗" }
        var ckptH: [MTLBuffer] = []
        for _ in 0 ..< nLayers { guard let b = RawMetalForward.makeResidentBuffer(H * 2) else { return "[A5] ckptH 失敗" }; ckptH.append(b) }
        let missCount = device.makeBuffer(length: nLayers * 4, options: .storageModeShared)!
        let missExperts = device.makeBuffer(length: nLayers * 8 * 4, options: .storageModeShared)!
        // ★ A4 in-kernel stop flag: miss 後の suffix gather(gqmm4/gqmm4_swiglu)を no-op 化。常時有効
        //   (residency_check が miss で stopFlag=1 を立てる→以降 gather skip)。lossless 維持・GPU-exec 削減。
        let stopFlag = device.makeBuffer(length: 4, options: .storageModeShared)!

        // ★ bookkeeping 最適化(1): GDN snapshot/restore を **永続 backup buffer** で(Data heap alloc 排除)。
        var gdnStateBak: [Int: MTLBuffer] = [:], gdnConvBak: [Int: MTLBuffer] = [:]
        for i in 0 ..< nLayers {
            guard let g = residentLayers[i].gdn else { continue }
            gdnStateBak[i] = device.makeBuffer(length: g.Hv * g.Dv * g.Dk * 4, options: .storageModeShared)!
            gdnConvBak[i] = device.makeBuffer(length: g.convKernel * g.convDim * 2, options: .storageModeShared)!
        }
        func snapshotGDN() {
            for i in 0 ..< nLayers {
                guard let g = residentLayers[i].gdn else { continue }
                memcpy(gdnStateBak[i]!.contents(), g.stateBuf.contents(), g.Hv * g.Dv * g.Dk * 4)
                memcpy(gdnConvBak[i]!.contents(), g.convInput.contents(), g.convKernel * g.convDim * 2)
            }
        }
        // ★ stop flag が miss 後層の state(recur/shiftConv)を凍結するので、resume 層 m **のみ** restore で足りる
        //   (m+1.. は token 開始のまま=frozen、0..m-1 は正しく前進済で保持)。per-pass の suffix memcpy を排除。
        func restoreGDN(from m: Int) {
            guard let g = residentLayers[m].gdn else { return }
            memcpy(g.stateBuf.contents(), gdnStateBak[m]!.contents(), g.Hv * g.Dv * g.Dk * 4)
            memcpy(g.convInput.contents(), gdnConvBak[m]!.contents(), g.convKernel * g.convDim * 2)
        }
        // ★ bookkeeping 最適化(2): slotTable/hotMask を **永続 buffer** にし、cache 変更層のみ in-place 更新
        //   (旧: 毎 pass 全40層 asMTLBuffer 再構築=80 alloc×passes)。
        var slotTableBufs: [MTLBuffer] = [], hotMaskBufs: [MTLBuffer] = []
        for _ in 0 ..< nLayers {
            slotTableBufs.append(device.makeBuffer(length: 256 * 4, options: .storageModeShared)!)
            hotMaskBufs.append(device.makeBuffer(length: 256 * 4, options: .storageModeShared)!)
        }
        func refreshLayer(_ i: Int) {
            let st = slotTableBufs[i].contents().bindMemory(to: Int32.self, capacity: 256)
            let hm = hotMaskBufs[i].contents().bindMemory(to: Int32.self, capacity: 256)
            memset(slotTableBufs[i].contents(), 0, 256 * 4); memset(hotMaskBufs[i].contents(), 0, 256 * 4)
            for (e, s) in caches[i].slotMap { st[e] = Int32(s); hm[e] = 1 }
        }
        for i in 0 ..< nLayers { refreshLayer(i) }   // 初期(cold)
        // ★ A5b: cross-layer 予測 prefetch。layer i の routing を hidden h から gate_i(h) top-(8+margin) で予測。
        //   resume の再 seed(prevStart 深化)で予測距離が縮み精度向上=自己補正。env QWISP_PREDICT=0 で無効、QWISP_MARGIN。
        let doPredict = (ProcessInfo.processInfo.environment["QWISP_PREDICT"] ?? "0") == "1"   // 既定 off(MLX gate+IO で wash)
        let predictEveryPass = (ProcessInfo.processInfo.environment["QWISP_PREDICT_EVERYPASS"] ?? "0") == "1"
        let predictK = 8 + (Int(ProcessInfo.processInfo.environment["QWISP_MARGIN"] ?? "16") ?? 16)
        func predictLayer(_ i: Int, _ h: MLXArray) -> MLXArray {
            let p = "language_model.model.layers.\(i).mlp.gate"
            let logits = MLX.quantizedMatmul(h, store.req("\(p).weight"), scales: store.req("\(p).scales"),
                                             biases: store.req("\(p).biases"), transpose: true, groupSize: 64, bits: 8).reshaped([256])
            let order = MLX.argPartition(logits, kth: 256 - predictK)
            return order[(256 - predictK)...].asType(.int32)
        }

        // 1 token の checkpoint-resume decode step。(予測 argmax, #passes) を返す。
        // GPU-exec(lastGPUExecMs) と pread(LayerExpertCache.preadNanos)を分離計測し CPU bookkeeping を切り分け。
        let maxPasses = nLayers + 20
        var gpuMsAccum = 0.0
        func decodeStep(_ inputTok: Int32, _ pos: Int) -> (Int, Int) {
            snapshotGDN()                                             // token 開始 GDN state(永続 backup へ)
            let embedX = model.embed(MLXArray([inputTok], [1, 1])); embedX.eval()
            var prevStart = 0, passes = 0
            while passes < maxPasses {
                passes += 1
                // 予測 prefetch: token 先頭(pass1)に全層を embed から予測しキャッシュ確保(1回/token=CPU 最小)。
                // QWISP_PREDICT_EVERYPASS=1 で旧来の per-pass 再予測(高精度・高 CPU)。
                if doPredict && (prevStart == 0 || predictEveryPass) {
                    let seedH = prevStart == 0 ? embedX.reshaped([1, H]) : RawMetalForward.readBuffer(ckptH[prevStart], H)
                    var preds: [MLXArray] = []
                    for L in prevStart ..< nLayers { preds.append(predictLayer(L, seedH)) }
                    MLX.eval(preds)
                    for (idx, L) in (prevStart ..< nLayers).enumerated() {
                        _ = caches[L].ensure(preds[idx].asArray(Int32.self).map { Int($0) })
                        refreshLayer(L)                                   // 予測 ensure で cache 変化→反映
                    }
                }
                restoreGDN(from: prevStart)
                if prevStart == 0 { RawMetalForward.writeBuffer(hb, embedX, H) }
                else { memcpy(hb.contents(), ckptH[prevStart].contents(), H * 2) }
                memset(missCount.contents(), 0, nLayers * 4); memset(missExperts.contents(), 0, nLayers * 8 * 4)
                stopFlag.contents().bindMemory(to: Int32.self, capacity: 1)[0] = 0   // ★ A4: pass 毎に stop 解除
                RawMetalForward.fusedForwardGPUStreamingResume(
                    hBuf: hb, layers: streamLayers, scratch: sc, slotTables: slotTableBufs, hotMasks: hotMaskBufs,
                    missCount: missCount, missExperts: missExperts, ckptH: ckptH, startLayer: prevStart,
                    H: H, E: 256, K: 8, eps: model.eps, decode: true, pos: pos, finalNormW: model.ensureFinalNorm())
                gpuMsAccum += RawMetalForward.lastGPUExecMs
                let mcp = missCount.contents().bindMemory(to: Int32.self, capacity: nLayers)
                var m = -1
                for l in prevStart ..< nLayers where mcp[l] > 0 { m = l; break }
                if m < 0 { break }
                let mep = missExperts.contents().bindMemory(to: Int32.self, capacity: nLayers * 8)
                let n = Int(mcp[m]); let missing = (0 ..< n).map { Int(mep[m * 8 + $0]) }
                _ = caches[m].ensure(missing); refreshLayer(m); prevStart = m   // 変更層のみ in-place 更新
            }
            let fn = RawMetalForward.readBuffer(sc.normed, H)
            let lg = model.headProj().apply(fn.reshaped([1, 1, H])); lg.eval()
            return (MLX.argMax(lg.reshaped([lg.size])).item(Int.self), passes)
        }

        // ===== ③ streaming decode（prompt prefill → N 生成, teacher-forced on refGen）=====
        model.resetGPUState(); pos = 0
        var promptPasses: [Int] = []
        var match = 0
        RawMetalForward.activeStopFlag = stopFlag                      // ★ A4: streaming 区間のみ guard 有効化
        // prefill: 各 prompt token を投入。最後の token の予測 = refGen[0] のはず。
        for (idx, t) in prompt.enumerated() {
            let (pred, p) = decodeStep(t, pos); promptPasses.append(p)
            if idx == prompt.count - 1 && pred == Int(refGen[0]) { match += 1 }
            pos += 1
        }
        // teacher-forced 生成: refGen[i] を投入し予測 == refGen[i+1] を照合。wall-clock/GPU/pread を分離計測。
        var genPasses: [Int] = []
        gpuMsAccum = 0.0; LayerExpertCache.preadNanos = 0
        let tStream0 = DispatchTime.now().uptimeNanoseconds
        for i in 0 ..< (N - 1) {
            let (pred, p) = decodeStep(refGen[i], pos); genPasses.append(p); pos += 1
            if pred == Int(refGen[i + 1]) { match += 1 }
        }
        RawMetalForward.activeStopFlag = nil                          // ★ A4: guard 解除(他経路に影響させない)
        let streamSecs = Double(DispatchTime.now().uptimeNanoseconds - tStream0) / 1e9
        let streamTokps = Double(N - 1) / streamSecs
        let gpuMsPerTok = gpuMsAccum / Double(N - 1)
        let preadMsPerTok = Double(LayerExpertCache.preadNanos) / 1e6 / Double(N - 1)
        let cpuMsPerTok = streamSecs / Double(N - 1) * 1000 - gpuMsPerTok - preadMsPerTok

        let coldPasses = promptPasses.first ?? 0
        let warmProm = promptPasses.dropFirst().map { $0 }
        let avgGen = genPasses.isEmpty ? 0.0 : Double(genPasses.reduce(0, +)) / Double(genPasses.count)
        let minGen = genPasses.min() ?? 0, maxGen = genPasses.max() ?? 0
        let genTrace = genPasses.prefix(16).map { String($0) }.joined(separator: ",")
        let matchable = N                       // prefill 末尾予測 1 + 生成 N-1
        let ok = match >= matchable
        return String(format: """
            [A5 raw-stream-decode] decode regime payoff 計測(C=%d, prompt T=%d, gen N=%d, teacher-forced)
              lossless(streaming argmax vs resident greedy): %d/%d 一致
              #sync(passes)/token:
                cold first token=%d(=#miss 40 近傍, 想定通り)
                prompt warm tail=%@
                generated: avg=%.1f  min=%d  max=%d  trace=%@
              wall-clock(Debug, ratio 有効/絶対値は遅い):
                streaming(C=%d, expert SSD)=%.1f tok/s(%.1f ms/tok, %.1f CB/tok)
                  内訳/tok: GPU-exec=%.1fms(suffix 再計算込) + pread=%.1fms + CPU bookkeeping=%.1fms
                resident(全 expert 常駐, 32GB+ tier)=%.1f tok/s(%.1f ms/tok, 1 CB/tok)
                → streaming/resident=%.2fx。GPU-exec 比で suffix 再計算コスト、CPU 比で bookkeeping を切り分け。
              %@
            """, C, prompt.count, N, match, matchable,
            coldPasses, warmProm.map { String($0) }.joined(separator: ","),
            avgGen, minGen, maxGen, genTrace,
            C, streamTokps, streamSecs / Double(N - 1) * 1000, avgGen,
            gpuMsPerTok, preadMsPerTok, cpuMsPerTok,
            refTokps, refSecs / Double(N) * 1000,
            streamTokps / refTokps,
            ok ? "✅ lossless 一致(A5 decode 配線 OK)" : "❌ argmax 乖離")
    }

    /// ★ issue#7 fix-2: inline demand-load 版 raw streaming decode。
    /// 各層 route→CPU ensure→gather を逐次（各層 1 回 dispatch、resume 無し）。resume 版(runRawStreamDecode)の
    /// no-op re-dispatch を排除し、small-C(高 miss)でも GPU-exec を ~resident 並みに抑えられるかを検証。
    public static func runRawStreamDecodeInline(modelDir: String, refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        guard let (_, _) = RawMetalForward.ensure(), RawMetalForward.compileSlotRemap() else {
            return "[fix-2] Metal/kernel init 失敗"
        }
        // ★ 本番: C は device 別自動選択(calibration layer, RAM tier 8→64/16→128/24→192/32+→256)。
        //   QWISP_STREAM_C で明示上書き可。
        let C = Int(ProcessInfo.processInfo.environment["QWISP_STREAM_C"] ?? "") ?? DeviceCalibration.defaultC()
        if ProcessInfo.processInfo.environment["QWISP_STREAM_C"] == nil {
            print("[calibration] " + DeviceCalibration.recommend().summary)
        }
        let N = Int(ProcessInfo.processInfo.environment["QWISP_GEN"] ?? "16") ?? 16
        let store = try WeightStore(modelDir: modelDir); store.residentNonExperts()
        let source = try ExpertSource(modelDir: modelDir); try source.warm()
        let model = QwispModel(store: store)
        let H = model.embed(MLXArray([Int32(0)], [1, 1])).dim(-1)
        let nLayers = model.numLayers
        guard model.buildGPULayers(MLXArray([Int32(1)], [1, 1]), H) != nil else { return "[fix-2] build 失敗" }
        guard model.gpuLayers != nil else { return "[fix-2] gpuLayers 未構築" }
        var residentLayers: [RawMetalForward.GPULayer]! = model.gpuLayers   // ★ step② 後に解放するため var

        var prompt: [Int32] = [1, 2, 3, 4, 5, 6, 7, 8]
        if let r = try? loadArrays(url: URL(fileURLWithPath: refPath)), let pa = r["spec_prompt"] {
            prompt = pa.asType(.int32).asArray(Int32.self)
        }

        // ① resident greedy 基準
        model.resetGPUState(); var pos = 0; var last: MLXArray? = nil
        for t in prompt { last = model.fusedDecodeStepGPU(t, pos: pos, H: H); pos += 1 }
        last?.eval()
        var cur = Int32(MLX.argMax(last!.reshaped([last!.size])).item(Int.self))
        var refGen: [Int32] = []
        let tRef0 = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< N {
            refGen.append(cur)
            guard let lg = model.fusedDecodeStepGPU(cur, pos: pos, H: H) else { break }
            lg.eval(); pos += 1; cur = Int32(MLX.argMax(lg.reshaped([lg.size])).item(Int.self))
        }
        let refSecs = Double(DispatchTime.now().uptimeNanoseconds - tRef0) / 1e9
        let refTokps = Double(N) / refSecs

        // ② streaming 構成（arena + slotTable）
        var caches: [LayerExpertCache] = []
        var streamLayers: [RawMetalForward.GPULayer] = []
        for i in 0 ..< nLayers {
            let cache = try LayerExpertCache(device: device, source: source, layer: i, C: C)
            guard let sMoE = RawMetalForward.prepareStreamingMoEBuffers(arena: cache.arena, resident: residentLayers[i].moe) else {
                return "[fix-2] layer \(i) streaming MoEBuffers 失敗"
            }
            caches.append(cache)
            let R = residentLayers[i]
            streamLayers.append(RawMetalForward.GPULayer(nw: R.nw, gdn: R.gdn, attn: R.attn,
                                                         moe: sMoE, gate: R.gate, sharedGate: R.sharedGate))
        }
        // ★ 二重確保解消(harness バグ修正): resident 参照(step①)は終了。resident の routed expert 全重み(~18GB,
        //   C=256 で arena と重複)を解放。streamLayers は nw/gdn/attn/gate/sharedGate(小) + sMoE(arena routed +
        //   resident.sh* shared を retain 済)を保持するので安全。これで step③(計測)は arena のみ=二重確保なし。
        residentLayers = nil
        model.gpuLayers = nil
        model.moeBufCache.removeAll()
        guard let sc = RawMetalForward.makeGPUScratch(H: H, E: 256, K: 8),
              let hb = RawMetalForward.makeResidentBuffer(H * 2) else { return "[fix-2] scratch/hBuf 失敗" }
        var slotTableBufs: [MTLBuffer] = []
        for _ in 0 ..< nLayers { slotTableBufs.append(device.makeBuffer(length: 256 * 4, options: .storageModeShared)!) }
        func refreshLayer(_ i: Int) {
            let st = slotTableBufs[i].contents().bindMemory(to: Int32.self, capacity: 256)
            memset(slotTableBufs[i].contents(), 0, 256 * 4)
            for (e, s) in caches[i].slotMap { st[e] = Int32(s) }
        }
        for i in 0 ..< nLayers { refreshLayer(i) }

        var gpuMsAccum = 0.0, missAccum = 0
        // route(i) 完了後: moe[i].binds(生 expert id)を読み→ensure→slotTable 更新。
        func routeAndEnsure(_ i: Int) {
            let bp = streamLayers[i].moe.binds.contents().bindMemory(to: Int32.self, capacity: 8)
            var routed: [Int] = []; routed.reserveCapacity(8)
            for k in 0 ..< 8 { routed.append(Int(bp[k])) }
            let before = caches[i].misses   // ★ misses は public cumulative（raw path は lastInds 無し＝missCount() 不可）
            let res = caches[i].ensure(routed)
            // ★ 最適化: O(C) full rebuild(refreshLayer)でなく **routed 8 expert の slot のみ** 更新。
            //   slot_remap は binds(=今 ensure 済 routed)の st だけ読む→evict された stale entry は未読ゆえ安全
            //   (次に同 expert が routed されれば ensure→res で再設定される)。CPU bookkeeping を ~O(8×40) に。
            let st = slotTableBufs[i].contents().bindMemory(to: Int32.self, capacity: 256)
            for (e, slot) in res { st[e] = Int32(slot) }
            if caches[i].misses != before { missAccum += 1 }
        }
        func decodeStep(_ inputTok: Int32, _ pos: Int) -> Int {
            let embedX = model.embed(MLXArray([inputTok], [1, 1])); embedX.eval()
            RawMetalForward.writeBuffer(hb, embedX, H)
            RawMetalForward.fusedForwardGPUInlineDemand(
                hBuf: hb, layers: streamLayers, scratch: sc, slotTables: slotTableBufs,
                H: H, E: 256, K: 8, eps: model.eps, decode: true, pos: pos,
                finalNormW: model.ensureFinalNorm(), routeAndEnsure: routeAndEnsure)
            gpuMsAccum += RawMetalForward.lastGPUExecMs
            let fn = RawMetalForward.readBuffer(sc.normed, H)
            let lg = model.headProj().apply(fn.reshaped([1, 1, H])); lg.eval()
            return MLX.argMax(lg.reshaped([lg.size])).item(Int.self)
        }

        // ③ streaming decode（teacher-forced on refGen）
        model.resetGPUState(); pos = 0
        var match = 0
        for (idx, t) in prompt.enumerated() {
            let pred = decodeStep(t, pos)
            if idx == prompt.count - 1 && pred == Int(refGen[0]) { match += 1 }
            pos += 1
        }
        gpuMsAccum = 0.0; LayerExpertCache.preadNanos = 0; missAccum = 0
        let tStream0 = DispatchTime.now().uptimeNanoseconds
        for i in 0 ..< (N - 1) {
            let pred = decodeStep(refGen[i], pos); pos += 1
            if pred == Int(refGen[i + 1]) { match += 1 }
        }
        let streamSecs = Double(DispatchTime.now().uptimeNanoseconds - tStream0) / 1e9
        let streamTokps = Double(N - 1) / streamSecs
        let gpuMsPerTok = gpuMsAccum / Double(N - 1)
        let preadMsPerTok = Double(LayerExpertCache.preadNanos) / 1e6 / Double(N - 1)
        let cpuMsPerTok = streamSecs / Double(N - 1) * 1000 - gpuMsPerTok - preadMsPerTok
        let missPerTok = Double(missAccum) / Double(N - 1)
        let matchable = N
        let tfOk = match >= matchable

        // ④ ★本番 free-run 実生成: 自身の argmax を次入力に戻す（ref 非依存、真の product 挙動）。
        //    cache は teacher-forced phase で warm 済 → cold IO を含まない steady-state tok/s。
        model.resetGPUState(); pos = 0
        var nextTok = refGen[0]
        for (idx, t) in prompt.enumerated() { let p = decodeStep(t, pos); if idx == prompt.count - 1 { nextTok = Int32(p) }; pos += 1 }
        var freeGen: [Int] = []
        gpuMsAccum = 0.0; LayerExpertCache.preadNanos = 0
        let tFree0 = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< N {
            freeGen.append(Int(nextTok))
            nextTok = Int32(decodeStep(nextTok, pos)); pos += 1
        }
        let freeSecs = Double(DispatchTime.now().uptimeNanoseconds - tFree0) / 1e9
        let freeTokps = Double(N) / freeSecs
        let freeMatch = zip(freeGen, refGen.map { Int($0) }).reduce(0) { $0 + ($1.0 == $1.1 ? 1 : 0) }
        let freeOk = freeMatch == N
        let ok = tfOk && freeOk
        return String(format: """
            [fix-2 raw-stream-decode-inline] inline demand-load(各層1回 dispatch, resume 無し)(C=%d, T=%d, gen N=%d)
              lossless: teacher-forced %d/%d, free-run(自回帰実生成) %d/%d 一致(vs resident greedy)
              ★本番 streaming(steady-state, C=%d, expert SSD)=%.1f tok/s(%.1f ms/tok, 41 CB/tok, miss=%.1f/tok)
                内訳/tok: GPU-exec=%.1fms + pread=%.1fms + CPU bookkeeping=%.1fms
              free-run(cold 再warming込, 参考)=%.1f tok/s(%.1f ms/tok)
              resident(全 expert 常駐, 32GB+ tier)=%.1f tok/s(%.1f ms/tok, 1 CB/tok)
                → steady-state/resident=%.2fx
              %@
            """, C, prompt.count, N, match, matchable, freeMatch, N,
            C, streamTokps, streamSecs / Double(N - 1) * 1000, missPerTok,
            gpuMsPerTok, preadMsPerTok, cpuMsPerTok,
            freeTokps, freeSecs / Double(N) * 1000,
            refTokps, refSecs / Double(N) * 1000,
            streamTokps / refTokps,
            ok ? "✅ lossless 一致(inline 本番配線 OK)" : "❌ argmax 乖離")
    }
}
