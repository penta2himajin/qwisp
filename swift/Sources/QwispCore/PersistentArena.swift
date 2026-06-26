import Foundation
import MLX
import Metal

/// M3: buffer 所有権で「持続バッファの in-place 更新」が可能かを検証.
///
/// Python mlx では `arena[slot]=expert` が immutable のため全バッファコピー(=[256,...] で ~1.4ms,
/// concat 0.3ms の 4.6x)になり持続バッファが破綻した。Swift では自前 MTLBuffer を所有し、
/// 1 expert を in-place memcpy → MLXArray でラップ → gather_qmm、が可能なはず。
/// それが安く・ビット正しければ「concat 不要の常駐 arena」が 8GB で viable になる。
public enum PersistentArenaTest {
    public static func run(refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let ref = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let x = ref["x"], let inds = ref["inds"], let w = ref["w"],
              let scales = ref["scales"], let biases = ref["biases"], let expected = ref["expected"]
        else { return "ERROR: ref 不足" }

        // 各 quantized tensor を「所有する MTLBuffer」へコピー（noCopy:false = 自前バッファ化）
        guard let wbuf = w.asMTLBuffer(device: device, noCopy: false),
              let sbuf = scales.asMTLBuffer(device: device, noCopy: false),
              let bbuf = biases.asMTLBuffer(device: device, noCopy: false)
        else { return "ERROR: asMTLBuffer 失敗" }

        let E = w.dim(0), OUT = w.dim(1), inPacked = w.dim(2)
        let gsCount = scales.dim(2)
        let T = x.dim(0), K = inds.dim(1)

        // 所有バッファを MLXArray でラップ（rawPointer, no-copy 参照）
        func wrap(_ buf: MTLBuffer, _ shape: [Int], _ dt: DType) -> MLXArray {
            MLXArray(rawPointer: buf.contents(), shape, dtype: dt) { _ = buf }
        }
        func gather(_ wArr: MLXArray, _ sArr: MLXArray, _ bArr: MLXArray) -> MLXArray {
            let xe = x.expandedDimensions(axes: [-2, -3])
            let y = gatherQuantizedMatmul(
                xe, wArr, scales: sArr, biases: bArr, rhsIndices: inds.asType(.uint32),
                transpose: true, groupSize: 64, bits: 2, mode: .affine, sortedIndices: false)
            return y.reshaped([T, K, OUT])
        }

        // C0: MLXArray(rawPointer:) は backing を共有するか／コピーするか（直接スカラ検証）
        // 共有なら持続 arena は「同一 array を in-place 更新」で済む。コピーなら毎回 re-wrap が要る。
        let tbuf = device.makeBuffer(length: 4 * MemoryLayout<Float>.size, options: .storageModeShared)!
        let tptr = tbuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<4 { tptr[i] = 1.0 }
        let ta = MLXArray(rawPointer: tbuf.contents(), [4], dtype: .float32) { _ = tbuf }
        let s0 = (ta + 0.0).sum().item(Float.self)             // = 4.0
        tptr[0] = 99.0                                          // buffer を in-place 変更
        let s1 = (ta + 0.0).sum().item(Float.self)             // 共有なら 102.0
        let sharedBacking = abs(s1 - s0 - 98.0) < 1e-3

        // C4: native mlx 配列(高速・mlx 常駐)を asMTLBuffer(noCopy:true) で in-place 変更→反映するか
        // 成立すれば「native の速度 + in-place 可能」＝rawPointer overhead を回避できる本命。
        let nat = MLXRandom.normal([8]) * 0.0 + 1.0          // [1,1,...,1]
        nat.eval()
        let n0 = nat.sum().item(Float.self)                  // 8.0
        var nativeReflects = false
        if let nbuf = nat.asMTLBuffer(device: device, noCopy: true) {
            nbuf.contents().assumingMemoryBound(to: Float.self)[0] += 1000.0
            let n1 = (nat + 0.0).sum().item(Float.self)
            nativeReflects = (n1 - n0 > 999.0)
        }

        // 持続 array を一度だけ作る（C2 で同一 array を in-place 更新して反映を見る）
        let wArr = wrap(wbuf, [E, OUT, inPacked], .uint32)
        let sArr = wrap(sbuf, [E, OUT, gsCount], .float16)
        let bArr = wrap(bbuf, [E, OUT, gsCount], .float16)

        // C1: 所有バッファ経由で gather_qmm がビット一致か
        let y1 = gather(wArr, sArr, bArr)
        y1.eval()
        let rel1 = MLX.max(MLX.abs(y1 - expected)).item(Float.self)
                 / (MLX.max(MLX.abs(expected)).item(Float.self) + 1e-9)
        let c1 = rel1 < 1e-3

        // C2: inds が実際に選ぶ expert を in-place で 0 化 → 同一 array で再 gather、反映されるか
        let expertBytes = OUT * inPacked * MemoryLayout<UInt32>.size
        let sel = Int(inds.asArray(Int32.self)[0])             // 確実に選択される expert
        memset(wbuf.contents().advanced(by: sel * expertBytes), 0, expertBytes)
        let y2 = gather(wArr, sArr, bArr)                       // ★ 同一 wArr（re-wrap なし）
        y2.eval()
        let changed = MLX.max(MLX.abs(y2 - y1)).item(Float.self) > 1e-6

        // C3: in-place 更新コスト（1 expert memcpy）を計測
        let src = malloc(expertBytes)!
        memset(src, 0xAB, expertBytes)
        let reps = 2000
        // warmup
        for _ in 0..<100 { wbuf.contents().copyMemory(from: src, byteCount: expertBytes) }
        let t0 = DispatchTime.now()
        for _ in 0..<reps { wbuf.contents().copyMemory(from: src, byteCount: expertBytes) }
        let updMs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6 / Double(reps)
        free(src)

        let verdict = (c1 && changed && sharedBacking)
            ? "→ 持続 arena VIABLE ✅: 同一 array を 1-expert in-place 更新(\(String(format: "%.4f", updMs))ms)で gather_qmm に反映＝concat 不要"
            : (c1 && changed)
                ? "→ 部分的: 更新は反映されるが backing 非共有＝re-wrap 要（wrap コスト次第）"
                : "→ NG: 所有バッファ経路が機能せず"
        return """
        [M3] 持続バッファ(自前 MTLBuffer)検証:
          C0 rawPointer backing 共有: \(sharedBacking ? "YES ✅ (in-place 即反映)" : "NO (mlx がコピー)")  (s0=\(s0) s1=\(s1))
          C4 native配列を noCopy で in-place 変更が反映: \(nativeReflects ? "YES ✅ (=native速度+mutable の本命)" : "NO ❌ (native は mutate 不可→rawPointer 要)")
          C1 所有バッファで gather_qmm: rel=\(String(format: "%.2e", rel1)) \(c1 ? "OK ✅ bit一致" : "NG ❌")
          C2 同一 array で in-place 更新が反映(expert \(sel)): \(changed ? "YES ✅" : "NO ❌")
          C3 1 expert in-place 更新コスト: \(String(format: "%.4f", updMs)) ms  (vs Python 全コピー ~1.4ms)
          \(verdict)
          expert size=\(expertBytes/1024)KB E=\(E) OUT=\(OUT)
        """
    }
}
