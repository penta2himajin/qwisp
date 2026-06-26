import Foundation
import MLX
import MLXRandom
import Metal

/// 40層 arena-MoE forward の速度（concat 排除＝持続 arena + gather_qmm のみ）を release で測る.
/// Python の streaming は毎層 concat(~16-25ms)を払うが、Swift arena は払わない（M3 実証）。
/// ここでは MoE compute floor（pipeline, sync/concat 無し）を測り、~29tok/s 試算の土台を確認。
public enum ArenaBench {
    public static func run(layers: Int = 40, T: Int = 2, K: Int = 8, B: Int = 64, reps: Int = 50)
        -> String
    {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let IN = 2048, I = 512

        func quantProj(_ outd: Int, _ ind: Int) -> [String: MLXArray] {
            let fp = MLXRandom.normal([B, outd, ind]) * 0.02
            let (wq, s, b) = quantized(fp, groupSize: 64, bits: 2)
            return ["weight": wq, "scales": s, "biases": b ?? wq]
        }

        var moe: [PersistentMoELayer] = []
        for _ in 0..<layers {
            var loaded: [String: MLXArray] = [:]
            for (p, sh) in [("gate_proj", (I, IN)), ("up_proj", (I, IN)), ("down_proj", (IN, I))] {
                for (k, v) in quantProj(sh.0, sh.1) { loaded["\(p).\(k)"] = v }
            }
            guard let l = PersistentMoELayer(device: device, loaded: loaded, bits: 2, gs: 64)
            else { return "ERROR: layer 構築失敗" }
            moe.append(l)
        }

        let x0 = MLXRandom.normal([T, IN]) * 0.1
        let inds = MLXRandom.randInt(0 ..< Int32(B), [T, K]).asType(.uint32)
        MLX.eval(x0, inds)

        // pipeline forward: 層間 eval なし → 末尾で1回 eval（GPU が 40層を流す）
        func forward() -> MLXArray {
            var x = x0
            for l in moe {
                let y = l(x, inds)          // [T,K,H], H==IN
                x = y.sum(axis: 1)          // K 合算 → [T,IN]
            }
            return x
        }

        for _ in 0..<10 { forward().eval() }
        let t0 = DispatchTime.now()
        for _ in 0..<reps { forward().eval() }
        let msPipe = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            / Double(reps)

        // 比較: native mlx 配列（rawPointer ラップ無し）で同じ forward → rawPointer overhead を分離
        struct NLayer { let g: (MLXArray, MLXArray, MLXArray); let u: (MLXArray, MLXArray, MLXArray)
                        let d: (MLXArray, MLXArray, MLXArray) }
        func nproj(_ o: Int, _ i: Int) -> (MLXArray, MLXArray, MLXArray) {
            let (w, s, b) = quantized(MLXRandom.normal([B, o, i]) * 0.02, groupSize: 64, bits: 2)
            return (w, s, b ?? w)
        }
        let nlayers = (0..<layers).map { _ in
            NLayer(g: nproj(I, IN), u: nproj(I, IN), d: nproj(IN, I)) }
        func nqmm(_ x: MLXArray, _ p: (MLXArray, MLXArray, MLXArray)) -> MLXArray {
            gatherQuantizedMatmul(x, p.0, scales: p.1, biases: p.2, rhsIndices: inds,
                                  transpose: true, groupSize: 64, bits: 2, mode: .affine)
        }
        func nforward() -> MLXArray {
            var x = x0
            for l in nlayers {
                let xe = x.expandedDimensions(axes: [-2, -3])
                let g = nqmm(xe, l.g); let u = nqmm(xe, l.u)
                let h = (g * MLX.sigmoid(g)) * u
                x = nqmm(h, l.d).squeezed(axis: -2).sum(axis: 1)
            }
            return x
        }
        for _ in 0..<10 { nforward().eval() }
        let tn = DispatchTime.now()
        for _ in 0..<reps { nforward().eval() }
        let msNative = Double(DispatchTime.now().uptimeNanoseconds - tn.uptimeNanoseconds) / 1e6
            / Double(reps)

        // 参考: 層間 eval（barrier）版 — sync を入れた時の上振れ感
        for _ in 0..<5 {
            var x = x0
            for l in moe { x = l(x, inds).sum(axis: 1); x.eval() }
        }
        let t1 = DispatchTime.now()
        for _ in 0..<reps {
            var x = x0
            for l in moe { x = l(x, inds).sum(axis: 1); x.eval() }
        }
        let msBarrier = Double(DispatchTime.now().uptimeNanoseconds - t1.uptimeNanoseconds) / 1e6
            / Double(reps)

        return """
        [bench] arena-MoE \(layers)層 forward (T=\(T) K=\(K) B=\(B), 2bit):
          arena(rawPointer所有, pipeline)  : \(String(format: "%.2f", msPipe)) ms/forward  (\(String(format: "%.0f", 1000/msPipe)) tok/s)
          native配列(rawPointer無, pipeline): \(String(format: "%.2f", msNative)) ms/forward  (\(String(format: "%.0f", 1000/msNative)) tok/s)
          arena per-layer barrier          : \(String(format: "%.2f", msBarrier)) ms/forward
          → rawPointer overhead = \(String(format: "%+.2f", msPipe-msNative)) ms（in-place 更新可能 vs 速度のトレードオフ）
          ※ Python: arena床11.4ms / concat(streaming)21.8ms。concat 排除が狙い。
        """
    }
}
