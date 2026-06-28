import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// A3 feasibility: dispatch tax の定量化。同一総 FLOPs/総 bytes を S 個の dispatch に分割し計時。
/// time が S に比例→dispatch-bound(融合=dispatch 削減が効く)。time が一定→compute-bound(融合無効)。
/// これで「1 forward の dispatch を N→M に減らすと何 ms 回収できるか」の天井を見積もる。
public enum DispatchBench {
    public static func run() -> String {
        MLXRandom.seed(0)
        let H = 2048
        let x = (MLXRandom.normal([1, H]) * 0.5).asType(.float16)

        // (1) 出力次元分割: [1,H]@[H,H] を S 個の [1,H]@[H,H/S] へ。総 FLOPs/bytes 一定、dispatch=S。
        func splitMatmul(_ S: Int) -> Double {
            let cols = H / S
            let ws = (0 ..< S).map { _ in (MLXRandom.normal([H, cols]) * 0.02).asType(.float16) }
            func run1() -> [MLXArray] { ws.map { MLX.matmul(x, $0) } }
            let w0 = run1(); MLX.eval(w0)                          // warmup
            let reps = 200
            let t = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< reps { let ys = run1(); MLX.eval(ys) }
            return Double(DispatchTime.now().uptimeNanoseconds - t) / Double(reps) / 1e6
        }

        // (2) 量子化版（実モデルの射影相当）: 同じ分割を 4bit qmm で。
        func splitQmm(_ S: Int) -> Double {
            let cols = H / S
            let qs = (0 ..< S).map { _ -> (MLXArray, MLXArray, MLXArray) in
                let w = (MLXRandom.normal([cols, H]) * 0.02).asType(.float16)
                let (wq, sc, b) = MLX.quantized(w, groupSize: 64, bits: 4); return (wq, sc, b!)
            }
            func run1() -> [MLXArray] {
                qs.map { MLX.quantizedMatmul(x, $0.0, scales: $0.1, biases: $0.2, transpose: true, groupSize: 64, bits: 4) }
            }
            let w0 = run1(); MLX.eval(w0)
            let reps = 200
            let t = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< reps { let ys = run1(); MLX.eval(ys) }
            return Double(DispatchTime.now().uptimeNanoseconds - t) / Double(reps) / 1e6
        }

        // (3) ★逐次依存チェーン（実層の構造）: N 個の elementwise/小matmul を依存させ critical path を測る。
        //   各 op が前の出力に依存 → overlap 不可、dispatch latency が直列に積算。これが A3 が削る対象。
        let wSmall = (MLXRandom.normal([H, H]) * 0.02).asType(.float16)
        let w0v = MLXArray.ones([H]).asType(.float16)
        func chainElem(_ N: Int) -> Double {                      // N 個の逐次 elementwise（matmul 無し）
            func run1() -> MLXArray { var y = x; for _ in 0 ..< N { y = MLXFast.rmsNorm(silu(y), weight: w0v, eps: 1e-6) }; return y }
            let w = run1(); w.eval()
            let reps = 100
            let t = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< reps { let y = run1(); y.eval() }
            return Double(DispatchTime.now().uptimeNanoseconds - t) / Double(reps) / 1e6
        }
        _ = wSmall

        var l1 = "  (A)独立分割 plain f16(総FLOPs一定,dispatch=S): "
        var l2 = "  (A)独立分割 quant 4bit                    : "
        for S in [1, 4, 16, 32] {
            l1 += String(format: "S=%d:%.3f  ", S, splitMatmul(S))
            l2 += String(format: "S=%d:%.3f  ", S, splitQmm(S))
        }
        var l3 = "  (B)逐次elementwise chain(matmul無, N個依存): "
        for N in [1, 8, 32, 96] { l3 += String(format: "N=%d:%.3fms  ", N, chainElem(N)) }
        return "[DispatchBench] batch=1, H=\(H):\n" + l1 + "\n" + l2 + "\n" + l3
            + "\n  → (B)が N に比例なら逐次 dispatch-bound＝A3 融合の回収対象。per-op = 傾き"
    }
}
