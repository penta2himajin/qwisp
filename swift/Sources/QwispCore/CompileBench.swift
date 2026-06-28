import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// mx.compile が batch=1 の per-layer launch overhead を削減するかの PoC microbench。
/// greedy 壁(~50ms=40層×~1.2ms/層)は kernel-launch-bound([[greedy-wall-fixed-overhead]])と判明。
/// compile がグラフ融合で launch 数を減らせば層あたりコストが落ちる → 全トークン lossless 高速化。
public enum CompileBench {
    /// 1 層相当の op chain(matmul+rmsNorm+silu を nOps 回)を compiled vs uncompiled で計時。
    /// plain f16 と quantized 4bit(実モデルの射影)両方で測る。
    public static func run() -> String {
        MLXRandom.seed(0)
        let H = 2048
        let nOps = 32
        let w0 = MLXArray.ones([H]).asType(.float16)

        // --- plain f16 chain ---
        let ws = (0 ..< nOps).map { _ in (MLXRandom.normal([H, H]) * 0.02).asType(.float16) }
        func chainPlain(_ x0: MLXArray) -> MLXArray {
            var x = x0
            for i in 0 ..< nOps {
                x = MLX.matmul(x, ws[i])
                x = MLXFast.rmsNorm(x, weight: w0, eps: 1e-6)
                x = silu(x)
            }
            return x
        }

        // --- quantized 4bit chain(実モデル射影相当) ---
        let qs = ws.map { w -> (MLXArray, MLXArray, MLXArray) in
            let (wq, s, b) = MLX.quantized(w, groupSize: 64, bits: 4); return (wq, s, b!)
        }
        func chainQuant(_ x0: MLXArray) -> MLXArray {
            var x = x0
            for i in 0 ..< nOps {
                x = MLX.quantizedMatmul(x, qs[i].0, scales: qs[i].1, biases: qs[i].2,
                                        transpose: true, groupSize: 64, bits: 4)
                x = MLXFast.rmsNorm(x, weight: w0, eps: 1e-6)
                x = silu(x)
            }
            return x
        }

        // --- elementwise-only chain(matmul 無し): compile が融合で効くはずの対照 ---
        let nEl = 96
        func chainElem(_ x0: MLXArray) -> MLXArray {
            var x = x0
            for _ in 0 ..< nEl { x = silu(x) * 1.001 + 0.0001; x = MLXFast.rmsNorm(x, weight: w0, eps: 1e-6) }
            return x
        }
        // --- matmul-only chain(norm/act 無し) ---
        func chainMM(_ x0: MLXArray) -> MLXArray {
            var x = x0
            for i in 0 ..< nOps { x = MLX.matmul(x, ws[i]) }
            return x
        }

        let x = (MLXRandom.normal([1, H]) * 0.5).asType(.float16)
        func bench(_ f: @escaping (MLXArray) -> MLXArray, _ reps: Int) -> Double {
            let w = f(x); w.eval()                                   // warmup(+compile trace)
            let t = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< reps { let y = f(x); y.eval() }
            return Double(DispatchTime.now().uptimeNanoseconds - t) / Double(reps) / 1e6
        }

        let pUnc = bench(chainPlain, 200), pCmp = bench(compile(chainPlain), 200)
        let qUnc = bench(chainQuant, 200), qCmp = bench(compile(chainQuant), 200)
        let eUnc = bench(chainElem, 200), eCmp = bench(compile(chainElem), 200)
        let mUnc = bench(chainMM, 200), mCmp = bench(compile(chainMM), 200)
        return String(format: """
            [COMPILE-LAUNCH] batch=1, H=%d, 200 reps:
              mixed(matmul+norm+silu)×%d  : unc=%.3f cmp=%.3f ms  %.2fx
              quant(qmm+norm+silu)×%d     : unc=%.3f cmp=%.3f ms  %.2fx
              elementwise(silu+norm)×%d   : unc=%.3f cmp=%.3f ms  %.2fx  ← compile が効く対照
              matmul-only×%d              : unc=%.3f cmp=%.3f ms  %.2fx  ← matmul は融合不可
              → elementwise で大きく効き matmul で効かないなら、実層の matmul 部は compile 不可
            """, H, nOps, pUnc, pCmp, pUnc / pCmp, nOps, qUnc, qCmp, qUnc / qCmp,
            nEl, eUnc, eCmp, eUnc / eCmp, nOps, mUnc, mCmp, mUnc / mCmp)
    }
}
