import Foundation
import Metal

/// raw-dispatch-bench: per-dispatch overhead の直接実測(kernel fusion キャンペーンの physics smoke)。
///
/// 質問: 1 command buffer 内で依存チェーンされた compute dispatch 1 本あたりの固定コストは何 µs か?
/// fusion 見積り(d1-handoff)は「~700 dispatch × ~5µs ≈ 3.5ms/forward」を仮定したが、実測の
/// dispatch 数は ~1354/forward。slope(µs/dispatch)× 削減可能本数が fusion キャンペーンの上限利得。
///
/// 方法: ping-pong buffer を читает→書く極小 kernel(32 float copy+add)を N 本チェーン
/// (逐次依存で並行実行を禁止)、GPU-exec 時間(gpuEndTime−gpuStartTime)の N に対する傾きを取る。
/// 仕事量が overhead を汚染しないことの確認に 16 倍仕事量の変種でも同じ N スイープを行う
/// (slope 差 = 仕事量寄与、切片/slope 一致 = overhead 支配の確認)。
public enum RawDispatchBench {
    static let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void tiny_chain(device const float* a [[buffer(0)]],
                           device float* b [[buffer(1)]],
                           constant uint& W [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
        if (tid < W) b[tid] = a[tid] + 1.0f;
    }
    """

    public static func run() -> String {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return "[dispatch-bench] no device" }
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: src, options: nil) }
        catch { return "[dispatch-bench] compile: \(error)" }
        guard let fn = lib.makeFunction(name: "tiny_chain"),
              let pso = try? device.makeComputePipelineState(function: fn) else {
            return "[dispatch-bench] pipeline nil"
        }

        var lines: [String] = ["[dispatch-bench] per-dispatch overhead(1-CB 逐次依存チェーン)"]
        // W=32(極小仕事)と W=8192(仕事 256 倍)で N スイープ → slope 差で仕事量寄与を分離。
        for W in [32, 8192] {
            let bytes = W * 4
            guard let bufA = device.makeBuffer(length: bytes, options: .storageModeShared),
                  let bufB = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                return "[dispatch-bench] buffer nil"
            }
            memset(bufA.contents(), 0, bytes)
            var results: [(n: Int, gpuMs: Double, wallMs: Double)] = []
            for n in [16, 256, 1024, 4096] {
                // warmup 1 + 実測 5 回の median
                var gpuTimes: [Double] = [], wallTimes: [Double] = []
                for rep in 0 ..< 6 {
                    guard let cb = queue.makeCommandBuffer(),
                          let enc = cb.makeComputeCommandEncoder() else { return "[dispatch-bench] cb nil" }
                    enc.setComputePipelineState(pso)
                    var w32 = UInt32(W)
                    enc.setBytes(&w32, length: 4, index: 2)
                    for i in 0 ..< n {
                        // ping-pong: 前段出力を次段入力に(逐次依存で GPU の並行 scheduling を禁止)
                        enc.setBuffer(i % 2 == 0 ? bufA : bufB, offset: 0, index: 0)
                        enc.setBuffer(i % 2 == 0 ? bufB : bufA, offset: 0, index: 1)
                        enc.dispatchThreadgroups(MTLSize(width: (W + 255) / 256, height: 1, depth: 1),
                                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                    enc.endEncoding()
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    cb.commit(); cb.waitUntilCompleted()
                    let t1 = DispatchTime.now().uptimeNanoseconds
                    if rep > 0 {
                        gpuTimes.append((cb.gpuEndTime - cb.gpuStartTime) * 1000.0)
                        wallTimes.append(Double(t1 - t0) / 1e6)
                    }
                }
                gpuTimes.sort(); wallTimes.sort()
                results.append((n, gpuTimes[gpuTimes.count / 2], wallTimes[wallTimes.count / 2]))
            }
            for r in results {
                lines.append(String(format: "  W=%5d N=%5d: GPU %8.3f ms (%6.3f µs/dispatch)  wall %8.3f ms",
                                    W, r.n, r.gpuMs, r.gpuMs * 1000.0 / Double(r.n), r.wallMs))
            }
            // slope: 大 N 2 点の差分(切片=CB 固定費を除去)
            let a = results[results.count - 2], b = results[results.count - 1]
            let slope = (b.gpuMs - a.gpuMs) * 1000.0 / Double(b.n - a.n)
            lines.append(String(format: "  W=%5d slope(N=%d→%d): %.3f µs/dispatch", W, a.n, b.n, slope))
        }
        lines.append("[dispatch-bench] 判定基準: slope_W32 ≈ slope_W8192 なら overhead 支配(仕事量非依存)。")
        lines.append("  fusion 上限利得 ≈ slope × 削減 dispatch 数(~1354→~200 なら ×~1150)/ M=1 step ~14ms。")
        return lines.joined(separator: "\n")
    }
}
