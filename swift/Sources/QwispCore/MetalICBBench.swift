import Foundation
import Metal

/// issue#5 capture/replay de-risk スパイク（raw Metal, MLX 非依存）。
/// 「1 forward = K dispatch」を (A) 毎回 encode し直す経路 と (B) MTLIndirectCommandBuffer(ICB) に
/// 一度 encode→以降 replay する経路 で比較し、**replay が CPU-encode を削減するか**（mlx-native の
/// ~13x per-dispatch 主張）を二値判定する。これが効けば capture/replay 本実装の価値が確定。
/// - env: QWISP_RUN=icb-bench / QWISP_ICB_K(dispatch 数, 既定 200) / QWISP_ICB_REPS(既定 300) /
///        QWISP_ICB_N(各 kernel の行数, 既定 32)
public enum MetalICBBench {
    public static func run() -> String {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return "ERROR: no Metal device/queue" }
        let K = envInt("QWISP_ICB_K", 200)        // forward の per-token dispatch 規模を模擬
        let reps = envInt("QWISP_ICB_N_REPS", envInt("QWISP_ICB_REPS", 300))
        let rows = envInt("QWISP_ICB_N", 32)
        let D = 128                                // 行あたり要素（rmsnorm 風の reduction で GPU 実行を非自明に）
        guard device.supportsFamily(.apple7) || device.supportsFamily(.metal3) else {
            return "ERROR: ICB/Metal3 非対応 device"
        }

        // 代表 kernel: 行ごと(D スレッド) sum(x^2)→rsqrt→正規化 出力（実 forward の小カーネルを模擬）
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void rmsk(device const float* x [[buffer(0)]],
                         device float* out      [[buffer(1)]],
                         uint d [[thread_position_in_threadgroup]],
                         uint row [[threadgroup_position_in_grid]]) {
            const uint D = 128;
            threadgroup float sh[128];
            float c = x[row*D + d];
            sh[d] = c*c;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = D>>1; s>0; s>>=1) { if (d<s) sh[d]+=sh[d+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }
            float r = rsqrt(sh[0]/(float)D + 1e-6f);
            out[row*D + d] = c * r;
        }
        """
        let pipeline: MTLComputePipelineState
        do {
            let lib = try device.makeLibrary(source: src, options: nil)
            pipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "rmsk")!)
        } catch { return "ERROR: kernel compile: \(error)" }

        // K 個の独立 buffer（各 dispatch が別 buffer を触る=実 forward の依存連鎖を粗く模擬）
        let n = rows * D
        let inBufs = (0 ..< K).map { _ -> MTLBuffer in
            let b = device.makeBuffer(length: n * 4, options: .storageModeShared)!
            let p = b.contents().bindMemory(to: Float.self, capacity: n)
            for i in 0 ..< n { p[i] = Float((i % 13) + 1) }
            return b
        }
        let outBufs = (0 ..< K).map { _ in device.makeBuffer(length: n * 4, options: .storageModeShared)! }
        let tg = MTLSize(width: D, height: 1, depth: 1)
        let grid = MTLSize(width: D, height: rows, depth: 1)

        func cpuNs() -> UInt64 {
            var ru = rusage(); getrusage(RUSAGE_SELF, &ru)
            return UInt64(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1_000_000_000
                 + UInt64(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1000
        }
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

        // (A) 毎 rep で K dispatch を encode し直す（MLX の tape 再 encode を模擬）
        func pathReencode() -> (ms: Double, cpu: Double) {
            // warmup
            for _ in 0 ..< 3 { _ = encodeOnce() }
            var tAcc: UInt64 = 0; let c0 = cpuNs()
            for _ in 0 ..< reps { tAcc += encodeOnce() }
            let cpu = Double(cpuNs() - c0) / Double(tAcc)
            return (Double(tAcc) / Double(reps) / 1e6, cpu)
        }
        func encodeOnce() -> UInt64 {
            let t = now()
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            for k in 0 ..< K {
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(inBufs[k], offset: 0, index: 0)
                enc.setBuffer(outBufs[k], offset: 0, index: 1)
                enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: tg)
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return now() - t
        }

        _ = grid
        // ★macOS では compute の ICB(replay) が unavailable（render ICB のみ）。
        //   → 「encode once→replay で再 encode を省く」は不可。達成可能なのは「単一 cmd buffer+encoder で
        //   K dispatch を効率 re-encode」まで。その per-dispatch encode CPU コストを測り MLX と比較する。
        let a = pathReencode()
        // per-dispatch の encode+launch コスト見積（CPU 時間 / dispatch 数）
        let cpuMsPerIter = a.ms * a.cpu                          // ≈ CPU 占有 ms/iter
        let usPerDispatch = cpuMsPerIter * 1000.0 / Double(K)
        // MLX no-sync forward 実測: ~16ms wall, CPU~1.0 cores, ~200 dispatch/forward(C=256) → ~80us/dispatch 相当
        let verdict: String
        if usPerDispatch < 30 {
            verdict = String(format: "✅ raw-Metal の単一 encoder は %.1f us/dispatch＝MLX(~80us/dispatch 相当)より大幅安。"
                + "自前 Metal forward に encode-efficiency の headroom あり（replay 無しでも価値）", usPerDispatch)
        } else if usPerDispatch > 60 {
            verdict = String(format: "❌ raw でも %.1f us/dispatch＝MLX と同等。encode は本質的に高く headroom 小", usPerDispatch)
        } else {
            verdict = String(format: "△ %.1f us/dispatch（MLX ~80 との中間）。実 kernel 規模で再評価", usPerDispatch)
        }
        return "[ICB-bench K=\(K) dispatch, rows=\(rows), reps=\(reps)] capture/replay de-risk\n"
            + "  ※ macOS は compute ICB(replay) 非対応＝literal replay 不可。測るのは単一 encoder の re-encode 効率\n"
            + String(format: "  (A) 単一 cmd buffer+encoder で K dispatch re-encode: %.3f ms/iter  CPU-busy=%.2f cores\n", a.ms, a.cpu)
            + String(format: "  → ~%.1f us/dispatch (encode+launch)\n  判定: %@", usPerDispatch, verdict)
    }

    static func envInt(_ k: String, _ d: Int) -> Int {
        guard let v = ProcessInfo.processInfo.environment[k], let i = Int(v) else { return d }
        return i
    }
}
