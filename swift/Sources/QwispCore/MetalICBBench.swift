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

        let bufs = envInt("QWISP_ICB_BUFS", 5)       // /dispatch の buffer 束縛数（quantized matmul 相当=5）
        // 実層相当の profile を encode するため 3 種 pipeline を用意（op 多様性=pipeline 切替コスト込み）。
        // 中身は同型(rmsnorm 風)だが encode コストは dispatch 数×buffer 数×pipeline 切替で決まり数学に依らない。
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        #define KDEF(NM) kernel void NM(device const float* x [[buffer(0)]], device float* out [[buffer(1)]], \
            device const float* b2 [[buffer(2)]], device const float* b3 [[buffer(3)]], device const float* b4 [[buffer(4)]], \
            uint d [[thread_position_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) { \
            const uint D = 128; threadgroup float sh[128]; float c = x[row*D + d]; sh[d] = c*c; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint s = D>>1; s>0; s>>=1) { if (d<s) sh[d]+=sh[d+s]; threadgroup_barrier(mem_flags::mem_threadgroup); } \
            out[row*D + d] = c * rsqrt(sh[0]/(float)D + 1e-6f); }
        KDEF(k0) KDEF(k1) KDEF(k2)
        """
        var pipelines: [MTLComputePipelineState] = []
        do {
            let lib = try device.makeLibrary(source: src, options: nil)
            for nm in ["k0", "k1", "k2"] {
                let pdesc = MTLComputePipelineDescriptor()
                pdesc.computeFunction = lib.makeFunction(name: nm)!
                pdesc.supportIndirectCommandBuffers = true
                pipelines.append(try device.makeComputePipelineState(descriptor: pdesc, options: [], reflection: nil))
            }
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
        let extra = (0 ..< 3).map { _ in device.makeBuffer(length: n * 4, options: .storageModeShared)! }  // b2,b3,b4 共有
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
                enc.setComputePipelineState(pipelines[k % pipelines.count])   // pipeline 切替コスト込み
                enc.setBuffer(inBufs[k], offset: 0, index: 0)
                enc.setBuffer(outBufs[k], offset: 0, index: 1)
                for bi in 2 ..< bufs { enc.setBuffer(extra[(bi - 2) % 3], offset: 0, index: bi) }
                enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: tg)
            }
            enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            return now() - t
        }

        _ = grid
        let a = pathReencode()
        // (B) compute ICB に K command を一度 encode → 以降 replay（再 encode 無し）。macOS 11+ で利用可
        //     (Swift accessor は macOS14 で indirectComputeCommandAt に改名)。
        let icbDesc = MTLIndirectCommandBufferDescriptor()
        icbDesc.commandTypes = [.concurrentDispatch]
        icbDesc.inheritBuffers = false
        icbDesc.inheritPipelineState = false
        icbDesc.maxKernelBufferBindCount = bufs
        var bResult: (ms: Double, cpu: Double)? = nil
        if let icb = device.makeIndirectCommandBuffer(descriptor: icbDesc, maxCommandCount: K, options: []) {
            for k in 0 ..< K {
                let c = icb.indirectComputeCommandAt(k)
                c.setComputePipelineState(pipelines[k % pipelines.count])
                c.setKernelBuffer(inBufs[k], offset: 0, at: 0)
                c.setKernelBuffer(outBufs[k], offset: 0, at: 1)
                for bi in 2 ..< bufs { c.setKernelBuffer(extra[(bi - 2) % 3], offset: 0, at: bi) }
                c.concurrentDispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: tg)
            }
            func replayOnce() -> UInt64 {
                let t = now()
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.useResources(inBufs, usage: .read)
                enc.useResources(outBufs, usage: .write)
                enc.executeCommandsInBuffer(icb, range: 0 ..< K)
                enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                return now() - t
            }
            for _ in 0 ..< 3 { _ = replayOnce() }
            var tAcc: UInt64 = 0; let c0 = cpuNs()
            for _ in 0 ..< reps { tAcc += replayOnce() }
            bResult = (Double(tAcc) / Double(reps) / 1e6, Double(cpuNs() - c0) / Double(tAcc))
        }
        let aCpuMs = a.ms * a.cpu, aUs = aCpuMs * 1000.0 / Double(K)
        var out = "[ICB-bench K=\(K) dispatch, rows=\(rows), reps=\(reps)] capture/replay de-risk（macOS）\n"
            + String(format: "  (A) 単一 encoder で K dispatch re-encode: %.3f ms wall, CPU %.3f ms/iter (~%.1f us/dispatch encode)\n", a.ms, aCpuMs, aUs)
        if let b = bResult {
            let bCpuMs = b.ms * b.cpu
            // ★公平比較は CPU 絶対時間（encode コスト本体）。wall は独立 kernel の GPU 並列で混入するため非代表。
            let cpuCut = aCpuMs > 0 ? (1 - bCpuMs / aCpuMs) * 100 : 0
            out += String(format: "  (B) compute ICB replay (再 encode 無): %.3f ms wall, CPU %.3f ms/iter\n", b.ms, bCpuMs)
            out += String(format: "  → ICB が CPU-encode を %.0f%% 削減（wall は独立 kernel の並列で混入ゆえ非代表）\n", cpuCut)
            out += abs(cpuCut) < 25
                ? "  判定: ICB replay は CPU-encode をほぼ削減せず（llama.cpp の null 結果と整合）。replay は本筋でない"
                : "  判定: ICB が CPU-encode を削減（要追加検証）"
        } else {
            out += "  (B) ICB 生成失敗（family 非対応?）\n"
        }
        // ★ forward 外挿（実層 ~25 dispatch×40 層 ≈ K_total）。raw encode CPU を 40 層へスケール。
        let layersEq = Double(K) / 25.0                              // この run が相当する「層数」
        let rawFwdEncodeMs = layersEq > 0 ? aCpuMs / layersEq * 40.0 : aCpuMs   // 40 層 forward の raw encode CPU
        let mlxFwdMs = 16.0, gpuFloorMs = 9.0                        // 実測: MLX no-sync forward 16ms, GPU-exec ~9ms
        let rawFwdWall = Swift.max(gpuFloorMs, rawFwdEncodeMs)       // raw は GPU-bound 化（encode<exec なら exec 床）
        let speedup = mlxFwdMs / rawFwdWall
        out += String(format: "\n  ★raw-Metal encode ~%.1f us/dispatch（実層相当 %.0f dispatch/層・%d buffer・3 pipeline）= MLX ~80us の ~%.0fx 安\n",
                      aUs, 25.0, bufs, 80.0 / Swift.max(0.1, aUs))
            + String(format: "  forward 外挿: raw encode %.1fms(40層) %@ GPU-exec %.0fms → wall ~%.1fms vs MLX %.0fms = ~%.2fx\n",
                     rawFwdEncodeMs, rawFwdEncodeMs < gpuFloorMs ? "<" : "≥", gpuFloorMs, rawFwdWall, mlxFwdMs, speedup)
            + (speedup >= 1.4
               ? "  判定: ✅ GREEN — raw-Metal forward で encode が GPU 床を下回り ~1.5-1.7x。rewrite の ROI 確認"
               : "  判定: △ encode 削減効くが GPU 床近く、ROI 限定。実 kernel で再評価")
            + "\n  機構: ICB replay でなく『MLX 迂回 raw-Metal forward(単一 encoder + pipeline cache + MTLResidencySet)』"
        return out
    }

    static func envInt(_ k: String, _ d: Int) -> Int {
        guard let v = ProcessInfo.processInfo.environment[k], let i = Int(v) else { return d }
        return i
    }
}
