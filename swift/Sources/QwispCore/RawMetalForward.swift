import Foundation
import Metal
import MLX
import MLXRandom

/// issue#5 raw-Metal forward 本実装の足場。MLX を迂回し forward を自前 Metal kernel + 単一 encoder で
/// 組むための基盤。第一歩 = quantized matmul(4-bit affine, gs=64)を MLX の quantizedMatmul と bit-exact
/// 照合（最難関の format + MLX weight buffer 共有 を検証）。
public enum RawMetalForward {
    nonisolated(unsafe) static var _qmmPipeline: MTLComputePipelineState?
    nonisolated(unsafe) static var _device: MTLDevice?
    nonisolated(unsafe) static var _queue: MTLCommandQueue?

    static func ensure() -> (MTLDevice, MTLCommandQueue)? {
        if let d = _device, let q = _queue { return (d, q) }
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        _device = d; _queue = q
        return (d, q)
    }

    /// 4-bit affine quantized matmul（decode gemv 一般: x[M,K] · Wq[N,K] → out[M,N], transpose=true）。
    /// dequant: w[n,k] = scales[n, k/gs]·nibble + biases[n, k/gs]、nibble=低位から 8 個/uint32。
    /// MLX weight buffer(wq/scales/biases)を asMTLBuffer(noCopy)で共有して読む。
    static func qmm(_ x: MLXArray, _ wq: MLXArray, scales: MLXArray, biases: MLXArray,
                    M: Int, K: Int, N: Int, bits: Int = 4, gs: Int = 64) -> MLXArray? {
        guard let (device, queue) = ensure() else { return nil }
        if _qmmPipeline == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void qmm4(device const half*  x      [[buffer(0)]],
                             device const uint*  wq     [[buffer(1)]],
                             device const half*  scales [[buffer(2)]],
                             device const half*  biases [[buffer(3)]],
                             device half*        out    [[buffer(4)]],
                             constant uint&       K      [[buffer(5)]],
                             constant uint&       N      [[buffer(6)]],
                             constant uint&       GS     [[buffer(7)]],
                             uint gid [[thread_position_in_grid]]) {
                uint m = gid / N, n = gid % N;
                uint kp = K / 8;          // uint32 / row（4bit×8）
                uint kg = K / GS;         // group / row
                float acc = 0.0f;
                for (uint k = 0; k < K; ++k) {
                    uint packed = wq[n*kp + (k >> 3)];
                    uint nib = (packed >> (4u * (k & 7u))) & 0xFu;
                    uint g = k / GS;
                    float w = (float)scales[n*kg + g] * (float)nib + (float)biases[n*kg + g];
                    acc += (float)x[m*K + k] * w;
                }
                out[m*N + n] = (half)acc;
            }
            """
            do {
                let lib = try device.makeLibrary(source: src, options: nil)
                _qmmPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "qmm4")!)
            } catch { print("[raw-qmm] compile error: \(error)"); return nil }
        }
        // MLX weight を MTLBuffer 共有（noCopy）。x も同様。out は新規。
        guard let bx = x.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bwq = wq.asMTLBuffer(device: device, noCopy: false),
              let bsc = scales.asType(.float16).asMTLBuffer(device: device, noCopy: false),
              let bbi = biases.asType(.float16).asMTLBuffer(device: device, noCopy: false)
        else { return nil }
        let outBuf = device.makeBuffer(length: M * N * 2, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(_qmmPipeline!)
        enc.setBuffer(bx, offset: 0, index: 0)
        enc.setBuffer(bwq, offset: 0, index: 1)
        enc.setBuffer(bsc, offset: 0, index: 2)
        enc.setBuffer(bbi, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var kk = UInt32(K), nn = UInt32(N), g = UInt32(gs)
        enc.setBytes(&kk, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        enc.setBytes(&g, length: 4, index: 7)
        let total = M * N
        let tgw = min(_qmmPipeline!.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tgw, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        // MTLBuffer → MLXArray（f16, [M,N]）
        let ptr = outBuf.contents().bindMemory(to: Float16.self, capacity: total)
        let arr = Array(UnsafeBufferPointer(start: ptr, count: total))
        return MLXArray(arr, [M, N])
    }

    /// 検証: ランダム W を MLX.quantized で量子化 → MLX.quantizedMatmul vs raw qmm を bit-exact 照合。
    /// - env: QWISP_RUN=raw-qmm-test / QWISP_QMM_K(既定2048) / QWISP_QMM_N(既定2048) / QWISP_QMM_M(既定1)
    public static func runQmmTest() -> String {
        let K = envInt("QWISP_QMM_K", 2048), N = envInt("QWISP_QMM_N", 2048), M = envInt("QWISP_QMM_M", 1)
        let gs = 64, bits = 4
        let x = MLXRandom.normal([M, K]).asType(.float16)
        let w = MLXRandom.normal([N, K]).asType(.float16)
        let (wq, scales, biasesOpt) = MLX.quantized(w, groupSize: gs, bits: bits, mode: .affine)
        guard let biases = biasesOpt else { return "[raw-qmm] ERROR: affine biases nil" }
        MLX.eval([x, wq, scales, biases])
        let ref = MLX.quantizedMatmul(x, wq, scales: scales, biases: biases, transpose: true,
                                      groupSize: gs, bits: bits, mode: .affine)
        MLX.eval([ref])
        guard let got = qmm(x, wq, scales: scales, biases: biases, M: M, K: K, N: N, bits: bits, gs: gs) else {
            return "[raw-qmm] ERROR: kernel 実行失敗"
        }
        MLX.eval([got])
        let d = MLX.max(MLX.abs(got.asType(.float32) - ref.asType(.float32))).item(Float.self)
        let scale = MLX.max(MLX.abs(ref.asType(.float32))).item(Float.self) + 1e-9
        let rel = d / scale
        let ok = rel < 2e-3
        return String(format: "[raw-qmm-test M=%d K=%d N=%d, 4bit affine gs=64] raw-Metal vs MLX quantizedMatmul\n"
            + "  max|Δ|=%.3e  rel=%.3e  %@", M, K, N, d, rel,
            ok ? "OK ✅ bit-exact(MLX weight buffer 共有 + format 一致)" : "MISMATCH ❌(format 要修正)")
    }

    static func envInt(_ k: String, _ d: Int) -> Int {
        guard let v = ProcessInfo.processInfo.environment[k], let i = Int(v) else { return d }
        return i
    }
}
