import Foundation
import MLX
import Metal

/// 持続 arena に常駐する quantized expert を1つの buffer として保持し、
/// gather_qmm×3 + swiglu で switch_mlp forward を行う層（M2a）。
/// 重みは自前 MTLBuffer 所有（in-place 更新可, M3 で実証）。ここでは forward の正しさを固める。
public final class PersistentMoELayer {
    struct Proj { let w: MLXArray; let s: MLXArray; let b: MLXArray; let wbuf: MTLBuffer }
    let gate: Proj, up: Proj, down: Proj
    let bits: Int, gs: Int

    /// loaded: {proj.part: MLXArray}（[E,...] 量子化済）を所有バッファ化して保持。
    public init?(device: MTLDevice, loaded: [String: MLXArray], bits: Int = 2, gs: Int = 64) {
        self.bits = bits; self.gs = gs
        func mk(_ p: String) -> Proj? {
            guard let w = loaded["\(p).weight"], let s = loaded["\(p).scales"],
                  let b = loaded["\(p).biases"],
                  let wbuf = w.asMTLBuffer(device: device, noCopy: false),
                  let sbuf = s.asMTLBuffer(device: device, noCopy: false),
                  let bbuf = b.asMTLBuffer(device: device, noCopy: false) else { return nil }
            let wa = MLXArray(rawPointer: wbuf.contents(), Array(w.shape), dtype: .uint32) { _ = wbuf }
            let sa = MLXArray(rawPointer: sbuf.contents(), Array(s.shape), dtype: .float16) { _ = sbuf }
            let ba = MLXArray(rawPointer: bbuf.contents(), Array(b.shape), dtype: .float16) { _ = bbuf }
            return Proj(w: wa, s: sa, b: ba, wbuf: wbuf)
        }
        guard let g = mk("gate_proj"), let u = mk("up_proj"), let d = mk("down_proj") else { return nil }
        gate = g; up = u; down = d
    }

    private func qmm(_ x: MLXArray, _ p: Proj, _ inds: MLXArray) -> MLXArray {
        gatherQuantizedMatmul(x, p.w, scales: p.s, biases: p.b, rhsIndices: inds,
                              transpose: true, groupSize: gs, bits: bits, mode: .affine,
                              sortedIndices: false)
    }

    /// switch_mlp forward: x:[T,IN], inds:[T,K] → [T,K,H]
    public func callAsFunction(_ x: MLXArray, _ inds: MLXArray) -> MLXArray {
        let xe = x.expandedDimensions(axes: [-2, -3])      // [T,1,1,IN]
        let g = qmm(xe, gate, inds)
        let u = qmm(xe, up, inds)
        let h = (g * MLX.sigmoid(g)) * u                   // swiglu = silu(gate)*up
        let d = qmm(h, down, inds)                         // [T,K,1,H]
        return d.squeezed(axis: -2)                        // [T,K,H]
    }
}

public enum MoELayerValidation {
    public static func run(refPath: String) throws -> String {
        guard let device = MTLCreateSystemDefaultDevice() else { return "ERROR: no Metal device" }
        let ref = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let x = ref["x"], let inds32 = ref["inds"], let expected = ref["expected_moe"] else {
            return "ERROR: ref に expected_moe 等が無い"
        }
        guard let layer = PersistentMoELayer(device: device, loaded: ref) else {
            return "ERROR: PersistentMoELayer 構築失敗（重み欠落？）"
        }
        let inds = inds32.asType(.uint32)
        let y = layer(x, inds)
        y.eval()
        let diff = MLX.max(MLX.abs(y - expected)).item(Float.self)
        let rel = diff / (MLX.max(MLX.abs(expected)).item(Float.self) + 1e-9)
        let ok = rel < 1e-3

        // 1層 forward の概算速度（持続 arena 経由）
        let reps = 200
        for _ in 0..<20 { layer(x, inds).eval() }
        let t0 = DispatchTime.now()
        for _ in 0..<reps { layer(x, inds).eval() }
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6 / Double(reps)

        return String(
            format: "[M2a] 持続 arena switch_mlp forward: out=%@ max|Δ|=%.3e rel=%.3e  %@  (%.4f ms/層)",
            "\(y.shape)", diff, rel, ok ? "OK ✅ bit一致" : "MISMATCH ❌", ms)
    }
}
