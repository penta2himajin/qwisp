import Foundation
import MLX
import MLXNN

/// Qwen3NextSparseMoeBlock の Swift 移植（M2b-3 MoE block wrapping）.
/// gate(8bit)→softmax precise→argpartition top-k→normalize→switch_mlp(4bit gather_qmm)→combine
/// + shared_expert(4bit dense MLP)+shared_expert_gate(8bit→sigmoid)。
public struct MoEBlock {
    let topK: Int
    let numExperts: Int
    let normTopk: Bool

    let gate: Proj                 // 8bit → [T, numExperts]
    // switch_mlp 量子化 expert（[E, OUT, IN/8] uint32, 4bit）
    let swGateW: MLXArray, swGateS: MLXArray, swGateB: MLXArray
    let swUpW: MLXArray, swUpS: MLXArray, swUpB: MLXArray
    let swDownW: MLXArray, swDownS: MLXArray, swDownB: MLXArray
    // shared_expert（4bit dense）
    let shGate: Proj, shUp: Proj, shDown: Proj
    let sharedGate: Proj           // 8bit → [T, 1]
    let expertBits: Int

    public init(topK: Int, numExperts: Int, normTopk: Bool, expertBits: Int,
                gate: Proj,
                swGateW: MLXArray, swGateS: MLXArray, swGateB: MLXArray,
                swUpW: MLXArray, swUpS: MLXArray, swUpB: MLXArray,
                swDownW: MLXArray, swDownS: MLXArray, swDownB: MLXArray,
                shGate: Proj, shUp: Proj, shDown: Proj, sharedGate: Proj) {
        self.topK = topK; self.numExperts = numExperts; self.normTopk = normTopk
        self.expertBits = expertBits; self.gate = gate
        self.swGateW = swGateW; self.swGateS = swGateS; self.swGateB = swGateB
        self.swUpW = swUpW; self.swUpS = swUpS; self.swUpB = swUpB
        self.swDownW = swDownW; self.swDownS = swDownS; self.swDownB = swDownB
        self.shGate = shGate; self.shUp = shUp; self.shDown = shDown; self.sharedGate = sharedGate
    }

    private func gatherQmm(_ x: MLXArray, _ w: MLXArray, _ s: MLXArray, _ b: MLXArray,
                           _ inds: MLXArray) -> MLXArray {
        gatherQuantizedMatmul(x, w, scales: s, biases: b, rhsIndices: inds,
                              transpose: true, groupSize: 64, bits: expertBits, mode: .affine,
                              sortedIndices: false)
    }

    /// x: [T, H] → [T, H]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = MLX.softmax(gate.apply(x), axis: -1, precise: true)  // [T, E]
        // top-k（kth=E-K で分割、後半 K 個が上位）。順序非依存（最後に sum）
        let order = MLX.argPartition(gates, kth: numExperts - topK, axis: -1)
        let inds = order[0..., (numExperts - topK)...].asType(.uint32)   // [T, K]
        var scores = MLX.takeAlong(gates, inds.asType(.int32), axis: -1) // [T, K]
        if normTopk {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        // switch_mlp: xe [T,1,1,H]
        let xe = x.expandedDimensions(axes: [-2, -3])
        let g = gatherQmm(xe, swGateW, swGateS, swGateB, inds)
        let u = gatherQmm(xe, swUpW, swUpS, swUpB, inds)
        let h = (g * MLX.sigmoid(g)) * u                                 // silu(g)*u
        let d = gatherQmm(h, swDownW, swDownS, swDownB, inds).squeezed(axis: -2)  // [T,K,H]
        let y = (d * scores.expandedDimensions(axis: -1)).sum(axis: -2)  // [T,H]

        // shared expert（dense swiglu）+ gate
        let sg = shGate.apply(x), su = shUp.apply(x)
        let sharedY = shDown.apply((sg * MLX.sigmoid(sg)) * su)          // [T,H]
        let gateScale = MLX.sigmoid(sharedGate.apply(x))                 // [T,1]
        return y + gateScale * sharedY
    }
}

public enum MoEBlockValidation {
    public static func run(refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        func q8(_ n: String) -> Proj {
            .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 8)
        }
        func q4(_ n: String) -> Proj {
            .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 4)
        }
        guard let x = r["x"], let expY = r["y"] else { return "ERROR: real-moe ref 不足" }
        let blk = MoEBlock(
            topK: 8, numExperts: 256, normTopk: true, expertBits: 4,
            gate: q8("gate"),
            swGateW: r["switch_mlp.gate_proj.weight"]!, swGateS: r["switch_mlp.gate_proj.scales"]!,
            swGateB: r["switch_mlp.gate_proj.biases"]!,
            swUpW: r["switch_mlp.up_proj.weight"]!, swUpS: r["switch_mlp.up_proj.scales"]!,
            swUpB: r["switch_mlp.up_proj.biases"]!,
            swDownW: r["switch_mlp.down_proj.weight"]!, swDownS: r["switch_mlp.down_proj.scales"]!,
            swDownB: r["switch_mlp.down_proj.biases"]!,
            shGate: q4("shared_expert.gate_proj"), shUp: q4("shared_expert.up_proj"),
            shDown: q4("shared_expert.down_proj"), sharedGate: q8("shared_expert_gate"))
        let y = blk(x)
        y.eval()
        let d = MLX.max(MLX.abs(y - expY)).item(Float.self)
            / (MLX.max(MLX.abs(expY)).item(Float.self) + 1e-9)
        let ok = d < 2e-3
        return String(format: "[M2b-3] 実 MoE block(gate8/switch4/shared4): y_rel=%.2e  %@",
                      d, ok ? "OK ✅ 実重み一致" : "MISMATCH ❌")
    }
}
