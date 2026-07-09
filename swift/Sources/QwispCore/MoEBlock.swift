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
    let expertGroupSize: Int

    public init(topK: Int, numExperts: Int, normTopk: Bool, expertBits: Int,
                expertGroupSize: Int = 64,
                gate: Proj,
                swGateW: MLXArray, swGateS: MLXArray, swGateB: MLXArray,
                swUpW: MLXArray, swUpS: MLXArray, swUpB: MLXArray,
                swDownW: MLXArray, swDownS: MLXArray, swDownB: MLXArray,
                shGate: Proj, shUp: Proj, shDown: Proj, sharedGate: Proj) {
        self.topK = topK; self.numExperts = numExperts; self.normTopk = normTopk
        self.expertBits = expertBits; self.expertGroupSize = expertGroupSize; self.gate = gate
        self.swGateW = swGateW; self.swGateS = swGateS; self.swGateB = swGateB
        self.swUpW = swUpW; self.swUpS = swUpS; self.swUpB = swUpB
        self.swDownW = swDownW; self.swDownS = swDownS; self.swDownB = swDownB
        self.shGate = shGate; self.shUp = shUp; self.shDown = shDown; self.sharedGate = sharedGate
    }

    private func gatherQmm(_ x: MLXArray, _ w: MLXArray, _ s: MLXArray, _ b: MLXArray,
                           _ inds: MLXArray) -> MLXArray {
        gatherQuantizedMatmul(x, w, scales: s, biases: b, rhsIndices: inds,
                              transpose: true, groupSize: expertGroupSize, bits: expertBits, mode: .affine,
                              sortedIndices: false)
    }

    /// x: [T, H] → [T, H]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inds: MLXArray, scores: MLXArray
        if SeedlessMetalForward.metalRoute {
            // ★ task#8 検証: routing(選択+normalize)を Metal(route_top8)で。gate logits は MLX(qmm8 と bit-exact)。
            //   argPartition→route_top8 だけを差し替え＝選択法の near-tie 差を実 decode で検証。T 行をループ。
            let logits = gate.apply(x)                                    // [T, E] f16（softmax 前）
            let T = x.dim(0)
            var iRows: [MLXArray] = [], sRows: [MLXArray] = []
            for t in 0 ..< T {
                guard let (ri, rs) = SeedlessMetalForward.routeTop8(logits[t].reshaped([numExperts]), N: numExperts, K: topK) else {
                    return callAsFunctionMLX(x)   // 失敗時は MLX 経路にフォールバック
                }
                iRows.append(ri.reshaped([1, topK])); sRows.append(rs.reshaped([1, topK]))
            }
            inds = MLX.concatenated(iRows, axis: 0).asType(.uint32)        // [T, K]
            scores = MLX.concatenated(sRows, axis: 0).asType(.float16)     // [T, K] f16（route_top8 で normalize 済）
            return moeExperts(x, inds: inds, scores: scores)
        }
        return callAsFunctionMLX(x)
    }

    /// MLX routing 経路（既存）。
    private func callAsFunctionMLX(_ x: MLXArray) -> MLXArray {
        let gates = MLX.softmax(gate.apply(x), axis: -1, precise: true)  // [T, E]
        // top-k（kth=E-K で分割、後半 K 個が上位）。順序非依存（最後に sum）
        let order = MLX.argPartition(gates, kth: numExperts - topK, axis: -1)
        let inds = order[0..., (numExperts - topK)...].asType(.uint32)   // [T, K]
        var scores = MLX.takeAlong(gates, inds.asType(.int32), axis: -1) // [T, K]
        if normTopk {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        return moeExperts(x, inds: inds, scores: scores)
    }

    /// expert 計算（gather swiglu + shared）。routing(inds/scores)は呼び元で決定。
    private func moeExperts(_ x: MLXArray, inds: MLXArray, scores: MLXArray) -> MLXArray {

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

extension MoEBlock {
    /// Build a MoEBlock from a quantized weight dict (8-bit gates, 4-bit experts).
    static func from(_ r: [String: MLXArray]) -> MoEBlock {
        func q8(_ n: String) -> Proj {
            .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 8)
        }
        func q4(_ n: String) -> Proj {
            .quantized(r["\(n).weight"]!, r["\(n).scales"]!, r["\(n).biases"]!, 4)
        }
        return MoEBlock(
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
    }
}
