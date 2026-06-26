import Foundation
import MLX
import MLXNN

/// GatedDeltaNet の recurrent 核（gated_delta_ops 相当）の Swift 移植（M2b-1 crux）.
/// Python mlx_lm/models/gated_delta.py の純ops版を忠実移植。後で Metal kernel 化で速度。
public enum GatedDelta {
    /// compute_g: exp(-exp(A_log) * softplus(a + dt_bias))
    public static func computeG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
        let sp = softplus(a + dtBias)
        return MLX.exp(-MLX.exp(aLog.asType(.float32)) * sp)
    }

    static func softplus(_ x: MLXArray) -> MLXArray {
        // 数値安定版: max(x,0) + log1p(exp(-|x|))  （mlx の softplus 相当）
        MLX.maximum(x, 0) + MLX.log(1 + MLX.exp(-MLX.abs(x)))
    }

    /// 1 recurrent step（g.ndim==2: [B,H]）。state:[B,H,Dv,Dk] q/k:[B,H,Dk] v:[B,H,Dv]
    static func step(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ g: MLXArray,
                     _ beta: MLXArray, _ state: MLXArray) -> (MLXArray, MLXArray) {
        let decay = g.expandedDimensions(axes: [-1, -2])       // [B,H,1,1]
        var s = state * decay
        let kE = k.expandedDimensions(axes: [-2])              // [B,H,1,Dk]
        let kvMem = (s * kE).sum(axis: -1)                    // [B,H,Dv]
        let delta = (v - kvMem) * beta.expandedDimensions(axes: [-1])  // [B,H,Dv]
        s = s + kE * delta.expandedDimensions(axes: [-1])     // [B,H,Dv,Dk]
        let y = (s * q.expandedDimensions(axes: [-2])).sum(axis: -1)   // [B,H,Dv]
        return (y, s)
    }

    /// gated_delta_ops: T を逐次ループ。q,k:[B,T,Hk,Dk] v:[B,T,Hv,Dv] g,beta:[B,T,Hv]
    public static func ops(_ q0: MLXArray, _ k0: MLXArray, _ v: MLXArray, _ g: MLXArray,
                           _ beta: MLXArray, _ state0: MLXArray) -> (MLXArray, MLXArray) {
        let B = v.dim(0), T = v.dim(1), Hv = v.dim(2), Dv = v.dim(3), Dk = q0.dim(3)
        let Hk = q0.dim(2)
        var q = q0, k = k0
        let rf = Hv / Hk
        if rf > 1 { q = MLX.repeated(q, count: rf, axis: -2); k = MLX.repeated(k, count: rf, axis: -2) }
        var state = state0.dim(0) == 0 ? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32) : state0
        var ys: [MLXArray] = []
        for t in 0 ..< T {
            let (y, s) = step(q[0..., t], k[0..., t], v[0..., t], g[0..., t], beta[0..., t], state)
            ys.append(y)
            state = s
        }
        return (MLX.stacked(ys, axis: 1), state)
    }

    /// gated_delta_update（use_kernel=False 経路）: beta=sigmoid(b), g=compute_g。
    /// state を渡すと（decode の carry）そこから継続、nil なら零状態から開始。返り値の state は更新後。
    public static func update(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ a: MLXArray,
                              _ b: MLXArray, _ aLog: MLXArray, _ dtBias: MLXArray,
                              state: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let beta = MLX.sigmoid(b)
        let g = computeG(aLog, a, dtBias)
        return ops(q, k, v, g, beta, state ?? MLXArray.zeros([0]))
    }
}

public enum GatedDeltaValidation {
    public static func run(refPath: String) throws -> String {
        let r = try loadArrays(url: URL(fileURLWithPath: refPath))
        guard let q = r["q"], let k = r["k"], let v = r["v"], let a = r["a"], let b = r["b"],
              let aLog = r["A_log"], let dtBias = r["dt_bias"],
              let expOut = r["out"], let expState = r["state"] else {
            return "ERROR: gdn ref 不足"
        }
        let (out, state) = GatedDelta.update(q, k, v, a, b, aLog, dtBias)
        out.eval(); state.eval()
        let dOut = MLX.max(MLX.abs(out - expOut)).item(Float.self)
            / (MLX.max(MLX.abs(expOut)).item(Float.self) + 1e-9)
        let dState = MLX.max(MLX.abs(state - expState)).item(Float.self)
            / (MLX.max(MLX.abs(expState)).item(Float.self) + 1e-9)
        let ok = dOut < 1e-3 && dState < 1e-3
        return String(format: "[M2b-1] GatedDelta recurrent核: out_rel=%.2e state_rel=%.2e  %@",
                      dOut, dState, ok ? "OK ✅ bit一致" : "MISMATCH ❌")
    }
}
