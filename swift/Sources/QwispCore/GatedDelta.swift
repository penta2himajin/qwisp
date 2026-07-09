import Foundation
import MLX
import MLXNN
import MLXRandom

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

    /// gated_delta_update（use_kernel=False 経路, ops）: beta=sigmoid(b), g=compute_g。
    /// state を渡すと（decode の carry）そこから継続、nil なら零状態から開始。返り値の state は更新後。
    public static func update(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ a: MLXArray,
                              _ b: MLXArray, _ aLog: MLXArray, _ dtBias: MLXArray,
                              state: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let beta = MLX.sigmoid(b)
        let g = computeG(aLog, a, dtBias)
        return ops(q, k, v, g, beta, state ?? MLXArray.zeros([0]))
    }

    // mlx_lm/models/gated_delta.py の _make_gated_delta_kernel(has_mask=False, vectorized=False) を移植.
    // 関数本体のみ（signature は input/output 名から自動生成）。recurrence 全体を kernel 内で回す。
    static let stepKernelSource = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
        """

    static let stepKernel = MLXFast.metalKernel(
        name: "gated_delta_step",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: stepKernelSource)

    /// fused Metal kernel 経路（use_kernel=True 相当）。q,k は repeat しない([B,T,Hk,Dk])。
    public static func updateKernel(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ a: MLXArray,
                                    _ b: MLXArray, _ aLog: MLXArray, _ dtBias: MLXArray,
                                    state: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let beta = MLX.sigmoid(b)
        let g = computeG(aLog, a, dtBias)                 // [B,T,Hv] float32
        let B = q.dim(0), T = q.dim(1), Hk = q.dim(2), Dk = q.dim(3)
        let Hv = v.dim(2), Dv = v.dim(3)
        let st = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
        let inT = q.dtype
        let out = stepKernel(
            [q, k, v, g, beta, st, T],
            template: [("InT", inT), ("StT", DType.float32),
                       ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
            grid: (32, Dv, B * Hv),
            threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], [B, Hv, Dv, Dk]],
            outputDTypes: [inT, .float32])
        return (out[0], out[1])
    }
}

extension GatedDelta {
    /// updateKernel(T=2) と 逐次 T=1×2（state carry）の bit 一致を検証。
    /// spec の batched verify vs sequential accept の drift 真因を切り分ける。
    public static func tConsistencyTest() -> String {
        let B = 1, T = 2, Hk = 16, Hv = 32, Dk = 128, Dv = 128
        // 決定論的な擬似乱数入力（mlx の random は seed 固定で再現可能）
        MLXRandom.seed(0)
        let q = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let k = MLXRandom.normal([B, T, Hk, Dk]).asType(.float16)
        let v = MLXRandom.normal([B, T, Hv, Dv]).asType(.float16)
        let a = MLXRandom.normal([B, T, Hv])
        let b = MLXRandom.normal([B, T, Hv])
        let aLog = MLXRandom.normal([Hv])
        let dtBias = MLXRandom.normal([Hv])

        // batched: 1 回の T=2 呼び出し
        let (yB, sB) = updateKernel(q, k, v, a, b, aLog, dtBias, state: nil)
        yB.eval(); sB.eval()

        // sequential: T=1 を 2 回、state を carry
        let (y0, s0) = updateKernel(q[0..., 0 ..< 1], k[0..., 0 ..< 1], v[0..., 0 ..< 1],
                                    a[0..., 0 ..< 1], b[0..., 0 ..< 1], aLog, dtBias, state: nil)
        let (y1, s1) = updateKernel(q[0..., 1 ..< 2], k[0..., 1 ..< 2], v[0..., 1 ..< 2],
                                    a[0..., 1 ..< 2], b[0..., 1 ..< 2], aLog, dtBias, state: s0)
        let yS = MLX.concatenated([y0, y1], axis: 1)
        yS.eval(); s1.eval()

        func rel(_ x: MLXArray, _ y: MLXArray) -> Float {
            MLX.max(MLX.abs(x.asType(.float32) - y.asType(.float32))).item(Float.self)
                / (MLX.max(MLX.abs(y.asType(.float32))).item(Float.self) + 1e-9)
        }
        // 位置別の out 差分（pos0 は両者 T=1 で必ず一致するはず、pos1 が問題位置）
        let relY0 = rel(yB[0..., 0 ..< 1], y0)
        let relY1 = rel(yB[0..., 1 ..< 2], y1)
        let relState = rel(sB, s1)
        let ok = relY1 < 1e-3 && relState < 1e-3
        return String(format: """
            [GDN-TTEST] updateKernel T=2 vs 逐次 T=1×2:
              out pos0 rel=%.3e  out pos1 rel=%.3e  final-state rel=%.3e  -> %@
            """, relY0, relY1, relState,
            ok ? "一致 ✅（kernel は sequential-consistent）" : "乖離 ❌（kernel の T>1 経路が逐次と非一致＝drift 真因）")
    }
}

