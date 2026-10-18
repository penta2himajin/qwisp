import Foundation
import MLX
import MLXRandom
import Metal

/// ①③ Step 3 real-model gate (notes/17 §検証 runner): raw SeedlessMTPHead vs MLX MTPHead
/// on the actual mtp.safetensors weights. QWISP_RUN=mtp-raw-validate.
///
/// Protocol: per probe i, draw (hPrev_i, tok_i); MLX reference drafts WITH cache
/// (appends the pair), raw drafts READ-ONLY then feedPairs the same pair — so both
/// sides walk the same growing KV positions. Gate: argmax match on every probe.
/// This is the only gate that exercises the real expert layout (4bit gs=32 vs the
/// synthetic tests' gs=64) and the real vocab-sized lm_head.
public enum SeedlessMTPValidate {
    /// Build the production WeightsSpec from mtp.safetensors + shared embed/lm_head
    /// (store must have residentNonExperts or residentAll done). Shared by the
    /// validate gate below and the run() draft-head wiring (①③ Step 4).
    public static func loadSpec(modelDir: String, store: WeightStore, maxSeqLen: Int = 256) throws
        -> SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec {
        let H = 2048
        // Grafted MTP head (mlx-lm/oMLX layout: `language_model.mtp.*` in model-mtp.safetensors,
        // 8-bit dense parts, stacked 4-bit gs64 experts, norms already +1) when there is no
        // MTPLX `mtp.safetensors`.
        let mtplx = URL(fileURLWithPath: modelDir).appendingPathComponent("mtp.safetensors")
        if !FileManager.default.fileExists(atPath: mtplx.path) {
            return try loadGraftedSpec(modelDir: modelDir, store: store, maxSeqLen: maxSeqLen)
        }
        // Raw head from the same arrays (mirror MTPHead.init recovery exactly)
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("mtp.safetensors")
        let w = try loadArrays(url: url)
        func g(_ k: String) -> MLXArray { w["mtp.\(k)"]! }
        func gn(_ k: String) -> MLXArray { w["mtp.\(k)"]! + MLXArray(Float(1)).asType(w["mtp.\(k)"]!.dtype) }
        func stackE(_ proj: String, _ part: String) -> MLXArray {
            MLX.stacked((0 ..< 256).map { w["mtp.layers.0.mlp.experts.\($0).\(proj).\(part)"]! }, axis: 0)
        }
        let embW = store.req("language_model.model.embed_tokens.weight")
        let embS = store.req("language_model.model.embed_tokens.scales")
        let embB = store.req("language_model.model.embed_tokens.biases")
        let lmW = store.req("language_model.lm_head.weight")
        let lmS = store.req("language_model.lm_head.scales")
        let lmB = store.req("language_model.lm_head.biases")
        let V = lmW.dim(0)

        return SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec(
            H: H, V: V, E: 256, I: 512, Ktop: 8,
            numHeads: 16, numKV: 2, headDim: 256, ropeDim: 64,
            ropeBase: 1e7, eps: 1e-6, maxSeqLen: maxSeqLen,
            expertGroupSize: 32,
            denseQ8: Tell.envInt("QWISP_MTP_Q8", 1) != 0,
            fc: g("fc.weight"),
            qW: g("layers.0.self_attn.q_proj.weight"),
            kW: g("layers.0.self_attn.k_proj.weight"),
            vW: g("layers.0.self_attn.v_proj.weight"),
            oW: g("layers.0.self_attn.o_proj.weight"),
            routerGate: g("layers.0.mlp.gate.weight"),
            shGate: g("layers.0.mlp.shared_expert.gate_proj.weight"),
            shUp: g("layers.0.mlp.shared_expert.up_proj.weight"),
            shDown: g("layers.0.mlp.shared_expert.down_proj.weight"),
            sharedGate: g("layers.0.mlp.shared_expert_gate.weight"),
            preEmb: g("pre_fc_norm_embedding.weight"),
            preHid: g("pre_fc_norm_hidden.weight"),
            inputLN: gn("layers.0.input_layernorm.weight"),
            postLN: gn("layers.0.post_attention_layernorm.weight"),
            qNorm: gn("layers.0.self_attn.q_norm.weight"),
            kNorm: gn("layers.0.self_attn.k_norm.weight"),
            finalNorm: g("norm.weight"),
            embedWq: embW, embedSc: embS, embedBi: embB,
            swGWq: stackE("gate_proj", "weight"), swGSc: stackE("gate_proj", "scales"),
            swGBi: stackE("gate_proj", "biases"),
            swUWq: stackE("up_proj", "weight"), swUSc: stackE("up_proj", "scales"),
            swUBi: stackE("up_proj", "biases"),
            swDWq: stackE("down_proj", "weight"), swDSc: stackE("down_proj", "scales"),
            swDBi: stackE("down_proj", "biases"),
            lmWq: lmW, lmSc: lmS, lmBi: lmB)
    }

    enum GraftedMTPError: Error { case missing(String) }

    /// MTP head stored the way mlx-lm / oMLX graft it: `language_model.mtp.*` tensors in
    /// `model-mtp.safetensors`. Dense projections (fc, attention, router, shared expert) are
    /// affine-quantized (8-bit gs64) and are dequantized to F16 here, because the raw head binds
    /// them as plain F16 matrices. Routed experts are stacked [256, …] 4-bit gs64 and are passed
    /// through (expertGroupSize 64). Norm weights are stored with the +1 offset already applied.
    static func loadGraftedSpec(modelDir: String, store: WeightStore, maxSeqLen: Int) throws
        -> SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec {
        let url = URL(fileURLWithPath: modelDir).appendingPathComponent("model-mtp.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GraftedMTPError.missing("no mtp.safetensors or model-mtp.safetensors in \(modelDir)")
        }
        let w = try loadArrays(url: url)
        func t(_ k: String) throws -> MLXArray {
            guard let a = w["language_model.mtp.\(k)"] else { throw GraftedMTPError.missing(k) }
            return WeightStore.metalF16(a)
        }
        // Dense matrix: dequantize when quantized (bits inferred from packed width), else F16.
        func dense(_ k: String) throws -> MLXArray {
            let wq = try t("\(k).weight")
            guard w["language_model.mtp.\(k).scales"] != nil else { return wq.asType(.float16) }
            let sc = try t("\(k).scales"), bi = try t("\(k).biases")
            let gs = 64
            let bits = wq.dim(-1) * 32 / (sc.dim(-1) * gs)
            let d = MLX.dequantized(wq, scales: sc, biases: bi, groupSize: gs, bits: bits, mode: .affine)
                .asType(.float16)
            d.eval()
            return d
        }
        func ex(_ proj: String, _ part: String) throws -> MLXArray { try t("layers.0.mlp.switch_mlp.\(proj).\(part)") }
        let lmW = store.req("language_model.lm_head.weight")
        return SeedlessFusedVerify.SeedlessMTPHead.WeightsSpec(
            H: 2048, V: lmW.dim(0), E: 256, I: 512, Ktop: 8,
            numHeads: 16, numKV: 2, headDim: 256, ropeDim: 64,
            ropeBase: 1e7, eps: 1e-6, maxSeqLen: maxSeqLen,
            expertGroupSize: 64,
            denseQ8: Tell.envInt("QWISP_MTP_Q8", 1) != 0,
            fc: try dense("fc"),
            qW: try dense("layers.0.self_attn.q_proj"),
            kW: try dense("layers.0.self_attn.k_proj"),
            vW: try dense("layers.0.self_attn.v_proj"),
            oW: try dense("layers.0.self_attn.o_proj"),
            routerGate: try dense("layers.0.mlp.gate"),
            shGate: try dense("layers.0.mlp.shared_expert.gate_proj"),
            shUp: try dense("layers.0.mlp.shared_expert.up_proj"),
            shDown: try dense("layers.0.mlp.shared_expert.down_proj"),
            sharedGate: try dense("layers.0.mlp.shared_expert_gate"),
            preEmb: try t("pre_fc_norm_embedding.weight"),
            preHid: try t("pre_fc_norm_hidden.weight"),
            inputLN: try t("layers.0.input_layernorm.weight"),
            postLN: try t("layers.0.post_attention_layernorm.weight"),
            qNorm: try t("layers.0.self_attn.q_norm.weight"),
            kNorm: try t("layers.0.self_attn.k_norm.weight"),
            finalNorm: try t("norm.weight"),
            embedWq: store.req("language_model.model.embed_tokens.weight"),
            embedSc: store.req("language_model.model.embed_tokens.scales"),
            embedBi: store.req("language_model.model.embed_tokens.biases"),
            swGWq: try ex("gate_proj", "weight"), swGSc: try ex("gate_proj", "scales"), swGBi: try ex("gate_proj", "biases"),
            swUWq: try ex("up_proj", "weight"), swUSc: try ex("up_proj", "scales"), swUBi: try ex("up_proj", "biases"),
            swDWq: try ex("down_proj", "weight"), swDSc: try ex("down_proj", "scales"), swDBi: try ex("down_proj", "biases"),
            lmWq: lmW, lmSc: store.req("language_model.lm_head.scales"), lmBi: store.req("language_model.lm_head.biases"))
    }

    public static func run(modelDir: String) throws -> String {
        let H = 2048
        guard let (device, _) = SeedlessMetalForward.ensure() else { return "[mtp-raw] no device" }

        // MLX reference = production MTPHead (loads + recovers norms itself)
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let mlxHead = try MTPHead(modelDir: modelDir, store: store)

        let spec = try loadSpec(modelDir: modelDir, store: store)
        let V = spec.V
        guard let raw = SeedlessFusedVerify.SeedlessMTPHead(spec: spec) else { return "[mtp-raw] init nil" }

        // Probes: both sides walk the same growing KV history.
        MLXRandom.seed(7)
        let kv = KVCache()
        let nProbe = 24
        var match = 0
        var firstMiss = ""
        for i in 0 ..< nProbe {
            let hPrev = (MLXRandom.normal([1, 1, H]) * 0.5).asType(.float16)
            let tok = Int32(MLXRandom.randInt(0 ..< V, [1]).item(Int32.self))
            hPrev.eval()
            let dl = mlxHead(hPrev, MLXArray([tok], [1, 1]), cache: kv)   // appends pair to kv
            let refD = MLX.argMax(dl[0, 0], axis: -1).item(Int.self)

            guard let hBuf = SeedlessMetalForward.mtlBuf(hPrev.reshaped([1, H]).asType(.float16), device)
            else { return "[mtp-raw] hBuf nil probe \(i)" }
            guard let rawD = raw.draftArgmax(hPrevBuf: hBuf, hPrevRow: 0, tok: tok)
            else { return "[mtp-raw] draftArgmax nil probe \(i)" }
            _ = raw.feedPairs(hBuf: hBuf, rowRange: 0 ..< 1, toks: [tok])  // commit same pair

            if rawD == refD { match += 1 }
            else if firstMiss.isEmpty { firstMiss = " firstMiss@\(i): raw=\(rawD) ref=\(refD)" }
        }
        let ok = match == nProbe && raw.len == nProbe
        return "[mtp-raw] real-weight argmax \(match)/\(nProbe) len=\(raw.len)/\(nProbe) "
            + (ok ? "OK ✅ MLX MTPHead 一致" : "MISMATCH ❌\(firstMiss)")
    }
}
