import Foundation
import MLX
import MLXFast
import Metal

/// Reusable raw inference engine for the resident tier (C=256 semantics).
/// Full expert tensors from the store, no arena/guard/cert/VSEQ.
/// Batched == sequential is bit-exact by construction (proven by raw-smoke U2a).
///
/// Factors out model-building from RawSmokeRunner so that Tell
/// and future raw-engine runners can share a single load path.
public struct SeedlessEngine {
    public static let eps: Float         = 1e-6
    public static let H: Int             = 2048
    public static let numLayers: Int     = 40
    static let fullAttnInterval: Int     = 4

    public let layers: [SeedlessVerifyForward.LayerSpec]
    public let moeI: Int

    // ── shared weight handles (extracted once at build time) ──────────────
    let embedW, embedS, embedB: MLXArray   // 4-bit quantised embedding
    let fnW: MLXArray                       // final rmsNorm weight [H]
    let lmW, lmS, lmB: MLXArray            // lm_head weights
    public let vocab: Int

    // ── helpers ───────────────────────────────────────────────────────────

    public static func isLinear(_ i: Int) -> Bool { (i + 1) % fullAttnInterval != 0 }

    // ── layer-spec construction ───────────────────────────────────────────

    public static func buildLayerSpec(_ i: Int, store: WeightStore, moeI: Int) -> SeedlessVerifyForward.LayerSpec {
        let p  = "language_model.model.layers.\(i)"
        let mp = "\(p).mlp"
        let lin = isLinear(i)

        let inputLN = store.req("\(p).input_layernorm.weight")
        let postLN  = store.req("\(p).post_attention_layernorm.weight")

        var gdnW:  SeedlessVerifyForward.GDNLayerW? = nil
        var attnW: SeedlessVerifyForward.AttnLayerW? = nil

        if lin {
            let la = "\(p).linear_attn"
            gdnW = SeedlessVerifyForward.GDNLayerW(
                qkvWq: store.req("\(la).in_proj_qkv.weight"),
                qkvSc: store.req("\(la).in_proj_qkv.scales"),
                qkvBi: store.req("\(la).in_proj_qkv.biases"),
                zWq:   store.req("\(la).in_proj_z.weight"),
                zSc:   store.req("\(la).in_proj_z.scales"),
                zBi:   store.req("\(la).in_proj_z.biases"),
                bWq:   store.req("\(la).in_proj_b.weight"),
                bSc:   store.req("\(la).in_proj_b.scales"),
                bBi:   store.req("\(la).in_proj_b.biases"),
                aWq:   store.req("\(la).in_proj_a.weight"),
                aSc:   store.req("\(la).in_proj_a.scales"),
                aBi:   store.req("\(la).in_proj_a.biases"),
                outWq: store.req("\(la).out_proj.weight"),
                outSc: store.req("\(la).out_proj.scales"),
                outBi: store.req("\(la).out_proj.biases"),
                conv1dW:    store.req("\(la).conv1d.weight"),
                normWeight: store.req("\(la).norm.weight"),
                aLog:   store.req("\(la).A_log"),
                dtBias: store.req("\(la).dt_bias"))
        } else {
            let sa = "\(p).self_attn"
            attnW = SeedlessVerifyForward.AttnLayerW(
                qWq: store.req("\(sa).q_proj.weight"),
                qSc: store.req("\(sa).q_proj.scales"),
                qBi: store.req("\(sa).q_proj.biases"),
                kWq: store.req("\(sa).k_proj.weight"),
                kSc: store.req("\(sa).k_proj.scales"),
                kBi: store.req("\(sa).k_proj.biases"),
                vWq: store.req("\(sa).v_proj.weight"),
                vSc: store.req("\(sa).v_proj.scales"),
                vBi: store.req("\(sa).v_proj.biases"),
                oWq: store.req("\(sa).o_proj.weight"),
                oSc: store.req("\(sa).o_proj.scales"),
                oBi: store.req("\(sa).o_proj.biases"),
                qNorm: store.req("\(sa).q_norm.weight"),
                kNorm: store.req("\(sa).k_norm.weight"))
        }

        // Shared gate is [1, H] 8-bit; qmm8 requires N%8==0.
        // Pad to [8, ...] by repeating the single row — moeBlockRows uses col 0 only.
        let sgW0 = store.req("\(mp).shared_expert_gate.weight")
        let sgS0 = store.req("\(mp).shared_expert_gate.scales")
        let sgB0 = store.req("\(mp).shared_expert_gate.biases")
        let sgWPad = MLX.concatenated(Array(repeating: sgW0, count: 8), axis: 0)
        let sgSPad = MLX.concatenated(Array(repeating: sgS0, count: 8), axis: 0)
        let sgBPad = MLX.concatenated(Array(repeating: sgB0, count: 8), axis: 0)
        MLX.eval([sgWPad, sgSPad, sgBPad])

        let moeW = SeedlessVerifyForward.MoEBlockW(
            gateWq: store.req("\(mp).gate.weight"),
            gateSc: store.req("\(mp).gate.scales"),
            gateBi: store.req("\(mp).gate.biases"),
            swGWq:  store.req("\(mp).switch_mlp.gate_proj.weight"),
            swGSc:  store.req("\(mp).switch_mlp.gate_proj.scales"),
            swGBi:  store.req("\(mp).switch_mlp.gate_proj.biases"),
            swUWq:  store.req("\(mp).switch_mlp.up_proj.weight"),
            swUSc:  store.req("\(mp).switch_mlp.up_proj.scales"),
            swUBi:  store.req("\(mp).switch_mlp.up_proj.biases"),
            swDWq:  store.req("\(mp).switch_mlp.down_proj.weight"),
            swDSc:  store.req("\(mp).switch_mlp.down_proj.scales"),
            swDBi:  store.req("\(mp).switch_mlp.down_proj.biases"),
            shGWq:  store.req("\(mp).shared_expert.gate_proj.weight"),
            shGSc:  store.req("\(mp).shared_expert.gate_proj.scales"),
            shGBi:  store.req("\(mp).shared_expert.gate_proj.biases"),
            shUWq:  store.req("\(mp).shared_expert.up_proj.weight"),
            shUSc:  store.req("\(mp).shared_expert.up_proj.scales"),
            shUBi:  store.req("\(mp).shared_expert.up_proj.biases"),
            shDWq:  store.req("\(mp).shared_expert.down_proj.weight"),
            shDSc:  store.req("\(mp).shared_expert.down_proj.scales"),
            shDBi:  store.req("\(mp).shared_expert.down_proj.biases"),
            sharedGateWq: sgWPad,
            sharedGateSc: sgSPad,
            sharedGateBi: sgBPad)

        return SeedlessVerifyForward.LayerSpec(
            isLinear: lin, inputLN: inputLN, postLN: postLN,
            gdn: gdnW, attn: attnW, moe: moeW, moeE: 256, moeI: moeI)
    }

    // ── public factory ────────────────────────────────────────────────────

    /// Build engine from a loaded WeightStore (caller must have called residentAll()).
    public static func build(store: WeightStore) -> SeedlessEngine {
        let gateW0 = store.req("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight")
        let moeI   = gateW0.shape[1]
        let layers = (0 ..< numLayers).map { i in buildLayerSpec(i, store: store, moeI: moeI) }
        return SeedlessEngine(
            layers:  layers,
            moeI:    moeI,
            embedW:  store.req("language_model.model.embed_tokens.weight"),
            embedS:  store.req("language_model.model.embed_tokens.scales"),
            embedB:  store.req("language_model.model.embed_tokens.biases"),
            fnW:     store.req("language_model.model.norm.weight"),
            lmW:     store.req("language_model.lm_head.weight"),
            lmS:     store.req("language_model.lm_head.scales"),
            lmB:     store.req("language_model.lm_head.biases"),
            vocab:   store.req("language_model.lm_head.weight").dim(0))
    }

    // ── core operations ───────────────────────────────────────────────────

    /// Fresh cold caches for all 40 layers.
    public func freshCaches() -> [SeedlessVerifyForward.LayerCaches] {
        (0 ..< Self.numLayers).map { i in
            if Self.isLinear(i) {
                return SeedlessVerifyForward.LayerCaches(
                    convState: MLX.zeros([3, 8192],          dtype: .float16),
                    recState:  MLX.zeros([1, 32, 128, 128],  dtype: .float32))
            } else {
                return SeedlessVerifyForward.LayerCaches(
                    kCache: MLX.zeros([2, 0, 256], dtype: .float16),
                    vCache: MLX.zeros([2, 0, 256], dtype: .float16))
            }
        }
    }

    /// Embed token ids (array of Int32) -> [M, H] float16 activations.
    public func embed(tokens: [Int32]) -> MLXArray {
        let ids = MLXArray(tokens)   // [M]
        let emb = ModelHead.embed(ids: ids, weight: embedW, scales: embedS, biases: embedB, bits: 4)
        return emb.reshaped([tokens.count, Self.H])
    }

    /// verifyForwardRows(x, ...) + final rmsNorm. Returns normed hidden [M, H] or nil.
    /// Caches are mutated in place (advancing the KV / recurrent state).
    public func forwardRows(_ x: MLXArray, caches: [SeedlessVerifyForward.LayerCaches], M: Int) -> MLXArray? {
        let mr = ProcessInfo.processInfo.environment["QWISP_RAW_METAL_ROUTE"] == "1"
        guard let h = SeedlessVerifyForward.verifyForwardRows(x, layers: layers, caches: caches, M: M, metalRoute: mr)
        else { return nil }
        return SeedlessMetalForward.rmsNormRows(h, fnW, M: M, eps: Self.eps, D: Self.H)
    }

    /// qmmTiled lm_head. Input normed [M, H]; returns logits [M, vocab] or nil.
    public func logits(_ normed: MLXArray, M: Int) -> MLXArray? {
        SeedlessMetalForward.qmmTiled(normed, lmW, scales: lmS, biases: lmB, M: M, K: Self.H, N: vocab)
    }

    // ── fused (single-CB) forward path — P3 speed plumbing ────────────────

    /// streaming fused engine: per-layer C-slot LRU arena + strict segmented forward。
    /// caller は store.residentNonExperts() を済ませておくこと(switch_mlp は mmap-lazy のまま)。
    /// Returns (forward, fnBuf, providers). providers は bolt calib 用に expose。
    public func makeFusedStreaming(modelDir: String, maxM: Int, maxSeqLen: Int, C: Int,
                                   existingProviders: [ArenaExpertProvider]? = nil)
        -> (SeedlessFusedVerify.SeedlessFusedForward, MTLBuffer, [ArenaExpertProvider])? {
        guard let (device, _) = SeedlessMetalForward.ensure() else { return nil }

        let providers: [ArenaExpertProvider]
        if let ep = existingProviders {
            // 既存 provider を再利用(arena/LRU 状態を継続): bolt phase 2 でフレッシュ forward を作る際
            providers = ep
        } else {
            // 新規 provider 構築
            guard let source = try? ExpertSource(modelDir: modelDir) else {
                print("[makeFusedStreaming] ERROR: ExpertSource init failed")
                return nil
            }
            do { try source.warm() } catch {
                print("[makeFusedStreaming] ERROR: ExpertSource.warm() failed: \(error)")
                return nil
            }
            var ps: [ArenaExpertProvider] = []
            for i in 0 ..< Self.numLayers {
                guard let cache = try? LayerExpertCache(device: device, source: source, layer: i, C: C) else {
                    print("[makeFusedStreaming] ERROR: LayerExpertCache init failed at layer \(i)")
                    return nil
                }
                ps.append(ArenaExpertProvider(cache: cache))
            }
            providers = ps
        }

        guard let fwd = SeedlessFusedVerify.SeedlessFusedForward(
            layers: layers, caches: freshCaches(),
            maxM: maxM, H: Self.H, maxSeqLen: maxSeqLen,
            providers: providers) else { return nil }

        let fnA = fnW.asType(.float16)
        fwd.retainedArrays.append(fnA)
        guard let fnBuf = SeedlessMetalForward.mtlBuf(fnA, device) else { return nil }
        _ = fwd.attachHead(embedW: embedW, embedS: embedS, embedB: embedB,
                           lmW: lmW, lmS: lmS, lmB: lmB, fnW: fnW, vocab: vocab)
        return (fwd, fnBuf, providers)
    }

    /// 全層 1-CB fused forward(cache 常駐)を cold cache で構築。final norm buffer も返す。
    /// forwardRows(x, M, finalNormW: fnBuf) で「40層+final norm」が 1 CB になる。
    public func makeFused(maxM: Int, maxSeqLen: Int) -> (SeedlessFusedVerify.SeedlessFusedForward, MTLBuffer)? {
        guard let (device, _) = SeedlessMetalForward.ensure(),
              let fwd = SeedlessFusedVerify.SeedlessFusedForward(layers: layers, caches: freshCaches(),
                                                       maxM: maxM, H: Self.H, maxSeqLen: maxSeqLen) else { return nil }
        let fnA = fnW.asType(.float16)
        fwd.retainedArrays.append(fnA)   // zero-copy buffer の寿命規約(変換一時 array の保持)
        guard let fnBuf = SeedlessMetalForward.mtlBuf(fnA, device) else { return nil }
        // head 同梱(embed→層→norm→lm_head→argmax の 1-CB step)。失敗しても forwardRows 経路は生きる。
        _ = fwd.attachHead(embedW: embedW, embedS: embedS, embedB: embedB,
                           lmW: lmW, lmS: lmS, lmB: lmB, fnW: fnW, vocab: vocab)
        return (fwd, fnBuf)
    }
}
