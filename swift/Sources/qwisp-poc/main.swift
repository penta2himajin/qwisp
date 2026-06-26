import Foundation
import QwispCore

print("[qwisp-poc] starting ...")
print(QwispCore.smoke())

// M1: gatherQuantizedMatmul の Python ビット一致検証
let refPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "/tmp/qwisp_ref.safetensors"
if FileManager.default.fileExists(atPath: refPath) {
    do {
        print(try GatherQMMValidation.run(refPath: refPath))
    } catch {
        print("[M1] error: \(error)")
    }
    do {
        print(try PersistentArenaTest.run(refPath: refPath))
    } catch {
        print("[M3] error: \(error)")
    }
    do {
        print(try MoELayerValidation.run(refPath: refPath))
    } catch {
        print("[M2a] error: \(error)")
    }
} else {
    print("[M1/M3] skip: ref not found at \(refPath) (run: PY -m qwisp.swift_ref)")
}

// M2b-0: config ロード検証
let modelDir = ProcessInfo.processInfo.environment["QWISP_MODEL"]
    ?? "\(FileManager.default.homeDirectoryForCurrentUser.path)/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"
if FileManager.default.fileExists(atPath: "\(modelDir)/config.json") {
    do {
        let cfg = try QwispConfig.load(modelDir: modelDir)
        let nLin = (0..<cfg.numHiddenLayers).filter { cfg.isLinearLayer($0) }.count
        print("[M2b-0] config OK: H=\(cfg.hiddenSize) L=\(cfg.numHiddenLayers) "
            + "(linear=\(nLin)/full=\(cfg.numHiddenLayers - nLin)) experts=\(cfg.numExperts)/top\(cfg.numExpertsPerTok) "
            + "linAttn(vH=\(cfg.linearNumValueHeads) kH=\(cfg.linearNumKeyHeads) hd=\(cfg.linearKeyHeadDim) convK=\(cfg.linearConvKernelDim))")
    } catch {
        print("[M2b-0] config error: \(error)")
    }
} else {
    print("[M2b-0] skip: config.json not found at \(modelDir)")
}

// M2b-1: GatedDeltaNet recurrent 核の検証
let gdnRef = "/tmp/qwisp_gdn_ref.safetensors"
if FileManager.default.fileExists(atPath: gdnRef) {
    do { print(try GatedDeltaValidation.run(refPath: gdnRef)) }
    catch { print("[M2b-1] error: \(error)") }
} else {
    print("[M2b-1] skip: gdn ref not found (run: PY -m qwisp.gdn_ref)")
}

// M2b-1: GatedDeltaNet 層 wrapping の検証
let gdnLayerRef = "/tmp/qwisp_gdn_layer_ref.safetensors"
if FileManager.default.fileExists(atPath: gdnLayerRef) {
    do { print(try GatedDeltaNetLayerValidation.run(refPath: gdnLayerRef)) }
    catch { print("[M2b-1 layer] error: \(error)") }
} else {
    print("[M2b-1 layer] skip: ref not found (run: PY -m qwisp.gdn_layer_ref)")
}

// M2b-2: full-attention 層の検証
let attnRef = "/tmp/qwisp_attn_ref.safetensors"
if FileManager.default.fileExists(atPath: attnRef) {
    do { print(try AttentionLayerValidation.run(refPath: attnRef)) }
    catch { print("[M2b-2] error: \(error)") }
} else {
    print("[M2b-2] skip: attn ref not found (run: PY -m qwisp.attn_ref)")
}

// M2b-3: 実モデル layer-0 を REAL 4bit 量子化重みで検証
let realLayerRef = "/tmp/qwisp_real_layer_ref.safetensors"
if FileManager.default.fileExists(atPath: realLayerRef) {
    do { print(try RealLayer0Validation.run(refPath: realLayerRef)) }
    catch { print("[M2b-3] error: \(error)") }
} else {
    print("[M2b-3] skip: real-layer ref not found (run: PY -m qwisp.real_layer_ref)")
}

// M2b-3: 実モデル layer-0 MoE block を REAL 量子化重みで検証
let realMoeRef = "/tmp/qwisp_real_moe_ref.safetensors"
if FileManager.default.fileExists(atPath: realMoeRef) {
    do { print(try MoEBlockValidation.run(refPath: realMoeRef)) }
    catch { print("[M2b-3 moe] error: \(error)") }
} else {
    print("[M2b-3 moe] skip: real-moe ref not found (run: PY -m qwisp.real_moe_ref)")
}

// M2b-3: 完全な DecoderLayer（linear 層0 / full-attn 層3）を REAL 量子化重みで検証
for (ref, lbl) in [("/tmp/qwisp_dec0_ref.safetensors", "DecoderLayer-0"),
                   ("/tmp/qwisp_dec3_ref.safetensors", "DecoderLayer-3")] {
    if FileManager.default.fileExists(atPath: ref) {
        do { print(try DecoderLayerValidation.run(refPath: ref, label: lbl)) }
        catch { print("[M2b-3 \(lbl)] error: \(error)") }
    } else {
        print("[M2b-3 \(lbl)] skip: ref not found (run: PY -m qwisp.real_decoder_ref)")
    }
}

// M2b-3: embed_tokens + final norm + lm_head を REAL 量子化重みで検証
let headRef = "/tmp/qwisp_head_ref.safetensors"
if FileManager.default.fileExists(atPath: headRef) {
    do { print(try ModelHeadValidation.run(refPath: headRef)) }
    catch { print("[M2b-3 head] error: \(error)") }
} else {
    print("[M2b-3 head] skip: head ref not found (run: PY -m qwisp.real_head_ref)")
}

// M2b-3: FULL forward(40層) を実モデルロードで Python と一致検証
let fullRef = "/tmp/qwisp_full_ref.safetensors"
if FileManager.default.fileExists(atPath: fullRef),
   FileManager.default.fileExists(atPath: "\(modelDir)/config.json") {
    do { print(try FullModelValidation.run(modelDir: modelDir, refPath: fullRef)) }
    catch { print("[M2b-3 full] error: \(error)") }
} else {
    print("[M2b-3 full] skip: full ref or model dir not found")
}

// 速度検証: 40層 arena-MoE pipeline（ref 不要）
print(ArenaBench.run())
print("[qwisp-poc] done.")
