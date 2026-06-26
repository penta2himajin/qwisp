import Foundation
import QwispCore

print("[qwisp-poc] starting ...")
print(QwispCore.smoke())

// GDN kernel T-consistency 検査: updateKernel(T=2) vs 逐次 T=1×2（state carry）が bit 一致するか。
// spec の accept-state drift の真因切り分け（QWISP_GDN_TTEST=1）。
if ProcessInfo.processInfo.environment["QWISP_GDN_TTEST"] == "1" {
    print(GatedDelta.tConsistencyTest())
    print(AttentionLayer.sConsistencyTest(dtype: .float16))
    print(AttentionLayer.sConsistencyTest(dtype: .float32))
    exit(0)
}

// streaming-only モード（クリーンな RSS 計測用）: `qwisp-poc stream` で起動
if CommandLine.arguments.contains("stream") {
    let md = ProcessInfo.processInfo.environment["QWISP_MODEL"]
        ?? "\(FileManager.default.homeDirectoryForCurrentUser.path)/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"
    do { print(try StreamingDecode.run(modelDir: md, refPath: "/tmp/qwisp_full_ref.safetensors")) }
    catch { print("[S3] error: \(error)") }
    let mtpRef = ProcessInfo.processInfo.environment["QWISP_MTP_REF"] ?? "/tmp/qwisp_mtp_ref.safetensors"
    if FileManager.default.fileExists(atPath: mtpRef),
       FileManager.default.fileExists(atPath: "\(md)/mtp.safetensors") {
        do { print(try StreamingDecode.runSpeculative(modelDir: md, refPath: mtpRef)) }
        catch { print("[M2c×stream] error: \(error)") }
        do { print(try StreamingDecode.runHybridFast(modelDir: md, refPath: mtpRef)) }
        catch { print("[fast] error: \(error)") }
        do { print(try Tell.runM0(modelDir: md, refPath: mtpRef)) }
        catch { print("[Tell] error: \(error)") }
        do { print(try Tell.runM2(modelDir: md, refPath: mtpRef)) }
        catch { print("[Tell M2] error: \(error)") }
        do { print(try Tell.runM5(modelDir: md, refPath: mtpRef)) }
        catch { print("[Tell M5] error: \(error)") }
        if ProcessInfo.processInfo.environment["QWISP_HOTCOLD_CALIB"] == "1" {
            do { print(try Tell.runHotColdCalib(modelDir: md, refPath: mtpRef)) }
            catch { print("[HotCold] error: \(error)") }
        }
        if ProcessInfo.processInfo.environment["QWISP_HOTCOLD_FAST"] == "1" {
            do { print(try Tell.runHotColdFast(modelDir: md, refPath: mtpRef)) }
            catch { print("[HotColdFast] error: \(error)") }
        }
        if ProcessInfo.processInfo.environment["QWISP_HOTCOLD_DIAG"] == "1" {
            do { print(try Tell.runHotColdDiag(modelDir: md, refPath: mtpRef)) }
            catch { print("[HotColdDiag] error: \(error)") }
        }
        if ProcessInfo.processInfo.environment["QWISP_HOTCOLD_ONLINE"] == "1" {
            do { print(try Tell.runHotColdOnline(modelDir: md, refPath: mtpRef)) }
            catch { print("[HotColdOnline] error: \(error)") }
        }
        if ProcessInfo.processInfo.environment["QWISP_HOTCOLD_ADAPT"] == "1" {
            do { print(try Tell.runHotColdAdaptive(modelDir: md, refPath: mtpRef)) }
            catch { print("[HotColdAdaptive] error: \(error)") }
        }
        if ProcessInfo.processInfo.environment["QWISP_HOTCOLD_AUTO"] == "1" {
            do { print(try Tell.runHotColdAuto(modelDir: md, refPath: mtpRef)) }
            catch { print("[HotColdAuto] error: \(error)") }
        }
        if ProcessInfo.processInfo.environment["QWISP_HOTCOLD_SPEC"] == "1" {
            do { print(try Tell.runHotColdSpec(modelDir: md, refPath: mtpRef)) }
            catch { print("[HotColdSpec] error: \(error)") }
        }
        do { print(try Tell.runM4(modelDir: md, refPath: mtpRef)) }
        catch { print("[Tell M4] error: \(error)") }
        do { print(try StreamingDecode.runCrossLayerFast(modelDir: md, refPath: mtpRef)) }
        catch { print("[xlayer] error: \(error)") }
    }
    exit(0)
}

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

// M2b-3: decode cache 正しさ + tok/s 粗計測
if FileManager.default.fileExists(atPath: fullRef),
   FileManager.default.fileExists(atPath: "\(modelDir)/config.json") {
    do { print(try DecodeValidation.run(modelDir: modelDir, refPath: fullRef)) }
    catch { print("[M2b-3 decode] error: \(error)") }
}

// S1: ExpertSource pread スライスが resident と bit 一致するか
if FileManager.default.fileExists(atPath: "\(modelDir)/config.json") {
    do { print(try ExpertSourceValidation.run(modelDir: modelDir)) }
    catch { print("[S1] error: \(error)") }
}

// S2: 持続 arena streaming MoE が resident と一致するか（concat 無し in-place）
if FileManager.default.fileExists(atPath: "\(modelDir)/config.json") {
    do { print(try StreamingMoEValidation.run(modelDir: modelDir)) }
    catch { print("[S2] error: \(error)") }
}

// M2c: MTP head の検証（mtp.safetensors を自前ロード）
if FileManager.default.fileExists(atPath: "/tmp/qwisp_mtp_ref.safetensors"),
   FileManager.default.fileExists(atPath: "\(modelDir)/mtp.safetensors") {
    do { print(try MTPHeadValidation.run(modelDir: modelDir, refPath: "/tmp/qwisp_mtp_ref.safetensors")) }
    catch { print("[M2c] error: \(error)") }
}

// M2c: MTP 投機デコード（実プロンプトで greedy 一致 + Python spec 一致 + speedup）
let mtpRefMain = ProcessInfo.processInfo.environment["QWISP_MTP_REF"] ?? "/tmp/qwisp_mtp_ref.safetensors"
if CommandLine.arguments.contains("spec"),
   FileManager.default.fileExists(atPath: mtpRefMain),
   FileManager.default.fileExists(atPath: "\(modelDir)/mtp.safetensors") {
    do { print(try SpeculativeDecode.run(modelDir: modelDir, refPath: mtpRefMain)) }
    catch { print("[M2c spec] error: \(error)") }
    exit(0)
}

// 速度検証: 40層 arena-MoE pipeline（ref 不要）
print(ArenaBench.run())
print("[qwisp-poc] done.")
