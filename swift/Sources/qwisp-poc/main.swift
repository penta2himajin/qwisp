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

// 速度検証: 40層 arena-MoE pipeline（ref 不要）
print(ArenaBench.run())
print("[qwisp-poc] done.")
