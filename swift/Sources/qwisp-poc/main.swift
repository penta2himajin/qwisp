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

// 速度検証: 40層 arena-MoE pipeline（ref 不要）
print(ArenaBench.run())
print("[qwisp-poc] done.")
