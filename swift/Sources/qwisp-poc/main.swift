import Foundation
import QwispCore

// issue#5: 1 command buffer に op を多く束ね commit/sync を削減（dispatch 律速の無料 ~4-5%, exact）。
// 未設定時のみ既定 2000（ユーザ上書き可）。純スケジューリングゆえ数値不変。
if ProcessInfo.processInfo.environment["MLX_MAX_OPS_PER_BUFFER"] == nil {
    setenv("MLX_MAX_OPS_PER_BUFFER", "2000", 1)
}

print("[qwisp-poc] starting ...")
print(QwispCore.smoke())

if CommandLine.arguments.contains("stream") {
    let md = ProcessInfo.processInfo.environment["QWISP_MODEL"]
        ?? "\(FileManager.default.homeDirectoryForCurrentUser.path)/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16"
    let env = ProcessInfo.processInfo.environment
    let mtpRef = env["QWISP_MTP_REF"] ?? "/tmp/qwisp_mtp_ref.safetensors"

    // Gate / bench entry points:
    //   test_raw.sh      → QWISP_RUN=raw-tests (RawVerifyTests, the correctness gate)
    //   bench_batch.sh   → QWISP_RUN=raw-spec  (RawSpecRunner.run, strict/bolt)
    let runners: [(String, (String, String) throws -> String)] = [
        ("raw-tests", { _, _ in RawVerifyTests.runAll() }),
        ("raw-spec",  { try RawSpecRunner.run(modelDir: $0, refPath: $1) }),
    ]
    if let name = env["QWISP_RUN"] {
        if let r = runners.first(where: { $0.0 == name }) {
            do { print(try r.1(md, mtpRef)) } catch { print("[\(name)] error: \(error)") }
        } else {
            print("unknown QWISP_RUN=\(name)\n  available: \(runners.map { $0.0 }.joined(separator: ", "))")
        }
        exit(0)
    }

    // 既定: strict/bolt とも raw engine（RawSpecRunner: resident=fused 1-CB / C<256=streaming、
    // C は QWISP_RAW_C 未指定なら RAM tier 自動）。QWISP_BOLT=1 は raw bolt(forceBolt, streaming 限定)。
    let boltDefault = env["QWISP_BOLT"] == "1"
    do {
        print(try RawSpecRunner.run(modelDir: md, refPath: mtpRef, forceBolt: boltDefault))
    } catch { print("[\(boltDefault ? "RawBolt" : "RawSpec")] error: \(error)") }
    exit(0)
}

print("[qwisp-poc] done.")
