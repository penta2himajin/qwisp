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

    // Measurement runner registry — ONE entry per metric. `QWISP_RUN=list` prints this
    // catalog; docs/measurement.md maps symptoms → runners. Format: (name, "what it
    // answers · requirements · knobs", closure). Keep descriptions to one line.
    struct Runner { let name: String; let desc: String; let run: (String, String) throws -> String }
    let runners: [Runner] = [
        // ── correctness gates (green = safe to commit) ──
        Runner(name: "raw-tests", desc: "RAWTESTS lossless gate (engine kernels; GPU, no model) — scripts/test_raw.sh",
               run: { _, _ in SeedlessVerifyTests.runAll() }),
        Runner(name: "prefix-cache-e2e", desc: "prefix cache lossless: reuse/extend/cross/reset byte-compare vs cache-off (#76; GPU+model; QWISP_PREFIX_E2E_C=<c> for streaming)",
               run: { md, _ in Tell.prefixCacheE2E(modelDir: md) }),
        Runner(name: "prefix-persist-e2e", desc: "disk-persist restart lossless gate (#89; GPU+model)",
               run: { md, _ in Tell.prefixPersistE2E(modelDir: md) }),
        Runner(name: "prefix-ram-e2e", desc: "RAM-tier conversation-switch lossless gate + ramHits assertions (#117; GPU+model)",
               run: { md, _ in Tell.prefixRAME2E(modelDir: md) }),
        Runner(name: "prefix-bolt-e2e", desc: "bolt prompt-prefix blob reuse byte-identity (#76 bolt side; GPU+model)",
               run: { md, _ in Tell.prefixBoltE2E(modelDir: md) }),
        // ── decode speed (regressions, long-context) ──
        Runner(name: "raw-spec", desc: "shipping decode bench, strict/bolt tok/s + token dump (GPU+model+refs) — bench_batch.sh cell",
               run: { try Tell.run(modelDir: $0, refPath: $1) }),
        Runner(name: "long-context-decay", desc: "per-stage (GDN/attn/MoE) GPU ms: prefill chunks by position + M=1 decode by ctx up to 48K (#119; QWISP_DECAY_MAX)",
               run: { md, _ in Tell.longContextDecayProbe(modelDir: md) }),
        Runner(name: "spec-width", desc: "verify forward stage ms by draft width M at fixed ctx (#119; QWISP_SPEC_CTX) — is speculation paying?",
               run: { md, _ in Tell.specWidthProbe(modelDir: md) }),
        Runner(name: "seqmt-m", desc: "M-row verify scaling r_M (#90) — cost of widening the verify",
               run: { md, _ in Tell.seqMTScalingProbe(modelDir: md) }),
        // ── prefill / TTFT ──
        Runner(name: "prefill-probe", desc: "prefill tok/s by chunk size (overhead- vs compute-bound discriminator)",
               run: { md, _ in Tell.prefillThroughputProbe(modelDir: md) }),
        Runner(name: "prefill-breakdown", desc: "prefill wall vs GPU time split (dispatch/sync tax; chunk + floor variants)",
               run: { md, _ in Tell.prefillBreakdownProbe(modelDir: md) }),
        Runner(name: "prefill-stage-profile", desc: "per-stage prefill GPU ms (GDN/attn/MoE) + MLX matrix-unit reference (QWISP_PREFILL_LEN)",
               run: { md, _ in Tell.prefillStageProfile(modelDir: md) }),
        Runner(name: "hybrid-estimate", desc: "steel-hybrid prefill win estimate (analytic, cheap)",
               run: { md, _ in Tell.hybridEstimate(modelDir: md) }),
        Runner(name: "hybrid-prefill-bench", desc: "steel-hybrid prefill measured tok/s vs baseline",
               run: { md, _ in Tell.hybridPrefillBench(modelDir: md) }),
        Runner(name: "prefix-cache-speed", desc: "TTFT cold vs cross-conversation vs intra-conversation (QWISP_PREFIX_SHARED)",
               run: { md, _ in Tell.prefixCacheSpeedProbe(modelDir: md) }),
        Runner(name: "prefix-cache-poc", desc: "snapshot/restore byte-identity micro-PoC (design validation, pre-e2e)",
               run: { md, _ in Tell.prefixCachePoC(modelDir: md) }),
        // ── kernel micro-benchmarks (no model) ──
        Runner(name: "grouped-moe-bench", desc: "grouped MoE expert kernel micro-bench",
               run: { _, _ in GroupedMoEPoC.bench() }),
        Runner(name: "dense-tiled-bench", desc: "dense tiled matmul micro-bench",
               run: { _, _ in GroupedMoEPoC.denseBench() }),
        Runner(name: "steel-route-bench", desc: "steel routing kernel micro-bench",
               run: { _, _ in GroupedMoEPoC.steelRouteBench() }),
        Runner(name: "gqmm2-bench", desc: "gqmm 2-bit kernel micro-bench (notes/18 W1)",
               run: { _, _ in SeedlessMetalForward.gqmm2Bench() }),
        Runner(name: "mlx-qmm-minv", desc: "MLX qmm M-invariance check (kernel-switch bit stability)",
               run: { md, _ in Tell.mlxQmmInvariance(modelDir: md) }),
    ]
    func catalog() -> String {
        "QWISP_RUN catalog (see docs/measurement.md for symptom → runner):\n"
            + runners.map { "  " + $0.name.padding(toLength: 22, withPad: " ", startingAt: 0) + $0.desc }.joined(separator: "\n")
            + "\n  server-side: QWISP_SPEC_PROFILE=1 on serve = per-step decode buckets (draft/chain/verify/rebuild)"
    }
    if let name = env["QWISP_RUN"] {
        if name == "list" {
            print(catalog())
        } else if let r = runners.first(where: { $0.name == name }) {
            do { print(try r.run(md, mtpRef)) } catch { print("[\(name)] error: \(error)") }
        } else {
            print("unknown QWISP_RUN=\(name)\n" + catalog())
        }
        exit(0)
    }

    // 既定: strict/bolt とも raw engine（Tell: resident=fused 1-CB / C<256=streaming、
    // C は QWISP_RAW_C 未指定なら RAM tier 自動）。QWISP_BOLT=1 は raw bolt(forceBolt, streaming 限定)。
    let boltDefault = env["QWISP_BOLT"] == "1"
    do {
        print(try Tell.run(modelDir: md, refPath: mtpRef, forceBolt: boltDefault))
    } catch { print("[\(boltDefault ? "RawBolt" : "RawSpec")] error: \(error)") }
    exit(0)
}

print("[qwisp-poc] done.")
