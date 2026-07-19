import Foundation
import MLX

// #98 phase 2b: real-checkpoint parity probe — replay oracle/dflash_parity.py's call
// sequence on DFlashDrafter and compare module outputs (final-normed hidden) against
// the Python reference bundle. QWISP_RUN=dflash-parity.
// Env: QWISP_DFLASH_DIR (default ~/.mtplx/models/z-lab--Qwen3.6-35B-A3B-DFlash),
//      QWISP_DFLASH_PARITY (default /tmp/dflash_parity.safetensors).
public enum DFlashParityProbe {
    static func cmp(_ got: MLXArray, _ ref: MLXArray, _ name: String) -> String {
        got.eval(); ref.eval()
        let g = got.asType(.float16).reshaped([-1]).asArray(Float16.self)
        let r = ref.asType(.float16).reshaped([-1]).asArray(Float16.self)
        guard g.count == r.count else { return "\(name): SHAPE MISMATCH \(g.count) vs \(r.count)" }
        var bitEq = 0
        var maxAbs: Float = 0, maxRel: Float = 0
        for i in 0 ..< g.count {
            if g[i].bitPattern == r[i].bitPattern { bitEq += 1 }
            let a = abs(Float(g[i]) - Float(r[i]))
            maxAbs = max(maxAbs, a)
            let d = max(abs(Float(r[i])), 1e-3)
            maxRel = max(maxRel, a / d)
        }
        let pct = 100.0 * Double(bitEq) / Double(g.count)
        return String(format: "%@: bitEq=%.2f%% maxAbs=%.3e maxRel=%.3e", name, pct, maxAbs, maxRel)
    }

    public static func run() -> String {
        let env = ProcessInfo.processInfo.environment
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let dir = env["QWISP_DFLASH_DIR"] ?? "\(home)/.mtplx/models/z-lab--Qwen3.6-35B-A3B-DFlash"
        let bundlePath = env["QWISP_DFLASH_PARITY"] ?? "/tmp/dflash_parity.safetensors"

        guard let drafter = DFlashDrafter.load(dir: URL(fileURLWithPath: dir)) else {
            return "[dflash-parity] FAIL: DFlashDrafter.load(\(dir)) nil"
        }
        guard let bundle = try? loadArrays(url: URL(fileURLWithPath: bundlePath)) else {
            return "[dflash-parity] FAIL: bundle load \(bundlePath) (run oracle/dflash_parity.py first)"
        }
        func need(_ k: String) -> MLXArray? { bundle[k] }
        guard let n1 = need("noise1"), let c1 = need("ctx1"), let o1 = need("out1"),
              let n2 = need("noise2"), let c2 = need("ctx2"), let o2 = need("out2"),
              let n3 = need("noise3"), let c3 = need("ctx3"), let o3 = need("out3"),
              let nB = need("noiseB"), let cB = need("ctxB"), let oB = need("outB")
        else { return "[dflash-parity] FAIL: bundle missing keys" }

        var lines: [String] = ["[dflash-parity] drafter=\(dir)"]

        // Case A: sequential blocks + trim rollback (mirror of dflash_parity.py)
        let caches = drafter.makeCaches()
        lines.append(cmp(drafter.forward(noise: n1.asType(.float16), ctx: c1.asType(.float16), caches: caches), o1, "out1"))
        lines.append(cmp(drafter.forward(noise: n2.asType(.float16), ctx: c2.asType(.float16), caches: caches), o2, "out2"))
        drafter.trimTo(committed: caches[0].offset - 2, caches: caches)
        lines.append(cmp(drafter.forward(noise: n3.asType(.float16), ctx: c3.asType(.float16), caches: caches), o3, "out3"))

        // Case B: crop + windowed-causal mask branch
        let cachesB = drafter.makeCaches()
        lines.append(cmp(drafter.forward(noise: nB.asType(.float16), ctx: cB.asType(.float16), caches: cachesB), oB, "outB"))

        return lines.joined(separator: "\n")
    }
}
