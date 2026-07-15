import Foundation
import Metal
import MLX

/// device 別自動構成（calibration layer）。起動時に物理 RAM から mode/C/maxK を静的に決め、
/// （オプションで）forward-cost / SSD probe を実測して cost-model 係数を埋める。
/// 方針=実測主軸ハイブリッド（[[device-matrix-engine]] / notes/02）: RAM で mode は静的 gate、
/// cost model は実測。RSS 実測 tier: 8GB→C64 / 16GB→C128 / 24GB→C192 / 32GB+→C256(full)。
public enum DeviceMode: String {
    case streaming      // 8GB: expert を SSD から pread demand-load
    case partial        // 16GB: 半数常駐
    case nearFull       // 24GB: 75% 常駐 miss 数%
    case fullResident   // 32GB+: 全 expert 常駐 streaming 無
}

public struct DeviceConfig {
    public let ramGB: Double
    public let mode: DeviceMode
    public let C: Int
    public let maxK: Int               // = C×3/8（cost-model で max 安全=最速と検証済）

    public var summary: String {
        String(format: "RAM=%.0fGB → mode=%@ C=%d maxK=%d (RSS~%.1fGB)",
               ramGB, mode.rawValue, C, maxK, DeviceCalibration.estRSS(C))
    }
}

public enum DeviceCalibration {
    /// 実効 RAM (GB)。QWISP_DEVICE_RAM=<GB> で上書き(他 device を模擬/テスト)、無ければ物理 RAM。
    public static func physicalRAMGB() -> Double {
        if let r = ProcessInfo.processInfo.environment["QWISP_DEVICE_RAM"], let g = Double(r) { return g }
        return Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
    }

    /// 確定 tier: 8GB→C64 / 16GB→C128 / 24GB→C192 / 32GB+→C256。閾値は中間値で丸め。
    public static func tier(_ ramGB: Double) -> (DeviceMode, Int) {
        if ramGB < 12 { return (.streaming, 64) }      // 8GB
        if ramGB < 20 { return (.partial, 128) }       // 16GB
        if ramGB < 28 { return (.nearFull, 192) }      // 24GB
        return (.fullResident, 256)                    // 32GB+
    }

    /// C から推定 RSS（実測 base 2.4GB + 64C ごと 4.5GB）。
    public static func estRSS(_ C: Int) -> Double { 2.4 + Double(C) / 64.0 * 4.5 }

    /// RAM のみから静的 config（既定）。env override 可。
    public static func recommend(ramGB: Double? = nil) -> DeviceConfig {
        let ram = ramGB ?? physicalRAMGB()
        let (mode, C) = tier(ram)
        return DeviceConfig(ramGB: ram, mode: mode, C: C, maxK: Swift.max(4, C * 3 / 8))
    }

    /// engine 既定 C（QWISP_CACHE_C 未指定時に使う）。実効 RAM(physicalRAMGB)から tier 決定。
    public static func defaultC() -> Int { tier(physicalRAMGB()).1 }

    /// C for the STRICT streaming path only (issue #69). The RAM tier's C=128 carries a
    /// ~11.4GB wired footprint; on a real 16GB Mac that starves the page cache / GPU
    /// working set and strict collapses ~10x (field: 1.5-3.4 tok/s on 5 machines;
    /// ballast-sim reproduced 1.9 and C=64 recovered 6x under identical pressure).
    /// Budget = recommendedMaxWorkingSetSize (≈ RAM×0.68 on small Macs) × 0.8 margin —
    /// the wired ceiling that leaves the rest of RAM breathing. bolt keeps the tier C:
    /// its frozen residency tolerates a big wired set (stable hot subset) and a smaller
    /// C makes it LOOPY-prone (sim-measured).
    /// Overrides: QWISP_STRICT_C (direct), QWISP_GPU_BUDGET_GB (budget, for tests/sims).
    public static func strictStreamingC(tierC: Int) -> Int {
        let env = ProcessInfo.processInfo.environment
        if let o = env["QWISP_STRICT_C"], let v = Int(o) { return v }
        let budgetGB: Double
        if let o = env["QWISP_GPU_BUDGET_GB"], let v = Double(o) {
            budgetGB = v
        } else if let dev = MTLCreateSystemDefaultDevice() {
            budgetGB = Double(dev.recommendedMaxWorkingSetSize) / 1e9
        } else {
            return tierC
        }
        return fitC(tierC: tierC, budgetGB: budgetGB)
    }

    /// Pure fitting rule (self-checked from configtest): largest C ≤ tierC, in steps of
    /// 32 down to 64, whose estimated wired footprint fits budget×0.8.
    public static func fitC(tierC: Int, budgetGB: Double) -> Int {
        var c = tierC
        while c > 64, estRSS(c) > budgetGB * 0.8 { c -= 32 }
        return c
    }


    /// device-config インスペクタ（モデル不要）。現機 + 各 RAM tier の構成を表示。
    public static func describeAll() -> String {
        var lines = ["[DeviceConfig] 本機: " + recommend().summary, "  tier 表:"]
        for ram in [8.0, 16, 24, 32, 64] {
            lines.append("    " + recommend(ramGB: ram).summary)
        }
        return lines.joined(separator: "\n")
    }
}
