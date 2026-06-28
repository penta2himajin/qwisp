import Foundation
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
    public var costModel: CostModel?   // 実測時のみ(a,b,c)
    public var ssdGBs: Double?         // streaming のみ実測

    public var summary: String {
        let cm = costModel.map { String(format: ", a=%.0fms b=%.1f", $0.a, $0.b) } ?? ""
        let ssd = ssdGBs.map { String(format: ", SSD=%.1fGB/s", $0) } ?? ""
        return String(format: "RAM=%.0fGB → mode=%@ C=%d maxK=%d (RSS~%.1fGB)%@%@",
                      ramGB, mode.rawValue, C, maxK, DeviceCalibration.estRSS(C), cm, ssd)
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
        return DeviceConfig(ramGB: ram, mode: mode, C: C, maxK: Swift.max(4, C * 3 / 8),
                            costModel: nil, ssdGBs: nil)
    }

    /// engine 既定 C（QWISP_CACHE_C 未指定時に使う）。実効 RAM(physicalRAMGB)から tier 決定。
    public static func defaultC() -> Int { tier(physicalRAMGB()).1 }

    /// cost-model 係数 a,b,c をオンデバイス実測（要 model+hot-pin 済）。起動時 calibration で DeviceConfig に埋める。
    /// a,b=forward-cost L-sweep の最小二乗、c=spec-step overhead(snapshot/restore/readback) - forward_ms(D+1)。
    public static func measureCostModel(model: StreamingQwispModel, ids: MLXArray, isLin: [Bool]) throws -> CostModel {
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        // (a,b) forward-cost L-sweep（teacher-forced, restore で同一 prefill 反復）
        var pts: [(L: Int, ms: Double)] = []
        for L in [1, 2, 4, 8, 16] {
            let bc = model.makeCaches(); _ = try model.prefillChunked(ids, caches: bc)
            MLX.eval(bc.flatMap { $0.stateArrays })
            let snaps = bc.map { $0.snapshot() }
            let seq = MLXArray(Array(repeating: Int32(100), count: L), [1, L])
            let (hw, _) = try model.forwardHidden(seq, caches: bc); MLX.eval([hw] + bc.flatMap { $0.stateArrays })
            for (i, cc) in bc.enumerated() { cc.restore(snaps[i], isLinear: isLin[i], trim: L) }
            var t: UInt64 = 0
            for _ in 0 ..< 10 {
                let s = now(); let (h, _) = try model.forwardHidden(seq, caches: bc)
                MLX.eval([h] + bc.flatMap { $0.stateArrays }); t += now() - s
                for (i, cc) in bc.enumerated() { cc.restore(snaps[i], isLinear: isLin[i], trim: L) }
            }
            pts.append((L, Double(t) / 10 / 1e6))
        }
        let ab = CostModel.fit(pts)
        // (c) spec-step overhead: snapshot + forward(D+1) + argmax readback(CPU) + restore - forward_ms(D+1)
        let D = 4
        let bc = model.makeCaches(); _ = try model.prefillChunked(ids, caches: bc)
        MLX.eval(bc.flatMap { $0.stateArrays })
        let seq = MLXArray(Array(repeating: Int32(100), count: D + 1), [1, D + 1])
        var t: UInt64 = 0; let reps = 10
        for _ in 0 ..< reps {
            let s = now()
            let snaps = bc.map { $0.snapshot() }
            let (_, vlg) = try model.forwardHidden(seq, caches: bc)
            let ev = MLX.argMax(vlg[0, 0 ..< (D + 1)], axis: -1)
            MLX.eval([ev] + bc.flatMap { $0.stateArrays })
            _ = ev.asArray(Int32.self)                                   // CPU readback(実 step と同じ)
            for (i, cc) in bc.enumerated() { cc.restore(snaps[i], isLinear: isLin[i], trim: D + 1) }
            t += now() - s
        }
        let stepMs = Double(t) / Double(reps) / 1e6
        let c = Swift.max(0, stepMs - ab.forwardMs(D + 1))
        return CostModel(a: ab.a, b: ab.b, c: c)
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
