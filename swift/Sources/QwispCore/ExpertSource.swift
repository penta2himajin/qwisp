import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#endif

/// safetensors から per-expert スライスを on-demand pread で取得（qwisp/expert_source.py の Swift 版）.
/// switch_mlp.{gate,up,down}_proj.{weight(U32),scales(F16),biases(F16)} は先頭次元に 256 experts
/// スタック → expert e は連続バイト。stride=(end-begin)/256、offset=dataStart+begin+e*stride を pread。
public final class ExpertSource {
    public static let projs = ["gate_proj", "up_proj", "down_proj"]
    public static let parts = ["weight", "scales", "biases"]
    static let prefix = "language_model.model.layers"

    struct TensorMeta { let dtype: String; let shape: [Int]; let begin: Int; let end: Int }

    // ★ SSD 帯域エミュレーション (slow-NAND=Neo の C/D 評価): preadInto を単一 chokepoint とした
    //   leaky-bucket active throttle。QWISP_SSD_THROTTLE_GBS=目標 BW(GB/s, 0=無効)。単一 NAND の直列性を
    //   lock+virtualClock で模擬し、fast pread 後に modeled 完了時刻まで nanosleep → engine の実 prefetch/
    //   overlap が自然応答=faithful。QWISP_SSD_ACCT=1 で bytes/reads/実 nanos 累積(cross-check)。既定 off。
    nonisolated(unsafe) public static var throttleGBs: Double = Double(ProcessInfo.processInfo.environment["QWISP_SSD_THROTTLE_GBS"] ?? "") ?? 0
    nonisolated(unsafe) public static var acct = ProcessInfo.processInfo.environment["QWISP_SSD_ACCT"] == "1"
    nonisolated(unsafe) public static var acctBytes = 0
    nonisolated(unsafe) public static var acctReads = 0
    nonisolated(unsafe) public static var acctNanos: UInt64 = 0        // 実 pread 時間(throttle sleep 除く)
    nonisolated(unsafe) static var virtualClockNs: UInt64 = 0          // 単一 NAND が free になる時刻
    static let throttleLock = NSLock()
    // ★ T2: throttle defer knob。QWISP_THROTTLE_DEFER=1 で「初期 ~18GB weight load / calib / prefill」を
    //   throttle 対象外にし、runner が timed decode 直前に throttleActive=true を立てるまで不活性化
    //   （steady-state decode tok/s の計測に load/prefill の throttle は無関係で計測が ~2x 遅いだけ）。
    //   既定 OFF(throttleActive=true)=現行挙動と完全同一。batch runner は cell 毎に !throttleDefer へ reset。
    public static let throttleDefer = ProcessInfo.processInfo.environment["QWISP_THROTTLE_DEFER"] == "1"
    nonisolated(unsafe) public static var throttleActive = ProcessInfo.processInfo.environment["QWISP_THROTTLE_DEFER"] != "1"
    public static func resetAcct() { throttleLock.lock(); acctBytes = 0; acctReads = 0; acctNanos = 0; virtualClockNs = 0; throttleLock.unlock() }

    let dir: URL
    let wm: [String: String]                          // tensor name -> shard
    var headers: [String: (meta: [String: TensorMeta], dataStart: Int)] = [:]
    var fds: [String: Int32] = [:]

    public init(modelDir: String) throws {
        dir = URL(fileURLWithPath: modelDir)
        let data = try Data(contentsOf: dir.appendingPathComponent("model.safetensors.index.json"))
        let top = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        wm = (top["weight_map"] as? [String: String]) ?? [:]
    }

    static func dtype(_ s: String) -> DType {
        switch s {
        case "U32": return .uint32
        case "F16": return .float16
        case "BF16": return .bfloat16
        case "F32": return .float32
        default: return .uint8
        }
    }

    static func itemSize(_ s: String) -> Int {
        switch s { case "U32", "F32": return 4; case "F16", "BF16": return 2; default: return 1 }
    }

    func key(_ layer: Int, _ proj: String, _ part: String) -> String {
        "\(ExpertSource.prefix).\(layer).mlp.switch_mlp.\(proj).\(part)"
    }

    func header(_ shard: String) throws -> (meta: [String: TensorMeta], dataStart: Int) {
        if let h = headers[shard] { return h }
        let url = dir.appendingPathComponent(shard)
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        let lenData = try fh.read(upToCount: 8)!
        let hlen = lenData.withUnsafeBytes { $0.load(as: UInt64.self).littleEndian }
        let hdrData = try fh.read(upToCount: Int(hlen))!
        let json = try JSONSerialization.jsonObject(with: hdrData) as? [String: Any] ?? [:]
        var meta: [String: TensorMeta] = [:]
        for (k, v) in json {
            guard let d = v as? [String: Any], let dt = d["dtype"] as? String,
                  let shape = d["shape"] as? [Int],
                  let off = d["data_offsets"] as? [Int], off.count == 2 else { continue }
            meta[k] = TensorMeta(dtype: dt, shape: shape, begin: off[0], end: off[1])
        }
        let h = (meta, 8 + Int(hlen))
        headers[shard] = h
        return h
    }

    func fd(_ shard: String) -> Int32 {
        if let f = fds[shard] { return f }
        let f = open(dir.appendingPathComponent(shard).path, O_RDONLY)
        fds[shard] = f
        return f
    }

    /// 全 layer の header/fd を逐次で先読み（並列 pread 前に dict 競合を避ける）。
    public func warm(numLayers: Int = 40) throws {
        for layer in 0 ..< numLayers {
            for proj in ExpertSource.projs {
                for part in ExpertSource.parts {
                    let name = key(layer, proj, part)
                    if let shard = wm[name] { _ = try header(shard); _ = fd(shard) }
                }
            }
        }
    }

    /// (layer,proj,part) の 1-expert スライスのバイト数（= 全 layer 共通の slot サイズ）。
    public func sliceBytes(_ layer: Int, _ proj: String, _ part: String) throws -> Int {
        let name = key(layer, proj, part)
        guard let shard = wm[name], let t = try header(shard).meta[name] else {
            throw NSError(domain: "ExpertSource", code: 2)
        }
        return (t.end - t.begin) / t.shape[0]
    }

    public func restShape(_ layer: Int, _ proj: String, _ part: String) throws -> [Int] {
        let name = key(layer, proj, part)
        guard let shard = wm[name], let t = try header(shard).meta[name] else {
            throw NSError(domain: "ExpertSource", code: 2)
        }
        return Array(t.shape.dropFirst())
    }

    public func partDType(_ layer: Int, _ proj: String, _ part: String) throws -> DType {
        let name = key(layer, proj, part)
        guard let shard = wm[name], let t = try header(shard).meta[name] else {
            throw NSError(domain: "ExpertSource", code: 2)
        }
        return ExpertSource.dtype(t.dtype)
    }

    /// expert e の (layer,proj,part) の絶対バイト範囲（shard パス, offset, length）。device probe 用。
    public func expertByteRange(_ layer: Int, _ proj: String, _ part: String, _ e: Int)
        throws -> (shardPath: String, offset: Int, length: Int) {
        let name = key(layer, proj, part)
        guard let shard = wm[name], let t = try header(shard).meta[name] else {
            throw NSError(domain: "ExpertSource", code: 2)
        }
        let (_, dataStart) = try header(shard)
        let stride = (t.end - t.begin) / t.shape[0]
        let offset = dataStart + t.begin + e * stride
        return (dir.appendingPathComponent(shard).path, offset, stride)
    }

    /// expert e の (layer,proj,part) バイトを buf に直接 pread（arena slot へのゼロコピー書込）。
    public func preadInto(_ buf: UnsafeMutableRawPointer, _ layer: Int, _ proj: String,
                          _ part: String, _ e: Int) throws {
        let name = key(layer, proj, part)
        guard let shard = wm[name], let t = try header(shard).meta[name] else {
            throw NSError(domain: "ExpertSource", code: 2)
        }
        let (_, dataStart) = try header(shard)
        let stride = (t.end - t.begin) / t.shape[0]
        let offset = dataStart + t.begin + e * stride
        let t0 = DispatchTime.now().uptimeNanoseconds
        let n = pread(fd(shard), buf, stride, off_t(offset))
        if n != stride { throw NSError(domain: "ExpertSource", code: 3) }
        let dt = DispatchTime.now().uptimeNanoseconds - t0
        if ExpertSource.acct {
            ExpertSource.throttleLock.lock()
            ExpertSource.acctBytes += stride; ExpertSource.acctReads += 1; ExpertSource.acctNanos += dt
            ExpertSource.throttleLock.unlock()
        }
        // ★ leaky-bucket throttle: 単一 NAND を直列にモデル化。serveNs = stride / (GB/s)（GB=1e9, ns=s×1e9 相殺）。
        //   throttleActive: T2 defer gate（既定 true=透過。QWISP_THROTTLE_DEFER=1 時のみ decode 開始まで false）。
        if ExpertSource.throttleGBs > 0 && ExpertSource.throttleActive {
            let serveNs = UInt64(Double(stride) / ExpertSource.throttleGBs)
            ExpertSource.throttleLock.lock()
            let now = DispatchTime.now().uptimeNanoseconds
            let start = Swift.max(now, ExpertSource.virtualClockNs)
            let end = start &+ serveNs
            ExpertSource.virtualClockNs = end
            ExpertSource.throttleLock.unlock()
            let after = DispatchTime.now().uptimeNanoseconds
            if end > after {
                let s = end - after
                var ts = timespec(tv_sec: Int(s / 1_000_000_000), tv_nsec: Int(s % 1_000_000_000))
                nanosleep(&ts, nil)
            }
        }
    }

    /// expert e の (layer,proj,part) スライスを [1, rest...] の MLXArray で返す（pread, 自前 buffer 所有）。
    public func slice(_ layer: Int, _ proj: String, _ part: String, _ e: Int) throws -> MLXArray {
        let name = key(layer, proj, part)
        guard let shard = wm[name] else { throw NSError(domain: "ExpertSource", code: 1) }
        let (meta, dataStart) = try header(shard)
        guard let t = meta[name] else { throw NSError(domain: "ExpertSource", code: 2) }
        let nExp = t.shape[0]
        let stride = (t.end - t.begin) / nExp
        let offset = dataStart + t.begin + e * stride
        let buf = UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 16)
        let n = pread(fd(shard), buf, stride, off_t(offset))
        if n != stride { buf.deallocate(); throw NSError(domain: "ExpertSource", code: 3) }
        let restShape = [1] + Array(t.shape.dropFirst())
        return MLXArray(rawPointer: buf, restShape, dtype: ExpertSource.dtype(t.dtype)) {
            buf.deallocate()
        }
    }
}

