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
        let n = pread(fd(shard), buf, stride, off_t(offset))
        if n != stride { throw NSError(domain: "ExpertSource", code: 3) }
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

public enum ExpertSourceValidation {
    public static func run(modelDir: String) throws -> String {
        let store = try WeightStore(modelDir: modelDir)
        store.residentNonExperts()
        let src = try ExpertSource(modelDir: modelDir)
        // layer 0,3,39 × proj × part × expert {0,7,255} を resident と bit 比較
        var worst: Float = 0
        var checks = 0
        for layer in [0, 3, 39] {
            for proj in ExpertSource.projs {
                for part in ExpertSource.parts {
                    let full = store.req("\(ExpertSource.prefix).\(layer).mlp.switch_mlp.\(proj).\(part)")
                    for e in [0, 7, 255] {
                        let s = try src.slice(layer, proj, part, e)
                        let ref = full[e ..< (e + 1)]
                        // uint32 は ==、float は abs 差
                        let d: Float
                        if s.dtype == .uint32 {
                            d = MLX.sum(MLX.notEqual(s, ref)).item(Int.self) > 0 ? 1 : 0
                        } else {
                            d = MLX.max(MLX.abs(s.asType(.float32) - ref.asType(.float32))).item(Float.self)
                        }
                        worst = max(worst, d); checks += 1
                    }
                }
            }
        }
        let ok = worst == 0
        return String(format: "[S1] ExpertSource pread スライス: %d 件検証 worst|Δ|=%.3e  %@",
                      checks, worst, ok ? "OK ✅ resident と bit一致" : "MISMATCH ❌")
    }
}
