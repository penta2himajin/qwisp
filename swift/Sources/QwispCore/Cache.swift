import Foundation
import MLX

/// full-attention の KV キャッシュ。keys/values は [B, kvHeads, L, headDim]。
public final class KVCache {
    public var keys: MLXArray?
    public var values: MLXArray?
    public private(set) var offset: Int = 0

    public init() {}

    /// k,v を追記し、全 keys/values を返す。offset は追記前の位置（RoPE 用に別途参照）。
    public func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        if let kk = keys, let vv = values {
            keys = MLX.concatenated([kk, k], axis: 2)
            values = MLX.concatenated([vv, v], axis: 2)
        } else {
            keys = k; values = v
        }
        offset += k.dim(2)
        return (keys!, values!)
    }

    /// 末尾 n 位置を巻き戻す（reject 用）。
    public func trim(_ n: Int) {
        guard let k = keys, let v = values else { return }
        let L = k.dim(2)
        if n >= L { keys = nil; values = nil; offset = 0; return }
        keys = k[0..., 0..., 0 ..< (L - n), 0...]
        values = v[0..., 0..., 0 ..< (L - n), 0...]
        offset -= n
    }
}

/// GatedDeltaNet の cache: conv 状態(直近 K-1 トークンの qkv) と recurrent 状態。
public final class GDNCache {
    public var convState: MLXArray?   // [B, K-1, convDim]
    public var recState: MLXArray?    // [B, Hv, Dv, Dk]
    public init() {}
}

/// 1 層分の cache（linear 層は gdn、full 層は kv を使う）。
public final class LayerCache {
    public let kv = KVCache()
    public let gdn = GDNCache()
    public init() {}

    /// この層 cache が保持する全状態（毎 step eval して lazy グラフの増殖を防ぐ）。
    public var stateArrays: [MLXArray] {
        [kv.keys, kv.values, gdn.convState, gdn.recState].compactMap { $0 }
    }

    // reject 用 snapshot/restore。GDN(線形)は state 参照を退避、KV(full)は trim で巻き戻す。
    public struct Snapshot { let conv: MLXArray?; let rec: MLXArray? }
    public func snapshot() -> Snapshot { Snapshot(conv: gdn.convState, rec: gdn.recState) }
    public func restore(_ s: Snapshot, isLinear: Bool, trim n: Int) {
        if isLinear { gdn.convState = s.conv; gdn.recState = s.rec } else { kv.trim(n) }
    }
}
