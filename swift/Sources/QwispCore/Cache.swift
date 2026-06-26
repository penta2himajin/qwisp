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
}
