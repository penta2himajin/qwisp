import Foundation
import MLX

/// QwispCore — Swift 版エンジンの基盤。まずは mlx-swift の疎通確認用の最小 API。
public enum QwispCore {
    /// mlx-swift が動作し GPU 演算が通るかの smoke。
    public static func smoke() -> String {
        let a = MLXArray(converting: [1.0, 2.0, 3.0, 4.0])
        let b = (a * 2.0).sum()
        b.eval()
        return "mlx-swift OK: sum(2*[1..4]) = \(b.item(Float.self))  device=\(Device.defaultDevice())"
    }
}
