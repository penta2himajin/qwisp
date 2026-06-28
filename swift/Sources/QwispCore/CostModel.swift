import Foundation

/// device 別 decode 速度の予測式（calibration の中核）。engine が起動時実測した係数から
/// (maxK, accept, IO) を入れて tok/s を予測し、最適 C/maxK/mode を選ぶ。
///
/// 形:  tok/s = (1 + p) / ( forward_ms(D+1) + io_ms ) * 1000
///   forward_ms(L) = a + b·L      … forward-cost で実測(a=固定床, b=marginal, device 依存)
///   io_ms         = miss(C)·bytes_expert / SSD_BW … 8GB streaming のみ。resident は ~0
///   p             = 期待受理トークン/step（draft accept × maxK 依存）
public struct CostModel {
    public let a: Double   // forward 固定費 (ms/forward)
    public let b: Double   // forward marginal (ms/token)

    public init(a: Double, b: Double) { self.a = a; self.b = b }

    /// forward_ms(L) = a + b·L
    public func forwardMs(_ L: Int) -> Double { a + b * Double(L) }

    /// 投機 decode の予測 tok/s。draftLen=D, acceptedPerStep=p（commit=1+p, verify は D+1 token forward）。
    public func tokPerSec(draftLen D: Int, acceptedPerStep p: Double, ioMsPerStep: Double = 0) -> Double {
        let stepMs = forwardMs(D + 1) + ioMsPerStep
        return (1.0 + p) / stepMs * 1000.0
    }

    /// (L, ms) 点群から a,b を最小二乗 fit（forward-cost ベンチの出力を食わせる）。
    public static func fit(_ points: [(L: Int, ms: Double)]) -> CostModel {
        let n = Double(points.count)
        let sx = points.reduce(0.0) { $0 + Double($1.L) }
        let sy = points.reduce(0.0) { $0 + $1.ms }
        let sxx = points.reduce(0.0) { $0 + Double($1.L * $1.L) }
        let sxy = points.reduce(0.0) { $0 + Double($1.L) * $1.ms }
        let denom = n * sxx - sx * sx
        let b = denom != 0 ? (n * sxy - sx * sy) / denom : 0
        let a = (sy - b * sx) / n
        return CostModel(a: a, b: b)
    }
}
