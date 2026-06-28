import Foundation
import Darwin

/// device calibration の SSD probe: expert streaming の実効 read 帯域を F_NOCACHE(キャッシュ回避=真の
/// SSD BW)で実測。3 経路を比較し engine の streaming 設計を確定:
///  (1) pread gather 順（実 streaming パターン: 散在 expert を順次 pread）
///  (2) pread sorted 順（offset 昇順=sequential 寄り。random ペナルティを切り分け）
///  (3) mmap+MAP_NOCACHE（page-fault 経路。研究の「16KB fault 断片化」リスクを実測）
/// 現状エンジンは (1) を採用済。pread≥mmap を確認＋実 BW を calibration 入力に。
public enum DeviceProbe {
    public static func run(modelDir: String) throws -> String {
        let source = try ExpertSource(modelDir: modelDir)
        try source.warm()
        let nLayers = max(1, Tell.envInt("QWISP_PROBE_LAYERS", 2))
        let nExperts = max(1, Tell.envInt("QWISP_PROBE_EXPERTS", 64))

        // working set: nLayers × nExperts × 9 tensor の byte 範囲を収集
        var ranges: [(path: String, offset: Int, length: Int)] = []
        for layer in 0 ..< nLayers {
            for e in 0 ..< nExperts {
                for proj in ExpertSource.projs {
                    for part in ExpertSource.parts {
                        let r = try source.expertByteRange(layer, proj, part, e)
                        ranges.append((r.shardPath, r.offset, r.length))
                    }
                }
            }
        }
        let totalBytes = ranges.reduce(0) { $0 + $1.length }
        let maxLen = ranges.map { $0.length }.max() ?? (1 << 20)
        let totalMB = Double(totalBytes) / 1e6

        func openNoCache(_ path: String) -> Int32 {
            let fd = open(path, O_RDONLY)
            if fd >= 0 { _ = fcntl(fd, F_NOCACHE, 1) }   // macOS: buffer cache 回避=disk へ
            return fd
        }
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

        // best-effort cold 化: working set のページを madvise(MADV_DONTNEED)で追い出す（sudo purge 不可環境用）。
        // 効けば cold SSD BW、効かねば warm(page cache)。BW>8GB/s は SSD 物理上限超ゆえ warm と判定。
        let shards = Set(ranges.map { $0.path })
        for path in shards {
            let fd = open(path, O_RDONLY); if fd < 0 { continue }
            var st = Darwin.stat(); _ = fstat(fd, &st); let len = Int(st.st_size)
            if let p = mmap(nil, len, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED {
                _ = madvise(p, len, MADV_DONTNEED); munmap(p, len)
            }
            close(fd)
        }

        // --- (1)(2) pread: gather 順 / sorted 順 ---
        func benchPread(_ rs: [(path: String, offset: Int, length: Int)]) -> Double {
            var fds: [String: Int32] = [:]
            for r in rs where fds[r.path] == nil { fds[r.path] = openNoCache(r.path) }
            defer { for (_, fd) in fds { close(fd) } }
            let buf = UnsafeMutableRawPointer.allocate(byteCount: maxLen, alignment: 16384)
            defer { buf.deallocate() }
            let t = now()
            for r in rs { _ = pread(fds[r.path]!, buf, r.length, off_t(r.offset)) }
            return Double(now() - t) / 1e9
        }
        let sPread = benchPread(ranges)
        let sSorted = benchPread(ranges.sorted { $0.offset < $1.offset })

        // --- (3) mmap + MAP_NOCACHE: gather 順に touch（page-in 強制）---
        func benchMmap(_ rs: [(path: String, offset: Int, length: Int)]) -> Double? {
            var maps: [String: (UnsafeMutableRawPointer, Int)] = [:]
            for r in rs where maps[r.path] == nil {
                let fd = open(r.path, O_RDONLY); if fd < 0 { return nil }
                var st = Darwin.stat(); _ = fstat(fd, &st); let len = Int(st.st_size)
                let p = mmap(nil, len, PROT_READ, MAP_PRIVATE | MAP_NOCACHE, fd, 0)
                close(fd)
                if p == MAP_FAILED { return nil }
                maps[r.path] = (p!, len)
            }
            defer { for (_, m) in maps { munmap(m.0, m.1) } }
            let t = now()
            var acc: UInt64 = 0
            for r in rs {
                let base = maps[r.path]!.0.advanced(by: r.offset).assumingMemoryBound(to: UInt8.self)
                var i = 0
                while i < r.length { acc &+= UInt64(base[i]); i += 4096 }   // page ごとに touch
            }
            if acc == 0xDEADBEEF { print("") }   // DCE 防止
            return Double(now() - t) / 1e9
        }
        let sMmap = benchMmap(ranges)

        func bw(_ s: Double) -> Double { totalMB / s / 1000 }   // GB/s
        let warm = bw(sPread) > 8.0   // SSD 物理上限(~7GB/s)超 = page cache warm
        let warmTag = warm
            ? "  ⚠️ BW>8GB/s = page cache WARM(SSD 実測でない)。cold には reboot/別device、または起動時=モデル未キャッシュで自然に cold"
            : "  ✅ cold(SSD 実測値)"
        let mmapLine = sMmap.map {
            String(format: "  mmap(MAP_NOCACHE,gather)   : %.1f ms  %.2f GB/s  (vs pread %.2fx)",
                   $0 * 1000, bw($0), bw(sPread) / bw($0))
        } ?? "  mmap: N/A"
        return String(format: """
            [DeviceProbe SSD] working set %.0f MB (%d 層 × %d expert × 9 tensor, ~%.2f MB/expert-tensor 平均)
              pread gather 順(実 streaming): %.1f ms  %.2f GB/s  ← calibration 入力
              pread sorted 順(sequential)  : %.1f ms  %.2f GB/s  (random ペナルティ %.2fx)
            %@
            %@
              → random≈sorted なら 1.7MB chunk は large-block で OK / mmap≥pread なら断片化リスク無
            """,
            totalMB, nLayers, nExperts, totalMB / Double(ranges.count),
            sPread * 1000, bw(sPread),
            sSorted * 1000, bw(sSorted), bw(sSorted) / bw(sPread),
            mmapLine, warmTag)
    }
}
