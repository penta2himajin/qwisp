import Foundation
import Metal

// qwisp simulate — emulate a small-RAM Mac on a big one (issue #71).
//
// Two ballasts, derived from one target (overridable):
//  - GPU: storageModePrivate MTLBuffers kept "recently used" by a tiny RMW touch
//    kernel every ~10ms, so the driver's working-set eviction (and the wired-memory
//    squeeze on unified memory) lands on the process under test. This alone
//    reproduced the #69 strict-streaming field collapse (29.3 → 1.9 tok/s).
//  - RAM: incompressible pages (a random 16KB page tiled — per-page WKdm sees random
//    bytes, and macOS has no page dedup) swept slowly so they stay resident.
//
// Foreground; ^C releases everything instantly. Run the workload in another terminal.
enum Simulate {
    /// Measured recommendedMaxWorkingSetSize / RAM on small Macs (16GB→10.9, 18GB→12.3).
    static let gpuBudgetFraction = 0.68
    /// Rough OS + apps working set on a small Mac — what the target machine does NOT have free.
    static let osReserveGB = 4.0

    static func run(args: [String]) -> Int32 {
        setbuf(stdout, nil)   // status lines must survive ^C / kill even when redirected
        var targetGB: Double? = nil
        var gpuOverride: Double? = nil
        var ramOverride: Double? = nil
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--gpu-gb": if i + 1 < args.count { gpuOverride = Double(args[i + 1]); i += 1 }
            case "--ram-gb": if i + 1 < args.count { ramOverride = Double(args[i + 1]); i += 1 }
            default:
                let a = args[i].lowercased().hasSuffix("gb") ? String(args[i].lowercased().dropLast(2)) : args[i]
                targetGB = Double(a)
            }
            i += 1
        }
        guard targetGB != nil || gpuOverride != nil || ramOverride != nil else {
            print("usage: qwisp simulate <N>gb [--gpu-gb X] [--ram-gb Y]   (e.g. qwisp simulate 16gb)")
            return 1
        }
        guard let dev = MTLCreateSystemDefaultDevice() else {
            FileHandle.standardError.write(Data("no Metal device\n".utf8)); return 1
        }
        let limitGB = Double(dev.recommendedMaxWorkingSetSize) / 1e9
        let physGB = Double(ProcessInfo.processInfo.physicalMemory) / 1e9

        // GPU ballast: shrink this machine's GPU budget to the target machine's.
        var gpuGB = gpuOverride ?? (targetGB.map { Swift.max(0, limitGB - $0 * gpuBudgetFraction) } ?? 0)
        // RAM trim: whatever the GPU ballast (which is wired RAM on unified memory)
        // doesn't already take, up to (this RAM − target free) where target free = N − reserve.
        let targetFree = targetGB.map { Swift.max(2, $0 - osReserveGB) }
        var ramGB = ramOverride ?? (targetFree.map { Swift.max(0, physGB - gpuGB - $0 - osReserveGB) } ?? 0)
        gpuGB = Swift.min(gpuGB, physGB - 6)   // sanity clamp: never wire the whole machine
        ramGB = Swift.min(ramGB, physGB - gpuGB - 6)

        if let t = targetGB {
            print(String(format: "[simulate] target %.0fGB Mac on a %.0fGB machine (GPU budget %.1f→%.1fGB)",
                         t, physGB, limitGB, limitGB - gpuGB))
        }
        print(String(format: "[simulate] ballast: GPU %.1fGB (wired, touched ~10ms) + RAM %.1fGB (incompressible, swept)", gpuGB, ramGB))
        print("[simulate] run your workload (e.g. `qwisp benchtest`) in another terminal; ^C here releases")

        // ── RAM ballast ──
        var ramChunks: [Data] = []
        if ramGB > 0.1 {
            var page = Data(count: 16384)
            page.withUnsafeMutableBytes { b in _ = SecRandomCopyBytes(kSecRandomDefault, 16384, b.baseAddress!) }
            let chunk = 256 << 20
            let tile = Data(repeating: 0, count: 0) + Data((0 ..< chunk / 16384).flatMap { _ in page })
            var alloc = 0
            let target = Int(ramGB * Double(1 << 30))
            while alloc < target {
                ramChunks.append(tile)   // Data is CoW — force unique pages:
                ramChunks[ramChunks.count - 1].withUnsafeMutableBytes { b in
                    b.storeBytes(of: UInt8(alloc & 0xff), toByteOffset: 0, as: UInt8.self)
                }
                alloc += chunk
            }
            print(String(format: "[simulate] RAM ballast resident: %.1fGB", Double(alloc) / Double(1 << 30)))
        }

        // ── GPU ballast ──
        var bufs: [MTLBuffer] = []
        var pso: MTLComputePipelineState? = nil
        let queue = dev.makeCommandQueue()!
        if gpuGB > 0.1 {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void touch(device uint* buf [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
                buf[gid * 4096] = buf[gid * 4096] + 1;
            }
            """
            guard let lib = try? dev.makeLibrary(source: src, options: nil),
                  let f = lib.makeFunction(name: "touch"),
                  let p = try? dev.makeComputePipelineState(function: f) else {
                FileHandle.standardError.write(Data("[simulate] Metal pipeline failed\n".utf8)); return 1
            }
            pso = p
            let chunk = 1 << 30
            for i in 0 ..< Int(gpuGB.rounded()) {
                guard let b = dev.makeBuffer(length: chunk, options: .storageModePrivate) else {
                    print("[simulate] GPU alloc stopped at \(i) GiB"); break
                }
                bufs.append(b)
            }
            print("[simulate] GPU ballast resident: \(bufs.count)GiB private")
        }

        // ── hold: touch GPU pages every ~10ms; sweep a RAM slice each pass ──
        var ramSweep = 0
        while true {
            if let pso, !bufs.isEmpty {
                let cb = queue.makeCommandBuffer()!
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pso)
                let pages = (1 << 30) / 16384
                for b in bufs {
                    enc.setBuffer(b, offset: 0, index: 0)
                    enc.dispatchThreads(MTLSize(width: pages, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
                }
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
            }
            if !ramChunks.isEmpty {
                var acc: UInt8 = 0
                ramChunks[ramSweep % ramChunks.count].withUnsafeBytes { b in
                    for off in stride(from: 0, to: b.count, by: 16384) { acc &+= b[off] }
                }
                ramSweep += 1
                _ = acc
            }
            usleep(10_000)
        }
    }
}
