import Foundation
import QwispCore

// qwisp benchtest — community benchmark for the call-for-testers workflow.
//
// Runs a fixed, deterministic (greedy) benchmark on the local machine and prints a
// paste-able markdown block: environment (chip / RAM / macOS / SSD read probe / tier)
// + per-prompt TTFT, decode rate, and a free-run stability signal. On streaming tiers
// (<32GB → bolt default) it also measures the strict (--lossless) pair so the table
// shows both modes. Progress goes to stderr; ONLY the markdown block goes to stdout,
// so `qwisp benchtest > report.md` captures a clean report.

// ── environment ──────────────────────────────────────────────────────────────

private func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    guard size > 0 else { return "?" }
    var buf = [CChar](repeating: 0, count: size)
    sysctlbyname(name, &buf, &size, nil, 0)
    return String(cString: buf)
}

private func runTool(_ path: String, _ args: [String]) -> String? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = Pipe()
    guard (try? p.run()) != nil else { return nil }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    return String(data: data, encoding: .utf8)
}

/// GPU core count via IORegistry — the brand string alone is ambiguous (e.g. "Apple M1
/// Max" ships as 24- or 32-core GPU), and decode is GPU-bound.
private func gpuCoreCount() -> Int? {
    guard let out = runTool("/usr/sbin/ioreg", ["-rc", "AGXAccelerator", "-d", "1"]) else { return nil }
    for line in out.split(separator: "\n") where line.contains("\"gpu-core-count\"") {
        if let v = line.split(separator: "=").last.flatMap({ Int($0.trimmingCharacters(in: .whitespaces)) }) {
            return v
        }
    }
    return nil
}

/// AC vs battery — MacBooks throttle GPU clocks on battery, a major variance source in
/// community-reported numbers.
private func powerSource() -> String {
    guard let out = runTool("/usr/bin/pmset", ["-g", "batt"]) else { return "?" }
    if out.contains("AC Power") { return "AC" }
    if out.contains("Battery Power") { return "battery" }
    return "?"
}

private func thermalStateName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "?"
    }
}

/// Total size of the volume holding the model — 256GB configs have fewer NAND channels
/// (slower sequential reads, the MacBook-floor tier), so it contextualizes the SSD probe.
private func diskSizeGB(modelDir: String) -> Int? {
    guard let attrs = try? FileManager.default.attributesOfFileSystem(forPath: modelDir),
          let size = attrs[.systemSize] as? Int else { return nil }
    return Int((Double(size) / 1e9).rounded())
}

/// Sequential read bandwidth of the model's biggest shard with the page cache bypassed
/// (F_NOCACHE) — the honest "what tier is this NAND" number (MacBook-class ≈ 1.5 GB/s,
/// desktop-class ≈ 5+ GB/s). Reads ≤256MB in 8MB chunks; nil if the model dir is odd.
private func ssdReadGBs(modelDir: String) -> Double? {
    let fm = FileManager.default
    guard let files = try? fm.contentsOfDirectory(atPath: modelDir) else { return nil }
    var best: (path: String, size: Int)? = nil
    for f in files where f.hasSuffix(".safetensors") {
        let p = modelDir + "/" + f
        let sz = ((try? fm.attributesOfItem(atPath: p))?[.size] as? Int) ?? 0
        if sz > (best?.size ?? 0) { best = (p, sz) }
    }
    guard let target = best, target.size > (64 << 20) else { return nil }
    let fd = open(target.path, O_RDONLY)
    guard fd >= 0 else { return nil }
    defer { close(fd) }
    _ = fcntl(fd, F_NOCACHE, 1)
    let chunk = 8 << 20
    let total = min(target.size, 256 << 20)
    let buf = UnsafeMutableRawPointer.allocate(byteCount: chunk, alignment: 4096)
    defer { buf.deallocate() }
    let t0 = DispatchTime.now()
    var done = 0
    while done < total {
        let n = pread(fd, buf, min(chunk, total - done), off_t(done))
        if n <= 0 { break }
        done += n
    }
    let secs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e9
    guard secs > 0, done > (16 << 20) else { return nil }
    return Double(done) / 1e9 / secs
}

// ── stability signal ─────────────────────────────────────────────────────────

/// Distinct n-gram ratio over the tail of the generated token ids. Greedy repetition
/// loops (the failure mode worth catching on real hardware) collapse the ratio toward
/// loopLen/window; healthy text stays near 1.0. Pure — self-checked below.
func stabilityRatio(_ ids: [Int], n: Int = 8, window: Int = 256) -> Double {
    let tail = Array(ids.suffix(window))
    guard tail.count >= n * 2 else { return 1.0 }
    var seen = Set<[Int]>()
    var total = 0
    for i in 0 ... (tail.count - n) {
        seen.insert(Array(tail[i ..< i + n])); total += 1
    }
    return Double(seen.count) / Double(total)
}

func stabilitySelfCheck() -> Bool {
    let unique = stabilityRatio(Array(0 ..< 300))                                   // no repeats → ~1.0
    let loop = stabilityRatio((0 ..< 20).flatMap { _ in Array(0 ..< 16) })          // 16-cycle → ~0.06
    let short = stabilityRatio([1, 2, 3])                                           // too short → benign 1.0
    return unique > 0.9 && loop < 0.5 && short == 1.0
}

// ── bench core ───────────────────────────────────────────────────────────────

private struct BenchRow {
    let name: String
    let mode: String
    let ttftMs: Int
    let tokPerSec: Double
    let tokens: Int
    let stable: Bool
    let tail: String
}

private func note(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

private func benchRun(name: String, mode: String, prompt: String, maxTokens: Int,
                      tokenizer: QwispTokenizer, backend: any LLMBackend) async -> BenchRow? {
    guard let promptIds = try? tokenizer.render(messages: [["role": "user", "content": prompt]])
    else { return nil }
    note("[benchtest] \(name) (\(mode), \(maxTokens) tok max) …")
    let opts = GenerateOptions(maxTokens: maxTokens, stopTokens: tokenizer.stopTokenIds)
    var outIds: [Int] = []
    let t0 = Date()
    var tFirst: Date? = nil
    for await id in backend.generate(promptIds, options: opts) {
        if tFirst == nil { tFirst = Date() }
        // Mirror runGeneration's defensive stop/max guards: the spec loop streams tokens
        // in accepted-draft/chain granularity, so the raw stream can overshoot EOS and
        // maxTokens — measuring past EOS times degenerate free-running text.
        if tokenizer.stopTokenIds.contains(id) { break }
        outIds.append(id)
        if outIds.count >= maxTokens { break }
    }
    let tEnd = Date()
    guard let tf = tFirst, outIds.count > 1 else { return nil }
    let rate = Double(outIds.count - 1) / max(1e-9, tEnd.timeIntervalSince(tf))
    let text = tokenizer.decode(outIds)
    let tail = String(text.suffix(70)).replacingOccurrences(of: "\n", with: " ")
    return BenchRow(name: name, mode: mode,
                    ttftMs: Int(tf.timeIntervalSince(t0) * 1000),
                    tokPerSec: rate, tokens: outIds.count,
                    stable: stabilityRatio(outIds) >= 0.5, tail: tail)
}

/// Entry point for `qwisp benchtest`. Returns the markdown report (stdout).
func runBenchtest(modelDir: String) async -> String {
    guard stabilitySelfCheck() else { return "[benchtest] FATAL: stability self-check failed" }
    let tok: QwispTokenizer
    do { tok = try await QwispTokenizer(modelDir: modelDir) }
    catch { return "[benchtest] tokenizer load failed: \(error)" }
    note("[benchtest] SSD read probe …")
    let ssd = ssdReadGBs(modelDir: modelDir)
    note("[benchtest] loading Seedless engine (loads the model) …")
    let backend: SeedlessBackend
    do { backend = try SeedlessBackend(modelDir: modelDir) }
    catch { return "[benchtest] engine load failed: \(error)" }

    let dc = DeviceCalibration.recommend()
    let isStreaming = dc.C > 0 && dc.C < 256
    let lossless = backend.lossless
    let tierDesc = isStreaming
        ? "streaming C=\(dc.C) → \(lossless ? "strict (lossless forced)" : "bolt (near-lossless) default")"
        : "resident C=\(dc.C) → strict (lossless)"

    // Warmup: page-cache/GPU warm + (streaming) the one-time bolt calibration, so the
    // timed rows measure steady-state, not first-request setup.
    note("[benchtest] warmup (16 tok; includes one-time bolt calibration on streaming tiers) …")
    // Deliberately a TRIVIAL prompt: exercises the worst-case calibration path
    // (BoltServe's minimum-calib-corpus guard) instead of flattering the numbers
    // with an on-topic calib.
    _ = await benchRun(name: "warmup", mode: "-", prompt: "Say hello.", maxTokens: 16,
                       tokenizer: tok, backend: backend)

    let defaultMode = isStreaming && !lossless ? "bolt" : "strict"
    let prompts: [(String, String, Int)] = [
        ("code-256", "Write a Python function that merges two sorted lists into one sorted list, then explain its time complexity briefly.", 256),
        ("nl-256", "Explain why the sky appears blue in plain English, in about three paragraphs.", 256),
        ("long-600", "Write a detailed step-by-step explanation of how quicksort works, with a Python implementation.", 600),
    ]
    let tWall = Date()
    var rows: [BenchRow] = []
    for (name, p, n) in prompts {
        if let r = await benchRun(name: name, mode: defaultMode, prompt: p, maxTokens: n,
                                  tokenizer: tok, backend: backend) { rows.append(r) }
    }
    // Streaming tiers: measure the strict (--lossless) pair on the first prompt, so the
    // report carries the bolt-vs-strict speed ratio for this hardware.
    if isStreaming && !lossless {
        backend.losslessForced = true
        if let r = await benchRun(name: "code-256", mode: "strict", prompt: prompts[0].1,
                                  maxTokens: 256, tokenizer: tok, backend: backend) { rows.append(r) }
        backend.losslessForced = false
    }
    let wall = Int(Date().timeIntervalSince(tWall))

    // ── markdown report ──────────────────────────────────────────────────────
    var md: [String] = []
    md.append("### qwisp benchtest v\(Config.version)")
    md.append("")
    md.append("| env | |")
    md.append("|---|---|")
    let gpu = gpuCoreCount().map { " (\($0)-core GPU)" } ?? ""
    md.append("| chip | \(sysctlString("machdep.cpu.brand_string"))\(gpu) |")
    md.append("| RAM | \(Int((Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0).rounded())) GB |")
    md.append("| macOS | \(ProcessInfo.processInfo.operatingSystemVersionString) |")
    md.append("| disk | \(diskSizeGB(modelDir: modelDir).map { "\($0) GB" } ?? "n/a"), read \(ssd.map { String(format: "%.1f GB/s", $0) } ?? "n/a") |")
    md.append("| power | \(powerSource()), thermal \(thermalStateName()) |")
    md.append("| model | \(URL(fileURLWithPath: modelDir).lastPathComponent) |")
    md.append("| tier | \(tierDesc) |")
    md.append("")
    md.append("| test | mode | TTFT | decode | tokens | stability |")
    md.append("|---|---|---|---|---|---|")
    for r in rows {
        md.append(String(format: "| %@ | %@ | %.1fs | %.1f tok/s | %d | %@ |",
                         r.name, r.mode, Double(r.ttftMs) / 1000.0, r.tokPerSec, r.tokens,
                         r.stable ? "ok" : "**LOOPY**"))
    }
    if let long = rows.first(where: { $0.name == "long-600" }) {
        md.append("")
        md.append("tail of long-600: `…\(long.tail)`")
    }
    md.append("")
    md.append("_greedy/deterministic; TTFT includes prefill; \(wall)s total. Post results: https://github.com/penta2himajin/qwisp/issues (call-for-testers)._")
    return md.joined(separator: "\n")
}
