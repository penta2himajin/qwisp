import Foundation

/// Tokenizer self-test (GPU-free integration check; needs the model's tokenizer files).
/// Mirrors the RAWTESTS pattern: prints "TOKTEST passed/total", nonzero-exit driven by the
/// shell wrapper on any FAIL.
func runTokenizerSelftest(modelDir: String) async -> String {
    var passed = 0, total = 0
    var lines: [String] = []
    func check(_ name: String, _ body: () throws -> Bool) {
        total += 1
        do {
            let ok = try body()
            lines.append("[tok-test] \(name): \(ok ? "PASS" : "FAIL")")
            if ok { passed += 1 }
        } catch {
            lines.append("[tok-test] \(name): FAIL(\(error))")
        }
    }

    let tok: QwispTokenizer
    do {
        tok = try await QwispTokenizer(modelDir: modelDir)
    } catch {
        return "[tok-test] load: FAIL(\(error))\nTOKTEST 0/1"
    }

    check("encode_decode_roundtrip") {
        let s = "Hello, world! def foo(x): return x + 1"
        let ids = tok.encode(s)
        let dec = tok.decode(ids)
        return !ids.isEmpty && dec.contains("Hello, world!") && dec.contains("def foo")
    }
    check("chat_template_nonempty") {
        let ids = try tok.render(messages: [["role": "user", "content": "Hi there"]])
        return !ids.isEmpty
    }
    check("chat_template_preserves_content") {
        let ids = try tok.render(messages: [["role": "user", "content": "MAGICWORD42"]])
        return tok.decode(ids).contains("MAGICWORD42")
    }

    return lines.joined(separator: "\n") + "\nTOKTEST \(passed)/\(total)"
}
