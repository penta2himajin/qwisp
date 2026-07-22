import Foundation
import Tokenizers
import Hub

// Server-layer tokenizer (productization step 5b). text/messages ↔ token ids for the
// token-id backend (Tell). Wraps swift-transformers `Tokenizer` + the model's
// chat_template.jinja — which on this model lives in a SEPARATE file, not inside
// tokenizer_config.json, so applyChatTemplate must be handed the template explicitly.
struct QwispTokenizer {
    let tokenizer: Tokenizer
    let chatTemplate: String?

    init(modelDir: String) async throws {
        let url = URL(fileURLWithPath: modelDir)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: url)
        let tpl = url.appendingPathComponent("chat_template.jinja")
        self.chatTemplate = try? String(contentsOf: tpl, encoding: .utf8)
    }

    /// Token ids that end generation: EOS + Qwen chat turn-end (<|im_end|>).
    var stopTokenIds: [Int] {
        var ids: [Int] = []
        if let e = tokenizer.eosTokenId { ids.append(e) }
        if let im = tokenizer.convertTokenToId("<|im_end|>") { ids.append(im) }
        return Array(Set(ids))
    }

    /// Encode raw text → token ids.
    func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Decode token ids → user-facing text (special tokens stripped).
    func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }

    /// Render chat messages (+ optional tool specs) → token ids (chat_template + generation prompt).
    /// Messages/tools are `[String: any Sendable]` so they can carry tool_calls, tool results, and
    /// arbitrary tool JSON schemas through to the Jinja template. `additionalContext` feeds extra
    /// Jinja variables (issue #77: chat_template_kwargs, e.g. enable_thinking=false); nil renders
    /// byte-identically to the previous signature.
    func render(messages: [[String: any Sendable]], tools: [[String: any Sendable]]? = nil,
                addGenerationPrompt: Bool = true,
                additionalContext: [String: any Sendable]? = nil) throws -> [Int] {
        let ct: ChatTemplateArgument? = chatTemplate.map { .literal($0) }
        return try tokenizer.applyChatTemplate(messages: messages, chatTemplate: ct,
                                               addGenerationPrompt: addGenerationPrompt, truncation: false,
                                               maxLength: nil, tools: tools,
                                               additionalContext: additionalContext)
    }
}

/// Incremental detokenizer for the streaming loops. The server re-decoded the FULL
/// output on every token (O(n²) — measured as the 14-25% per-token decode tax,
/// HANDOFF 2026-07-22); byte-level BPE decode is byte concatenation, so a finalized
/// prefix may be cut wherever the decoded tail ends on a codepoint boundary. A tail
/// whose decode ends in U+FFFD is a dangling multi-byte sequence — it stays pending
/// until later tokens complete it (a genuine mid-tail U+FFFD finalizes as-is: decode
/// is deterministic, later tokens cannot repair it).
/// Contract (locked by TOKTEST stream_detok_stepwise_equals_full): after n pushes the
/// returned text == decode(first n ids), so swapping it into the loops is
/// byte-identical to the old full re-decode by induction.
struct StreamDetok {
    let decode: ([Int]) -> String
    private var stable = ""          // finalized text — never changes retroactively
    private var pending: [Int] = []  // ids not yet safely cut
    init(decode: @escaping ([Int]) -> String) { self.decode = decode }
    /// Finalized text: append-only, never ends in a dangling U+FFFD. Streaming deltas
    /// MUST be derived from THIS, not from the full view — a view's trailing
    /// replacement char gets rewritten by the next token, and a prefix-guarded delta
    /// stream then stalls (streamSSE suppressed every later delta once `sent` held a
    /// U+FFFD — the OpenCode mid-stream stall, 2026-07-22) or duplicates
    /// (runGeneration re-sent the full text).
    var finalized: String { stable }
    /// Full text so far (== decode(all ids pushed)) — final-result use only.
    var text: String { pending.isEmpty ? stable : stable + decode(pending) }
    /// Push one id; returns the full text so far (== decode(all ids pushed)).
    @discardableResult
    mutating func push(_ id: Int) -> String {
        pending.append(id)
        let tail = decode(pending)
        if tail.last == "\u{FFFD}" { return stable + tail }   // dangling bytes — hold the cut
        stable += tail
        pending.removeAll(keepingCapacity: true)
        return stable
    }

    /// Pure self-check (COMPTEST, no model): a fake byte-level "tokenizer" whose ids map
    /// to UTF-8 byte fragments (🚀 split across two ids) — stepwise views must equal the
    /// full fake decode at every step, and the pending window must drain.
    static func selfCheck() -> [(String, Bool)] {
        let frags: [[UInt8]] = [
            Array("a".utf8),                       // 0: plain ASCII
            [0xF0, 0x9F],                          // 1: first half of 🚀
            [0x9A, 0x80],                          // 2: second half of 🚀
            Array("日本語".utf8),                   // 3: complete multi-byte run
            [0xE2],                                // 4: first byte of ✓
            [0x9C, 0x93],                          // 5: rest of ✓
            Array("!".utf8),                       // 6
        ]
        func fakeDecode(_ ids: [Int]) -> String {
            String(decoding: ids.flatMap { frags[$0] }, as: UTF8.self)
        }
        let seq = [0, 1, 2, 3, 4, 5, 6, 0]
        var d = StreamDetok(decode: fakeDecode)
        var stepwise = true
        for k in 0 ..< seq.count {
            stepwise = stepwise && (d.push(seq[k]) == fakeDecode(Array(seq[0 ... k])))
        }
        var d2 = StreamDetok(decode: fakeDecode)
        let mid = seq.prefix(4).map { d2.push($0) }.last ?? ""
        return [
            ("stepwise_equals_full", stepwise),
            ("split_rocket_rendered", mid.contains("🚀") && mid.contains("日本語")),
            ("final_text", d.push(6) == fakeDecode(seq + [6])),
        ]
    }
}
