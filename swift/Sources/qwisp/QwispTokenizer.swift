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
    /// arbitrary tool JSON schemas through to the Jinja template.
    func render(messages: [[String: any Sendable]], tools: [[String: any Sendable]]? = nil) throws -> [Int] {
        let ct: ChatTemplateArgument? = chatTemplate.map { .literal($0) }
        return try tokenizer.applyChatTemplate(messages: messages, chatTemplate: ct,
                                               addGenerationPrompt: true, truncation: false,
                                               maxLength: nil, tools: tools)
    }
}
