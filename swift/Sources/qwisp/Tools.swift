import Foundation

// OpenAI function/tool calling. Qwen3.6's chat template renders `tools` and emits calls as
// `<tool_call><function=NAME><parameter=KEY>\nVALUE\n</parameter>…</function></tool_call>`.
// This file: the request/response tool types, a JSON passthrough value, the output parser, and
// the message conversion that feeds tool_calls / tool results back through the template.

// ── JSON passthrough (arbitrary tool schemas / arguments through Codable + the renderer) ──────
enum JSONValue: Codable {
    case null, bool(Bool), int(Int), double(Double), string(String)
    case array([JSONValue]), object([String: JSONValue])

    init(from d: Decoder) throws {
        let c = try d.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }        // before Int: true/false
        else if let i = try? c.decode(Int.self) { self = .int(i) }
        else if let x = try? c.decode(Double.self) { self = .double(x) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else if let o = try? c.decode([String: JSONValue].self) { self = .object(o) }
        else { throw DecodingError.dataCorruptedError(in: c, debugDescription: "unsupported JSON") }
    }
    func encode(to e: Encoder) throws {
        var c = e.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let x): try c.encode(x)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }
    // For the chat-template renderer, which takes [String: any Sendable].
    var sendable: any Sendable {
        switch self {
        case .null: return NSNull()
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let x): return x
        case .string(let s): return s
        case .array(let a): return a.map { $0.sendable }
        case .object(let o): return o.mapValues { $0.sendable }
        }
    }
}

// ── Request tool types ────────────────────────────────────────────────────────
struct Tool: Codable { let type: String?; let function: ToolFunction }
struct ToolFunction: Codable { let name: String; let description: String?; let parameters: JSONValue? }
struct ReqFunctionCall: Codable { let name: String; let arguments: String }   // OpenAI: arguments = JSON string
struct ReqToolCall: Codable { let id: String?; let type: String?; let function: ReqFunctionCall }

// ── Response tool types ───────────────────────────────────────────────────────
struct FunctionCall: Codable { let name: String; let arguments: String }      // arguments = JSON string
struct ToolCall: Codable { let id: String; let type: String; let function: FunctionCall }
struct ToolCallDelta: Codable { let index: Int; let id: String?; let type: String?; let function: FunctionCall }

// Convert request messages / tool specs into the `[String: any Sendable]` shape the template wants.
extension ChatMessage {
    var renderDict: [String: any Sendable] {
        var d: [String: any Sendable] = ["role": role]
        if let content { d["content"] = content }
        if let tool_calls {
            d["tool_calls"] = tool_calls.map { tc -> [String: any Sendable] in
                let fn: [String: any Sendable] = ["name": tc.function.name,
                                                  "arguments": ToolParse.argsToSendable(tc.function.arguments)]
                return ["id": tc.id ?? "", "type": tc.type ?? "function", "function": fn]
            }
        }
        if let tool_call_id { d["tool_call_id"] = tool_call_id }
        if let name { d["name"] = name }
        return d
    }
}

extension Tool {
    // The OpenAI tool object as a Sendable dict; the template dumps it with `tool | tojson`.
    var spec: [String: any Sendable] {
        guard let data = try? JSONEncoder().encode(self),
              let v = try? JSONDecoder().decode(JSONValue.self, from: data),
              case let .object(o) = v else { return [:] }
        return o.mapValues { $0.sendable }
    }
}

enum ToolParse {
    // Parse the model's <tool_call> blocks into OpenAI tool_calls; return the remaining text as content.
    static func parse(_ text: String) -> (content: String?, toolCalls: [ToolCall]) {
        guard text.contains("<tool_call>") else {
            let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return (t.isEmpty ? nil : text, [])
        }
        var calls: [ToolCall] = []
        for (i, block) in allMatches(text, #"<tool_call>(.*?)</tool_call>"#).enumerated() {
            guard let name = firstMatch(block, #"<function=([^>]+)>"#) else { continue }
            var args: [String: JSONValue] = [:]
            for (k, v) in paramPairs(block) { args[k] = coerce(v) }
            calls.append(ToolCall(id: "call_\(i)_\(UUID().uuidString.prefix(8))",
                                  type: "function",
                                  function: FunctionCall(name: name, arguments: encodeArgs(args))))
        }
        let stripped = replaceAll(text, #"<tool_call>.*?</tool_call>"#, "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (stripped.isEmpty ? nil : stripped, calls)
    }

    // Qwen emits parameter values as text; coerce scalars to JSON, else keep as string.
    static func coerce(_ s: String) -> JSONValue {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if t == "true" { return .bool(true) }
        if t == "false" { return .bool(false) }
        if t == "null" { return .null }
        if let i = Int(t) { return .int(i) }
        if let d = Double(t), !t.isEmpty { return .double(d) }
        return .string(t)
    }

    static func encodeArgs(_ args: [String: JSONValue]) -> String {
        let enc = JSONEncoder(); enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return (try? enc.encode(args)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    // Convert an OpenAI arguments JSON string back into a Sendable dict for the template's |items.
    static func argsToSendable(_ json: String) -> any Sendable {
        guard let data = json.data(using: .utf8),
              let v = try? JSONDecoder().decode(JSONValue.self, from: data) else { return [String: any Sendable]() }
        return v.sendable
    }

    // ── regex helpers (dotall) ──
    private static func re(_ p: String) -> NSRegularExpression {
        try! NSRegularExpression(pattern: p, options: [.dotMatchesLineSeparators])
    }
    private static func allMatches(_ s: String, _ p: String) -> [String] {
        let ns = s as NSString
        return re(p).matches(in: s, range: NSRange(location: 0, length: ns.length)).compactMap {
            $0.numberOfRanges > 1 ? ns.substring(with: $0.range(at: 1)) : nil
        }
    }
    private static func firstMatch(_ s: String, _ p: String) -> String? { allMatches(s, p).first }
    private static func replaceAll(_ s: String, _ p: String, _ r: String) -> String {
        let ns = s as NSString
        return re(p).stringByReplacingMatches(in: s, range: NSRange(location: 0, length: ns.length), withTemplate: r)
    }
    private static func paramPairs(_ block: String) -> [(String, String)] {
        let ns = block as NSString
        return re(#"<parameter=([^>]+)>\n?(.*?)\n?</parameter>"#)
            .matches(in: block, range: NSRange(location: 0, length: ns.length))
            .compactMap { m in
                guard m.numberOfRanges > 2 else { return nil }
                return (ns.substring(with: m.range(at: 1)), ns.substring(with: m.range(at: 2)))
            }
    }

    // Pure self-check (GPU-free, no model) — folded into COMPTEST.
    static func selfCheck() -> [(String, Bool)] {
        let (c1, t1) = parse("let me check</tool_call> oops")   // malformed → no closing before open? treat as text
        _ = (c1, t1)
        let (content, calls) = parse("Sure.<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
        var out: [(String, Bool)] = []
        out.append(("tool_name", calls.first?.function.name == "get_weather"))
        out.append(("tool_args_json", calls.first?.function.arguments == #"{"city":"Tokyo","days":3}"#))
        out.append(("tool_content", content == "Sure."))
        let (c2, calls2) = parse("just text, no tools")
        out.append(("no_tools_passthrough", c2 == "just text, no tools" && calls2.isEmpty))
        out.append(("coerce_bool", { if case .bool(true) = coerce("true") { return true }; return false }()))
        out.append(("coerce_string", { if case .string("hi") = coerce("hi") { return true }; return false }()))
        return out
    }
}
