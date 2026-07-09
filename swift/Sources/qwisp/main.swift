import Foundation
import Hummingbird
import Tokenizers
import Hub

// qwisp — product CLI + OpenAI-compatible server (productization step 5).
// Scaffold: proves the dependency graph (Hummingbird + swift-transformers) resolves
// and links. `serve` / `chat` are wired in follow-up sub-steps.

let args = Array(CommandLine.arguments.dropFirst())
switch args.first {
case "serve":
    print("qwisp serve — not yet implemented (step 5c/5d)")
case "chat":
    print("qwisp chat — not yet implemented (step 6)")
default:
    print("usage: qwisp [serve|chat]")
}
