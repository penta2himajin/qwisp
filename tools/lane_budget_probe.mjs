#!/usr/bin/env node
// WS-B Stage A GO-bar probe (notes/21): fires a "steady decode" stream (A) concurrently
// with a large-prompt admit (B) against a running qwisp server, and reports the
// inter-token-latency (ITL) distribution of stream A — the metric the spec's GO bar is
// defined on ("a steady decode lane's p99 ITL during a concurrent admit"). Built-in
// http only (ponytail — mirrors tools/opencode-repro/replay_concurrent.mjs).
//
// Usage: node tools/lane_budget_probe.mjs <host:port> [bigPromptTokens]
// Output: one JSON line on stdout: {n, p50, p90, p99, max, gapsMs}
import http from "node:http";
import { pathToFileURL } from "node:url";

const hostport = process.argv[2] || "127.0.0.1:8080";
const bigTokens = parseInt(process.argv[3] || "24000", 10);
// Steady stream A's length. The ITL percentiles are computed over exactly these gaps,
// so this IS the percentile denominator: prefill pollutes ~k gaps (k = rounds the admit
// spans, set by promptLen/budget and independent of this) out of `steadyTokens - 1`.
// Short streams therefore report a worse p90 for identical engine behaviour — the 45
// default is the pathological end of that, not a neutral choice (notes/21 p90 = 9,983ms).
const steadyTokens = parseInt(process.env.STEADY_TOKENS || "45", 10);
const [host, port] = hostport.split(":");

function fireStream(body, onDelta) {
  const payload = Buffer.from(JSON.stringify({ ...body, stream: true, temperature: 0 }));
  return new Promise((resolve, reject) => {
    const req = http.request({
      host, port, method: "POST", path: "/v1/chat/completions",
      headers: { "content-type": "application/json", "content-length": payload.length,
                 authorization: "Bearer sk-noauth" },
    }, (res) => {
      let buf = "";
      res.on("data", (c) => {
        buf += c.toString("utf8");
        let nl;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
          if (!line.startsWith("data:")) continue;
          const d = line.slice(5).trim();
          if (d === "[DONE]") continue;
          try {
            const j = JSON.parse(d);
            const dl = j.choices?.[0]?.delta;
            const text = dl?.reasoning_content || dl?.content;
            if (text) onDelta(Date.now());
          } catch {}
        }
      });
      res.on("end", () => resolve());
      res.on("error", reject);
    });
    req.on("error", reject);
    req.end(payload);
  });
}

// Filler ≈1.3 tok/word for English — a repeated sentence is enough to force a real
// multi-chunk prefill; exact token count doesn't matter for the ITL comparison. `reps`
// is the SENTENCE repeat count (bugfix: the original version repeated the whole
// multi-word sentence `words` times, inflating the prompt ~14x past the target).
// Exported so lane_stage_b_probe.mjs shares ONE copy of this formula — the
// sentence-repeat bug above must never be re-derived in a second file.
export function makeFiller(bigTokens) {
  const sentence = "The quick brown fox jumps over the lazy dog near the riverbank at dusk.";
  const sentenceWords = sentence.split(" ").length;
  const reps = Math.max(1, Math.round(bigTokens / 1.3 / sentenceWords));
  return Array(reps).fill(sentence).join(" ");
}
const filler = makeFiller(bigTokens);

async function main() {
  const gaps = [];
  let last = null;
  const aDone = fireStream(
    { model: "qwisp", messages: [{ role: "user", content: `Count slowly from 1 to ${steadyTokens}, one number per line, nothing else.` }], max_tokens: steadyTokens },
    (t) => { if (last !== null) gaps.push(t - last); last = t; }
  );
  // Let A's stream start before B's admit lands — mirrors "lane 0 already decoding".
  await new Promise((r) => setTimeout(r, 600));
  // Count B's tokens: a dropped admit still returns a normal empty 200, so discarding
  // this stream makes the whole probe silently measure "no concurrent load at all"
  // (#151 — that is exactly how the flag-off arm read as a 13s pass with no stall).
  let bTokens = 0;
  const bDone = fireStream(
    { model: "qwisp", messages: [{ role: "user", content: filler }], max_tokens: 3 },
    () => { bTokens += 1; }
  );
  await Promise.all([aDone, bDone]);
  if (bTokens === 0) {
    console.error("[lane_budget_probe] FATAL: the big-prompt stream produced 0 tokens — " +
                  "the admit was dropped, so stream A measured an idle server. Results are void.");
  }
  gaps.sort((x, y) => x - y);
  const pct = (p) => gaps.length ? gaps[Math.min(gaps.length - 1, Math.floor(p * gaps.length))] : null;
  console.log(JSON.stringify({
    n: gaps.length, p50: pct(0.5), p90: pct(0.9), p99: pct(0.99),
    max: gaps.length ? gaps[gaps.length - 1] : null, bTokens, gapsMs: gaps,
  }));
}
// Only run when invoked directly — importing makeFiller must not fire a probe.
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((e) => { console.error("[lane_budget_probe] error:", e); process.exit(1); });
}
