#!/usr/bin/env node
// #154 M1 — shared-prefix KV duplication vs peak footprint (Step 0 kill criterion 1).
//
// Unlike lane_stage_b_probe conc (distinct fillers so the prefix cache cannot collapse),
// this fires B streams that share one long prefix and differ only in a short suffix —
// the #121 fan-out shape. After a warm capture, later admits restore the blob but each
// lane still materialises its own KV (the byte cost #154 asks about).
//
// Usage:
//   node tools/lane_issue154_m1_probe.mjs <host:port> <sharedPrefixTokens> <B> [maxTokens]
// Output: one JSON line on stdout.
import http from "node:http";
import { makeFiller } from "./lane_budget_probe.mjs";

const [hostport = "127.0.0.1:8099", sharedStr = "8192", bStr = "3", maxStr = "16"] =
  process.argv.slice(2);
const [host, port] = hostport.split(":");
const sharedTokens = parseInt(sharedStr, 10);
const B = parseInt(bStr, 10);
const maxTokens = parseInt(maxStr, 10);
// Issue #154 accounting: 10 attn layers × KV=2 × D=256 × 2B × {K,V} ≈ 20KB/token.
const KV_BYTES_PER_TOKEN = 20 * 1024;

const sharedPrefix = makeFiller(sharedTokens);

function fireStream(content, maxTok) {
  const payload = Buffer.from(JSON.stringify({
    model: "qwisp", messages: [{ role: "user", content }],
    max_tokens: maxTok, stream: true, temperature: 0,
  }));
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    const req = http.request({
      host, port, method: "POST", path: "/v1/chat/completions",
      headers: {
        "content-type": "application/json", "content-length": payload.length,
        authorization: "Bearer sk-noauth",
      },
    }, (res) => {
      let buf = "", deltas = 0, ttft = null;
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
            const t = dl?.reasoning_content || dl?.content;
            if (t) { if (ttft === null) ttft = Date.now() - t0; deltas += 1; }
          } catch {}
        }
      });
      res.on("end", () => resolve({
        status: res.statusCode, deltas, ttftMs: ttft, totalMs: Date.now() - t0,
      }));
      res.on("error", reject);
    });
    req.on("error", reject);
    req.end(payload);
  });
}

async function main() {
  // Warm: capture the shared prefix into sharedStore (recurrence / end-of-prompt save).
  const warm = await fireStream(`${sharedPrefix}\nWarm capture. Reply: ok.`, 8);
  // Concurrent fan-out: identical shared prefix, unique short suffixes.
  const streams = await Promise.all(Array.from({ length: B }, (_, i) =>
    fireStream(
      `${sharedPrefix}\nRequest id ${i}. Reply with exactly the id number and nothing else.`,
      maxTokens,
    )));
  const promptLenEst = sharedTokens; // filler targets this; suffix is negligible for accounting
  const dupBytes = (B - 1) * promptLenEst * KV_BYTES_PER_TOKEN;
  const arenaBytesEst = B * promptLenEst * KV_BYTES_PER_TOKEN;
  console.log(JSON.stringify({
    issue: 154,
    step: "M1",
    sharedPrefixTokens: sharedTokens,
    B,
    kvBytesPerToken: KV_BYTES_PER_TOKEN,
    warm: { status: warm.status, deltas: warm.deltas, ttftMs: warm.ttftMs, totalMs: warm.totalMs },
    streams: streams.map((r) => ({
      status: r.status, deltas: r.deltas, ttftMs: r.ttftMs, totalMs: r.totalMs,
    })),
    ok: warm.deltas > 0 && streams.every((r) => r.deltas > 0),
    arenaBytesEst,
    duplicatedBytes: dupBytes,
    duplicatedMB: +(dupBytes / (1024 * 1024)).toFixed(1),
    arenaMBEst: +(arenaBytesEst / (1024 * 1024)).toFixed(1),
  }));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
