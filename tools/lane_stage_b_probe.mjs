#!/usr/bin/env node
// WS-B Stage B bench probe (notes/22 "Bench verification"): the three Stage B
// measurements against a running qwisp server. Built-in http only (ponytail —
// same shape as lane_budget_probe.mjs, whose filler generator it imports so the
// sentence-repeat formula lives in exactly one place).
//
// Usage:
//   node tools/lane_stage_b_probe.mjs e2e  <host:port> <promptTokens> [maxTokens]
//   node tools/lane_stage_b_probe.mjs tps  <host:port> <B>            [maxTokens]
//   node tools/lane_stage_b_probe.mjs conc <host:port> <promptTokens> <N> [maxTokens]
// Output: one JSON line on stdout.
import http from "node:http";
import { makeFiller } from "./lane_budget_probe.mjs";

const [mode, hostport = "127.0.0.1:8080"] = process.argv.slice(2);
const [host, port] = hostport.split(":");

// Fires one streaming completion; resolves with per-stream timing. `deltas` counts
// SSE content deltas — a token-count PROXY (the server emits one delta per decoded
// token today), good enough for a paired ±5% A/B, not an absolute tok/s claim.
function fireStream(content, maxTokens) {
  const payload = Buffer.from(JSON.stringify({
    model: "qwisp", messages: [{ role: "user", content }],
    max_tokens: maxTokens, stream: true, temperature: 0,
  }));
  const t0 = Date.now();
  return new Promise((resolve, reject) => {
    const req = http.request({
      host, port, method: "POST", path: "/v1/chat/completions",
      headers: { "content-type": "application/json", "content-length": payload.length,
                 authorization: "Bearer sk-noauth" },
    }, (res) => {
      let buf = "", deltas = 0, ttft = null, text = "";
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
            if (t) { if (ttft === null) ttft = Date.now() - t0; deltas += 1; text += t; }
          } catch {}
        }
      });
      res.on("end", () => {
        const totalMs = Date.now() - t0;
        // Decode rate excludes prefill: (deltas-1) gaps over the post-TTFT window.
        const decodeMs = ttft === null ? 0 : totalMs - ttft;
        resolve({
          status: res.statusCode, deltas, ttftMs: ttft, totalMs,
          tps: deltas > 1 && decodeMs > 0 ? +((deltas - 1) / (decodeMs / 1000)).toFixed(2) : null,
          sample: text.slice(0, 120),
        });
      });
      res.on("error", reject);
    });
    req.on("error", reject);
    req.end(payload);
  });
}

async function main() {
  if (mode === "e2e") {
    // (1) The motivating E2E: a prompt far past the old 16K lane cap must ADMIT
    // and stream real content. On main this returns an empty 200 (silent no-op).
    const tokens = parseInt(process.argv[4] || "35000", 10);
    const maxTokens = parseInt(process.argv[5] || "32", 10);
    const r = await fireStream(makeFiller(tokens), maxTokens);
    console.log(JSON.stringify({ mode, promptTokens: tokens, ...r, ok: r.deltas > 0 }));
    return;
  }
  if (mode === "tps") {
    // (2) No-regression paired A/B: B concurrent SHORT prompts (well under 16K so
    // adaptive sizing is immaterial) — isolates the sizing overhead itself.
    const B = parseInt(process.argv[4] || "1", 10);
    const maxTokens = parseInt(process.argv[5] || "64", 10);
    const rs = await Promise.all(Array.from({ length: B }, (_, i) =>
      fireStream(`Count slowly from 1 to ${60 + i}, one number per line, nothing else.`, maxTokens)));
    const tpsAll = rs.map((r) => r.tps).filter((x) => x != null);
    const mean = tpsAll.length ? +(tpsAll.reduce((a, b) => a + b, 0) / tpsAll.length).toFixed(2) : null;
    console.log(JSON.stringify({ mode, B, meanTps: mean, perStream: rs.map((r) => ({ tps: r.tps, deltas: r.deltas, ttftMs: r.ttftMs, status: r.status })) }));
    return;
  }
  if (mode === "conc") {
    // (3) Footprint gate: N SIMULTANEOUS large admits. N>=3 is the point — the
    // round-1 reservedBytes TOCTOU guard only bites when budgetedStep peeks at
    // >=3 free slots in ONE round (2 passes even with the bug).
    const tokens = parseInt(process.argv[4] || "35000", 10);
    const N = parseInt(process.argv[5] || "3", 10);
    const maxTokens = parseInt(process.argv[6] || "16", 10);
    const filler = makeFiller(tokens);
    // Distinct suffixes so the prefix cache cannot collapse them into one lane.
    const rs = await Promise.all(Array.from({ length: N }, (_, i) =>
      fireStream(`${filler} Request id ${i}. Reply with the id.`, maxTokens)));
    console.log(JSON.stringify({
      mode, promptTokens: tokens, N,
      ok: rs.every((r) => r.deltas > 0),
      streams: rs.map((r) => ({ deltas: r.deltas, ttftMs: r.ttftMs, totalMs: r.totalMs, status: r.status })),
    }));
    return;
  }
  if (mode === "ratchet") {
    // Memory-growth probe for #148. Its ORIGINAL hypothesis — that per-request arena
    // sizing ratchets MLX's buffer pool, fixable with Memory.clearCache() — is DISPROVED:
    // bounding the pool (Memory.cacheLimit=2GB) pinned cacheMemory at ~2,050MB and did
    // not stop the growth. Measured law instead:
    //     nonMLXmetal ~= 19,750MB + concurrency * promptLen(this round) * ~2.03MB/token
    // i.e. it tracks the PROMPT LENGTH of the round being sampled, not arena size and not
    // cumulative admissions. Root cause was autoreleased noCopy MTLBuffer wrappers from
    // the steel-hybrid prefill, held until the decode thread exited; fixed by draining per
    // prefill chunk (QWISP_LANE_CHUNK_POOL). This probe is now the #148 regression check.
    //   vary      = prompt grows each round (prompt AND arena size vary TOGETHER — this
    //               A/B therefore never isolated arena size; the two are confounded)
    //   fixeduniq = constant SIZE, unique CONTENT -> equal prefill work, no prefix reuse
    //   fixed     = constant size, SHARED prefix -> the prefix cache serves it and the
    //               pass does almost no prefill. CONFOUNDED; do not quote as a control.
    const rounds = parseInt(process.argv[4] || "10", 10);
    const shape = process.argv[5] || "vary";   // vary | fixed | fixeduniq
    const base = parseInt(process.argv[6] || "1500", 10);
    const conc = parseInt(process.argv[7] || "2", 10);
    const out = [];
    for (let r = 0; r < rounds; r++) {
      const toks = shape === "vary" ? base + r * 1000 : base;
      const filler = makeFiller(toks);
      // The round marker must lead, not trail: with a trailing marker every "fixed"
      // round shares the same long prefix and the prefix cache serves it, so the pass
      // does almost no prefill and the vary/fixed comparison comes out confounded
      // (observed 2026-07-25: fixed finished in 8 samples vs vary's 62).
      // "fixeduniq" = constant SIZE, unique CONTENT -> same prefill work as vary,
      // same arena size every round. That is the one that isolates size churn.
      const lead = shape === "fixed" ? "" : `Session ${r}-${base}-${r * 7919 % 10007}. `;
      const rs = await Promise.all(Array.from({ length: conc }, (_, i) =>
        fireStream(`${lead}${filler} Round ${r} item ${i}. Reply with the round number.`, 8)));
      out.push({ round: r, promptTokens: toks, deltas: rs.map((x) => x.deltas), ttftMs: rs.map((x) => x.ttftMs) });
      console.error(`[ratchet] round ${r} size=${toks} done`);
    }
    console.log(JSON.stringify({ mode, shape, rounds, out }));
    return;
  }
  console.error("usage: lane_stage_b_probe.mjs e2e|tps|conc <host:port> ...");
  process.exit(2);
}
main().catch((e) => { console.error("[lane_stage_b_probe] error:", e); process.exit(1); });
