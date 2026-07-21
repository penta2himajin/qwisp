#!/usr/bin/env node
// Deterministic replay of a captured OpenCode trace against a qwisp server (issue: parallel-agent
// reproduction). Fires the captured request bodies either CONCURRENTLY at their recorded relative
// arrival times (reproducing OpenCode's fan-out load) or SERIALLY one-at-a-time (the bit-identity
// reference). Collects each request's streamed text + timing. A separate `diff` compares two runs
// token-for-text to quantify how often the batch path diverges from the serial/strict reference.
//
// Usage:
//   node replay_concurrent.mjs run <capture.jsonl> <host:port> <concurrent|serial> <out.json> [--greedy]
//   node replay_concurrent.mjs diff <ref.json> <cmp.json>
//
// --greedy overrides temperature/top_p to force the greedy path (so the identity comparison is
// meaningful even if OpenCode sampled). Compares streamed text (the wire has no token ids); text
// divergence is a strong proxy for a token flip. ponytail: built-in http only.
import http from "node:http";
import fs from "node:fs";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function fire(hostport, body, greedy) {
  const [host, port] = hostport.split(":");
  let req;
  try { req = JSON.parse(body); } catch { return Promise.resolve({ ok: false, text: "", err: "bad-body" }); }
  req.stream = true;
  if (greedy) { req.temperature = 0; delete req.top_p; delete req.top_k; }
  const payload = Buffer.from(JSON.stringify(req));
  const t0 = Date.now();
  return new Promise((resolve) => {
    const r = http.request({ host, port, method: "POST", path: "/v1/chat/completions",
      headers: { "content-type": "application/json", "content-length": payload.length, authorization: "Bearer sk-noauth" } },
      (res) => {
        let text = "", ttft = null, buf = "";
        res.on("data", (c) => {
          if (ttft === null) ttft = Date.now() - t0;
          buf += c.toString("utf8");
          let nl;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
            if (!line.startsWith("data:")) continue;
            const d = line.slice(5).trim();
            if (d === "[DONE]") continue;
            try {
              const j = JSON.parse(d); const dl = j.choices?.[0]?.delta;
              if (dl?.reasoning_content) text += dl.reasoning_content;   // thinking models
              if (dl?.content) text += dl.content;
            } catch {}
          }
        });
        res.on("end", () => resolve({ ok: true, text, ttft, wall: Date.now() - t0, status: res.statusCode }));
      });
    r.on("error", (e) => resolve({ ok: false, text: "", err: String(e) }));
    r.end(payload);
  });
}

async function run(capPath, hostport, mode, outPath, greedy) {
  const recs = fs.readFileSync(capPath, "utf8").trim().split("\n").map((l) => JSON.parse(l))
    .filter((r) => r.method === "POST" && r.path.includes("chat/completions"));
  recs.sort((a, b) => a.tArrival - b.tArrival);
  console.error(`[replay] ${recs.length} completions, mode=${mode}, target=${hostport}, greedy=${!!greedy}`);
  const results = {};
  const wall0 = Date.now();
  if (mode === "concurrent") {
    const base = recs[0].tArrival;
    await Promise.all(recs.map(async (r) => {
      await sleep(Math.max(0, r.tArrival - base));   // reproduce arrival timing
      const t = Date.now() - wall0;
      const res = await fire(hostport, r.body, greedy);
      results[r.id] = { ...res, tStart: t, inflightAtArrival: r.inflightAtArrival };
      console.error(`  #${r.id} done ttft=${res.ttft ?? "-"}ms wall=${res.wall ?? "-"}ms len=${res.text.length}`);
    }));
  } else {
    for (const r of recs) {
      const res = await fire(hostport, r.body, greedy);   // one at a time = reference
      results[r.id] = { ...res, inflightAtArrival: 1 };
      console.error(`  #${r.id} done ttft=${res.ttft ?? "-"}ms wall=${res.wall ?? "-"}ms len=${res.text.length}`);
    }
  }
  const totalWall = Date.now() - wall0;
  const genChars = Object.values(results).reduce((s, r) => s + (r.text?.length ?? 0), 0);
  fs.writeFileSync(outPath, JSON.stringify({ mode, hostport, greedy: !!greedy, totalWall, genChars, results }, null, 2));
  console.error(`[replay] totalWall=${totalWall}ms genChars=${genChars} → ${outPath}`);
}

function diff(refPath, cmpPath) {
  const ref = JSON.parse(fs.readFileSync(refPath, "utf8"));
  const cmp = JSON.parse(fs.readFileSync(cmpPath, "utf8"));
  let ids = 0, identical = 0, diverged = [];
  for (const id of Object.keys(ref.results)) {
    const a = ref.results[id]?.text, b = cmp.results[id]?.text;
    if (a == null || b == null) continue;
    ids++;
    if (a === b) identical++;
    else {
      let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++;
      diverged.push({ id, divergeAtChar: i, refLen: a.length, cmpLen: b.length,
                      refTail: a.slice(i, i + 40), cmpTail: b.slice(i, i + 40) });
    }
  }
  console.log(`identity: ${identical}/${ids} streams byte-identical`);
  console.log(`throughput: ref totalWall=${ref.totalWall}ms  cmp totalWall=${cmp.totalWall}ms  speedup=${(ref.totalWall / cmp.totalWall).toFixed(2)}x`);
  for (const d of diverged) console.log(`  #${d.id} diverges @char ${d.divergeAtChar} (ref ${d.refLen} / cmp ${d.cmpLen})\n    ref: …${JSON.stringify(d.refTail)}\n    cmp: …${JSON.stringify(d.cmpTail)}`);
}

const [, , cmd, ...a] = process.argv;
if (cmd === "run") await run(a[0], a[1], a[2], a[3], a.includes("--greedy"));
else if (cmd === "diff") diff(a[0], a[1]);
else { console.error("usage: run <cap.jsonl> <host:port> <concurrent|serial> <out.json> [--greedy]  |  diff <ref.json> <cmp.json>"); process.exit(1); }
