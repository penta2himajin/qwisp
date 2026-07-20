#!/usr/bin/env node
// OpenCode → qwisp capturing reverse proxy (issue: OpenCode parallel-agent reproduction).
//
// Sits between OpenCode and the qwisp server so we capture the EXACT wire requests OpenCode
// emits when it fans out parallel sub-agents — ground truth for a deterministic replay harness.
// Streams (SSE) pass through untouched via pipe; nothing is buffered on the response side, so
// OpenCode's token streaming and timing are unaffected. Each request is logged as one JSONL
// line with the full body (for replay) plus timing + concurrency (in-flight count at arrival).
//
// Usage:  PORT=8081 UPSTREAM=127.0.0.1:8080 CAPTURE=./oc-capture.jsonl node capture_proxy.mjs
// Point OpenCode's qwisp provider baseURL at http://127.0.0.1:8081/v1 and run a fan-out task.
//
// ponytail: built-in http only, no deps. Bodies are buffered request-side (chat JSON, not huge);
// responses stream. Upgrade path if a body ever exceeds memory: spill to a sidecar file by id.
import http from "node:http";
import fs from "node:fs";

const PORT = parseInt(process.env.PORT ?? "8081", 10);
const [UP_HOST, UP_PORT] = (process.env.UPSTREAM ?? "127.0.0.1:8080").split(":");
const CAPTURE = process.env.CAPTURE ?? "./oc-capture.jsonl";
const out = fs.createWriteStream(CAPTURE, { flags: "a" });
const t0 = process.hrtime.bigint();
const ms = () => Number(process.hrtime.bigint() - t0) / 1e6;

let seq = 0;
let inflight = 0;

const server = http.createServer((req, res) => {
  const id = ++seq;
  const tArrival = ms();
  inflight++;
  const inflightAtArrival = inflight;

  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    const body = Buffer.concat(chunks);
    const upReq = http.request(
      { host: UP_HOST, port: UP_PORT, method: req.method, path: req.url, headers: req.headers },
      (upRes) => {
        let tFirstByte = null;
        let respBytes = 0;
        let sseEvents = 0;
        res.writeHead(upRes.statusCode ?? 502, upRes.headers);
        upRes.on("data", (c) => {
          if (tFirstByte === null) tFirstByte = ms();
          respBytes += c.length;
          // Count SSE "data:" frames to sanity-check streaming shape (not parsed).
          for (let i = 0; i < c.length - 4; i++) {
            if (c[i] === 0x64 && c[i + 1] === 0x61 && c[i + 2] === 0x74 && c[i + 3] === 0x61 && c[i + 4] === 0x3a) sseEvents++;
          }
        });
        upRes.pipe(res); // stream through, no buffering
        upRes.on("end", () => {
          inflight--;
          let parsed = null;
          try {
            const j = JSON.parse(body.toString("utf8"));
            parsed = {
              model: j.model,
              stream: j.stream,
              n: j.n ?? null,
              parallel_tool_calls: j.parallel_tool_calls ?? null,
              messages: Array.isArray(j.messages) ? j.messages.length : null,
              tools: Array.isArray(j.tools) ? j.tools.length : null,
              lastRole: Array.isArray(j.messages) && j.messages.length ? j.messages[j.messages.length - 1].role : null,
              approxPromptChars: body.length,
            };
          } catch {}
          out.write(JSON.stringify({
            id, method: req.method, path: req.url,
            tArrival, tFirstByte, tDone: ms(),
            inflightAtArrival, status: upRes.statusCode,
            respBytes, sseEvents, parsed,
            body: body.toString("utf8"),   // full body for replay
          }) + "\n");
          console.error(`#${id} ${req.method} ${req.url} inflight=${inflightAtArrival} ttfb=${tFirstByte === null ? "-" : (tFirstByte - tArrival).toFixed(0)}ms sse=${sseEvents} ${upRes.statusCode}`);
        });
      }
    );
    upReq.on("error", (e) => { inflight--; res.writeHead(502); res.end(String(e)); });
    upReq.end(body);
  });
});

server.listen(PORT, "127.0.0.1", () => {
  console.error(`[capture] :${PORT} → ${UP_HOST}:${UP_PORT}  capture=${CAPTURE}`);
});
