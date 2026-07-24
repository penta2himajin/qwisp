# 21 — Spec: WS-B Stage A — token-budget admission scheduler

Phase 1 of WS-B. Scheduling-only: zero numeric/kernel changes, lane path stays
bit-exact. Goal: stop a large incoming prefill from stalling decode of
already-active requests on the same server.

## Why (measured)

- notes/19 §8 (`long-context-decay`): 0.42s→17.72s per 1024-chunk, pos 0→47K;
  attention share 12%→76%. WS-A tried to shrink that number directly and is
  NO-GO on M1-class (notes/20 §VERDICT) — the only lever left is scheduling:
  hide the cliff instead of flattening it.
- Real workload (HANDOFF `Decisions`): OpenCode sends full history every turn
  (no truncation), explore subagents inject ~35K tok/turn. Under today's
  `ContinuousScheduler.loop()` (ContinuousBatch.swift:73-118), admits are NOT
  interleaved with steps — the loop drains the whole queue into free slots,
  and each `admit()` call runs to completion (LaneBatchSlots.admit's internal
  `while pos < prompt.count` loop, LaneServe.swift:193-207, chunks at
  1024/64 tokens internally but the whole loop executes inside one call)
  before a single batched decode step runs for slots that were already
  active. A steady decode stream is fully stalled for the duration of any
  other request's admit — this is the actual UX pain, not raw prefill speed.
- `lane-batch-bench` (notes/19 §8, ctx=1024/lane): B=1 → 87.7 tok/s ≈
  serialize decode at comparable shallow ctx (~81-88 tok/s) — parity holds,
  the lane path itself costs nothing; only the missing interleaving is the
  gap. B-axis verdict there is already GO: "lanes can subsume serialize;
  scheduler = admission policy, not mode switch."
- Industry precedent (notes/19 §3): vLLM V1's one per-step `token_budget`
  where RUNNING (decode + in-flight prefill chunks) spends first and WAITING
  prefill gets the remainder; "serial" is the degenerate zero-load case of
  this, not a separate mode.

## Contract

Replace `ContinuousScheduler.loop()`'s "drain queue → run each admit to
completion → one step" sequence with a single per-iteration token budget
(default 2048 — matches the existing hybrid chunk size and `QWISP_LANE_PREFIX`
snapshot stride, so boundary alignment is free; tunable via
`QWISP_TOKEN_BUDGET`):

1. RUNNING slots (active decode) get their share first — 1 token each, up to
   `slotCount` tokens total.
2. Remaining budget is spent on WAITING/partially-admitted prefill, FIFO
   (matches vLLM's simple default; no need for a fancier policy at this
   scale), split across one or more free/mid-admission slots.
3. **Prefill must become resumable.** `BatchSlots.admit` is currently atomic
   (blocks until the whole prompt is prefilled). New surface, e.g.:
   `func admitStep(prompt: [Int32], slot: Int, tokenBudget: Int) -> AdmitProgress`
   returning `.prefilling(consumed: Int)` or `.done(firstToken: Int)`.
   `LaneBatchSlots` needs the loop currently local to `admit()`
   (`pos`/`lastNormed`/`plan`, LaneServe.swift:191-207) turned into per-slot
   state that survives across calls, resumed at `pos` each time the
   scheduler grants that slot budget.
4. **Bit-exactness constraint**: never split *inside* an existing internal
   chunk (1024 hybrid / 64 raw). The existing composition-invariance property
   (PREFIXE2E, `LaneAdmitPlan` boundaries) already holds at those boundaries —
   interleaving *between* chunks preserves it for free; interleaving *within*
   one would require new invariance work this phase does not need.
5. `ContinuousBatchEngine`/`BatchBackend` (MLX batch mode) are OUT OF SCOPE —
   already flagged slower on agentic traffic and not bit-exact (#121); Stage A
   targets `LaneBackend` only.

## Locked tests (COMPTEST — pure logic, no GPU; pattern: `ContinuousScheduler.selfCheck()` / `LaneBatchSlots.admitSelfCheck()`, Selftest.swift:136-139)

New self-check group (name prefix `tokenbudget_`), added to the same
`check(...)` tally Selftest.swift uses (COMPTEST total increases by however
many cases land — no fixed-total lock like RAWTESTS, mirror the existing
`batch_*`/`laneadmit_*` style):

- `tokenbudget_no_starvation`: a large-prompt admit (simulated via the
  existing `Fake: BatchSlots` pattern in `ContinuousScheduler.selfCheck`,
  extended with a multi-chunk fake prefill) does not delay an active decode
  slot's next token beyond one budget-iteration.
- `tokenbudget_fifo_fairness`: two waiting requests admitted in submission
  order when budget can't cover both in one iteration.
- `tokenbudget_output_identical`: same fake-engine script run with
  interleaving ON vs today's non-interleaved path → byte-identical per-stream
  output (interleaving changes only ORDER of internal calls, never results).
- `tokenbudget_chunk_boundary_respected`: fake admit sequence never receives
  a `tokenBudget` grant that would split inside an existing internal chunk.

## Bench verification (driver runs after GREEN, GPU-exclusive, paired same-session A/B — notes/20's measurement doctrine: no isolated-probe or cross-day claims)

Scenario: 2 lanes; lane 0 holds a steady decode stream, lane 1 admits a 24K
prompt mid-stream. Metric: p99 inter-token latency of lane 0's stream during
lane 1's admit, `QWISP_TOKEN_BUDGET_SCHED=1` vs unset, via `bench_decay_ab.sh`
harness shape.

**GO bar**: interleaved p99 ITL on the steady stream regresses by no more
than ~1 chunk-worth of decode latency (bounded by `ceil(budget / chunk_size)`
token-times), not by the other request's entire prefill duration (today's
behavior — unbounded stall).

## Bench verification results (2026-07-25, `scripts/bench_lane_budget_ab.sh 24000`, real model, QWISP_LANE_CTX=32768 for the measurement only)

Paired same-session run: lane 0 streams 45 tokens (`Count slowly from 1 to 45...`),
lane 1 admits a real ~24K-token prompt ~600ms in. Inter-token gaps (ms) of lane 0's
stream, `QWISP_TOKEN_BUDGET_SCHED` unset vs `=1`:

| | n | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| OFF | 44 | 12 | 17 | **93,668** | **93,668** |
| ON (budget 2048) | 44 | 12 | 9,983 | **13,516** | **13,516** |

**Qualitative GO**: the motivating failure mode is fixed. OFF stalls lane 0 for one
unbounded block (~93.7s = the other request's entire prefill — and this duration
scales with THAT prompt's size, unboundedly). ON caps the worst single stall at
13.5s regardless of how large the admitted prompt is — a ~7x reduction in max/p99,
and the bound no longer grows with the admitting request's total prompt length.

**The spec's literal GO-bar text above ("~1 chunk-worth of decode latency") does
NOT hold at depth, and that's expected, not a bug**: `ON`'s 44 gaps are NOT one
small stall per round — the last 11 are a MONOTONICALLY GROWING series (5.0s →
5.4s → 6.0s → 6.7s → 7.9s → 9.3s → 10.0s → 10.9s → 11.9s → 12.9s → 13.5s), one per
scheduler round, because `budgetedStep` runs a full budget-worth of `admitStep`
prefill work SYNCHRONOUSLY before the round's `step()` — and per notes/19 §8, a
1024-token prefill chunk itself costs 0.42s→17.72s depending on DEPTH (12%→76% attn
share by 47K). So the real bound Stage A delivers is "≤1 budget-worth of prefill
compute AT THE ADMITTING REQUEST'S CURRENT DEPTH", not a constant decode-token
time — correct the spec's mental model to this, not "~1 chunk-worth of decode
latency" (that phrase implicitly assumed flat per-chunk cost, which WS-A's own
findings (notes/20) already contradicted).

**Distribution tradeoff, stated plainly**: OFF concentrates all pain into ONE
sample (p90 = 17ms, only the last 1-2 of 44 gaps are large). ON spreads it across
~25% of samples (p90 = 9,983ms — a single measurement 587x worse than OFF's p90,
even though max improved 7x). This is the expected shape for a token-budget
scheduler (bounded-but-frequent beats unbounded-but-rare) but it is a real,
measurable regression on the p90 percentile specifically — record it, don't bury
it under the max/p99 win.

**Actionable lever for a follow-up**: default budget is 2048 (2 hybrid chunks);
halving it to 1024 (1 chunk) would roughly halve the max single-round stall at the
cost of admitting the other request's prefill in twice as many rounds (more total
rounds, each smaller) — untested this pass, a candidate tuning knob before any
default-ON decision.

Script: `scripts/bench_lane_budget_ab.sh` (orchestrator) + `tools/lane_budget_probe.mjs`
(HTTP/SSE client, built-in Node `http` only) — kept as the regression tool per the
project's measure-first doctrine. Known gotcha hit while building this probe:
macOS's bash 3.2 evaluates all RHS expansions in a multi-var `local a=x b=$a` line
BEFORE any assignment (`$a` reads as unbound under `set -u`) — split into separate
`local` statements, don't chain them.

## Prohibitions (this phase)

- Do not touch `ContinuousBatchEngine`/`BatchBackend` (MLX batch mode).
- Do not change chunk size (1024 hybrid / 64 raw) or any bit-exactness
  invariant of `LaneBatchSlots.admit` — Stage A is scheduling-only.
- Do not touch `QWISP_LANE_CTX` or `PrefixRAMStore` admission logic (#121) —
  that is Stage B (ctx-adaptive admission + prefix-aware lane restore).
- Default OFF via `QWISP_TOKEN_BUDGET_SCHED` (default 0) until GO — mirrors
  WS-A's flag-off-by-default discipline (notes/20).
