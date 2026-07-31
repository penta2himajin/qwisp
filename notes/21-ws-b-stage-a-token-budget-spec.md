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

> **READ THE 2026-08-01 ADDENDUM AT THE END OF THIS FILE BEFORE QUOTING ANY NUMBER
> BELOW.** These measurements are real, but the p90 and p99 figures describe the
> 45-token probe stream as much as the scheduler: percentile cleanliness is set by
> `k/L` (rounds spanned / stream length), so both invert at realistic stream lengths.
> `max` is the only stream-length-independent metric here. The addendum also records
> that this run's harness had the OFF arm mis-wired from the GO onward.

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
  WS-A's flag-off-by-default discipline (notes/20). **GO declared 2026-07-25**
  (see "Bench verification results" above) — `QWISP_TOKEN_BUDGET_SCHED` now
  defaults ON (`=0` opts back out to the old atomic path).

---

## Addendum 2026-08-01: AC re-measurement, budget sweep, and what the 2026-07-25 table actually showed

The 2026-07-25 table above is **not wrong**, but it was read wrong, by us, for a week.
It was measured with a 45-token steady stream, and at that length two of its three
headline numbers are artifacts of the stream length rather than properties of the
scheduler. This addendum records the AC re-measurement that establishes which claims
survive. It supersedes the "Actionable lever for a follow-up" paragraph above.

### Three harness defects found first (all fixed, PR #152)

Every one of these made a pass look valid while measuring something else. They are
listed first because the p90 topic existed for a week on top of them.

1. `scripts/bench_lane_budget_ab.sh` ran its OFF arm with **no env at all**. The
   2026-07-25 GO flipped `QWISP_TOKEN_BUDGET_SCHED`'s default to 1, so from that
   commit onward the "OFF" arm silently ran the **ON** path. Both arms then measured
   k=12 and summed stalls within 2.5% of each other — identical, because they *were*
   identical.
2. `tools/lane_budget_probe.mjs` discarded the big-prompt stream entirely, so a
   dropped admit was invisible: stream A measured an **idle server** and reported it
   as a clean latency profile. This is how the flag-off arm read as a 13s pass with
   no stall, which is what surfaced #151.
3. `max_tokens: 45` was hardcoded in both the request and its prompt text. Since the
   percentiles are computed over exactly those gaps, **the stream length is the
   percentile denominator** — see the law below.

### The law (confirmed at 6 measured points)

Let `S` = total prefill compute the admitting request needs, `k` = the number of
scheduler rounds it spans, `L` = the steady stream's length in tokens.

- `k = promptLen / QWISP_TOKEN_BUDGET`. Measured at 12K prompt: budget
  1024/2048/4096/8192 → k = 12/7/4/3. **k does not depend on L** (k=11 at L=45 and
  k=12 at L=500 for the same 24K/2048 configuration).
- One decode token is emitted per round, so exactly `k` of the stream's gaps carry
  prefill work. **Pollution rate = k/L.**
- Percentile `p` is clean iff `k/L < 1-p`. Measured crossover for p90 sits between
  k=4 (clean, 24ms) and k=7 (broken, 4,590ms) at L=45 — i.e. straddling the predicted
  `k ≤ 0.1L = 4.4`.
- **`S` is conserved**: the scheduler redistributes stall, it does not remove it.
  Summed stall across the whole sweep was 32.8 / 34.6 / 33.8 / 33.4 / 33.0s for
  OFF / 1024 / 2048 / 4096 / 8192 — within 5%. The ~13% excess ON carries over OFF at
  L=500 (32.1 → 36.4s) is chunking overhead.
- Therefore `max ≥ S/k`: fewer, larger slices trade percentile cleanliness for a worse
  worst case, monotonically. There is no setting that improves both.

### AC measurements (2026-08-01, `bench_lane_budget_ab.sh 12000`, real model, AC power)

12,000-token admit, not 24,000: **since #151 the flag-off path caps eligibility at
`legacyArenaCap` = 16,384**, so the 24K A/B of 2026-07-25 is no longer reproducible on
the OFF arm — it is now refused visibly. Any future paired A/B must stay under 16K.

Steady stream L=500 (a realistic agentic stream), `QWISP_TOKEN_BUDGET=2048`:

| | n | p50 | p90 | p99 | max | k | summed stall |
|---|---|---|---|---|---|---|---|
| OFF | 499 | 12 | 13 | 14 | **31,888** | 2 | 32.1s |
| ON | 499 | 12 | 13 | 4,733 | **8,471** | 7 | 36.4s |

Stall series — OFF `[204, 31888]` (one atomic block), ON `[201, 2303, 4733, 5341,
6984, 8326, 8471]` (the growing per-round series, as at 24K).

Budget sweep at L=45 (the 2026-07-25 stream length, kept for comparability):

| budget | k | pollution | p50 | **p90** | p99 = max | summed stall |
|---|---|---|---|---|---|---|
| OFF | 2 | 5% | 12 | **13** | 32,589 | 32.8s |
| 1024 | 12 | 27% | 12 | 3,359 | 4,397 | 34.6s |
| 2048 (default) | 7 | 16% | 12 | **4,590** | 8,328 | 33.8s |
| 4096 | 4 | 9% | 12 | **24** | 13,579 | 33.4s |
| 8192 | 3 | 7% | 12 | **19** | 22,780 | 33.0s |

Same sweep at L=500 (the realistic-stream counterpart; both sweeps are the same
12,000-token admit, AC power, same session per sweep):

| budget | k | pollution | p50 | p90 | p99 | max | summed stall |
|---|---|---|---|---|---|---|---|
| OFF | 2 | 0.4% | 12 | 13 | 22 | **33,879** | 34.1s |
| 1024 | 12 | 2.4% | 12 | 13 | 3,658 | **4,544** | 36.9s |
| 2048 (default) | 7 | 1.4% | 12 | 13 | 4,802 | **8,511** | 35.0s |
| 4096 | 4 | 0.8% | 12 | 13 | 23 | **13,655** | 34.0s |
| 8192 | 3 | 0.6% | 12 | 13 | 19 | **23,392** | 33.7s |

The stall series is the decision-relevant form of this table — a budget choice is a
choice of how many waits and how long each is, for the same conserved total:

| budget | stalls (s), L=500 |
|---|---|
| OFF | 0.2, **33.9** |
| 1024 | 0.2, 2.1, 2.6, 2.6, 2.6, 2.9, 3.1, 3.7, 4.0, 4.2, 4.3, **4.5** |
| 2048 | 0.2, 2.2, 4.8, 5.1, 6.6, 7.6, **8.5** |
| 4096 | 0.2, 10.0, 10.2, **13.7** |
| 8192 | 0.2, 10.1, **23.4** |

Note `p90 = 13ms` for EVERY arm including OFF at this stream length: at L=500 p90 no
longer discriminates at all, and `p99` is the percentile the budget moves. 4096 is the
notable point — it reproduces OFF's whole percentile profile (p99 23ms vs OFF's 22ms)
while cutting the worst stall 2.5x.

### What this changes

1. **The p90 regression is a stream-length artifact.** At L=500 the default already
   measures p90 = 13ms, identical to OFF. The 9,983ms in the table above is what a
   45-token stream reports for the same engine behaviour.
2. **So is the p99 win.** At L=45, ON beats OFF on p99 (both polluted, ON's stalls are
   smaller). At L=500 the ordering **inverts** — OFF's p99 is clean at 14ms while ON's
   is 4,733ms, because OFF pollutes 2 gaps (0.4%) and ON pollutes 7 (1.4%). Neither
   direction is a property of the scheduler; both are k/L crossing 1%.
3. **Percentile comparisons between OFF and ON are therefore not stable and must not be
   quoted without L.** The L-independent statements are exactly three: `p50` is
   identical (12ms), summed stall is conserved, and **`max` — the worst single stall —
   is where ON wins unconditionally** (31.9s → 8.5s at L=500), with a bound that does
   not grow with the admitting prompt's length. **Adopt `max` as the SLO metric for
   this scheduler** (owner agreed 2026-08-01); report percentiles only alongside L.
4. **Correction to the analysis that opened this topic.** "Restore p90 while keeping the
   p99/max win is arithmetically impossible" was too strong. The identities
   (`max ≥ S/k`, p90-clean ⟺ `k ≤ 0.1L`) hold, but "keeping the win" was anchored to
   ON@2048's specific max rather than to *beating OFF*. Under the honest definition
   `budget = 4096` satisfies both at L=45: p90 4,590 → **24ms** while max stays
   **2.4x better than OFF** (13,579 vs 32,589). The frontier is real and reachable;
   only the target was mis-stated.
5. **The "halve the budget to 1024" lever suggested above is the wrong direction for
   p90** and is now measured: it triples pollution (16% → 27%) and moves p90 from
   4,590ms to 3,359ms — better only because *every* stall shrinks, while more of them
   land in the sample. It is the right direction for `max` (8,328 → 4,397).

### Open: default budget

`QWISP_TOKEN_BUDGET` default is still 2048. 4096 is the only measured point that
restores p90 at short L while keeping a max win over OFF; 2048 keeps the smallest max.
Deferred to an owner decision — do not change the default on the strength of this
addendum alone.
