# 22 — Spec: WS-B Stage B — ctx-adaptive lane admission (lift the 16K cap)

Stage B of WS-B (roadmap settled with owner 2026-07-23; Stage A = token-budget
scheduler, shipped default-ON in PR #140 / notes/21). Scheduling + allocation
only: zero numeric/kernel changes, lane decode stays bit-exact.

**Audience note**: this spec is written for a Sonnet implementation session.
Every design ambiguity is resolved here (see "Resolved ambiguities — binding");
do not re-open them. For engine-physics questions not covered here, consult
Fable read-only (`/askfable`) rather than guessing.

## Preconditions (check before starting)

1. PR #140 merged (Stage A on main, `QWISP_TOKEN_BUDGET_SCHED` default ON). ✓ 2026-07-25.
2. **PR #138 (WS-A archive) must be merged first.** On main RAWTESTS total=92
   (SeedlessVerifyTests.swift:52); PR #138 adds locked tests 93–96. Stage B adds
   locked test **97** — starting before #138 merges creates a numbering collision.
   If #138 is closed unmerged instead (owner's call), renumber Stage B's test to 93.
3. Baselines green on main: RAWTESTS 92/92 (96/96 post-#138), COMPTEST 83/83,
   BENCHBATCHTEST PASS.

## Why (measured)

- The real workload (HANDOFF, OpenCode trial): explore subagents ingest ~35K
  tok/turn; OpenCode sends full history every turn. `QWISP_LANE_CTX` default
  16384 (LaneServe.swift:313-314) makes those prompts **silently unservable**
  in lane mode: `generate()` computes `headroom = laneCtx - prompt.count - 1`
  → ≤ 0 → ceiling 0 → the scheduler's `maxTokens > 0` guard finishes the
  request with an **empty stream, no error** (LaneServe.swift:333-336 +
  ContinuousBatch.swift `budgetedStep` pend-creation guard). This exact silent
  no-op burned two false-started bench passes on 2026-07-25 (notes/21 "Failed
  approaches" in issue #139) — it is a real trap, not a theoretical one.
- The serialize path already fought and won this same battle (PR #135, the
  64K cliff): unbounded arena sizing collapsed the box (observed 2026-07-22:
  41GB footprint, prefill 138→41 tok/s), fixed by a bounded gen budget
  (`QWISP_PREFIX_GEN_MAX` default 16384) + eligibility cap at model context.
  Stage B mirrors that shipped, validated policy onto lanes.
- Memory arithmetic (why per-request sizing is cheap): KV cost is ~20KB/token
  (10 full-attn layers × KV=2 heads × D=256 × 2 bytes × K+V; the
  LLMBackend.swift:152 comment's "10 full-attn layers ≈ 20KB/token" and the
  alloc at SeedlessFusedVerify.swift:4825 agree). A 35K-token admit with a 16K
  gen budget ≈ 1.0GB per lane — fine on the resident tier (≥32GB) lanes
  require, IF bounded. Unbounded (model ctx 262K × B lanes) is the PR #135
  collapse again — hence the aggregate gate below.

## Verified load-bearing facts (do not re-derive; cite these)

1. **Heterogeneous arena sizes across lanes are safe by construction.** Every
   batched-step kernel takes `maxLen` from the LANE'S OWN cache object
   (`lanes[b].layers[li].kvCache!` → `kv.maxLen`, SeedlessLaneBatch.swift
   ~697-710), and `SeedlessLaneBatch.init` requires only `layers.count`
   equality across lanes (SeedlessLaneBatch.swift:483-487) — no uniform
   maxSeqLen assumption anywhere. Lanes with different arena sizes can ride
   in one batch today.
2. **Persist blobs are arena-size-independent.** `persistentStateData()` saves
   only the USED slice (`kv.len` rows/head); `restorePersistentState()`
   validates `KV`/`D` and `len <= kv.maxLen`, then writes at the DESTINATION's
   stride (`base + h * kv.maxLen * kv.D * 2`) (SeedlessFusedVerify.swift:
   3849-3921). A blob saved from a 16K-arena lane restores bit-correctly into
   a 32K-arena lane. Oversized blob → `false` → existing cold-prefill fallback
   at the LaneServe restore site. Nothing to change in the format.
3. **The sizing policy already exists as a pure function.**
   `SeedlessBackend.cachedGenBudget(promptLen:ceiling:arenaMax:genCap:)` =
   `max(1, min(ceiling, min(genCap, arenaMax - promptLen)))`
   (LLMBackend.swift:161-163, public static, self-checked). REUSE it — do not
   reimplement.
4. **Arena allocation is already per-admit.** `makeLane` calls
   `engine.makeFused(maxM: 1024, maxSeqLen: maxSeqLen)` on every admission
   (LaneServe.swift:169) — per-request sizing is a parameter change, not a
   lifecycle change. Idle slots hold no arena.
5. `readContextLen` (LLMBackend.swift:190) reads
   `text_config.max_position_embeddings`, fallback 32768.

## Contract — B1: ctx-adaptive admission

### Sizing policy (LaneBackend.generate, LaneServe.swift:333-336)

```
ctxMax     = min(readContextLen(modelDir), QWISP_LANE_CTX if explicitly set)
             // NEW default: model context. QWISP_LANE_CTX stays honored as an
             // override; its old 16384 default is GONE (that was the cap).
genCap     = max(1024, QWISP_LANE_GEN_MAX default 16384)
             // NEW knob, mirrors QWISP_PREFIX_GEN_MAX; separate knob because
             // lanes multiply the cost by B.
headroom   = max(0, ctxMax - prompt.count - 1)
ceiling    = maxTokens < 0 ? headroom : min(maxTokens, headroom)
genBudget  = SeedlessBackend.cachedGenBudget(promptLen: prompt.count,
                 ceiling: ceiling, arenaMax: ctxMax, genCap: genCap)   // reuse (fact 3)
seqBudget  = prompt.count + 1 + genBudget      // the arena size this request needs
```

`generate()` submits with `maxTokens: genBudget` (was: ceiling). Env reads stay
in `LaneBackend.init`/`generate` ONLY — never inside `ContinuousScheduler` or
`LaneBatchSlots` (Stage A discipline; keeps COMPTEST deterministic).

### Plumbing seqBudget to the arena (protocol change, Stage A's pattern)

- `ContinuousScheduler.Request` already carries `maxTokens`; `budgetedStep`
  computes `seqBudget = req.prompt.count + 1 + req.maxTokens` (pure arithmetic
  on fields it has) and passes it to a new `admitStep` overload:
  `admitStep(prompt:slot:tokenBudget:seqBudget:)`.
- **`seqBudget = 0` is the legacy sentinel**: size the arena at the init-time
  `maxSeqLen`, exactly today's behavior. The default protocol extension (which
  keeps `ContinuousBatchEngine`/`BatchBackend` untouched — same trick as Stage
  A) and `LaneBatchSlots.admit()`'s atomic wrapper both pass 0.
- Consequence (deliberate, see Resolved #1): ctx-adaptive sizing is a feature
  of the budgeted-scheduler path only. `QWISP_TOKEN_BUDGET_SCHED=0` keeps the
  legacy fixed-cap path **bit-for-bit untouched** — the escape hatch stays a
  true escape hatch. The frozen `loop()` tokenBudget==0 body is NOT edited.
- `LaneBatchSlots.admitStep` first-call setup: `makeLane(hybrid:seqLen:)` where
  `seqLen = seqBudget > 0 ? min(seqBudget, ctxMaxFromInit) : maxSeqLen`. The
  guard at LaneServe.swift:185 becomes `prompt.count < seqLen`.

### Aggregate memory gate (the PR #135 collapse guard)

- `LaneBatchSlots` gains `func kvBytesPerToken() -> Int` computed once from
  `engine.layers` (Σ over attn layers of `numKV × headDim × 2 × 2`) and tracks
  `activeArenaBytes` = Σ over non-nil lanes/prefills of `seqLen × kvBytesPerToken()`.
- New protocol member `func canAdmit(promptLen: Int, seqBudget: Int) -> Bool`
  (default extension: `true` — MLX engines unaffected). LaneBatchSlots returns
  whether `activeArenaBytes + candidate ≤ QWISP_LANE_KV_MB × 1MB` — budget knob
  read in `LaneBackend.init` and passed into `LaneBatchSlots.init` as a plain
  `Int` (env discipline), default `physicalMemory ≥ 48GB ? 8192 : 4096`.
- Scheduler (budgeted path only): before creating a Pend for the queue head,
  check `canAdmit`. If false, **stop admitting this round — do NOT skip ahead**
  (FCFS head-of-line wait, vLLM semantics; preserves `tokenbudget_fifo_fairness`).
  Memory frees when lanes release; the head retries every round.
- Clamp-to-fit: if the candidate ALONE exceeds the whole budget, clamp its
  `seqBudget` down to fit (floor: `prompt.count + 2` — at least 1 generated
  token). If even the floor doesn't fit, fail the request (return `.failed` /
  today's nil-admit path) and `FileHandle.standardError.write` ONE clear line —
  never the silent empty stream.

## Contract — B2: hardening carried from the Stage A Fable review

Small, low-risk items recorded in PR #140 comments; fold them in here rather
than leaving them as lore:

1. `Prefill.lastNormed` gets a one-line comment: "must never be consumed across
   a pause — the final chunk and its logits always execute in the same
   admitStep call" (implicit invariant today, verified true; noCopy-lifetime
   history says make it explicit).
2. The #121 fan-out restore ordering under the budgeted scheduler is an
   EMERGENT property of ceil-grant semantics (pendings serialize because any
   `.prefilling` grant drives pool ≤ 0). Add a comment at the `budgetedStep`
   grant site saying exactly that, so a future grant-policy change knows what
   it silently breaks.
3. The self-check `Fake`'s floor-grant semantics intentionally differ from the
   real ceil-grant `LaneBatchSlots.admitStep`; comment both sides so nobody
   "fixes" one to match the other.

## Explicitly REJECTED for Stage B (do not implement; reasons are terminal)

- **Live warm-lane reuse** (keep released lanes' state alive; adopt on prefix
  match): L1-incompatible. The lane's end-of-decode KV rows for generated
  tokens are produced by the M=1 decode kernels, NOT by the canonical hybrid
  chunked prefill — restoring/continuing from them diverges from the canonical
  greedy stream (same physics as the makeLane comment: "raw chunked prefill
  produces the pre-hybrid stream and diverges ~100 tokens in"). Rewinding to
  the prompt boundary is impossible: GDN state is a recurrence, it cannot be
  truncated. #121's blob-at-prompt-boundary design (pure-prefill-produced
  state, exact-boundary replay-gated) is the correct and only L1-safe shape.
- **Warmth-based admission reordering** (admit best-prefix-match pendings
  first): breaks FCFS (`tokenbudget_fifo_fairness` is locked), diverges from
  vLLM's default, and the aggregate win is speculative. Not without a
  measurement first, and not this stage.
- **PrefixPersist (disk) tier for lanes**: real value (lane warmth across
  restarts) but scope-coupled to the owner's pending "PrefixPersist
  default-ON" decision — explicitly backlog (HANDOFF "Then" §4), not Stage B.

## Locked tests

### RAWTESTS (SeedlessVerifyTests.swift; total bumps 96 → 97 post-#138)

97. `lane_admit_restore_cross_arena_bitexact`: save `persistentStateData` from
    a lane forward built with `maxSeqLen = S1`, restore into a fresh lane
    forward built with `maxSeqLen = S2 > S1`, decode N steps on each (same
    tokens, same synthetic weights as the test-90/91/92 lane family), assert
    bit-identical output streams AND that restore into `S3 < len` correctly
    returns false (cold-fallback path). Pattern/harness: extend the test-92
    (`lane_admit_restore_bitexact`) fixture, ADD a new test — never modify 92.

### COMPTEST (pure logic, no GPU/model; wire after the `tokenbudget_*` block, Selftest.swift:142)

`lanesize_*` — pure-function checks on the sizing chain (extract the sizing
into a testable static, e.g. `LaneBackend.sizePlan(promptLen:maxTokens:ctxMax:genCap:) -> (genBudget: Int, seqBudget: Int)`):
- `lanesize_unset_maxtokens`: maxTokens=-1, prompt 35_000, ctxMax 262_144,
  genCap 16_384 → genBudget 16_384, seqBudget 51_385.
- `lanesize_explicit_capped`: maxTokens 100_000, prompt 1_000, ctxMax 32_768 →
  genBudget ≤ min(16_384, 31_767); assert exact value via cachedGenBudget.
- `lanesize_prompt_too_big`: prompt ≥ ctxMax → genBudget floor behavior
  (headroom 0 → the request must NOT silently succeed; assert the plan flags it).
- `lanesize_legacy_sentinel`: seqBudget 0 → init maxSeqLen used (document via
  the LaneBatchSlots seam, or assert in the scheduler fake below).

`lanemem_*` — scheduler gate via the Stage A `Fake` extended with
`canAdmit`/per-slot seqBudget recording:
- `lanemem_head_of_line_wait`: head request too big for the fake budget while
  a smaller one queues behind → NEITHER admits this round (FCFS preserved);
  head admits after a release frees budget.
- `lanemem_seqbudget_passthrough`: the fake records
  `seqBudget == prompt.count + 1 + maxTokens` for a budgeted-path admit.
- `lanemem_atomic_legacy`: the tokenBudget==0 path never passes a non-zero
  seqBudget (legacy sentinel honored).

All existing `tokenbudget_*` cases stay byte-untouched and green.

## Bench verification (driver/owner runs after GREEN; GPU-exclusive, paired same-session per notes/20 doctrine)

1. **The motivating E2E**: `QWISP_LANES=2`, default env, a ~35K-token prompt
   (reuse `tools/lane_budget_probe.mjs`'s filler generator — mind its
   sentence-repeat units bugfix, notes/21) → must ADMIT and stream a correct
   completion (today: silent empty). Record TTFT. PASS = completes.
2. **No-regression paired A/B**: same session, same binary, short-prompt
   workload (well under 16K so sizing is immaterial):
   `QWISP_TOKEN_BUDGET_SCHED=0` (legacy arena) vs `=1` (adaptive) at B=1..4 —
   per-stream tok/s Δ within noise (±5%). Isolates sizing overhead.
3. **Footprint gate**: during 2 concurrent 35K admits, `footprint <pid>`
   (NOT `ps rss` — HANDOFF gotcha) stays within QWISP_LANE_KV_MB + model
   residency + ~200MB×B scratch expectation. No swap storm.

## Resolved ambiguities (binding — do not re-litigate in-session)

1. **Adaptive sizing is budgeted-path-only.** The `QWISP_TOKEN_BUDGET_SCHED=0`
   opt-out keeps the ENTIRE legacy behavior including the fixed 16K cap. Reason:
   Stage A froze `loop()`'s tokenBudget==0 body byte-for-byte as the escape
   hatch; threading seqBudget through it would void that guarantee. The default
   is ON, so real users get the lift.
2. **`QWISP_LANE_CTX` semantics change**: from "cap AND arena size, default
   16384" to "eligibility cap override, default = model context". This is
   deliberate (the 16K default IS the bug). Update the main.swift help text
   (~L43-46) accordingly.
3. **Memory gate is FCFS with head-of-line wait**, not skip-ahead. Matches
   vLLM, preserves the locked fairness test. Head-of-line blocking on a huge
   request is accepted behavior, not a bug to fix this stage.
4. **`cachedGenBudget` is reused as-is** (rung 2 of the ladder). If its
   semantics need to differ for lanes, that is a spec change — stop and ask.
5. **seqBudget flows scheduler→admitStep as a parameter**; the arena-size
   decision lives in LaneBatchSlots (it owns the engine facts), the POLICY
   (genCap/ctxMax) lives in LaneBackend (it owns the env). Nothing reads env
   in between.
6. **The silent-empty-stream path must die** for oversized prompts: fail
   visibly (one stderr line + failed stream), never an empty 200.

## Prohibitions (Stage B)

- Frozen forward path (CLAUDE.md #3) — untouched; this is allocation/scheduling.
- Do not modify: `ContinuousBatchEngine` / `BatchBackend`, chunk sizes
  (1024/64), `LaneAdmitPlan`/`admitPlan`/`admitSelfCheck` semantics, the
  `loop()` tokenBudget==0 body, any existing locked test (RAWTESTS 90-96,
  `tokenbudget_*`, `laneadmit_*`).
- No cloud offload of anything (memory `qwisp-no-cloud-principle`).
- No default flips beyond the specified `QWISP_LANE_CTX` semantic change.
- Commit convention: `feat(seedless):` / `test(bench):`, Co-Authored-By
  trailer, push after every commit, branch `claude/lane-ctx-adaptive`.

## Suggested session shape (mirrors Stage A's, which converged in 1 round)

1. Recon (Opus or careful Sonnet read): verify facts 1-4 above against HEAD
   line numbers (they WILL have drifted post-#138 merge); confirm
   `SeedlessLaneBatch` has no other uniform-maxLen assumption
   (`grep -n maxSeqLen swift/Sources/QwispCore/SeedlessLaneBatch.swift`).
2. devloop loop phase: locked tests RED (RAWTESTS 97 + the COMPTEST cases
   above) → implement → adversarial review. Gates green at every commit.
3. Driver runs the bench section, records results in an addendum to THIS file
   (same shape as notes/21's "Bench verification results"), then owner decides
   the default (nothing to flip here — adaptive sizing has no separate flag;
   it rides the Stage A scheduler flag).
