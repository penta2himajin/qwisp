# Qwisp

## Overview

Qwisp is a single-model-specialised local inference engine for **Qwen3.6-35B-A3B (MoE)** on
Apple Silicon (MLX). It streams MoE expert weights from flash and keeps only the active slice
resident, extending the reachable model size on RAM-constrained Macs. The decode core is
**Seedless** — a raw-Metal engine (persistent buffers, hand-issued command buffers, int32
readback) that runs *outside* the MLX op-graph, peer to "MLX" as a backend.

Positioning: fastest practical-accuracy local LLM for power-users + researchers, with
**bit-exact lossless** (strict L1: reproduces the quantised greedy token stream) exposed as an
option. See @README.md.

> **Status: productization.** The research phase is closed and the engine is frozen. Work is
> converging the ~29K-line research PoC into a shippable OpenAI-compatible local server. Volatile
> campaign state lives in `HANDOFF.md` and `notes/`, not here.

## Project Structure

```
swift/            # the product — Swift package
  Sources/QwispCore/    # Tell runtime + Seedless engine (raw-Metal forward, arena/streaming, spec-verify) + locked tests
  Sources/qwisp/        # OpenAI server + `qwisp chat` CLI + tokenizer (swift-transformers)
  Sources/qwisp-poc/    # bench/gate binary (RAWTESTS + bench harness)
scripts/          # shell gate + benchmark scripts
oracle/           # Python reference/bench oracle (bit-compare only; NEVER in the serving path)
notes/            # engine design rationale (referenced by number from source comments)
docs/             # process docs (handoff-protocol, i18n-policy)
refs/             # canonical measurement refs (raw-greedy) — GITIGNORED, regenerate locally
```

The boundary that matters: **Swift = product + engine; Python = reference oracle only.** The
server is Swift, in-process — the engine holds GBs of resident Metal buffers a language boundary
cannot cheaply reach.

## Development Setup

- **Xcode + Metal Toolchain** required (raw-Metal kernels). SourceKit shows "No such module 'MLX'"
  on QwispCore files — LSP-only noise; `xcodebuild` is the truth.
- Model: a Qwen3.6-35B-A3B MTPLX checkpoint; point `QWISP_MODEL` at its directory.
- Python reference oracle needs an MLX-capable python (numpy/safetensors/mlx_lm), not Homebrew
  python3 — see `oracle/README.md`.

## Build & Test

```bash
# build (Release; Metal Toolchain required; ~minutes)
#   scheme `qwisp` = product (server + CLI) ; scheme `qwisp-poc` = bench/gate binary
cd swift && xcodebuild build -scheme qwisp -configuration Release \
  -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel \
  -skipPackagePluginValidation

# correctness gates (must stay green through every commit)
scripts/test_raw.sh          # → RAWTESTS 79/79     (engine, GPU, no model)
scripts/test_bench_batch.sh  # → BENCHBATCHTEST PASS (fixture, no GPU)
scripts/test_tokenizer.sh    # → TOKTEST 3/3        (needs model tokenizer files)
scripts/test_completion.sh   # → COMPTEST 4/4       (needs model tokenizer files)
```

`refs/*.safetensors` are gitignored. A fresh checkout / `git clean` loses them and makes the
strict fidelity gate false-red on longctx/shortnl — regenerate per `HANDOFF.md` before trusting a
red strict cell.

## Development Principles

- **Lossless is defined at L1: bit-exact reproduction of the quantised greedy token stream.** The
  strict path is the reference; `bolt`/near-lossless is an opt-in speed tier. Never weaken the
  lossless definition to make a number look better.
- **RAWTESTS 79/79 is the campaign-wide safety gate.** It must stay green through every
  delete/rename/refactor commit. A red gate blocks the commit, not the other way around.
- **Predictive levers lose to mechanical levers on the same slack** (engine doctrine, measured
  repeatedly). Don't re-propose prediction/prefetch schemes as speedups without new measurement.
- **Measure before implementing.** Heavy Metal/kernel changes get a Python physical-plausibility
  check first (the research phase burned weeks on plausible-but-wrong kernel ideas).

## Architectural Boundaries

- The Seedless decode core does **not** use the MLX op-graph — its speed comes from raw command
  buffers and persistent buffers. MLX compatibility is coarse (load + generate + tier via a
  backend protocol), never op-level.
- Python stays a reference/bit-compare oracle. It is never on the serving path.
- `refs/` is the canonical raw-greedy measurement set — regenerate from Swift raw greedy only,
  never from MLX or a bootstrap.

## Prohibitions

1. Do not regenerate `refs/*.safetensors` from MLX or any non-raw-greedy source.
2. Do not weaken, skip, or delete the `WRITE-LOCKED` tests in
   `swift/Sources/QwispCore/SeedlessVerifyTests.swift` (guarded by `total = 79`). They are the lossless safety net.
3. Do not rewrite the shipped forward path (SeedlessEngine / SeedlessMetalForward / SeedlessFusedVerify /
   Tell / ExpertArena / ExpertSource + model layers). It is frozen — refactor/rename only, never rewrite.
4. Do not rewrite `main`'s history. Work on a topic branch (`claude/<topic>`, etc.) and open a PR to `main`.
5. Do not run two `qwisp-poc` processes at once — the GPU is exclusive; heavy runs must be standalone.

## Git Conventions

- **Conventional Commits** with a scope: `feat(seedless):`, `feat(server):`, `fix(arena):`,
  `docs:`, `refactor:`, `test:`, `chore:`. Research scopes like `feat(ghost):` are historical.
- When an agent authors a commit, append a `Co-Authored-By:` trailer for the agent.
- Push after every commit (working convention: don't leave commits unpushed).

## Session Handoff

Cross-session continuity uses `HANDOFF.md` at the repo root (overwritten each session; the
SessionStart hook re-injects it) plus the file-based memory index. The GitHub-issue protocol in
`docs/handoff-protocol.md` is the canonical spec for the issue-based variant if/when a workstream
moves to issues; the `session-handoff` label and `.github/ISSUE_TEMPLATE/handoff.md` support it.

## Internationalisation

qwisp ships a Japanese-facing README. Follow `docs/i18n-policy.md`:

- Translations are suffix files (`README.ja.md` next to `README.md`); no language directories.
- Only `README.md` and the user-facing introduction tier of `docs/` are in scope. Engineering docs
  and notes stay English-only (or the author's working language).
- Each translated file carries a `> Source: <name>.md @ <sha>` header. PRs are never blocked on
  translation parity.

---

<!-- Common rules below this line apply to every project. -->

## Common Development Rules

### TDD (Red → Green → Refactor)

All implementation work proceeds in this cycle:

1. **Red**: write a failing test that captures the intended behaviour.
2. **Green**: write the minimum code that makes the test pass.
3. **Refactor**: tidy up while keeping tests green.

When a test fails, fix the production code — do not delete, skip, or weaken the test.

### Measure, Don't Conjecture

Base decisions on observed data, not assumptions. Before optimising, claiming a bottleneck, or asserting that something is slow or broken, measure it — profile, benchmark, log, or reproduce. When you report a cause, cite the measurement that supports it.

### Git Conventions

- **Conventional Commits**: `feat:` `fix:` `docs:` `refactor:` `test:` `ci:` `chore:`. Project-specific prefixes (e.g. `data:`, `experiments:`) live in the project's `AGENTS.md`.
- **Branch naming**: use a short prefix for the agent or author followed by a topic, e.g. `claude/<topic>`, `codex/<topic>`, or `human/<topic>`.
- **Trailer**: when an AI agent authors the commit, append a trailer crediting the agent. Do not embed model name or session info in the trailer; put those in the commit body if needed.

### Pull Requests

- **Always ready for review.** Open PRs in the "ready" state, never as drafts. Draft PRs do not fire review-requested events and slow the loop.
- **Auto-subscribe after creating a PR.** Immediately after the PR is created, subscribe to its activity without asking the user. Rationale: the user explicitly opted into the "agent opens and watches its own PRs" workflow at the template level, so the per-PR confirmation is noise. Unsubscribe only when the user says to stop, when the PR merges, or when it is closed unmerged.
- **One PR per workstream**, matching the handoff issue. Reference the issue with `Closes #N` per `.github/PULL_REQUEST_TEMPLATE.md`.

### Stream Idle Timeout Mitigation

Cloud agent sessions occasionally fail with `Stream idle timeout - partial response received` on long output. To reduce risk:

1. **Stage long writes.** For long documents or source files, write the skeleton (headings, function signatures, trait stubs) first, then fill each section in follow-up edits. Avoid single blocks larger than ~200 lines.
2. **Watch out after large reads.** Reading a big file (e.g. `Cargo.lock`, large generated modules) and then immediately producing long output is a common trigger. Split into separate turns or excerpt only the relevant portion.
3. **Recover carefully.** A timeout can still leave the file write completed. Run `git status` before retrying so the same content is not written twice.

### Common Prohibitions

1. Do not delete, skip, or comment out existing tests.
2. Do not modify CI configuration without explicit instruction.
3. Do not weaken production code merely to make tests pass.
4. Do not commit credentials, API keys, signed URLs, or anything in `.env*`.
