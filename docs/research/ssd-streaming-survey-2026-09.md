# SSD-streaming methods relevant to Qwisp (2026-09)

Status: research survey; no implementation decision  
Date: 2026-09-29  
Qwisp baseline: `b7dffc15f148d072a96f652d5d4907a2b6a4d06d`

## Question

Which newer MoE weight-streaming techniques could improve Qwisp on non-resident Apple
Silicon tiers without weakening its strict L1 contract, and which ones require a separate
near-lossless model or execution tier?

This note separates three classes that are easy to conflate:

1. **Mechanical, output-preserving scheduling**: execute resident work before waiting for
   cold weights, merge reads, or change prefill traversal without changing weights or routing.
2. **Speculative I/O with exact fallback**: predict only what to read; the real router still
   decides what executes, and a bad prediction costs time/bytes but cannot change output.
3. **Model-changing approximation**: replace routing, drop experts, or change precision. These
   may be valid for an opt-in tier but cannot be called strict/lossless.

External results below are author-reported and have not been reproduced in Qwisp. Throughput
numbers are not comparable across models, quantization, hardware, cache warmth, or fidelity
contracts.

## Executive result

The strongest new evidence supports work Qwisp already has open, but has not implemented:

- **Resident-first compute overlapped with cold reads** is now used by both Colibri and
  streamlx. It maps directly to [#88](https://github.com/penta2himajin/qwisp/issues/88).
- **Larger, coalesced expert reads** are used by streamlx and by Colibri's expert-major
  container. They sharpen [#147](https://github.com/penta2himajin/qwisp/issues/147): the first
  question is no longer whether one read is possible, but whether Qwisp's current nine-way
  parallel `pread` is already saturating the target SSD.
- **Expert-major prefill** is a distinct opportunity. streamlx reports about 5x over
  row-chunking when a prompt's routed union exceeds the pool. Qwisp has no equivalent path.
- **One-layer-ahead read hints** now have positive external evidence, but Qwisp #88(b)'s old
  premise needs correction: Bolt freezes residency/remap tables, not each token's route. A strict
  or Bolt predictor is useful only if it creates lead time without adding a GPU drain.

The current memory failures must be fixed or controlled before timing these techniques. Arena
coexistence ([#165](https://github.com/penta2himajin/qwisp/issues/165)) and the Bolt OOM cluster
([#169](https://github.com/penta2himajin/qwisp/issues/169)) can turn a storage experiment into a
driver paging experiment. A faster loader cannot make an overcommitted working set correct.

## Qwisp baseline

The strict streaming loop currently has this dependency chain:

```text
route on GPU -> wait/read indices -> partition rows -> blocking ensure/pread
             -> encode gather -> wait before a slot may be overwritten -> next chunk/layer
```

Relevant implementation facts:

- [`runStrictLayers`](../../swift/Sources/QwispCore/SeedlessFusedVerify.swift) calls
  `provider.ensure(chunk.experts)` before encoding each chunk, and flushes before an arena slot
  can be reused.
- [`ExpertArena.loadMany`](../../swift/Sources/QwispCore/ExpertArena.swift) already issues every
  miss's three projections times weight/scales/biases as concurrent operations: nine `pread`s
  per expert, not nine sequential syscalls.
- [`BoltServe`](../../swift/Sources/QwispCore/BoltServe.swift) has a bounded two-arena background
  pipeline for rolling recalibration refreshes. It does **not** overlap ordinary demand misses
  in the strict chunk loop.
- Bolt freezes its resident set and buddy/slot tables, but still computes the real route inside
  every layer. The next layer's routed expert IDs are therefore not known in advance. Ordinary
  Bolt decode then remaps cold IDs to resident slots, so it has refresh I/O rather than strict's
  per-step demand-miss I/O.
- The shipped engine already owns persistent fixed-slot expert arenas, exact safe-prefix
  handling for batched union overflow, calibration artifacts, and prefix-state reuse. A new
  design should reuse those seams rather than rebuild an MLX-style wrapper in Swift.

## Systems surveyed

| System | Relevant mechanism | Fidelity claim | Transferability |
|---|---|---|---|
| [Colibri](https://github.com/JustVugg/colibri/tree/ce370e87d7b623d7759b52ec2007d75fc5b0e87e) | Deferred cold-read pipeline, resident-first Metal work, expert-major records, learned pinning, optional router lookahead | Decode Metal path is token-exact vs its CPU path; some GPU prefill modes explicitly are not byte-exact | High for scheduling; model and scale differ |
| [streamlx](https://github.com/srcterm/streamlx/tree/146507b59293058b3a1cecc43f47947b25012297) | Span-merged reads, resident/cold overlap, lookahead read hints, expert-major prefill | Claims bitwise identity with the stock model | High conceptually; MLX graph mechanics differ from Seedless |
| [Godwit](https://github.com/rayl15/Godwit/tree/094a91fbb202cb294d355a052964092ccce98f49) | Native Swift/Metal fixed slots and direct SSD streaming on a 16 GB Mac | Per-engine correctness checks; not a claim of identity with a differently quantized base | Useful Apple baseline and bottleneck evidence |
| [FlashMoE](https://arxiv.org/abs/2601.17063v1) | Expert-major prefill and a learned recency/frequency cache policy | Executes routed experts; cache policy changes placement, not routing | Useful algorithmic evidence; CUDA/discrete-GPU evaluation |
| [SSD-LLaMA](https://arxiv.org/abs/2609.18110v1) | Explicit SSD/RAM/VRAM hierarchy, optimized delivery, CPU/GPU hybrid execution | Executes selected experts without pruning/substitution | Architectural evidence; evaluated on discrete-GPU systems, not Apple unified memory |
| [Edge0](https://arxiv.org/abs/2609.18063v1) | Trained one-token-ahead prerouter consumed as the real router, plus recovery LoRA | Quality-bounded trained variant, not original-model bit identity | Separate checkpoint/tier only; incompatible with strict L1 |

Source snapshots were read on 2026-09-29. The repository SHAs above make this survey auditable if
the projects change later.

Godwit is a useful boundary case rather than a speed comparison: on its base M4 MacBook Air it
reports expert reads as 71.6% of decode wall and GPU busy for 17.6%. On that slow internal SSD,
overlap alone cannot hide all storage time; byte reduction, hit rate, or storage bandwidth must
also move. Qwisp must measure its own decomposition before selecting the same lever.

## Technique assessment

### 1. Resident-first compute and deferred cold reads

Colibri submits resident expert Metal work before issuing missed-expert disk reads, then joins the
cold results before the layer completes. Its bounded I/O pool reports both disk service time and
the smaller foreground-visible wait, which is the right overlap metric. streamlx uses the same
shape through `async_eval`: resident experts begin on the GPU while missing ranges are fetched.

This is the closest fit to Qwisp's current architecture. A strict layer can classify a chunk as
all-resident without changing routing or weights. All-resident work can be submitted first, while
cold bytes are read into slots that are not referenced by the in-flight command buffer.

The hard invariant is slot lifetime, not numerical correctness: a background read must never
overwrite a slot still consumed by Metal. A safe design needs either reserved victim slots or a
staging arena followed by an explicit swap. Qwisp's Bolt refresh pipeline already demonstrates the
staging form.

**Assessment:** strongest first candidate after the memory cluster is stable. It is mechanical,
compatible with strict L1, and already scoped by #88. The old #88 estimate of 38% synchronization
tax predates later fusion/no-sync work and must not be reused; a fresh decomposition is mandatory.

### 2. Read coalescing and expert-major layout

streamlx span-merges adjacent safetensors ranges into multi-megabyte reads. Colibri stores each
expert's matrices adjacent and reads them as one record. Qwisp instead reads nine exact slices per
miss. Those nine operations are already parallel, so syscall count alone is not evidence of a
bottleneck.

There are two implementation levels:

- **Range coalescing without a new artifact**: merge only genuinely adjacent source ranges and
  scatter/copy into the existing slot buffers. Low disk cost, but Qwisp's projection-major source
  layout may expose little adjacency.
- **Expert-major sidecar**: store all planes for one `(layer, expert)` in one contiguous record and
  make the slot buffers views of that record. This gives one large read but duplicates roughly the
  expert checkpoint unless installation converts the canonical serving artifact.

**Assessment:** keep #147's microbenchmark gate. Measure batch wall time around `loadMany`, not the
sum of concurrent `pread` durations. Compare the real nine-slice pattern with one contiguous record
at cold-cache and throttled-NAND settings before changing the format.

### 3. Expert-major prefill

Large prefill differs from decode: a chunk can route to most experts in a layer even when each row
uses only top-k. Row-major chunking may evict and later reload the same expert. streamlx instead
sweeps each needed expert once through a temporary buffer, accumulates its rows, and adopts the
hottest experts into the decode pool. It reports roughly 5x over slicing prefill to fit the pool.
FlashMoE independently uses the same expert-major principle: across the prompt batch, each required
expert is loaded once and its outputs are indexed back to the routed token positions.

For Qwisp this should be a separate prefill path, not an edit to decode scheduling. The first probe
does not require a new kernel: trace, per layer and prefill window, how many bytes are loaded today,
how many are duplicate `(layer, expert)` loads, and the lower bound if every required expert were
read once. Only build an expert-major accumulator if that lower bound is material.

**Assessment:** high potential for long prompts and TTFT, but greater implementation risk than
resident-first overlap because accumulation order can affect L1. The acceptance gate must compare
the quantized greedy stream, not only tensor tolerance.

### 4. Bolt refresh overlap and the stale #88(b) premise

#88(b) says Bolt's next-layer expert set is known from frozen routing. That is not true of the
current implementation. [`encodeLayerBolt`](../../swift/Sources/QwispCore/SeedlessFusedVerify.swift)
still runs the router for every layer and token; the uploaded table remaps its resulting expert IDs
to resident slots. What is frozen is residency and substitution, not the route itself.

Bolt's predictable I/O is instead the rolling refresh plan built from an observed routing window.
Qwisp already reads that plan through a bounded two-staging-arena background pipeline and publishes
swaps on the decode thread. Any new claim for exact Bolt next-layer prefetch must first identify
bytes that are not already covered by this refresh pipeline.

**Assessment:** mark #88(b) for re-scoping rather than implementation. The resident-first half of
#88 remains valid for strict streaming; Bolt lookahead belongs in the predictor experiment below,
not in the exact-mechanical bucket.

### 5. Predicted lookahead with exact demand fallback

Colibri applies layer `L+1`'s router early to layer `L`'s state and reports 71.6% top-8 recall.
streamlx reports about 70% and starts missing reads while attention computes. Both retain the real
route as authority, so a wrong hint should only waste I/O or fail to hide a miss.

This evidence is new enough to reopen the *I/O-hint* question, but not Qwisp's old approximate
execution paths. Seedless strict keeps the dependent layer chain inside command buffers; exposing
the exact intermediate state to CPU storage I/O can add the synchronization that consumes the lead
time. Bolt has the same routing dependency even though its substitution table is frozen. A useful
Qwisp design must show where the prediction runs, which cache/refresh action consumes the hint, and
how the read starts without a new per-layer drain.

**Assessment:** Step 0 only. Record predictor recall, false-positive bytes, demand-wait saved, and
added synchronization. Kill it if it loses to the same-residency mechanical overlap on the same
slack.

### 6. Learned pinning and live placement

Colibri persists routing heat, pins hot experts at startup, and optionally replaces a few pinned
experts at safe request boundaries using a decayed frequency/recency score. streamlx also supports
a warm-start trace, but notes that it helps mainly when the pool nearly covers the model; at small
budgets LRU reaches the same set within a few tokens.

Qwisp already has calibration artifacts, rolling recalibration, LRU arenas, and a measured
decayed-LFU no-go. FlashMoE is stronger evidence than a plain LFU retry: its small learned policy
approximates Belady from recency and frequency, and reports about 7% end-to-end improvement over LRU
for Qwen3-30B-A3B on its test system. That result is workload-trained and discrete-GPU-specific, so
it does not establish a Qwisp win.

**Assessment:** do not prioritize a cache-policy rewrite ahead of mechanical overlap. First make
existing arena ownership and budgeting correct, then use Qwisp traces to compute the Belady gap. A
small gap kills the idea without training; a large gap justifies comparing the learned policy with
LRU under the same I/O budget.

### 7. Multi-request expert-union amortization

Batching independent streams can load the union of their experts once, but may also destroy hit
rate, overflow a small arena, and multiply KV memory. streamlx deliberately serializes requests
because batching collapses its hit rate. Qwisp [#160](https://github.com/penta2himajin/qwisp/issues/160)
correctly requires an offline trace study before a streaming-lane design.

**Assessment:** retain #160's Step 0. This is workload-dependent and should follow the single-stream
I/O work, not lead it.

### 8. Trained prediction-as-routing

Edge0 goes beyond prefetch: a trained head predicts the next token/layer routing, and that
prediction becomes the route actually executed. This makes staged and executed experts identical
by construction and moves the approximation into training; an unmerged recovery LoRA compensates
for routing replacement and int4 error. The paper reports 80--84% decode gains over its on-demand
streamer and 20.4 tok/s for its 35B tier in 2.9 GiB peak active memory.

Those results are interesting but describe a different model artifact and fidelity contract. The
base router is no longer authoritative, a recovery adapter is required, and the published quality
comparison is benchmark-level rather than bit identity with the original quantized greedy stream.

**Assessment:** not a fix or optimization for current Qwisp. It could only be a future, explicitly
named near-lossless checkpoint/tier after the productization campaign, with its own training and
quality evaluation.

### 9. Explicit three-tier execution

SSD-LLaMA coordinates SSD, RAM, and discrete GPU memory and balances CPU/GPU expert execution. Its
paper reports 2.10--15.58x decode improvements over evaluated baselines while executing every
selected expert. It is useful evidence that delivery, placement, and execution need one planner.

Apple unified memory removes the PCIe RAM-to-GPU boundary but not the working-set limit. The
transferable lesson is the explicit byte ledger and placement policy, not a literal SSD/RAM/VRAM
port. Qwisp should represent trunk, expert slots, KV, scratch, allocator cache, and OS reserve in
one plan rather than derive mode from total physical RAM alone.

**Assessment:** apply the planning principle to the memory cluster; do not import the discrete-GPU
execution design.

## Items not promoted by this survey

- **Metal I/O as a presumed win:** no surveyed result establishes a general Apple-Silicon win.
  One [Colibri transport campaign](https://github.com/JustVugg/colibri/issues/594) reports an
  instrumented no-go for MTLIO on its M5 Max workload. Qwisp should require a local A/B before
  adopting MTLIO or another API solely for its name.
- **A new eviction heuristic:** Qwisp has already measured decayed LFU against LRU without a speed
  win. External LFRU/S3-FIFO results do not override a model-specific measurement.
- **Prediction that changes execution on a miss:** this violates strict L1 and repeats the failure
  mode in which corrupted hidden states drive later predictions.
- **KV quantization as a streaming-memory fix:** [#168](https://github.com/penta2himajin/qwisp/issues/168)
  shows the scale mismatch. It may help long-context decode bandwidth, but it saves hundreds of MiB
  where arena lifetime errors cost multiple GiB.
- **More residency without a ledger:** the current C/tier heuristic can already overcommit on long
  prompts. Raising cache coverage before fixing accounting can move the OOM threshold earlier.

## Measurement sequence for the later design discussion

No implementation ordering is decided here. The following sequence produces the evidence needed
to choose one:

1. **Stabilize the subject.** Resolve or explicitly work around #165/#169, record the actual arena
   mode, and run each arm in a fresh process. A command-buffer failure invalidates a timing row.
2. **Re-profile current streaming.** Per layer/step record route wait, `ensure` wall time, `pread`
   batch wall time, GPU compute, chunks, misses, bytes, and foreground-visible wait. Measure strict
   and generic Bolt separately at C=64 and C=128.
3. **Run two cheap probes.** First, simulate resident-first ordering from the trace to bound
   overlap. Second, microbenchmark nine scattered real-size reads against one contiguous record on
   the internal SSD, cold and under the existing 1.5 GB/s throttle.
4. **Measure prefill duplication.** For 2K/8K/32K prompts, compare actual expert bytes read with a
   one-read-per-`(layer, expert, window)` lower bound.
5. **Prototype one lever at a time.** Keep flag-off byte-unchanged. Resident-first overlap and
   coalescing must not land in the same experiment.
6. **Gate correctness at L1 where claimed.** Compare output tokens against canonical Swift raw
   greedy refs. Tensor tolerance or agreement with a different kernel family is insufficient.
7. **Gate speed on real wall time.** Report p50 across paired/interleaved runs, SSD state, AC power,
   miss rate, bytes, and foreground wait. A reduced sum of worker durations is not a speedup.

The decision discussion should answer these questions from the new measurements:

- Is current decode still I/O-bound after the memory fixes, and at which C/device tiers?
- How much cold-read time is actually hideable behind resident work?
- Is Qwisp limited by bytes, request scatter, synchronization, or driver paging?
- Does expert-major prefill save enough repeated I/O to justify a separate accumulation path?
- Can any strict lookahead start I/O without a new GPU-to-CPU drain?

## Primary sources

- Colibri:
  [README and core techniques](https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/README.md),
  [Metal backend](https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/docs/metal.md),
  [tuning and lookahead](https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/docs/tuning.md),
  [environment/I/O controls](https://github.com/JustVugg/colibri/blob/ce370e87d7b623d7759b52ec2007d75fc5b0e87e/docs/ENVIRONMENT.md).
- streamlx:
  [README at surveyed commit](https://github.com/srcterm/streamlx/blob/146507b59293058b3a1cecc43f47947b25012297/README.md).
- Godwit:
  [README](https://github.com/rayl15/Godwit/blob/094a91fbb202cb294d355a052964092ccce98f49/README.md),
  [design notes](https://github.com/rayl15/Godwit/blob/094a91fbb202cb294d355a052964092ccce98f49/docs/DESIGN.md).
- Kim et al., [FlashMoE](https://arxiv.org/abs/2601.17063v1), arXiv:2601.17063v1.
- Liang et al., [SSD-LLaMA](https://arxiv.org/abs/2609.18110v1), arXiv:2609.18110v1.
- Lin et al., [The Other Half of the Memory Wall / Edge0](https://arxiv.org/abs/2609.18063v1),
  arXiv:2609.18063v1.
- Qwisp tracking:
  [#88](https://github.com/penta2himajin/qwisp/issues/88),
  [#147](https://github.com/penta2himajin/qwisp/issues/147),
  [#160](https://github.com/penta2himajin/qwisp/issues/160),
  [#165](https://github.com/penta2himajin/qwisp/issues/165),
  [#168](https://github.com/penta2himajin/qwisp/issues/168), and
  [#169](https://github.com/penta2himajin/qwisp/issues/169).
