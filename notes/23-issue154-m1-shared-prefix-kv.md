# #154 M1 — shared-prefix KV duplication vs peak footprint

Step 0 kill criterion 1 for
[issue #154](https://github.com/penta2himajin/qwisp/issues/154).

## How to reproduce

```bash
# Release qwisp binary required; GPU exclusive; AC power preferred.
QWISP_MODEL=~/models/Ornith-1.5-35B-A3B-MLX-4bit scripts/bench_issue154_m1.sh 8192
```

Probe: `tools/lane_issue154_m1_probe.mjs` (warm capture → B concurrent admits
sharing one filler prefix). Footprint via `/usr/bin/footprint` (never `ps rss`).
Duplicated bytes = `(B-1) × sharedPrefixTokens × 20 KiB` (issue accounting).

## Result (2026-09-12, M1 Max 64GB, Ornith-1.5 MLX-4bit, QWISP_LANES=4)

| B | peak footprint | dup (formula) | arena≈ | dup / peak |
|---|---:|---:|---:|---:|
| 2 | 23 552 MB | 160 MB | 320 MB | **0.68%** |
| 3 | 24 576 MB | 320 MB | 480 MB | **1.30%** |
| 4 | 25 600 MB | 480 MB | 640 MB | **1.88%** |

Raw: `/tmp/issue154-m1-20260912-121753/`. Server reported ~7226 prompt tokens for
the nominal 8192-token filler (sentence-repeat estimate); using 8192 in the
formula is the conservative overestimate and still ≪ 10%.

Prefix cache was live: warm TTFT ~23.5 s → subsequent warms ~1.4 s.

## Verdict

**Kill criterion 1 trips** (duplicated prefix KV < 10% of peak for all B).
Combined with the issue’s mechanism kill (no third path past frozen `sdpa_rows`
/ L1-unsafe split softmax), close **not-planned**. No design issue.
