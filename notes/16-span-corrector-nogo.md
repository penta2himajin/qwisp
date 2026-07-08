# notes/16 — lever② resident-span corrector (gate_fold): NO-GO 判定書

Date: 2026-07-08 | Branch: feat/raw-verify @d94c50a | Runner: `qwisp/span_residual.py`
Logs: `~/.claude/projects/-Users-penta2himajin-repos-qwisp/span-residual-2026-07-08/`

## 問い（handoff の go/no-go 基準）

miss expert e の出力を常駐集合の span で再構成する corrector
`f_e(x) ≈ Σ_b c_{e,b}·f_b(x) + β·x + v`（gate_fold 形）の品質天井
`dist(f_e, span(常駐出力)⊕affine)²` は小さいか。文献（routing 冗長性）は「小さい」を示唆していた。

## 方法

- refs/{regime}.safetensors の prompt+greedy 128 を teacher-forced 1 forward、per-layer MoE 入力を捕獲。
- residency proxy = per-layer top-64 by prompt 域 gate 質量（prefill 後凍結の analog）。
- fit = gen 前半 64 token（rolling recalib analog）、eval = gen 後半の miss イベント（gate 質量加重）。
  ridge 65 係数（64 buddy + β、per-dim 切片は demean）、e 非依存の共有 Gram で全 miss expert 一括。
- corner 比較: novice(v) / affine(β,v) / best-single-buddy / span top-j (j=2..64)。

## 結果（相対残差 ‖f_e−pred‖²/‖f_e‖²、質量加重、eval=held-out）

| regime | group | missM | top1mass | buddy1 | span8 | span64 |
|---|---|---|---|---|---|---|
| code | late(30-39) | .325 | .17 | .925 | .923 | **.923** |
| code | early/mid | .31-.46 | .18-.23 | 1.01 | 1.01 | 1.01 |
| agentic | mid/late | .35-.41 | .13-.16 | .93 | .93 | **.925** |
| longctx | late | .445 | .19 | .942 | .941 | **.941** |
| shortnl | mid/late | .35-.38 | .18-.19 | .97-.98 | .97 | **.97** |

- **span64 − buddy1 ≤ 0.002 全 cell** — corrector が現行 buddy 代替に対して買う fidelity ≈ 0。
- in-sample floor ですら 0.68–0.93（過学習ではなく物理）。
- 切り分け診断（span_diag.py, L2/L20/L38）:
  - SELF（自分込み span で自己再構成）= 0.0000 → 配管正常。
  - **LOO: 常駐 expert 自身を他 63 本+x から再構成しても eval 残差 0.84–0.95、ridge 非感応**
    → expert 出力は相互にほぼ直交。routing 冗長性仮説はこのモデルで反証。
  - combine レベル（token 出力誤差）: drop 0.40–0.76 → span64 0.40–0.76（改善 0–10% 相対）。
- 仮説検証: top-1 miss の質量シェアは 0.13–0.26（懸念した ~0.30 より小、件数シェア 0.08–0.11）。
- residency proxy の missMass は Swift BoltDiag 実測より悲観的（code mean ~.35 vs 実測 .216;
  凍結 top-64 vs 実機の warmup+async refresh の差）— 方向は一致（longctx 最悪）で、
  LOO の直交性は residency 選択に依存しない。

## 案C（low-rank shadow expert）も同時閉路

miss expert 自身の重み SVD rank-r 再構成（rank_shadow_diag.py, 24 expert × 3 層, code）:
**r=16: 0.99 / r=32: 0.98 / r=64: 0.94 / r=128: 0.65–0.91** — expert 行列は実効フルランク。
文献 2512.17073 の文脈は「量子化残差の補償」（微小摂動）であり全 expert 再構成には非転移。
RAM 収支も不成立（r=32 ≈ 0.5MB/expert × 192 miss × 40 層 ≈ 3.8GB）。

## 判定

**② corrector campaign 全体を NO-GO で閉路。** io=0 の miss 補償は
span/affine/novice/buddy/low-rank shadow の全形態で品質天井が「ほぼ無補正」と同等。
miss expert の情報は decode 時にその重みを読む以外に存在しない
（Lean read_lower_bound の fidelity 版が実務発現）。

- Lean の in-sample 支配定理（residentClass_extends_combined）は成立しているが margin ≈ 0
  — 「支配」は「有用」を含意しない。定理+実測の組で初めて go/no-go。
- Neo C64 fidelity hole（81–84%）は構造的: 埋める手段は C 増（RAM）か escalation（IO/sync 再導入
  = bolt の io=0 1-CB 前提と非両立）のみ。fidelity/RAM は製品 knob であって工学バグではない。
- doctrine 再確認: 「重実装の前に python 物理検証」が campaign 1 本を 1 session で回避
  （lm_head cert NO-GO と同型）。
