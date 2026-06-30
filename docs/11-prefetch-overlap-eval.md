# 11 — prefetch overlap 評価（SP-MoE cutoff）: streaming tier で不発、speed 天井不変

## 結論（先出し）
mixed+MTP(D1) は全 RAM tier(4–12GB) で all4-noMTP(D0) を上回る（cutoff bracket で 1.3–1.7×）。
**だが勝因は prefetch overlap ではなく、MTP compute-amortization + mixed-precision IO 削減**
（serial bracket で既に 1.26–1.55×、[[mixed-precision]] docs/08 既出）。
**prefetch overlap（cutoff−serial）の純価値は draft 窓律速でほぼゼロ**であり、
**[[positioning]]（docs/07）の speed 天井（churn/IO/同期）は本評価でも不変**。
＝「prefetch でもう一段速くなるか」の不確実性が「やってみないと分からない」から
**「窓律速 or coverage 律速のどちらでも追加では勝てない」**へ降りた（ネガティブ結果の確定）。

## 背景
外部 3 本を Qwisp の streaming tier（＝SSD↔unified memory の expert offload regime）で評価:
- **SP-MoE**（arXiv:2510.10302）/ **SpecMD**（arXiv:2602.03921）= corpus 新規（[[research-notes]] docs/01 C 節に追記）。
- **MoE-SpeQ**（arXiv:2511.14102）= 既収録（[[verify-forward]] docs/09）。
いずれも「draft 段が verify より先に走る構造を使い verify の expert を先読み prefetch」が核。

## 手法（sim-only, engine 非変更）
`tools/step4_streaming/sim_mtp_mixed.py` に cutoff ブラケットを追加（commit d1dc9fa）。
SP-MoE の cutoff-layer を Qwisp の MTP 窓構造へ再導出:

- `T_cycle(φ) = T_comp + (1−φ)·T_flash`、`φ = min(φmax, hideable/T_flash)`。
- `hideable = min(T_comp, t_draft + overlap_eff·(T_comp − t_draft))`（h ≤ T_comp 保証＝
  per-row 単調性 serial ≤ cutP ≤ cutXL ≤ ideal、self-check PASS）。
- **sync は T_comp に内包**（base_tps=54 / T(W1) 実測由来）ゆえ hideable から差引かない（二重計上回避）。
  prefetch-exactness の miss 検出 sync は φmax に反映。
- **φmax = 予測 coverage**: prev-token 再利用 0.66 / cross-layer 予測 0.77（[[positioning]] docs/07:235-243）。
- `T_flash` は union-miss バイトのみ（cache hit 85% 除外済）。
- t_draft = `t_draft_frac·T_comp`、上限 ~(2-3)/40·T_comp（MTP ヘッド ~1層。SP-MoE の別 draft-model
  〔L_all 層フル forward〕とは別で窓 ≪ L_all·t_comp）。[assumed/sweep]

## (a) overlap 純価値（mixed, D1 固定, 同一 miss バイト, Δ=cutoff−serial[tok/s]）
| 総GB | T_flash | overlap_eff=0（draft 窓のみ） | overlap_eff=1.0 |
| --- | --- | --- | --- |
| 4 | 35.9ms | +0.8 | +14.0 |
| 6 | 17.6ms | +1.6 | +10.8 |
| 8 | 8.0ms | +2.4 | +6.6 |
| 12 | 2.2ms | +2.2 | +2.2 |

→ **draft 窓単独（overlap_eff=0）では overlap 純価値ほぼゼロ**。draft 窓（~2.5/40 = 6% compute）は狭い。
overlap の価値は **verify 中の cross-layer prefetch（overlap_eff>0）に依存**する。

## (b) φ 律速分析（本評価の核心）
`budget_phi = hideable/T_flash`（時間上限）、`φ = min(φmax, budget_phi)`:
- **overlap_eff=0**: mixed budget_phi 0.06–0.26（4–8GB）< φmax → **draft 窓律速**
  （MTP D1 窓 ~6% compute では IO をほぼ隠せない）。
- **overlap_eff=1**: mixed budget_phi ≥ 0.94 ≥ φmax → **coverage 律速**（0.66/0.77 で頭打ち）。

## 二段の不発理由
1. **draft 窓を広げる路線（D 増 / draft 厚化）は無駄**: 窓 overlap 純価値ゼロ（上表）に加え、
   D2/D3 は streaming 下で悪化（[[step4-poc]] docs/06・[[mixed-precision]] docs/08 既出）。→ 閉路。
2. **verify 中 cross-layer overlap を上げる路線は coverage 律速**: coverage 改善は
   [[positioning]] docs/07 で **trained 0.54 < 既存 cache hit 0.85** ＝ ROI 負と既測。→ 動機薄。

## 留保
- `overlap_eff` / `t_draft_frac` は sweep（assumed）。両端（0 と 1）で同方向ゆえ中間も向き不変。
- verify 中 cross-layer の実 coverage は自モデル未実測（docs/07 の 77% は zero-shot 単発）。
  ただし上限が既存 cache 85% 未満ゆえ実装動機薄。
- `T_flash`=flash 4.18GB/s[measured]。`io 0.146ms`（stream_hitrate.py）は 12.1GB/s 相当＝
  SSD random で非現実（warm 汚染疑い）ゆえ不採用。
- MTP exactness は [[mixed-precision]] docs/08 §7 で 96/96 検証済＝**本評価は性能上限推定のみ、lossless 不変**。
- 比率（Step5）は分母 all4-D0 も同 bracket の overlap を受ける（D0 は IO 律速強→overlap 利得大）ため、
  cutoff/ideal で比率が serial より縮むのは正常（T_cycle は単調）。

## 最終
[[positioning]]（docs/07）の「speed は churn/IO/同期 天井で確定」を**上書きせず追認**。
reach（作業集合を DRAM に収める）は **mixed-precision が本命**（[[mixed-precision]] docs/08）、
prefetch overlap は speed 天井を動かさない。

相互参照: [[positioning]]（docs/07）/ [[step4-poc]]（docs/06）/ [[mixed-precision]]（docs/08）/
[[research-notes]]（docs/01）/ [[verify-forward]]（docs/09）。
