# 09 — verify forward の高速化分析と関連研究

実施 2026-06。MTP 投機の overhead 削減後、時間の **97% が verify(main 2-token forward)**。
その内訳を実測し（`qwisp/verify_profile.py`）、関連研究と突き合わせて次の最適化方針を定める。

## 1. verify forward 内訳（実測, ctx512, hot64/cold-B37, 6GB）

| 要素 | ms/fwd | 割合 | 性質 |
|:-----|------:|:----:|:-----|
| cache.gather（IO + concat 再 stack） | 114.8 | 54% | per-forward の expert slice 再連結が主、cold-miss pread が一部 |
| **per-layer `inds.tolist()` 同期（40層）** | 88.2 | 42% | GPU を毎層 drain＝直列化（churn 天井, docs/07） |
| lm_head（vocab 248320） | 2.4 | 1% | 無視可 |
| **合計 verify** | 213 | 100% | = 9.4 tok/s(2/fwd) |

**切り分け**: cold-B=256（ほぼ全 resident）でも gather=120ms（=37 の 114ms とほぼ同）→ gather は
**IO でなく concatenate（毎 forward の再 stack）が主**。`ExpertCache.gather` が U の expert slice を
9 テンソル分 `mx.concatenate` し直すコスト。

**天井の正体**: MTP ヘッド方式（持続 stacked 配列に GPU inds で `gather_qmm`、sync/concat 無し）の
**full-resident AR=31 tok/s** に対し streaming=4-6 tok/s。**streaming 税（per-layer 同期＋per-forward
concat）が 5-6× ギャップ**。lm_head/attention/MTP-head は無関係。

## 2. 関連研究（2025–2026）

- **MoE-SpeQ**（arxiv 2511.14102）: 投機デコード×MoE。verify で k トークンが異なる expert に routing
  → **union を全ロード**（我々の union-miss と同型）。proactive expert prefetch + offloading で対処。
- **Blink (CPU-Free LLM Inference)**: host-device 同期の除去。「**MoE routing は data-dependent だが
  shape-dependent でない**」→ 固定 shape の単一グラフ capture で router 出力を host で解釈せず GPU 内で
  dispatch/gather。→ 我々の per-layer `tolist` 同期を消す原理。MLX 対応物＝`mx.compile`＋固定バッファ。
- **MoEpic**: next-layer の活性 expert を予測 prefetch し転送-計算 overlap。
- **共通知見**: 転送≫計算だと prefetch overlap が効かない。**ただし mixed-precision で cold 転送を
  半減した我々は overlap が効きやすい側**（docs/08 Stage B の overlap ブラケットが効く根拠）。

## 3. 最適化方針（ROI×リスク）

1. **持続 hot-expert バッファ + GPU remap（Blink 流, 低リスク・推奨初手）**: hot64 は常時 resident
   なので [64,...] の persistent stacked 配列を一度だけ作り、`gather_qmm` を **GPU 上の
   expert_id→slot LUT** で引く。hot 側の **concat と CPU 依存を除去**。cold 側のみ従来 cache.gather。
   → gather(54%) の hot 分を削減。hot は frequent ＝ top-8 の多くを占めるので効果大。
2. **async cold prefetch + overlap（MoEpic/MoE-SpeQ 流）**: cold-miss pread を背景スレッド化し GPU
   compute と overlap（Stage B の serial→overlap、sim で 45→79 tok/s 相当）。mixed で cold 半減済が追い風。
3. **mx.compile / 固定 shape 化（Blink 流, 高難度）**: per-layer 同期(42%) の根治。streaming の動的
   shape と Python 副作用が障壁。過去の SlottedExpertCache は mlx in-place 書込で -15% 回帰。

**推奨**: まず (1) 持続 hot バッファ（低リスク、concat 削減）→ (2) async cold prefetch（overlap で IO 隠蔽）。
(3) は mlx の制約調査が要る根治策。lm_head/MTP-head は触らない（無関係）。
