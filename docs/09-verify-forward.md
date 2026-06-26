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

## 4. 実施結果（overhead 削減）

### 4.1 MTP ループ overhead（light rollback）
hybrid の非trimmable KV を **KVCache=trim / ArraysCache=shallow snapshot** で巻き戻し（docs/08 §7）。
ループ overhead 42%→3%（verify 支配に）。

### 4.2 持続 hot バッファ（fast_hot）= 回帰（負の結果）
hot64 を持続 [64,...] バッファ化し GPU remap で gather。**7.7→4.6 tok/s に回帰**（96/96 は維持）。
mlx は大バッファ gather が小 subset の都度 gather より遅い（prior SlottedExpertCache -15% と同根）。
→ default off。**「concat が遅い」前提が誤りだった**（下記）。

### 4.3 真因は pure-python でなく IO syscall 数（精密プロファイル）
gather 91ms の内訳: **pread(IO)=89ms（682 slice=76miss×9 tensor の syscall）／ pure-python（ensure
ループ+concat 構築）=わずか 2.4ms**。`inds.tolist()`→`np.array` 変換も全40層で 50µs（無視可）。
帯域は 76MB/4GB/s≈19ms ＝ **残り 70ms は 682-syscall の latency**。→ **python 高速化でなく IO 並列化**が解。

### 4.4 並列 pread（採用、大きな実勝ち）
`os.pread` は GIL 解放 → `ExpertCache` が miss を **ThreadPoolExecutor で一括並列ロード**
（`ExpertSource.load_expert_slices`）。worker は **8 が最適**（多いと GIL 競合で悪化: 8→149ms,
32→170ms）。

| 指標 | 前 | 後（並列 pread, 8w） |
|:-----|---:|---:|
| gather | 114.8ms | **59.8ms** |
| verify forward | 213ms | **~149ms（1.4×）** |
| AR greedy(6GB/ctx512) | 4.2 tok/s | **6.4 tok/s（1.52×）** |
| MTP spec light | 6.3 tok/s | **7.9 tok/s** |
| 正しさ | — | 96/96 維持 |

### 4.5 async cold prefetch = 無効（負の結果）
前 forward の各層 cold 集合をヒントに背景 warm（`Prefetcher`, `ExpertCache.warm`, スレッド安全化）。
**結果: 効果なし。`prefetch_hits=0`**（warm が新規ロードゼロ）。見かけの +20% は計測順の warmup で、
pf を先に計測すると逆転（pf 6.1 / no-pf 7.7）＝順序効果。**正しさ 96/96**。
- **理由**: 時間的ヒント＝前 forward の cold＝**直前に使った＝まだ resident**（LRU が保持）→ warm スキップ。
  実 miss は「最近未使用の新規 expert」で履歴から予測不能。**LRU が時間的局所性を既に汲み尽くしている**
  （学習予測器 0.54 < cache 0.85 だった理由と同根, [[predictor_eval]]）。
- streaming の miss は本質的に churn で、予測には >0.85 coverage の cross-layer 予測器が要る（未達）。
- コードは opt-in（`--prefetch`, default off）で負の結果として保持。

### 4.6 残る天井
並列 pread 後の内訳: **per-layer tolist 同期 86ms/56%**（GPU-drain churn, docs/07）／gather 60ms/38%。
低リスク策（持続 hot バッファ=回帰 / async prefetch=無効）は出尽くした。**残るは §3(3) Blink 流
GPU-side routing**（固定 shape グラフで host 同期除去）一択で、mlx 制約の正面突破（`mx.compile`+
固定バッファ gather の feasibility 実験、最悪 Metal カーネル自作）が要る高難度策。または許容 RAM を
上げて resident 比率を増やす（streaming 税そのものを減らす）かの設計判断。
