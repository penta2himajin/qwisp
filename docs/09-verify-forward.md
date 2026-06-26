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

### 4.6 段階分解で真因確定（`qwisp/full_profile.py`）
同じ 2-token verify を engine 構成で段階測定（ctx512）:

| 構成 | model ms | tok/s | |
|:-----|------:|:----:|:--|
| A resident（純正4bit, GPU gather, **tolist無し**） | 19.4 | **94** | 床（MTPLX 94.5 と一致）|
| B stream@256（全常駐・**tolistあり**） | 115.9 | 16.8 | IO ゼロでも 6× 遅い |
| C all4@37（miss） | 224.8 | 8.8 | |
| D mixed@256 | 126.8 | 15.5 | |
| E mixed@37（現行 streaming） | 148.3 | 13.3 | |
| **F GPU-routed mixed（全常駐, tolist排除）** | **24.1** | **77.9** | |

**税の分解**: B−A=**+96ms 同期税**（streaming税130の74%）／ E−D=**+8ms mixed IO税（ほぼ解決）**／
D−B=+26ms two-gather税。**IO ゼロの B でも A の 6× 遅い＝ボトルネックは IO でなく per-layer 同期**
（tolist のデータ50µs でなく、毎層 GPU パイプライン drain による損失）。

### 4.7 GPU-routing = 実証成功（`qwisp/gpu_routed.py`）
`GPURoutedMixedSwitchGLU`: hot(4bit)/cold(2bit) 全 expert を**持続 GPU バッファ**に置き、**GPU の
inds LUT で直接 gather_qmm**（CPU 往復・tolist ゼロ）。結果 **F=24.1ms/77.9tok/s**＝D(15.5) から
同期税 102.7ms を回収（**5×**）、ネイティブ A(94) との差 ~5ms は two-gather のみ。**正しさ 64/64**
（resident-4bit 参照と一致, mixed GREEN）。fast_hot 回帰の原因（cold 側が cache+tolist のまま）を
両側 GPU-route で解消。

**適用条件と射程**: GPU-routing は per-layer の miss 同期が不可能＝**全 expert 常駐前提**。
mixed 全常駐 = 12GB → **18–24GB Mac**。全4bit常駐(18GB)なら 24–32GB。**8–16GB の真 streaming 域は
per-layer 同期が構造的に残る**（experts がディスク）。

### 4.8 end-to-end 実測（MTP 統合 + 最適化ループ）
`mtp_decode --gpu-routed`。**tok/s は decode-only**（prefill 除外, stream_generate と同条件）。正しさ **192/192**。

| ループ | AR tok/s | +MTP D1 |
|:-----------|:--------:|:-------:|
| naive（per-token .item, prefill 込み計測）| 20 | 25（計測アーティファクト）|
| stream_generate（AR 参照）| 58.3 | — |
| **tight（配列直接 feed + async, 同期1/step, decode-only）** | **52.1** | **69.9（1.34×）** |

**ループ最適化の肝**: ① decode-only 計測（prefill を分母から除外）② d を materialize せず GPU 配列で
`[u,d]` を verify に渡し d/v/w を1回の tolist で（同期 2→1/step）③ 配列直接 feed + `async_eval`。
→ **GPU-routed mixed resident + MTP D1 = 70 tok/s**（AR 52、stream_generate 58）。

### 4.9 デプロイ確定値（18–24GB Mac, mixed-resident 12GB）

| 構成 | decode tok/s |
|:-----|:--:|
| 旧 streaming | 6–13 |
| GPU-routed mixed（AR） | 52–58 |
| **GPU-routed mixed + MTP D1** | **~70** |

→ デプロイは「**載るなら GPU-routed resident + (MTP)**、載らないなら streaming」の二段。
残課題: (2 保留) two-gather 融合カーネルで 78→~90 / forward 天井引き上げ、(3) 16GB に 12GB を収める KV/OS 切り詰め。
