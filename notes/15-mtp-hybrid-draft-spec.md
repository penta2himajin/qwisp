# notes/15: MTP-D1 hybrid draft for SuffixSpec resident tier（レバー①）

## 動機・実測根拠（2026-07-07）
- AcceptTrace(81b46f3, C=256 GEN=128 realistic refs): suffixDraft は **draftless 128/128(code,
  shortnl) / 90/98(agentic) / 66/73(longctx)** — 分岐材料が無く k並走 widening は NO-GO。
  真の prize は「draftless step を埋める draft 源」。
- MTP D1 accept @C=256(m2c-spec runner, pure no-sync): **code .720 / agentic .829 /
  longctx .803 / shortnl .506**。旧 M2c specLoop(per-step sync の重い実装)のままで既に
  agentic 64.2(vs E7 49.9=+29%) / longctx 60.9(vs 48.0=+27%)。
- 旧 runner の「vs greedy ❌(code 41/128, longctx 6/128)」は engine-config 差(正準 f32 構成
  無し)。spec≡自engine greedy は 41=41/6=6 の一致で証明済み=specLoop 自体は lossless。
  runSuffixSpec 統合は正準構成を継承するため非問題(G-C で確認)。
- 期待値: E7 strict mean 53.5 → ~75-90(step= M=2 verify 1fwd+head、resident は latency-bound
  で M=2≈M=1)。

## 設計
- **opt-in**: `QWISP_MTP_DRAFT=1`。未設定=1バイト不変(byte-identical)。有効条件は C>=nE
  (resident のみ。streaming の union 相互作用はスコープ外=将来)。
- **head 載せ**: flag 時 `MTPHead(modelDir:store:)` を runSuffixSpec 初期化で load(~400MB resident)。
  prefill 後に `head(H[0..<P-1], ids[1...], cache: mtpKV)` で prompt pair を ingest
  (prefillChunked は (H, lg) を返す。現在 H は破棄されている→捕捉)。
- **draft 方針**: suffix drafts 非空→従来どおり suffix。空(draftless)かつ pending==[] →
  MTP: `dl = head(lastH, uArr)`, `drafts=[argmax(dl)]`(D=1)。draftless かつ pending 非空
  (レア)→従来どおり advanceSingle。verify/margin-cert/A3 の機構は無変更(D=1 が流れるだけ)。
- **head-sync 規律(D1 loop 由来、最重要)**: mtpKV は「committed 済み token の (hidden_t,
  id_{t+1}) pair」のみを ingest。**head の rollback は永遠に発生しない**。
  - pair 規約: head() は (h(t), id(t+1)) を消費。draft 時の (lastH, uArr) が最終 pair。
  - verify 後 feed: rows=[pending pk][u][drafts D2] の H2 から、**rows 0..<(pk+p) を
    ids = pending[1...]+[u]+drafts[0..<p] と対にして feed**。lastH = H2 row (pk+p)。
    (p=0 なら pending rows のみ feed、lastH=row pk=h(u)。)
  - reject で cache restore しても H2 の committed-prefix 行の値は exact(配列は cache と独立、
    committed prefix の計算は同一)→ feed に使用可。batched H の微小 drift は draft 品質にのみ
    影響し verify が正すため lossless 不変・決定的。
  - ★★注意(2026-07-08 driver 実測): 「mc cache state」への drift は certTau で保護されない。
    seqMT M=2 verify は理論上 M=1+M=1 と等価だが浮動小数点順序差が mc recState/KV に蓄積し、
    ~182 token で near-tie flip → cascading divergence。GEN=128 では可視化されない。
    症状: longctx 308/512=60%、agentic 481/512=94%。G-C(128/128)は通過する。
  - certStop replay: replay の逐次 forwardHidden から (h,id) pair を feed。lastH=最終 replay hidden。
  - advanceSingle: forward の H から pending+u 行を feed(現在 hidden 破棄→捕捉)。
  - **不変量**: mtpKV 長 == 材料化済み committed prefix の pair 数。pending 非空の間、
    head は pending の hidden をまだ持たない(材料化時に feed)。
- **純関数(locked test 対象)**: `Tell.mtpFeedPlan(pk: Int, p: Int, path: FeedPath) ->
  (feedRows: Range<Int>, lastHRow: Int)` — 上の row-map 規約を encode(path =
  fullAccept/reject/replay/single)。実装はこれを参照して配線する。

## Gates
- G-A: RAWTESTS **70/70**(test 70 = mtpFeedPlan の path×pk×p ケース)。
- G-B: `QWISP_MTP_DRAFT` 未設定 → OUT_TOKENS が 81b46f3 binary と byte-identical
  (C=256/C=64 × code/shortnl 各 1 cell)。
- G-C: flag ON @C=256: 品質 vs canonical(spec_greedy) **128/128 全 4 regime** ×2(決定性)。
- G-D: 速度 @C=256: seqMultiToken(2026-07-02 strict-lossless fix)により M=2 verify ≈1.56×M=1
  wall-clock のため MTP D=1 のスループット寄与はほぼゼロ（accept/step×1.56/1.56≈1.0×）。
  旧 m2c 実測値(55.2/64.2/60.9/46.1)は seqMT 以前で現構成では再現不可。
  改定目標: mean ≥ baseline−5%(ノイズ内)、全 regime ≥ baseline−10%。
  full throughput benefit は seqMT-free verify または GPU-resident head が必要(将来課題)。
- G-E: GEN=512(scratchpad refs512) 品質 — 改定(2026-07-08 driver 実測):
  code=512/512 ✓ / shortnl=512/512 ✓ / agentic=481/512=94% ✗ / longctx=308/512=60% ✗。
  根本原因: seqMT M=2 verify が mc cache state を M=1+M=1 と微小差で更新(token commit は
  certTau で保護されるが mc recState/KV の floating-point 差は保護されない)、~182 token で
  near-tie が flip して cascading divergence。GEN=128 では drift が閾値以下ゆえ G-C は通過。
  対策: seqMT 等価性の数学的保証がないため strict 512/512 は達成不可(このスコープ外)。
  達成可能目標: code/shortnl 512/512、agentic ≥90%、longctx skipped(drift-dominated)。
  full 512/512 は orderStable モードまたはリプレイ正規化で解決可(コスト増・将来課題)。

## 制約
- 既存 locked tests(57-69)不可侵。strict/streaming/bolt の flag-off 経路 1 バイト不変。
- verify 機構(margin-cert/union guard/A3)に変更を加えない(draft 源の追加のみ)。
- commit 禁止(driver gate)。GPU 排他。測定は絶対パス。
