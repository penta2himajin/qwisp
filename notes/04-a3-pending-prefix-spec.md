# A3 pending-prefix (raw backend) — 形式仕様 / 検証可能ゴール定義

**Author (goal owner): Opus.** Branch `feat/raw-verify`. This document is the *contract*.
実装(Haiku)・テスト著者(Sonnet)・敵対的レビュー(Sonnet)・最終確認(Opus)は全て本書の
「§6 検証可能な受け入れゲート」を唯一の合否基準とする。曖昧点は Opus に差し戻す。

---

## 1. ゴール(一文)

raw-Metal spec ループ(`RawSpecRunner`)の **partial-reject 時の "rebuild forward"(logits
不使用の cache 再構築 1 回)を撤廃**し、`[u]+accepted` を **pending prefix** として次 verify
バッチ先頭に融合する(A3)。raw kernel は per-row order-stable(RAWTESTS 保証)ゆえ、MLX 版
(near-lossless L3 のみ許容)と違い **strict lossless を保ったまま** reject-heavy regime を高速化する。

## 2. 現状(非A3)の reject 経路 — 撤廃対象

`swift/Sources/QwispCore/RawSpecRunner.swift` `runSpecLoop`(~275–327)partial-reject 分岐:

```swift
} else {  // partial reject (p < D)
    backend.rollback(snap)                 // ① 直前 1 forward(verify)を undo
    out.append(u); hist.append(u)
    for d in drafts.prefix(p) { out.append(d); hist.append(d) }
    accTok += p; steps += 1
    let rebuildTokens: [Int32] = [Int32(u)] + drafts.prefix(p).map { Int32($0) }
    guard let _ = backend.forward(rebuildTokens) else { return nil }  // ② ★撤廃対象★
    u = evals[p]
}
```

② の `backend.forward(rebuildTokens)` は logits を捨て cache 前進のみ目的の**無駄な forward**。
A3 はこれを消し、`pending += [u] + drafts.prefix(p)` に置換する。

## 3. A3 機構(MLX 版 `TellBolt.swift` ~146–219 / `Tell.swift` ~230–461 を raw へ移植)

- 状態: `var pending: [Int] = []`(コミット済みだが cache 未実現のトークン列)。
- 各反復先頭で cache は **直前コミット境界 B**(pending は cache 未反映)。`snap = backend.snapshot()` は B。
- verify 入力を融合: `verifyTokens = pending.map{Int32} + [Int32(u)] + drafts.map{Int32}`(M=`pk+1+D`, `pk=pending.count`)。
- 判定行オフセット(★訂正: 当初 `pk+1+p` と誤記→正は `pk+p`。row `pk`=u 行が「u の次」の予測=比較起点
  であり skip しない。TellBolt は `vlg[0, pk ..< pk+D+1]` を slice 後 `evals[p]` ゆえ、unsliced raw では
  `evals[pk+p]`)。すなわち `accept prefix p` は
  `while p < D && drafts[p] == evals[pk + p] { p += 1 }`、次 token `u' = evals[pk + p]`。
  (`evals[pk]`=u の次の予測 vs `drafts[0]`。`pk+1+p` にすると drafts[0] を「drafts[0] の次」と比較する
  循環誤りになり accept 崩壊=13/128 mismatch。実装・第2ループの敵対レビューで検出・修正済。)
- full accept(p==D): batched forward が既に cache を B+pk+1+D へ前進済 → `pending=[]` にクリア、
  u+drafts を commit、`u' = evals[pk+D]`。**rollback も rebuild も無し**(A3 の主利得)。
- partial reject(p<D): `backend.rollback(snap)` で cache を B へ戻し、u+accepted を commit、
  `pending += [u] + drafts.prefix(p)`、`u' = evals[pk+p]`。**rebuild forward 無し**。
- **pending cap + flush**: `pending.count` が上限(MLX と同じ **24**)を超えたら flush =
  `backend.forward(pending)` で実現し `pending=[]`(境界 B を前進)。flush 後は snapshot も更新される。
- D==0 経路: pending が非空なら `[pending, u]` を融合(`stepArgmax`)、空なら現状どおり `stepArgmax([u])`。

## 4. 正しさ論拠(なぜ raw では strict lossless か)

**不変量(position-wise causal)**: pre-pending 状態 B から `[pending, u, drafts]` を **1 回 batched
forward** した結果の各行 logits は、B から `forward(pending)` → `stepArgmax([u]+drafts)` と **逐次
2 回** 実行した結果と *数学的に同一*。attention は過去+現在のみ参照、GDN conv/recurrence は causal
ゆえ、行 i の出力は行 `0..i` のみに依存する。

MLX 版はこの等式が **kernel 累積順の batch-shape 依存 drift** で bit では崩れる(near-tie flip)→
margin-cert / M=1 replay で救う(L3 or strict-with-cert)。**raw kernel は per-row order-stable
(RAWTESTS が batched≡M=1-loop を bit で保証)ゆえ drift ゼロ = A3 は cert 不要で bit-exact strict
lossless**。これが本移植の核心的主張であり、§6 の identity oracle が実測で保証する。

**rollback 規約の要検証点**: `RawFusedForward.rollbackOneStep`(`RawFusedVerify.swift` ~1835)は
KV `len` 巻き戻し + GDN ping-pong `swapState()` の **1 forward 限定** undo。A3 では各反復が
「snapshot(B) → batched forward 1 回 → (reject 時)rollback」で **1 forward しか挟まない**ため
rollbackOneStep で足りる *はず*(ping-pong は B を convHist slot に保持し続け、reject 連続でも B は
不変)。**ただしこれは仮説であり実装で保証すべき事項**: flush 経路・full-accept 後の境界前進・
D==0 融合が rollback 規約(snapshot と rollback の間に forward はちょうど 1 回)を破らないことを
テストで固定する。破る設計になったら multi-step KV-len/GDN スナップショットへ拡張する(§6-U3 が検出)。

## 5. 実装制約

- **ファイル**: 変更は原則 `RawSpecRunner.swift`(spec ループ)に限定。rollback 規約拡張が必要と
  判明した場合のみ `RawFusedVerify.swift`(RawFusedForward snapshot/rollback)を触る。
- **フラグ**: `QWISP_RAW_A3=1` で A3 経路、未設定(既定)で現行(非A3)経路。既定は非A3 のまま
  (default 昇格は owner 判断)。両経路を同一ビルドに共存させ、環境変数だけで切替。
- **既存不変**: 非A3 経路の挙動・出力は byte 単位で不変。RAWTESTS 既存 28 本は全 PASS 維持。
- **禁止**: テストを緩める/oracle を書き換える/lossless 判定の閾値を導入する(raw は bit-exact のみ)。
- **[[buddy-determinism-canonical-refs]] の noCopy 寿命罠 / self-check 単独不十分(実重み diff 必須)** 厳守。

### ★5.1 構造要件(必須・G5 の前提)— "flush-before-verify" は不合格
A3 の verify は **必ず 1 本の融合バッチ `verifyTokens = pending + [u] + drafts`**(判定行 `evals[pk...]`)で
行う。**pending を verify とは別に `backend.forward(pending)` で先に実現する "flush-before-verify" 実装は
不合格**(reject ごとに 2 forward = 撤廃したはずの rebuild forward の復活で、A3 の速度目的を全く達成しない)。
- 各反復ちょうど 1 forward:full-accept / partial-reject / D==0(pending 有)いずれも `[pending,(u,)drafts]`
  を 1 回 forward。partial-reject のみ rollbackOneStep で undo。**別 flush forward は cap>24 の安全弁のみ**。
- **maxM の拡張**: 融合で verify 行数は最大 `pendingCap + maxK + 1`(=24+96+1=121)。現状 `maxM=max(maxK+1,64)`
  では足りない → **`maxM = max(pendingCap + maxK + 1, 64)` に拡張**(fused backend の scratch 容量)。
- pending は連続 reject で累積し、full-accept か cap flush で解消。累積が maxM を溢れさせないよう cap で bound。

## 6. 検証可能な受け入れゲート(唯一の合否基準)

すべて **bit-exact**(閾値・確率的許容なし)。Opus が最終的に直接 Bash(GPU 排他)で再実行して確認する。

### G1 — RAWTESTS 単体(書き換え不可・Sonnet 著)
新規 write-locked テストを `RawVerifyTests.swift` に追加(total を更新、`qwisp/test_raw.sh` が緑):
- **T-A3-fuse**: 乱数重みの RawFusedForward で、snapshot B から `stepArgmax(pending+[u]+drafts)` の
  判定行 `evals[pk...]` が、B から `forward(pending)` 後の `stepArgmax([u]+drafts)` と **bit 一致**
  (M を {pk∈[0,3,7,17]} × {D∈[1,4,8]} で振る)。= §4 不変量の kernel 実測。
- **T-A3-reject-rollback**: reject→rollback→pending 拡張→次 verify の系列が、非A3(rollback+rebuild)
  経路の commit 列と **同一トークン列** を出すことを乱数重みで固定(1-step rollback 規約が保たれる証拠)。
- **T-A3-flush**: pending が cap(24)超過時の flush が境界前進を正しく行い、以降 lossless を破らないこと。
- 期待: `RAWTESTS 31/31`(28 + 3)全 PASS。**このテスト群は Haiku が変更してはならない**(§7 で write-lock)。

### G2 — 実重み identity oracle(最重要・偽造不能)
同一 prompt・同一 GEN=128 で **A3 出力 == 非A3 出力**(両者とも exact greedy ゆえ *必ず* byte-identical)。
resident は **必ず fused backend**(`QWISP_RAW_FUSED=1`。未設定だと composed の低速経路 1.3 tok/s になる)。
```
QWISP_RUN=raw-spec QWISP_RAW_FUSED=1 QWISP_RAW_A3=0 QWISP_RAWSPEC_CHECK=1 QWISP_DUMP_TOKENS=1 QWISP_GEN=128 \
  QWISP_MTP_REF=refs/<regime>.safetensors  $BIN stream    # 非A3: OUT_TOKENS_ref, self-check LOSSLESS
QWISP_RUN=raw-spec QWISP_RAW_FUSED=1 QWISP_RAW_A3=1 QWISP_RAWSPEC_CHECK=1 QWISP_DUMP_TOKENS=1 QWISP_GEN=128 \
  QWISP_MTP_REF=refs/<regime>.safetensors  $BIN stream    # A3:   OUT_TOKENS_a3
```
合格条件(regime ごと):
- **A3 の self-check spec-vs-greedy = 128/128 LOSSLESS**。
- **OUT_TOKENS_a3 == OUT_TOKENS_ref(128/128 byte-identical)**。← A3 の正しさの決定的証拠。

**★A3 を実際に駆動する regime を必ず含めること**(非A3 baseline 実測 @fused resident):
- **longctx: accept/step=3.52, 131.9 tok/s** — draft 多発=reject 多発ゆえ A3 の主検証 regime(必須)。
- **agentic: accept/step=0.31, 59.0 tok/s** — 中程度 reject(必須)。
- code: accept/step=0.00(suffixDraft が ≥4-match 無し=pure greedy)→ A3 未駆動で identity は自明に成立
  =**vacuous**。回して LOSSLESS 確認はするが A3 の gate にはならない。shortnl も低 draft。
→ inner loop の smoke は **longctx**(最も A3 を stress)。Opus 最終は 4 regime 全て。

### G3 — stream-vs-resident(A3 on)
`QWISP_RAW_A3=1 QWISP_RAW_C=64 QWISP_RAWSTREAM_CHECK=1` で streaming(C=64)出力が resident(C=256)
出力と **IDENTICAL 128/128**(最低 code+agentic の 2 regime)。

### G4 — 非退行
`QWISP_RAW_A3=0`(既定)で G2 の非A3 出力が本仕様着手前と不変(=既存 canonical と一致)。RAWTESTS の
既存 28 本 PASS 維持。

### G5 — 速度(★ハードゲート・go/no-go)
A3 は lossless ゆえ accept/step は非A3 と同じ。A3 の存在意義は **reject 時 rebuild forward 消去による
高速化**のみ。したがって:
- **構造ゲート(必須)**: §5.1 の融合バッチ `[pending,u,drafts]`(1 forward/反復)であること。
  flush-before-verify(2 forward)は自動不合格。レビューは impl を grep し verify に pending が
  含まれ・別 `forward(pending)` が cap 安全弁以外に無いことを確認する。
- **速度ゲート(go/no-go)**: longctx(reject-heavy)で **A3 tok/s ≥ 非A3 tok/s × 1.05**(明確な改善)。
  - 満たす → A3 採用候補(default 昇格は owner)。
  - **at-parity / 下回る → A3-on-raw は NO-GO**(rebuild forward が拡大 verify に対し十分安く、融合の
    利得が出ない)。この場合も **正しさ(G1–G4)が緑なら実装は温存**し、`QWISP_RAW_A3=1` は opt-in の
    ままとし「raw では速度中立/微負=非採用」と honest に記録(flush 版の −11% は棄却済の別物)。
- 実測参考(非A3 baseline fused resident): longctx 133.3 / agentic 58.4 tok/s。flush-A3(棄却)は
  longctx 118.4(−11%)。真の融合版がこれを上回るかが本ゲートの争点。

## 7. 作業ループ(役割分担)

1. **Opus(本書)**: ゴール定義(完了)。
2. **Sonnet(テスト著者)**: G1 の 3 テストを `RawVerifyTests.swift` に追加、G2/G3 を実行する検証
   スクリプト `qwisp/test_a3.sh`(非A3 と A3 の OUT_TOKENS を dump し byte 比較 + self-check LOSSLESS
   確認、4 regime ループ)を新規作成。**テストは期待挙動を固定するもので、実装に合わせて緩めない。**
   作成後、テスト関連ファイルのハッシュを記録(write-lock 基準)。
3. **Haiku(実装)**: `RawSpecRunner.swift` に `QWISP_RAW_A3` 経路を実装、ビルド、G1(`test_raw.sh`)を
   緑にする。**テストファイル(`RawVerifyTests.swift` の新規テスト・`test_a3.sh`)は変更禁止**。
4. **Sonnet(敵対的レビュー)**: (a) テストファイルが改変されていない(ハッシュ一致)、(b) テストが
   骨抜きでない(assertion が bit-exact・regime を実際に回す)、(c) 実装が §3/§4 に忠実(判定行
   オフセット・cap/flush・rollback 規約)、(d) lossless 違反や rollback 破綻の潜在バグ、を審査。
   必要な修正を Haiku に戻す。**PASS まで 3↔4 をループ**。
5. **Opus(最終確認)**: 直接 Bash(GPU 排他)で G1–G4 を **自ら再実行**し bit で確認。全緑で受理。

## 8. 環境(agents へ渡す確定情報)

- BIN: `swift/.xcode-build-rel/Build/Products/Release/qwisp-poc`(既存・緑)。
- Build: `cd swift && xcodebuild build -scheme qwisp-poc -destination 'platform=macOS' -derivedDataPath ./.xcode-build-rel -configuration Release -skipPackagePluginValidation`(Metal Toolchain 要)。
- MODEL: 既定 `~/.mtplx/models/Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Speed-FP16`(env 不要)。
- refs: `refs/{code,agentic,longctx,shortnl}.safetensors`(present)。
- RAWTESTS: `QWISP_RUN=raw-tests $BIN stream`(乱数重み・model load 無し・数秒)。baseline 28/28。
- 重い実測は**単独プロセス・直列**(GPU 排他)。並列ビルド/実行は artifact 衝突するので禁止。
- multi-session: push 前に必ず `git pull --ff-only origin feat/raw-verify`。commit 毎 push(owner 指示)。auto-commit 禁止(Opus/Fable が gate)。

---

## 9. 最終判定(Opus gate, 2026-07-04)— 正しさ GREEN / 速度 NO-GO(neutral)

**プロセス**: Opus 仕様定義 → Sonnet 書換不可テスト著 → Haiku 実装 → Sonnet 敵対レビュー+修正 →
Opus 直接 Bash 最終確認、の2ループ。第1ループの Haiku 実装は "flush-before-verify"(reject 毎に
2 forward)= A3 の速度目的を達成せず(longctx −11%)ゆえ Opus が却下、真の融合バッチ §5.1 を義務化して
再ループ。敵対レビューが2バグを検出・修正(重複 pendingCap 宣言 / **仕様§3 の off-by-one `pk+1+p`→`pk+p`**)。

**correctness = 完全 GREEN(Opus 実測、単独プロセス GPU 排他)**:
- G1 RAWTESTS **31/31**(a3_fuse_invariant / a3_reject_rollback_equiv / a3_flush_boundary + 既存28)。
- G2 **4 regime(code/agentic/longctx/shortnl)全て A3 出力 == 非A3 出力 byte-identical・self-check 128/128 LOSSLESS**。
- 構造 §5.1 準拠(単一融合 `[pending,u,drafts]`・別 flush は cap≥24 のみ・maxM=pendingCap+maxK+1・非A3 経路不変)。

**speed = NO-GO(neutral、+5% 未達)** — clean interleaved warm(単独ジョブ):
- longctx(accept 3.52, reject-heavy): 非A3 ~117.5 / A3 ~117.7 = **parity(~1.00×)**。
- agentic(accept 0.31): 非A3 ~57.4 / A3 ~59.3 = **+3.3%**。
- code/shortnl(accept 0.00): A3≡非A3(D==0 で経路同一)=速度同一(vacuous)。

**機構(finding)**: MLX では撤廃対象の rebuild forward が重い `forwardHidden`(op 多数+sync 障壁)ゆえ A3 は明確
勝ち(commit b8e0cd4)。**raw は forward が既に dispatch 極小の 1-CB step ゆえ、1 本消しても得は小さく、融合で
verify が pending 分だけ毎 step 肥大するコストと相殺** → 高 accept で wash・低 accept で微勝ち。**A3 の payoff は
backend 依存で、order-stable raw では headroom がほぼ無い**([[nl-encode-bound-fusion]] の「raw は dispatch
最小化済」/ roofline と整合)。

**disposition**: 正しさ完全・速度中立・無害。**default 昇格せず opt-in `QWISP_RAW_A3=1` のまま温存**を推奨
(3テストは order-stability 不変量の回帰ガードとして有用、flag は harmless、将来 backend 変化時に再評価可能)。
別案=impl 撤去し finding のみ記録(中立の複雑さを負わない)。commit 可否含め owner 判断。
