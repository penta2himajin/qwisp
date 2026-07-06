# notes/10 — expert-reuse-aware draft rerank(案A)実装仕様 / devloop 契約

**性質**: 研究実験(go/no-go, publishable if demonstrated / negative result も成果)。純粋な高速化タスクでない。
**branch**: feat/raw-verify。**背景**: memory [[expert-reuse-drafting-research]] / [[expert-reuse-drafting-devloop]]。
**target tier**: raw strict streaming spec のみ(bolt/resident は対象外)。owner 判定 2026-07-05。

---

## 0. 狙いと framing(honest)
raw strict streaming spec の draft を「**現在 cache 常駐している expert 側**」に寄せて rerank し、verify の
expert union を縮める。**これは lossless な speed-only 実験**(fidelity でなく)。出力トークン列は不変。

### 勝ち筋 mechanism(driver 一次照合済み, `RawFusedVerify.swift:2981-3112`)
- raw strict streaming の verify は M 行を `partitionChunks`(2981)で「各 chunk の distinct expert
  union ≤ C」に貪欲分割。chunk 毎に `provider.ensure(chunk.experts)`(3103, miss→pread IO)+ chunk 間に
  `flushCB()`(3111 = GPU sync barrier)。
- ∴ verify の per-layer コスト = **(chunk 数)×(ensure+flush sync) + (総 miss)×pread**。
- rerank で draft を **少数 distinct expert** に寄せる → 1 chunk に多く収まる → **chunk 数↓ = sync 税↓**
  (streaming recon の per-step 38% sync)。draft を **常駐 expert** に寄せる → ensure miss↓ = **pread IO↓**
  (32% IO、slow-NAND throttle で顕在化)。

### lossless 不変条件(最重要)
rerank は「どの draft を verify batch に入れるか」だけを変える。accept されるのは exact verify と一致する
token のみ ⇒ **出力は greedy 列に不変**。chunking は実行戦略で数値に無影響。∴ rerank on/off で OUT_TOKENS
byte-identical、かつ両者とも strict-greedy ref に LOSSLESS。**これがゲートの中核**。

### go/no-go の本質リスク
reuseScore は **予測** expert 集合(on-the-fly 観測、不完全)を使う。予測が外れると実 union は縮まず速度利得
が出ない(correctness は verify が保証=無害)。accept 低下(reuse-optimal≠greedy)が chunk/IO 削減を上回れば
no-go。**判定は driver が step5 で bench 実測**(loop は数値を報告するのみ)。

---

## 1. 設計(既存部品の最小改造)

### 1a. on-the-fly token→expert map(calib 不要)
標準 streaming 経路に calib phase は無い。map は **decode 中に観測**して構築する:
- `RawFusedForward.indsCaptureHook: ((Int, [Int32]) -> Void)`(既存, `RawFusedVerify.swift:2846/3096`)は
  per-layer に flat inds `[M*Ktop]` を渡す。row m の expert = `inds[m*Ktop ..< (m+1)*Ktop]`。
- 呼び出し側(RawSpecRunner の verify)は各 verify の batch token 列 `verifyTokens = [u] + drafts` を知る。
  row m ↔ verifyTokens[m]。∴ hook 内で **row→token を対応付け**、`tokenExperts[token][li]` に expert を
  accumulate(Set union)。
- cold-start: map に無い token は reuseScore = neutral(下記)。map は生成が進むほど密になる。

### 1b. reuseScore(t, providers) → Double
```
residentSet(li) = Set(providers[li].cache.slotOf.keys)   // draft 時に query、per-layer
score(t) = Σ_li  |tokenExperts[t][li] ∩ residentSet(li)|          // 常駐との重なり数
       (t が map に無ければ neutral: 全層で「重なり = 期待 Ktop の半分」相当の定数、下記 α で無害化)
```
正規化やスケールの詳細は実装裁量。**要件は「常駐と重なるほど高スコア」の単調性 + 決定性のみ**。

### 1c. suffixDraft の rerank(唯一の数値変更点, `Tell.swift:519-563`)
現状の多数決(547-551)は `counts[t]`(頻度=accept-proxy)のみ。rerank は:
```
weight(t) = Double(counts[t]) * (1.0 + alpha * reuseScore_normalized(t))
```
- `alpha = 0` で **既存挙動と byte-identical**(strict generalization、テストで pin)。
- tie / near-tie を reuse で崩す保守設定〜aggressive まで `alpha`(env `QWISP_REUSE_ALPHA`, 既定は spec で
  0 に固定 = flag-off 相当、experiment で sweep)。
- 実装形: `suffixDraft` に optional 引数 `reuseCtx: ReuseContext?` を追加。nil の時は現行コードパスと
  **完全に同一**(既存 4 呼出箇所は nil 渡しで無改変挙動)。非 nil の時のみ weight 式を使う。

### 1d. 配線(standard streaming path)
- `RawSpecRunner.swift:374-383` の `mkBackend` は `streamingBackend(...).map { $0.0 }` で **fwd/providers を捨てている**。
  streaming の時のみ `fwd` と `providers` を捕捉して保持する(resident 経路は不要ゆえ触らない)。
- `RawSpecRunner.swift:417` の draft 呼出を、env `QWISP_REUSE_RERANK=1` かつ streaming の時のみ
  `reuseCtx` 付きに切替。flag 未設定 or resident では **現行と完全一致**(nil 渡し)。
- `fwd.indsCaptureHook` を verify の直前に「今の verifyTokens で row→token 対応付けする closure」に設定し、
  map を更新。**hook は測定専用で数値に無影響**(既に bolt calib で使われている無害フック)。

### 1e. 効果計測用カウンタ(measurement のみ、数値/lossless に無影響)
- 既存: `LayerExpertCache.missTotal`(miss 数)。
- 追加: `LayerExpertCache.chunkTotal`(partitionChunks が返した chunk 数の累計)を `forwardRows` の
  chunk ループで加算(static, reset は既存 accounting と同所)。rerank on/off で比較する。

---

## 2. Stub signatures(テストが pin する固定 API)
実装者はまず nil/neutral 返しの stub を置き RED を作る(RawVerifyTests idiom)。以下シグネチャは固定:

```swift
// ReuseContext: draft rerank に渡す観測状態(値型 or final class)
public struct ReuseContext {
    // token -> per-layer 観測 expert set。nLayers 個の [Int: Set<Int>] 等、実装裁量。
    public mutating func observe(rowTokens: [Int], layer: Int, inds: [Int32], Ktop: Int)  // 1b/1a: map 更新
    public func reuseScore(token: Int, residentPerLayer: [Set<Int>]) -> Double            // 1b: 常駐重なりスコア
}

// suffixDraft に optional reuse 引数を追加(既存呼出は reuseCtx: nil で無改変)
static func suffixDraft(_ seq: [Int], maxMatch: Int, draftK: Int, minMatch: Int,
                        reuseCtx: (ctx: ReuseContext, residentPerLayer: [Set<Int>], alpha: Double)?) -> [Int]
```
(引数の正確な受け渡し形は実装裁量だが、**reuseCtx=nil で既存挙動 byte-identical** が固定要件。)

---

## 3. GATE(loop の pass/fail 基準 — reviewer はこの節のみを根拠にする)

### G-A. 決定性 unit tests(model-free, RawVerifyTests.swift に追加, 全 PASS 必須)
1. **rerank α=0 恒等**: 合成 seq で `suffixDraft(..., reuseCtx: nil)` と `reuseCtx:(…, alpha:0)` が
   **返り値 byte-identical**(draft 配列一致)。α=0 が strict generalization であることの証明。
2. **reuseScore 単調性/正しさ**: 合成 tokenExperts map + residentSet で reuseScore が期待重なり数を返す
   (手計算値と一致)。常駐重なりが増えるほどスコア単調増。
3. **rerank tie-break**: `counts` 同数の 2 token・reuseScore 異なる合成ケースで、α>0 の時 **高 reuse token が
   勝つ**(決定的)。α=0 では既存 tie-break(最近位置)に一致。
4. **observe 帰属**: verify batch tokens + per-row inds を与え、observe 後の map が各 token に正しい per-layer
   expert set を持つ(row↔token 対応の検証)。
- 既存 56 テストは全て PASS のまま(additive、既存 kernel 無改変)。新 total = 56 + 追加本数。

### G-B. lossless integration(model 要, reviewer が実行)
- `QWISP_REUSE_RERANK=1 QWISP_REUSE_ALPHA=<非0>` で raw-spec self-check(spec≡greedy)が **LOSSLESS 維持**
  (`QWISP_RAWSTREAM_CHECK=1` 経路 or runSelfCheck、既存の 32/32 相当)。
- **OUT_TOKENS byte-identical(rerank on vs off)**: 同一 prompt/C/seed で `QWISP_REUSE_RERANK=1` と未設定の
  OUT_TOKENS が完全一致。rerank が出力を変えない証明。**これが最重要ゲート**。
- flag-off byte-unchanged: `QWISP_REUSE_RERANK` 未設定時、現行バイナリと挙動不変(resident 経路含む)。

### G-C. wiring 検証(reviewer が encode/CPU コードを読む, unit green ≠ 配線)
- standard streaming path が実際に reuseCtx を通して draft しているか(RawSpecRunner:417 経路)を **コード
  読解で確認**。indsCaptureHook が verify 毎に map 更新しているか。resident/bolt 経路が無改変か。
- `alpha=0`/flag-off で nil 経路に落ちることをコードで確認。

### G-D. 効果計測(数値報告のみ、pass/fail にしない — driver が step5 で go/no-go 判定)
- `bench_batch.sh 64 128 1.5 "suffix-spec"` を rerank off / on(α sweep) で実行し報告:
  `LayerExpertCache.chunkTotal`(chunk 数)/ `missTotal`(miss 数)/ tok/s / accept/step。
- throttle 0(fast-SSD)と 1.5(slow-NAND)両方。**loop はこの数値を report するだけ**。

---

## 4. Doctrine(devloop 実装者・reviewer 向け, memory 由来)
- **unit green ≠ 配線**: RAWTESTS が緑でも本番未配線があり得る(GDN wave1 の教訓)。reviewer は G-C で encode/
  CPU パスを読んで配線を確認。
- **歪んだ oracle は実装を歪める**: テスト参照は本番ロジック基準(CPU 再実装で別定義を作らない)。矛盾は STOP して報告。
- **flag-off/α=0 は byte-unchanged/byte-identical が絶対**: 既存挙動を1 bit も動かさない。
- **driver step5 実測が真実**: 単一 run は ±5-6% ノイズ。効果判定(G-D)は loop でなく driver が bench 実測で。
- **no commit / テスト不可侵 / delegation 禁止**: 実装者は commit しない。locked test を書き換えない。
- **GPU 排他**: build/bench は1ジョブずつ(他の重ジョブと同時実行しない)。

## 5. Out of scope(この devloop でやらない)
- 案B(bolt fidelity = residency-biased decoding): 別スレッド(park, owner 指示で recon 別途)。
- MLX StreamingModel 経路(Tell.swift:347)の safeMaxK 統合: raw が product ゆえ raw のみ。
- E' 予測器の高度化(ridge/MTP): まず on-the-fly 観測 map で go/no-go。効けば full 版で。
- default 昇格: 実験成功後に owner 判断。既定は flag-off(α=0)。
