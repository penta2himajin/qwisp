# GDN 層融合(fusion campaign 第2原子群)— 形式仕様

**Author (goal owner): Fable.** Branch `feat/raw-verify`. Contract = 本書 §4。
ループ: Sonnet 書換不可テスト → GLM-5.2(`--allow` 自走 build、subagent 委譲禁止)→ Sonnet 敵対レビュー → Fable gate。
前提 = notes/06(PoC GREEN、per-dispatch 2.0-2.3µs 実証、paired A/B prof 整備済み)。

## 1. ゴール
GDN 層(30 層 × **21 dispatch** = 630 本/forward = 最大鉱脈)を **21 → 8/層** に融合し
(−390 本 ≈ −0.85ms ≈ M=1 step **+6%**)、全段 **bit-exact**(既存 kernel 連鎖と全要素 bit 一致)を保つ。
`QWISP_FUSE_GDN=1` opt-in(既定 off、既存経路 byte 不変)。M>1 register 圧迫が出た原子は encode 時 M 分岐。

## 2. 現状の GDN 層 dispatch 列(grounding 済、encodeGdnLayerRows ~1281-1322 + encodePreMoE)
pre: ①rmsnorm(input) / 本体: ②qmm4 qkv ③qmm4 z ④qmm4 b ⑤qmm4 a ⑥conv1d_silu_hist ⑦shift_conv
⑧slice q ⑨slice k ⑩slice v ⑪rmsnorm qn(ones) ⑫rmsnorm kn(ones) ⑬scale_mul q ⑭scale_mul k
⑮compute_g_beta ⑯gated_delta_step ⑰rmsnorm coreOut ⑱gate(silu(z)⊙normed) ⑲qmm4 out-proj /
post: ⑳resid_add ㉑rmsnorm(post)。

## 3. 融合原子(risk 順に wave 分割)

### Wave 1(low-risk、−5/層)
- **F1 in-proj 統合(②③④⑤ → 1)**: qkv/z/b/a は同一入力 normed を読む。**重み行列を N 軸で連結**
  (`[convDim+valueDim+Hv+Hv, H]` を build 時に 1 本の 4-bit バッファへ並べる)し、既存 qmm4_rows を
  **1 回**呼ぶ。各出力列の dot は独立ゆえ**連結は bit-exact by construction**(演算不変・layout のみ)。
  下流は offset 読みに変更(sc.qkv/z/bP/aP は同一バッファの slice に)。−3 dispatch + x 3 回再読を削減。
  scales/biases も同様に連結(gs=64 の行独立性を確認して連結)。
- **F3 conv+shift 統合(⑥⑦ → 1)**: 両者は同じ hist+qkv を読む。convOut と histOut を 1 kernel で書く。
  各出力の式は既存と同一(演算順不変)。−1。
- **F4 coreNorm+gate 統合(⑰⑱ → 1)**: per-head rmsnorm(reduction 順を既存 rmsnorm kernel と同一に)
  → 続けて silu(z)⊙ を register 内で。−1。promoteF32 オプションの precision 経路も忠実に。

### Wave 2(F2 は rmsnorm reduction 再現が要注意、−8/層 追加)
- **F2 gdn_prep 統合(⑧⑨⑩⑪⑫⑬⑭⑮ → 1)**: convOut の slice q/k/v + per-head rmsnorm(qn/kn, ones 重み)
  + scale ×2 + compute_g_beta(aP/bP 読み)を 1 kernel に。rmsnorm の **reduction tree を既存 kernel と
  bit 一致**させること(threadgroup サイズ・加算順)。−7。
- **F5 resid+postNorm 統合(⑳㉑ → 1)**: hBuf += mixerOut と postNorm を 1 kernel で両方書く。−1。
  (①input rmsnorm は前層の F5 に畳めるが層境界を跨ぐため campaign 後段へ。)

到達: 21 → 16(wave1)→ 8(wave2)。30 層で −150 → −390 dispatch。

## 4. 受け入れゲート(原子ごと、PoC と同 idiom)
- **G1**: 各融合原子に bit-exact テスト(既存 kernel 連鎖 = 参照、CPU 再実装禁止)+ M 不変性。
  RAWTESTS 33 → 33+N(wave1: +3 = F1/F3/F4、wave2: +2)。書換不可(lock dir locked4)。
  F1 は「連結重み qmm ≡ 4 回個別 qmm」の bit 一致(連結 build 関数も検証対象)。
- **G2**: `QWISP_FUSE_GDN=1` vs `=0` で OUT_TOKENS byte-identical + 128/128 LOSSLESS(code+longctx)。
- **G3**: flag-off 全 byte 不変・既存テスト PASS。
- **G5**: **paired A/B prof(QWISP_PROF_AB 方式を fuseGDN にも拡張)** で M=1 delta を実測。
  gate: wave1 ≥ +250µs、wave1+2 ≥ +600µs(理論: 150/390 本 × 2µs、保守 80%)。
  M=8/17 の delta も報告し、退行 >200µs の原子は encode 時 M 分岐(M≤4 のみ fused 等)を入れる。

## 5. 環境・運用
- notes/06 §5 と同一 + GLM への注意: **内部 subagent への委譲禁止**(ハング実績)。wave 単位で
  glm-code 委譲、各 wave 後に Fable が build+RAWTESTS+G2 smoke。idle-kill されても編集は保存済み
  → まず driver が build+test してから再開判断。
- lock dir: `$CLAUDE_JOB_DIR/tmp/locked4`。commit 毎 push・auto-commit 禁止。

---

## 6. Wave 1 レビュー結果(2026-07-04)と F1/F4 再設計

敵対レビュー(Sonnet): G1 37/37・G2 全 identity(code/longctx/両flag同時)・M分岐 M=8 delta≈0 ✓。
**G5 FAIL(+118µs < 250µs)**: F3 のみ稼働(レビューが未配線を発見し修正、+118µs は F3 理論値どおり)。
F1 は helper のみで本番未配線かつ **concat+slice×4=5 dispatch の自滅設計**。F4 は既存 2 kernel の
wrapper で dispatch 削減ゼロ。**教訓: unit テストは helper を gate するが本番配線は gate しない →
G5 が本番の真実**(RAWTESTS 37/37 は必要条件にすぎない)。

### F1 再設計(demux 型・下流無変更)
`qmm4_inproj_demux_rows`: concat 重み [totalN, H] で qmm する 1 kernel が、**列 n の範囲で出力先を
qkv/z/bP/aP の 4 buffer に書き分ける**(境界は convDim/valueDim/Hv 境界、N/8 threadgroup 境界に整列)。
下流 kernel(conv/gate/compute_g_beta)は従来 buffer をそのまま読む=変更ゼロ。dot 演算は既存
qmm4_rows と同一順=bit-exact。4→1(−3/層)。
### F4 再設計(真融合)
`gdn_norm_gate_rows`: 1 threadgroup per (m, head): coreOut[m,head,:Dv] を既存 rmsnorm と同一の
reduction tree で正規化 → normWeight 適用 → silu(z)⊙ を register で → outV へ。f16/promoteF32 両変種。
2→1(−1/層)。
### 追加修正
- concat/demux 用 buffer 構築は `fuseGDN || QWISP_PROF_AB` 時のみ(現状 ~190MB 無条件確保は 8GB tier に有害)。
- gate: G5 wave1 M=1 median ≥ +250µs(F3+F1+F4 = −5/層 × 30 = −150 dispatch ≈ 330µs 理論)。

## 7. Wave 1 最終判定(Fable gate, 2026-07-04)— 全ゲート GREEN
- G1 39/39(lock byte-identical)/ G2 code+longctx+全flag同時 byte-identical + 128/128 LOSSLESS。
- **G5 fuseGDN M=1 median-of-medians +459µs ≥ 250µs PASS**(prof 2回: +435/+483µs。理論 330µs +
  encoder 側 CPU 削減の上乗せ)。M=8 は noise(σ>>|delta|)=demux/norm_gate に register 退行なし →
  fuseGDN の M 分岐不要。fuseGU M 分岐は M=8 delta≈0 で正常。
- faithfulness: demux dot=qmm4_rows verbatim・境界 straddle なし(1024/1536/1540 全て 8 整列)・
  F4 reduction tree=既存 rmsnorm と byte 一致・flag-off 不変・lazy concat 済。
- 実装=GLM-5.2(redesign は試行2で完遂)。残 concern: F4 pipeline nil 時の silent fallback(将来
  fuseGDN の速度が消えたら最初に疑う)/ A/B prof は M=1 lane のみ gate に使用可。
- **累計(fuseGU M=1 +156µs + fuseGDN +459µs ≈ 0.6ms/step ≈ +4-5%)。次 = Wave 2(F2 gdn_prep 8→1, F5)**。

## 8. Wave 2 最終判定(Fable gate, 2026-07-04)— 全ゲート GREEN
- **真の単一 dispatch kernel**: gdn_prep_rows(⑧-⑮ 8op→1、rmsnorm tree/scale_mul half丸め/compute_g_beta を
  厳密再現)+ gdn_resid_postnorm_rows(⑳㉑→1、全層タイプ適用=attn も −1)。GDN 層 **21→8 dispatch**。
- G1 41/41(lock 照合)/ G2 code+longctx+全flag同時 IDENTICAL + 128/128 LOSSLESS(reviewer + Fable 独立実測)。
- **G5 fuseGDN M=1 median +1402〜1439µs(3実測: reviewer×2 + Fable×1 が相互裏付け)≥ 600µs gate 大幅クリア**。
  M=8 も +995µs(退行なし)。**e2e code 71.5→78.2 tok/s = +9.4%(全flag)**。
- 経緯: GLM 2連続ハング(編集ゼロ)→ Sonnet 切替。Sonnet 1回目は chain 実装+無断 commit(c3b616d)で
  差戻し — その報告から**テスト oracle の欠陥**(scale_mul 参照が MLX 演算で Metal kernel と 1 ULP 不一致
  = Metal は s を half 丸めしてから乗算)を発見・修正(テスト著者、RED 証明付き)→ 真融合で再実装。
- ★学び: **歪んだ oracle は実装を歪める**(実装者は oracle に合わせて本番意味論を曲げた)。
  「テストと本番 kernel の矛盾は STOP して報告」を実装者規約に昇格。レビュー数値への疑義は
  goal-owner の独立実測で解消するのが正(今回 reviewer 1-tool-call 疑惑→実測で真正確認)。
- 残 concern: silent nil-fallback(F2/F5 とも)= fuseGDN の速度が消えたら最初に疑う / gdn_prep_rows は
  headVDim==headKDim(=128)前提(V copy stride)— 形状変更時に要注意。
- **campaign 累計: fuseGU(M=1)+GDN wave1+2 ≈ 1.4-1.6ms/step、code +9.4% 実測。残鉱脈 = attn 層(~150本)
  / MoE shared expert(200本)/ II期 megakernel(I/J)**。
