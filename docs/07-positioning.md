# 07 — ポジショニングと v0.1 実装計画

competitive 精査（2026-06）を経た Qwisp の確定ポジショニングと実装計画。

## 哲学：極限まで高められた "実用" 性能

**制約機（Apple Silicon・限られた RAM）で、実用に直結する {能力・文脈・速度} を物理限界まで高める。**

「実用」が核：synthetic な速度でなく、実タスク（コーディング／エージェント／長文脈）で使えること。
- **能力**：Qwen3.6-35B-A3B（SWE-bench 73.4＝active の 10倍級に勝つ）。
- **文脈**：64K 以上（ネイティブ 262K。DeltaNet ハイブリッドで長文脈が安い）。
- **速度**：制約機で ≥30 tok/s（インタラクティブに使える実用速度）。

## DS4・SwiftLM の鏡像

| | 目的 | モデル(active) | 速度 |
| --- | --- | --- | --- |
| DwarfStar4 | 巨大を高メモリ機にギリ載せる | 284B | 載れば可 |
| SwiftLM | 巨大 MoE を**汎用に**小機へ | 122-397B (10-22B) | 遅い (5.2 tok/s) を許容 |
| **Qwisp** | **十分賢いモデルを制約機で極限速度** | **35B-A3B (3B)** | **速さが第一目的** |

**モデル選択が thesis**：active 3B＝ストリーム負荷が SwiftLM の 1/3。「能力を保つ最小 active の sweet spot を選び、速さに全振り」＝汎用 SwiftLM には構造的に取れない位置。

## 競合（SharpAI: SwiftLM / TurboQuant-MLX）

- **SwiftLM**：ネイティブ MLX-Swift。expert SSD streaming、**OS ページキャッシュで eviction**（app-LRU は OS-cache に負けると測定し棄却）、**前トークン routing で先読み prefetch（~70%hit）**、**別 draft（Qwen3.5-9B, ~10.6GB 常駐）**で投機、汎用 MoE、runtime **top-k 削減**で高速化（品質劣化）。122B-A10B/M1 Ultra で 5.2 tok/s。
- **TurboQuant-MLX**：3-bit 重み量子化＋KV 量子化。「sparse MoE のメモリ壁＝ディスク帯域の壁」（＝我々の flash-bound 結論）。
- **手元 mtplx**：MTP 投機＋TurboQuant（KV/重み量子化）。**expert streaming は無い**＝補完的。

## 差別化（MTP ヘッドに集約）

1. **MTP-head を prefetch オラクル＋(高B時)投機に**：SwiftLM の ~10.6GB 常駐は**別 draft 9B**が占める。Qwisp はモデル内蔵 **MTP ヘッド（~0.5GB, `mtp.safetensors`）**を使う → 常駐を大幅圧縮、より小さい機へ。
2. **先読みの質**：SwiftLM は前トークン流用（reactive ~70%）。Qwisp は MTP で**実次トークンを予測**→その experts を prefetch（受理 ~88%）→ **hit を 80-90% へ**。
3. **混合精度 expert**：SwiftLM は top-k 削減で高速化（expert を捨て品質劣化）。Qwisp は **top-k 維持で cold expert を低bit化**＝品質保持で速度。
4. **単一モデル特化**（DS4 哲学）：Qwen3.6 の実測ルーティングに全チューニング。

**やらないこと**：汎用化、top-k 削減での品質劣化、reach のためだけの低速化。

**注意（cache policy）**：SwiftLM の「app-LRU < OS-cache」は彼らの**zero-copy mmap** 文脈の話。Python/MLX で expert を copy する我々の経路では OS-cache の無料再利用が効かず**明示キャッシュが必要**＝eviction 方策（LRU→MTP 予測器）の余地は我々の文脈では残る。v0.1 で zero-copy mmap→gather_qmm が Python で可能かを確認し、cache 戦略を決める。

## 4つ組（決定④、速さ第一に reframe）

> **`{Qwen3.6-35B-A3B 4bit（cold→混合精度）, 最低 ≥30 tok/s（狙い 50）, 64K ctx（KV量子化）, フロア 12GB(KV量子化)/16GB(余裕) Apple Silicon}`**

「RAM 予算の制約下で tok/s を最大化」が主、reach は従。

## v0.1 実装計画（Python/MLX、製品化フェーズで Swift）

**目標**：expert を materialize しないカスタムロードで、出力一致を保ちつつ、bounded resident で実生成 → 実 tok/s・実 peak を Max-reach 動作点で測り、シミュ gate を実機検証。prefetch 無しの baseline。

`qwisp/` パッケージ:
- `loader.py` — モデルロード surgery（mlx_lm 骨格＋非expert のみ常駐、switch_mlp.* は未ロード）。ExpertSource（mmap/pread で per-expert スライス）。
- `cache.py` — ExpertCache（明示・bounded、v0.1 は LRU、予測器を差せる IF）。
- `streaming_moe.py` — StreamingSwitchGLU（PoC 4.1 を製品化、cache 経由 gather_qmm）。
- `engine.py` — streaming モデル組立＋生成ループ。
- `verify.py` — full との出力一致（安全網）。
- `bench.py` — tok/s＋peak RSS 計測。

マイルストーン:
1. **ロード surgery**：非expert 常駐、switch_mlp 未ロード → load 後 RSS ≪ 20GB を確認（要点）。
2. **正しさ**：生成して full と出力一致（許容誤差）。
3. **計測**：tok/s＋peak RSS を B・context（64K 含む）で → sim 予測と突き合わせ（gate の実機検証）。
4. **方式決定**：zero-copy mmap（OS-cache）が Python/MLX で可能か → cache 戦略確定。

v0.2 以降：MTP-head prefetch（差別化）／混合精度 expert／KV 量子化（64K）／Speed 動作点（高B で MTP 投機）。

> 関連: go/no-go [[go-no-go-first-read]]、Step4 PoC [[step4-poc]]、MTP×streaming [[mtp-streaming]]。
