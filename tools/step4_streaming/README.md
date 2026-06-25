# Step 4 — streaming エンジン（最難関・工学）

go/no-go GREEN（`../../docs/05`）を受けて、AFM3-CA 流の expert streaming を MLX 上に実装する段。
docs `01` E-2 の「MLX に独立 VRAM 無し、offload 先が host DRAM でなく NAND」が核心リスク。

## poc_expert_stream.py — PoC 4.0（E-2 機構の可否、✅PASS）

「expert e をディスクから on-demand で読み、常駐モデルと bit 一致し、読むバイトは
~1.77MB/expert に限定される」を検証。

格納形式（実モデル確認）: `switch_mlp.{gate,up,down}_proj.{weight(U32 4bit), scales(F16), biases(F16)}`、
先頭次元に 256 experts スタック → expert e は連続バイトスライス（safetensors オフセットから pread）。

```bash
PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
"$PY" poc_expert_stream.py --model "$HOME/.mtplx/models/Youssofal--...-FP16"
```

結果（2026-06）:
- **bit 一致**: expert 0/5/100/255 × 全9テンソルが `mx.load`（MLX 正規）と完全一致。✅
- **per-expert = 1.769 MB**（gate の expert_bytes 1.6MB 仮定と整合）。
- VERDICT **PASS** ＝ IO/メモリ機構は成立。compute は MLX 標準 quantized_matmul（bits=4, group_size=64）なので risk 外。

**重要発見**: `mx.load`(5.4GB) が 0.1s ＝ **MLX は safetensors をネイティブに遅延/mmap** ロードする。
→ スタック expert テンソルを mmap 済み遅延配列のまま保持し slice→eval すれば、
**OS ページキャッシュが expert DRAM キャッシュになる**（eviction は OS-LRU）。

> 注意: PoC の cold 計測（8GB/s）は直前 `mx.load` の mmap でページが温まり過大評価。
> gate の帯域は bench_flash の clean 値 **4 GB/s** を使う。

## 設計の分岐（次の意思決定）

| 方式 | 内容 | trade-off |
|---|---|---|
| (a) 手動 pread＋自前キャッシュ | 自分で expert を読み LRU/予測器キャッシュを実装 | **ポリシー制御が効く**（Step 2 の oracle gap を取りに行ける）／実装多い |
| (b) mx.load 遅延 mmap に乗る | スタック tensor を mmap のまま slice→eval、OS にキャッシュ委任 | **実装激減**／eviction は OS-LRU でポリシー制御不可 |

## Step 4 の残り（PoC 後）

1. 1層の streaming MoE forward を組み（top-k experts を cache/mmap から取り compute）full と出力一致を確認。
2. 全40層に拡張＋非expert 常駐の分離ロード。
3. キャッシュ方策（(a) なら予測器寄り、Step2 の勝ち方策）。
4. 混合精度 expert（hot 高bit/cold 低bit）で速度を殺さずサイズ落とし。
5. MTP/投機デコード統合（draft の先読みで expert prefetch）。
