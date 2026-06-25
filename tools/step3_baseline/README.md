# Step 3 — 素の MLX ベースライン + 実効 flash 帯域

go/no-go（`../../docs/05-go-no-go-first-read.md`）の未検証仮定を実測で潰す。

## ① bench_flash.py — 実効 flash 帯域

gate の帯域は「**ミス＝キャッシュに無い expert を cold NAND から読む**」速度。
~1.6MB（1 expert 相当）のランダム読み持続スループットを **page cache バイパス**
（macOS `fcntl F_NOCACHE`）で測り、`--flash-bw` に差し戻す。

```bash
python3 bench_flash.py --model "$HOME/.mtplx/models/Youssofal--...-FP16" --reads 300
```

**重要な落とし穴（cache 汚染）**: 直前にモデルをロードしているとシャードが RAM 常駐し、
`nocache≈cached` になって**過大評価**（実測で 15.9GB/s と出たが汚染）。gate に使うのは
cold 値なので、測る前に **`sudo purge`** でキャッシュを落とすこと。purge 後に即再実行。

## ② bench_mlx.py — 素の AR ベースライン

フロア機で素の mlx_lm（hook 無し AR）を回し、context 長別に prefill/decode tok/s と
peak memory を記録。streaming 版が超えるべき基準値＆「載るか」の実測。

```bash
PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
"$PY" bench_mlx.py --model "$HOME/.mtplx/models/Youssofal--...-FP16" --ctx 128,2048,8192 --gen 64
```

実測（4bit、このマシン）: decode ~50-54 tok/s、peak 19.8–21.9GB（@8K で 21.9GB）。
→ 全載りフロア ~22GB＝24GB 機。target 15 tok/s は保守的（実 AR は ~54）。

## 使い方の流れ

1. `sudo purge` → `bench_flash.py` で cold 帯域を実測。
2. その帯域で Step 2 を引き直し: `simulate.py --flash-bw <実測>`。
3. `bench_mlx.py` を量子化版（4bit / unsloth UD-MLX-3bit 等）で回し、tok/s・peak を比較。
4. 「超えるべき基準値」「現実フロア」「実帯域での gate」から 4つ組を確定。
