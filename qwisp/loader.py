"""load_streaming — expert を materialize しないロード surgery（(A) アーキの核心）.

mlx_lm.load(lazy=True) は mx.eval(model.parameters()) をスキップ → 全 param が lazy(mmap)
のまま低 RSS でロードされる。その後:
  1. 各 Qwen3NextSparseMoeBlock の switch_mlp を StreamingSwitchGLU に差し替え
     （lazy な expert 重みは参照が外れて materialize されない）。
  2. 非expert だけ mx.eval で materialize。
→ ~16.4GB の expert を常駐させずに起動する。

使い方:
  PY="$HOME/Library/Application Support/MTPLX/runtime-venv/bin/python"
  "$PY" -m qwisp.loader "$HOME/.mtplx/models/Youssofal--...-FP16"   # 自己テスト
"""

from __future__ import annotations

import re
import resource
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .expert_source import ExpertSource
from .streaming_moe import StreamingSwitchGLU

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS: bytes


def load_streaming(model_dir: str, cache=None):
    """expert を常駐させずに streaming モデルをロードして (model, tokenizer, source) を返す。"""
    model, tok = load(model_dir, lazy=True)  # lazy: param は未 eval（低 RSS）
    src = ExpertSource(model_dir)

    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, Qwen3NextSparseMoeBlock):
            m = _LAYER_RE.search(name)
            layer = int(m.group(1)) if m else n
            # switch_mlp を差し替え（lazy な expert 重みの参照を外す → materialize されない）
            mod.switch_mlp = StreamingSwitchGLU(src, layer, cache=cache)
            n += 1
    if n == 0:
        raise RuntimeError("Qwen3NextSparseMoeBlock が見つからない（差し替え失敗）")

    mx.eval(model.parameters())  # 非expert のみ materialize
    return model, tok, src


def _selftest(model_dir: str):
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"[loader] before:        RSS={rss_gb():.2f} GB", file=sys.stderr)
    model, tok, src = load_streaming(model_dir)
    print(f"[loader] after surgery: RSS={rss_gb():.2f} GB  "
          f"(per-expert {src.per_expert_bytes()/1e6:.2f} MB)", file=sys.stderr)

    sampler = make_sampler(temp=0.0)
    ids = tok.encode("def fib(n):\n    ")[:8]
    out = []
    for resp in stream_generate(model, tok, prompt=ids, max_tokens=8, sampler=sampler):
        out.append(resp.text)
    print(f"[loader] after gen 8:   RSS={rss_gb():.2f} GB  mx_peak={mx.get_peak_memory()/1e9:.2f} GB",
          file=sys.stderr)
    print(f"[loader] generated: {''.join(out)!r}", file=sys.stderr)

    rss = rss_gb()
    print(f"\n[loader] VERDICT: "
          + ("PASS — expert 非常駐ロード成立（RSS << 20GB）" if rss < 12
             else f"要確認 — RSS={rss:.1f}GB（experts が materialize された可能性）"),
          file=sys.stderr)


if __name__ == "__main__":
    _selftest(sys.argv[1])
