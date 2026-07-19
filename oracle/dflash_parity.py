# DFlash drafter parity reference generator (#98 phase 2b).
#
# Runs the vendored upstream drafter (dflash_model_mlx.py, verbatim from z-lab/dflash)
# on the real checkpoint with weights cast to float16 (matching the Swift port's dtype),
# on deterministic random inputs, and writes inputs + module-level outputs (final-normed
# hidden, NO embed/lm_head — the exact contract of DFlashDrafter.forward) to a
# safetensors bundle. The Swift side (QWISP_RUN=dflash-parity) replays and compares.
#
# Cases:
#   A: call1 (S=17, L=8) -> out1; call2 (S=5, L=8) -> out2;
#      trim 2 rows (reject rollback) -> call3 (S=3, L=8) -> out3
#   B: fresh cache, S=4100 > sliding_window-1 (crop + windowed-causal mask branch), L=8 -> outB
#
# Usage: <mlx-python> oracle/dflash_parity.py [ckpt_dir] [out_path]

import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
from dflash_model_mlx import DFlashConfig, DFlashDraftModel, _trim_recent_cache


def load_f16(ckpt: Path) -> DFlashDraftModel:
    cfg = json.loads((ckpt / "config.json").read_text())
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_parameters"]["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["dflash_config"]["block_size"],
        target_layer_ids=tuple(cfg["dflash_config"]["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=cfg["dflash_config"]["mask_token_id"],
        rope_scaling=cfg.get("rope_scaling"),
        layer_types=tuple(cfg["layer_types"]),
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=cfg.get("final_logit_softcapping"),
    )
    weights = {}
    for f in ckpt.glob("*.safetensors"):
        for k, v in mx.load(str(f)).items():
            weights[k] = v.astype(mx.float16)  # match the Swift port's f16 cast
    model = DFlashDraftModel(config)
    model.load_weights(list(weights.items()))
    return model


def forward_module(model, noise, ctx, cache):
    """Module-level forward matching DFlashDrafter.forward: final-normed hidden,
    no embed/lm_head (verbatim slice of DFlashDraftModel.__call__)."""
    h_ctx = model.hidden_norm(model.fc(ctx))
    h = noise
    for layer, c in zip(model.layers, cache):
        h = layer(h, h_ctx, model.rope, c)
    return model.norm(h)


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path.home() / ".mtplx/models/z-lab--Qwen3.6-35B-A3B-DFlash")
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/dflash_parity.safetensors"

    model = load_f16(ckpt)
    H = model.config.hidden_size
    F = len(model.config.target_layer_ids) * model.config.hidden_size  # ctx feature dim
    L = 8

    mx.random.seed(0)
    bundle = {}

    def rnd(shape):
        return mx.random.normal(shape).astype(mx.float16)

    # Case A: sequential blocks + trim rollback
    cache = model.make_cache()
    noise1, ctx1 = rnd((1, L, H)), rnd((1, 17, F))
    out1 = forward_module(model, noise1, ctx1, cache)
    noise2, ctx2 = rnd((1, L, H)), rnd((1, 5, F))
    out2 = forward_module(model, noise2, ctx2, cache)
    _trim_recent_cache(cache, 2)  # reject rollback: drop last 2 committed rows
    noise3, ctx3 = rnd((1, L, H)), rnd((1, 3, F))
    out3 = forward_module(model, noise3, ctx3, cache)
    bundle.update(noise1=noise1, ctx1=ctx1, out1=out1,
                  noise2=noise2, ctx2=ctx2, out2=out2,
                  noise3=noise3, ctx3=ctx3, out3=out3)

    # Case B: crop + windowed-causal mask branch (S > sliding_window-1)
    cacheB = model.make_cache()
    noiseB, ctxB = rnd((1, L, H)), rnd((1, 4100, F))
    outB = forward_module(model, noiseB, ctxB, cacheB)
    bundle.update(noiseB=noiseB, ctxB=ctxB, outB=outB)

    mx.eval(*bundle.values())
    mx.save_safetensors(out_path, bundle)
    print(f"[dflash-parity-ref] wrote {out_path}")
    for k in ("out1", "out2", "out3", "outB"):
        a = bundle[k]
        print(f"  {k}: shape={a.shape} mean|x|={mx.abs(a).mean().item():.4f}")


if __name__ == "__main__":
    main()
