#!/usr/bin/env python3
"""Dump an exact, strided sample of weights per layer for the web UI (Mac only, needs MLX).

  uv run --extra mlx python -m neurolume.tools.dump_weights --candidate devstral-small-2
  uv run --extra mlx python -m neurolume.tools.dump_weights --repo org/model --key my-model   # a library model

For each layer and each of 4 matrices, takes a GRID x GRID sample at evenly spaced
(row, col) positions of the dequantized weight. Values are the model's real weights.
Output: <data_dir>/weights/weights-<key>.json (or --out DIR). The UI also starts this
from the model library ("生成权重图").
"""
import argparse
import json
import sys
from functools import reduce
from pathlib import Path


MATS = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"]


def pick(n: int, grid: int) -> list[int]:
    return [round(k * (n - 1) / (grid - 1)) for k in range(grid)]


def dump_from_model(model, model_id: str, key: str, grid: int, out_dir: Path, log=print) -> Path:
    """Strided exact sample of each layer's 4 weight matrices from an already loaded model.

    Used by the CLI above and by the server, which runs it on the model that is already in memory
    (no second load). Works through the layer probe's wrapper, which forwards attributes.
    """
    import mlx.core as mx
    from neurolume.backends import mlxutil

    blocks, where = mlxutil.find_layers(model)
    layers = []
    for i, blk in enumerate(blocks):
        entry = []
        for name in MATS:
            try:
                mod = reduce(getattr, name.split("."), blk)
            except AttributeError:       # e.g. linear-attention layers have no q_proj
                entry.append({"name": name, "shape": [1, 1], "std": 1e-6, "v": [0.0] * (grid ** 2)})
                continue
            w = mod.weight
            if hasattr(mod, "scales"):
                w = mx.dequantize(w, mod.scales, getattr(mod, "biases", None),
                                  group_size=mod.group_size, bits=mod.bits)
            rows, cols = w.shape
            ri, ci = pick(rows, grid), pick(cols, grid)
            sub = w[mx.array(ri)][:, mx.array(ci)].astype(mx.float32)
            sd = float(mx.std(w.astype(mx.float32)))
            entry.append({"name": name, "shape": [rows, cols], "std": round(sd, 6),
                          "v": [round(x, 5) for x in sub.reshape(-1).tolist()]})
            del w, sub
        layers.append({"i": i, "m": entry})
        log(f"layer {i + 1}/{len(blocks)}")
    out = Path(out_dir) / f"weights-{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"candidate": key, "model": model_id, "grid": grid,
                               "layers_attr": where, "layers": layers}, separators=(",", ":")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", help="candidate name from the config")
    ap.add_argument("--repo", help="Hugging Face repo id (for library models)")
    ap.add_argument("--key", help="output key when using --repo")
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--out", help="output folder (default: <data_dir>/weights)")
    a = ap.parse_args()
    if not a.candidate and not (a.repo and a.key):
        ap.error("give --candidate, or --repo and --key")

    import mlx.core as mx
    from neurolume.backends import mlxutil
    from neurolume.config import apply_model_env, load_config

    cfg = load_config(None, None)
    apply_model_env(cfg)
    if a.candidate:
        c = cfg.candidate(a.candidate)
        model_id, tok, key = c.model, c.tokenizer_config, a.candidate
    else:
        model_id, tok, key = a.repo, None, a.key
    model, _ = mlxutil.load_model(model_id, tok)
    out = dump_from_model(model, model_id, key, a.grid, Path(a.out or (cfg.data_dir / "weights")))
    print("saved:", out, f"({out.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
