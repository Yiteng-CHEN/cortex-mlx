#!/usr/bin/env python3
"""Dump an exact, strided sample of weights per layer for the web UI (Mac only, needs MLX).

  uv run --extra mlx python scripts/dump_weights.py --candidate devstral-small-2
  uv run --extra mlx python scripts/dump_weights.py --repo org/model --key my-model   # a library model

For each layer and each of 4 matrices, takes a GRID x GRID sample at evenly spaced
(row, col) positions of the dequantized weight. Values are the model's real weights.
Output: <project>/data/weights-<key>.json  (data/ is gitignored). The UI also starts this
from the model library ("生成权重图").
"""
import argparse
import json
import sys
from functools import reduce
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MATS = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"]


def pick(n: int, grid: int) -> list[int]:
    return [round(k * (n - 1) / (grid - 1)) for k in range(grid)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", help="candidate name from config/cortex.toml")
    ap.add_argument("--repo", help="Hugging Face repo id (for library models)")
    ap.add_argument("--key", help="output key when using --repo")
    ap.add_argument("--grid", type=int, default=12)
    a = ap.parse_args()
    if not a.candidate and not (a.repo and a.key):
        ap.error("give --candidate, or --repo and --key")

    import mlx.core as mx
    from cortex.backends import mlxutil
    from cortex.config import apply_model_env, load_config

    cfg = load_config(None, None)
    apply_model_env(cfg)
    if a.candidate:
        c = cfg.candidate(a.candidate)
        model_id, tok, key = c.model, c.tokenizer_config, a.candidate
    else:
        model_id, tok, key = a.repo, None, a.key
    model, _ = mlxutil.load_model(model_id, tok)
    blocks, where = mlxutil.find_layers(model)

    layers = []
    for i, blk in enumerate(blocks):
        entry = []
        for name in MATS:
            try:
                mod = reduce(getattr, name.split("."), blk)
            except AttributeError:       # e.g. linear-attention layers have no q_proj
                entry.append({"name": name, "shape": [1, 1], "std": 1e-6, "v": [0.0] * (a.grid ** 2)})
                continue
            w = mod.weight
            if hasattr(mod, "scales"):
                w = mx.dequantize(w, mod.scales, getattr(mod, "biases", None),
                                  group_size=mod.group_size, bits=mod.bits)
            rows, cols = w.shape
            ri, ci = pick(rows, a.grid), pick(cols, a.grid)
            sub = w[mx.array(ri)][:, mx.array(ci)].astype(mx.float32)
            sd = float(mx.std(w.astype(mx.float32)))
            entry.append({"name": name, "shape": [rows, cols], "std": round(sd, 6),
                          "v": [round(x, 5) for x in sub.reshape(-1).tolist()]})
            del w, sub
        layers.append({"i": i, "m": entry})
        print(f"layer {i + 1}/{len(blocks)}", flush=True)

    out = ROOT / "data" / f"weights-{key}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"candidate": key, "model": model_id, "grid": a.grid,
                               "layers_attr": where, "layers": layers}, separators=(",", ":")))
    print("saved:", out, f"({out.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
