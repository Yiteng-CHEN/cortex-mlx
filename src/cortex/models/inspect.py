"""Inspect a model's block structure: which layers are full attention, linear attention, MLP, MoE.

Verifies the architecture claims in D13 and seeds M1's brain-layout.json.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from ..config import Config
from ..paths import DataPaths

LETTER = {"linear_attention": "L", "attention": "A", "unknown": "?"}


def classify_layer(param_keys: list[str]) -> dict[str, str]:
    """Classify one transformer block from its flattened parameter names."""
    keys = [k.lower() for k in param_keys]
    has = lambda *frags: any(any(f in k for f in frags) for k in keys)  # noqa: E731
    if has("linear_attn.", "gated_delta", "deltanet", "conv1d"):
        mixer = "linear_attention"
    elif has("self_attn.", "attention.", "attn."):
        mixer = "attention"
    else:
        mixer = "unknown"
    if has(".experts.", "switch_mlp", "block_sparse_moe", ".router.", "mlp.gate.weight"):
        ffn = "moe"
    elif has("mlp.", "feed_forward.", "ffn."):
        ffn = "mlp"
    else:
        ffn = "unknown"
    return {"mixer": mixer, "ffn": ffn}


def compress_pattern(letters: str) -> str:
    """'LLLALLLA' -> '(LLLA)x2' when the string is a repeated unit; else run-length encode."""
    n = len(letters)
    for size in range(1, n // 2 + 1):
        if n % size == 0 and letters[:size] * (n // size) == letters:
            return f"({letters[:size]})x{n // size}" if size > 1 else f"{letters[0]}x{n}"
    out, i = [], 0
    while i < n:
        j = i
        while j < n and letters[j] == letters[i]:
            j += 1
        out.append(f"{letters[i]}{j - i if j - i > 1 else ''}")
        i = j
    return "".join(out)


def summarize_layout(layers: list[dict]) -> dict:
    mixers = Counter(layer["mixer"] for layer in layers)
    ffns = Counter(layer["ffn"] for layer in layers)
    letters = "".join(LETTER.get(layer["mixer"], "?") for layer in layers)
    return {"num_layers": len(layers), "mixers": dict(mixers), "ffn": dict(ffns),
            "pattern": compress_pattern(letters) if letters else ""}


def inspect_model(cfg: Config, paths: DataPaths, name: str) -> tuple[dict, Path]:
    from mlx.utils import tree_flatten

    from ..backends import mlxutil

    c = cfg.candidate(name)
    if c.backend != "mlx":
        raise ValueError(f"inspect needs an mlx candidate, {name} is {c.backend}")
    start = time.monotonic()
    model, _ = mlxutil.load_model(c.model)
    load_s = time.monotonic() - start
    blocks, where = mlxutil.find_layers(model)
    layers = []
    for i, block in enumerate(blocks):
        flat = tree_flatten(block.parameters())
        info = classify_layer([k for k, _ in flat])
        info.update(index=i, bytes=int(sum(v.nbytes for _, v in flat)),
                    modules=sorted({k.split(".")[0] for k, _ in flat}))
        layers.append(info)
    total = int(sum(v.nbytes for _, v in tree_flatten(model.parameters())))
    report = {
        "candidate": name, "model": c.model, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "load_seconds": round(load_s, 1), "layers_attr": where,
        "weights_gb": round(total / 1e9, 2), "peak_memory_gb": mlxutil.peak_memory_gb(),
        "summary": summarize_layout(layers), "layers": layers,
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    out = paths.results / f"inspect-{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report, out
