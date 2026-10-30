"""Helpers around MLX APIs whose names moved between versions (mx.metal.* -> mx.*).

UNVERIFIED on hardware: written without an Apple Silicon machine. Every lookup is defensive;
run `cortex models inspect` on the Mac first and fix here if something is missing.
"""
from __future__ import annotations


def _mx():
    import mlx.core as mx
    return mx


def _fn(name: str):
    mx = _mx()
    fn = getattr(mx, name, None)
    if fn is None and hasattr(mx, "metal"):
        fn = getattr(mx.metal, name, None)
    return fn


def peak_memory_gb() -> float | None:
    fn = _fn("get_peak_memory")
    return round(fn() / 1e9, 2) if fn else None


def reset_peak_memory() -> None:
    fn = _fn("reset_peak_memory")
    if fn:
        fn()


def set_memory_limit_gb(gb: float) -> bool:
    fn = _fn("set_memory_limit")
    if not fn:
        return False
    fn(int(gb * 1e9))
    return True


def load_model(model: str):
    from mlx_lm import load
    return load(model)


def find_layers(model):
    """The list of transformer blocks, wherever the architecture keeps it."""
    for path in ("layers", "model.layers", "language_model.model.layers", "language_model.layers",
                 "model.language_model.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (list, tuple)) and obj:
            return list(obj), path
    raise AttributeError("could not locate transformer layers on this model")
