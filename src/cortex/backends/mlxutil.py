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


def active_memory_bytes() -> int | None:
    fn = _fn("get_active_memory")
    return int(fn()) if fn else None


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


def load_model(model: str, tokenizer_config: dict | None = None, adapter_path: str | None = None):
    from mlx_lm import load
    kwargs = {}
    if tokenizer_config:
        kwargs["tokenizer_config"] = tokenizer_config
    if adapter_path:
        kwargs["adapter_path"] = adapter_path
    return load(model, **kwargs)


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


def find_layer_container(model):
    """The actual mutable list of transformer blocks the forward pass iterates.

    mlx-lm models often expose `layers` as a property that returns a fresh slice
    (e.g. `pipeline_layers`), so wrapping that copy would never be seen by the forward pass.
    Only a list that comes back as the *same object* on every access is the stored one.
    """
    for path in ("language_model.model.layers", "model.language_model.layers", "model.layers",
                 "language_model.layers", "layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            again = model
            for part in path.split("."):
                again = getattr(again, part)
        except AttributeError:
            continue
        if isinstance(obj, list) and obj and obj is again:
            return obj
    raise AttributeError("could not locate the stored list of transformer layers")
