"""Per-layer activity probe for MLX models (Apple Silicon only).

Wraps every transformer block. For each forward pass it records, at the LAST position only,
    contribution = mean|block_out - block_in| / mean|block_in|
i.e. how much that block changed the residual stream for the token being produced. This is a
real signal from the running model (not a stylized one); it is not a claim about "importance".

UNVERIFIED on hardware until first run on the Mac. Wrapper forwards unknown attributes to the
wrapped block, so architectures that read e.g. `layer.is_linear` keep working.
"""
from __future__ import annotations


class _Wrapped:
    def __init__(self, inner, index: int, probe: "LayerProbe"):
        self._inner = inner
        self._index = index
        self._probe = probe

    def __call__(self, x, *args, **kwargs):
        out = self._inner(x, *args, **kwargs)
        if self._probe.enabled:
            self._probe.record(self._index, x, out)
        return out

    def __getattr__(self, name):          # only called for names not found on the wrapper
        return getattr(self._inner, name)


class LayerProbe:
    def __init__(self, container: list):
        if not isinstance(container, list):
            raise TypeError("layer container must be a mutable list")
        self.n = len(container)
        self.enabled = False
        self._rec: dict[int, object] = {}
        for i, block in enumerate(container):
            if not isinstance(block, _Wrapped):
                container[i] = _Wrapped(block, i, self)

    def record(self, i: int, x, out) -> None:
        import mlx.core as mx

        o = out[0] if isinstance(out, tuple) else out
        xl = x[..., -1, :].astype(mx.float32)
        ol = o[..., -1, :].astype(mx.float32)
        self._rec[i] = mx.mean(mx.abs(ol - xl)) / (mx.mean(mx.abs(xl)) + 1e-6)

    def collect(self) -> list[float] | None:
        """Newest value per layer once every layer has reported; clears the buffer."""
        import mlx.core as mx

        if len(self._rec) < self.n:
            return None
        vals = [self._rec[i] for i in range(self.n)]
        mx.eval(vals)
        self._rec = {}
        return [round(float(v), 5) for v in vals]
