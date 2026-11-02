"""Per-layer activity probe for MLX models (Apple Silicon only).

Wraps every transformer block. For each forward pass it records, at the LAST position only,
    contribution = mean|block_out - block_in| / mean|block_in|
i.e. how much that block changed the residual stream for the token being produced. This is a
real signal from the running model (not a stylized one); it is not a claim about "importance".

Optional logit lens: for a subset of layers it also keeps the hidden state at the last position and,
on collect(), runs it through the model's final norm + output head to get "the token this layer would
predict if the model stopped here" (nostalgebraist, 2020). Costs roughly one extra output-head matmul per
lens layer per token, so it is off unless the UI asks for it.

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
        self.lens = None                 # callable(hidden[k, d]) -> (ids[k], probs[k]) or None
        self.lens_layers: set[int] = set()
        self._hid: dict[int, object] = {}
        for i, block in enumerate(container):
            if not isinstance(block, _Wrapped):
                container[i] = _Wrapped(block, i, self)

    def record(self, i: int, x, out) -> None:
        import mlx.core as mx

        o = out[0] if isinstance(out, tuple) else out
        xl = x[..., -1, :].astype(mx.float32)
        ol = o[..., -1, :].astype(mx.float32)
        self._rec[i] = mx.mean(mx.abs(ol - xl)) / (mx.mean(mx.abs(xl)) + 1e-6)
        if self.lens is not None and i in self.lens_layers:
            self._hid[i] = ol.reshape(-1)        # batch size is 1 while chatting

    def collect(self) -> list[float] | None:
        """Newest value per layer once every layer has reported; clears the buffer."""
        import mlx.core as mx

        if len(self._rec) < self.n:
            return None
        vals = [self._rec[i] for i in range(self.n)]
        mx.eval(vals)
        self._rec = {}
        return [round(float(v), 5) for v in vals]

    def collect_lens(self) -> list[tuple[int, int, float]] | None:
        """[(layer, token_id, prob)] for the lens layers of the newest token; clears the buffer."""
        if self.lens is None or not self._hid:
            return None
        import mlx.core as mx

        layers = sorted(self._hid)
        h = mx.stack([self._hid[i] for i in layers])
        self._hid = {}
        ids, probs = self.lens(h)
        mx.eval(ids, probs)
        return [(l, int(t), round(float(p), 3)) for l, t, p in zip(layers, ids.tolist(), probs.tolist())]
