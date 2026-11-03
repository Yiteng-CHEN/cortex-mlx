"""Per-layer activity probe and logit lens for MLX models (Apple Silicon only).

Wraps every transformer block. For each forward pass it records, at the LAST position only,
    contribution = mean|block_out - block_in| / mean|block_in|
i.e. how much that block changed the residual stream for the token being produced. This is a
real signal from the running model (not a stylized one); it is not a claim about "importance".

Optional logit lens: it also keeps every layer's hidden state at the last position and, on take(),
runs them through the model's final norm + output head in ONE batched matmul (the head's weights are
read once, so all layers cost about as much as one extra output pass) to get "the token this layer
would predict if the model stopped here" (nostalgebraist, 2020).

Alignment: mlx-lm computes the forward pass for token t+1 *before* it yields token t (async eval).
So every forward pass is kept as a snapshot and, when token t is yielded, the snapshot of the pass
that PRODUCED token t (the second newest) is reported. See Aligner.

UNVERIFIED parts on hardware are marked in docs/SIGNALS.md. The wrapper forwards unknown attributes
to the wrapped block, so architectures that read e.g. `layer.is_linear` keep working.
"""
from __future__ import annotations

from collections import deque


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


class Aligner:
    """Forward-pass snapshots -> the one that produced the token being yielded.

    Order of forward passes in mlx-lm: prefill chunks..., last prompt step (-> token 0), step(token 0)
    (-> token 1), and only then token 0 is yielded. So at every yield the second newest snapshot is the
    pass that produced the yielded token; older ones (prefill chunks) are dropped.
    """

    def __init__(self):
        self.snaps: deque = deque(maxlen=64)

    def push(self, snap) -> None:
        self.snaps.append(snap)

    def take(self):
        if not self.snaps:
            return None
        if len(self.snaps) >= 2:
            snap, newest = self.snaps[-2], self.snaps[-1]
            self.snaps.clear()
            self.snaps.append(newest)
            return snap
        return self.snaps.popleft()       # last token of a reply: no further pass was started

    def reset(self) -> None:
        self.snaps.clear()


class LayerProbe:
    def __init__(self, container: list):
        if not isinstance(container, list):
            raise TypeError("layer container must be a mutable list")
        self.n = len(container)
        self.enabled = False
        self.lens = None                 # callable(hidden[k, d]) -> (ids[k], probs[k]) or None
        self.lens_layers: set[int] = set()
        self._e: dict[int, object] = {}
        self._h: dict[int, object] = {}
        self.aligner = Aligner()
        # backfill mode: keep every layer's output for the global positions [lo, hi) (see capture_range)
        self.capture: dict | None = None
        for i, block in enumerate(container):
            if not isinstance(block, _Wrapped):
                container[i] = _Wrapped(block, i, self)

    def start(self) -> None:
        self._e, self._h = {}, {}
        self.aligner.reset()

    def capture_range(self, lo: int, hi: int) -> None:
        """Start keeping hidden states of positions lo..hi-1 (counted over all following forward calls)."""
        self.capture = {"lo": lo, "hi": hi, "off": 0, "h": [[] for _ in range(self.n)]}

    def end_capture(self) -> list:
        """Per layer: list of [T_i, d] slices, in position order."""
        c, self.capture = self.capture, None
        return c["h"] if c else []

    def record(self, i: int, x, out) -> None:
        o = out[0] if isinstance(out, tuple) else out
        c = self.capture
        if c is not None:
            t = o.shape[-2]
            a, b = max(c["lo"], c["off"]), min(c["hi"], c["off"] + t)
            if a < b:
                c["h"][i].append(o[0, a - c["off"]:b - c["off"], :])
            if i == self.n - 1:
                c["off"] += t
            return
        import mlx.core as mx

        xl = x[..., -1, :].astype(mx.float32)
        ol = o[..., -1, :].astype(mx.float32)
        self._e[i] = mx.mean(mx.abs(ol - xl)) / (mx.mean(mx.abs(xl)) + 1e-6)
        if self.lens is not None and i in self.lens_layers:
            self._h[i] = ol.reshape(-1)        # batch size is 1 while chatting
        if i == self.n - 1:                    # one forward pass finished: keep it as a snapshot
            self.aligner.push((self._e, self._h))
            self._e, self._h = {}, {}

    def take(self):
        """(energy list or None, [(layer, token_id, prob)] or None) for the token just yielded."""
        import mlx.core as mx

        snap = self.aligner.take()
        if snap is None:
            return None, None
        e, h = snap
        energy = None
        if len(e) == self.n:
            vals = [e[i] for i in range(self.n)]
            mx.eval(vals)
            energy = [round(float(v), 5) for v in vals]
        lens = None
        if self.lens is not None and h:
            layers = sorted(h)
            ids, probs = self.lens(mx.stack([h[i] for i in layers]))
            mx.eval(ids, probs)
            lens = [(l, int(t), round(float(p), 3)) for l, t, p in zip(layers, ids.tolist(), probs.tolist())]
        return energy, lens

    # kept for callers that only want energy (older code paths / tests)
    def collect(self) -> list[float] | None:
        return self.take()[0]
