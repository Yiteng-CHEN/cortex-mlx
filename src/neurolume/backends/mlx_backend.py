"""In-process MLX backend (Apple Silicon only).

UNVERIFIED on hardware: follows the mlx-lm API (load / stream_generate / make_sampler and the
GenerationResponse fields text, generation_tokens, generation_tps, peak_memory). The first
real run is `scripts/m0-bench.sh`; fix this file if the installed mlx-lm differs.
"""
from __future__ import annotations

import time
from typing import Any

from . import mlxutil
from .base import Generation


class MLXBackend:
    def __init__(self, model: str, name: str | None = None,
                 chat_template_kwargs: dict[str, Any] | None = None,
                 tokenizer_config: dict[str, Any] | None = None,
                 memory_limit_gb: float | None = None,
                 probe: bool = False, adapter_path: str | None = None):
        if memory_limit_gb:
            mlxutil.set_memory_limit_gb(memory_limit_gb)
        self.name = name or model
        self._template_kwargs = dict(chat_template_kwargs or {})
        self._model, self._tokenizer = mlxutil.load_model(model, tokenizer_config, adapter_path)
        # LoRA layers must be collected before the probe wraps the blocks (the wrapper is not an nn.Module).
        self._lora = mlxutil.lora_modules(self._model) if adapter_path else []
        self._lora_scale = [m.scale for m in self._lora]
        self.has_adapter = bool(self._lora)
        self._probe = None
        if probe:
            from ..telemetry.probe import LayerProbe
            self._probe = LayerProbe(mlxutil.find_layer_container(self._model))
        self.num_layers = self._probe.n if self._probe else None
        self._lens_fn = None

    # ---- adapter on/off without reloading (for before/after comparisons)
    def set_adapter_enabled(self, on: bool) -> None:
        for m, s in zip(self._lora, self._lora_scale):
            m.scale = s if on else 0.0

    # ---- logit lens
    def _lens(self):
        if self._lens_fn is None:
            norm, head = mlxutil.find_output_head(self._model)
            import mlx.core as mx

            def fn(h):
                logits = head(norm(h)).astype(mx.float32)
                probs = mx.softmax(logits, axis=-1)
                ids = mx.argmax(probs, axis=-1)
                return ids, mx.take_along_axis(probs, ids[:, None], axis=-1)[:, 0]
            self._lens_fn = fn
        return self._lens_fn

    def decode_token(self, tid: int) -> str:
        try:
            return self._tokenizer.decode([tid])
        except Exception:
            return "?"

    def _format(self, prompt: str) -> str:
        return self._format_messages([{"role": "user", "content": prompt}])

    def _format_messages(self, messages: list[dict]) -> str:
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self._template_kwargs)
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 task: Any = None) -> Generation:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        formatted = self._format(prompt)
        mlxutil.reset_peak_memory()
        chunks: list[str] = []
        last = None
        start = time.monotonic()
        for resp in stream_generate(self._model, self._tokenizer, formatted,
                                    max_tokens=max_tokens, sampler=make_sampler(temp=temperature)):
            chunks.append(resp.text)
            last = resp
        seconds = time.monotonic() - start
        peak = getattr(last, "peak_memory", None)
        return Generation(
            text="".join(chunks),
            seconds=round(seconds, 3),
            prompt_tokens=getattr(last, "prompt_tokens", None),
            completion_tokens=getattr(last, "generation_tokens", None),
            tokens_per_second=round(getattr(last, "generation_tps", 0.0) or 0.0, 2) or None,
            peak_memory_gb=round(peak, 2) if peak else mlxutil.peak_memory_gb(),
        )

    def stream_chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.2,
                    lens: bool = False):
        """Yield {"text", "energy" (per-layer list or None), "tps", "lens"} per generated chunk."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        formatted = self._format_messages(messages)
        if self._probe:
            self._probe.start()
            self._probe.enabled = True
            if lens:
                try:
                    self._probe.lens = self._lens()
                    self._probe.lens_layers = set(range(self._probe.n))   # all layers: one batched head pass
                except AttributeError:
                    self._probe.lens = None
        try:
            for resp in stream_generate(self._model, self._tokenizer, formatted,
                                        max_tokens=max_tokens, sampler=make_sampler(temp=temperature)):
                out = {"text": resp.text, "energy": None,
                       "tps": round(getattr(resp, "generation_tps", 0.0) or 0.0, 1) or None}
                if self._probe:
                    energy, got = self._probe.take()
                    out["energy"] = energy
                    if got:
                        out["lens"] = [[l, self.decode_token(t), p] for l, t, p in got]
                    tok = getattr(resp, "token", None)
                    if tok is not None:
                        out["token"] = self.decode_token(int(tok))
                yield out
        finally:
            if self._probe:
                self._probe.enabled = False
                self._probe.lens = None
