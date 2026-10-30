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
        self._probe = None
        if probe:
            from ..telemetry.probe import LayerProbe
            self._probe = LayerProbe(mlxutil.find_layer_container(self._model))
        self.num_layers = self._probe.n if self._probe else None

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

    def stream_chat(self, messages: list[dict], *, max_tokens: int = 1024, temperature: float = 0.2):
        """Yield {"text", "energy" (per-layer list or None), "tps"} per generated chunk."""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        formatted = self._format_messages(messages)
        if self._probe:
            self._probe.enabled = True
        try:
            for resp in stream_generate(self._model, self._tokenizer, formatted,
                                        max_tokens=max_tokens, sampler=make_sampler(temp=temperature)):
                yield {"text": resp.text,
                       "energy": self._probe.collect() if self._probe else None,
                       "tps": round(getattr(resp, "generation_tps", 0.0) or 0.0, 1) or None}
        finally:
            if self._probe:
                self._probe.enabled = False
