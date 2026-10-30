from __future__ import annotations

from ..config import Config
from .base import Backend


def make_backend(cfg: Config, name: str) -> Backend:
    c = cfg.candidate(name)
    if c.backend == "mock":
        from .mock import make_mock
        return make_mock(name, c.mode or "reference", cfg)
    if c.backend == "mlx":
        from .mlx_backend import MLXBackend
        return MLXBackend(c.model, name=name, chat_template_kwargs=c.chat_template_kwargs)
    if c.backend == "openai_compat":
        from .openai_compat import OpenAICompatBackend
        if not c.base_url:
            raise ValueError(f"candidate {name}: base_url is required for openai_compat")
        return OpenAICompatBackend(c.base_url, c.model, name=name)
    raise ValueError(f"unknown backend {c.backend!r} for candidate {name}")
