"""Any OpenAI-compatible /v1/chat/completions server (mlx_lm.server, LM Studio, llama.cpp)."""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from .base import Generation


class OpenAICompatBackend:
    def __init__(self, base_url: str, model: str, name: str | None = None, timeout_s: int = 1800):
        self.name = name or model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = timeout_s

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 task: Any = None) -> Generation:
        body = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(self._url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read())
        seconds = time.monotonic() - start
        usage = data.get("usage") or {}
        completion = usage.get("completion_tokens")
        return Generation(
            text=data["choices"][0]["message"].get("content") or "",
            seconds=round(seconds, 3),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=completion,
            tokens_per_second=round(completion / seconds, 2) if completion and seconds else None,
        )
