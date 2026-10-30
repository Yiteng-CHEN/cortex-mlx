from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Generation:
    text: str
    seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None
    peak_memory_gb: float | None = None


class Backend(Protocol):
    name: str

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 task: Any = None) -> Generation: ...
