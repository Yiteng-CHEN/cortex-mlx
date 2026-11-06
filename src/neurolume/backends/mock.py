"""Deterministic backends for tests and pipeline demos (no model needed)."""
from __future__ import annotations

from typing import Any, Callable

from ..config import Config
from ..evals.prompt import fence
from ..evals.sandbox import base_files, reference_solution
from .base import Generation


class MockBackend:
    def __init__(self, solver: Callable[[Any], str], name: str = "mock"):
        self.name = name
        self._solver = solver

    def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                 task: Any = None) -> Generation:
        text = self._solver(task)
        n = max(1, len(text) // 4)
        return Generation(text=text, seconds=0.0, prompt_tokens=len(prompt) // 4,
                          completion_tokens=n, tokens_per_second=None)


def make_mock(name: str, mode: str, cfg: Config) -> MockBackend:
    def files_text(files: dict[str, str | None]) -> str:
        return "\n\n".join(fence(p, c or "") for p, c in files.items())

    if mode == "reference":
        return MockBackend(lambda t: files_text(reference_solution(cfg.repo(t.repo), t)), name)
    if mode == "noop":
        return MockBackend(lambda t: files_text(base_files(cfg.repo(t.repo), t)), name)
    if mode == "garbage":
        return MockBackend(lambda t: "I am not sure how to fix this.", name)
    raise ValueError(f"unknown mock mode {mode!r}")
