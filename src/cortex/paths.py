"""Layout of the runtime data directory (D10, D14)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SPLITS = ("candidates", "dev", "holdout", "rejected")


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def evals(self) -> Path:
        return self.root / "evals"

    def split(self, name: str) -> Path:
        if name not in SPLITS:
            raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
        return self.evals / name

    @property
    def holdout_hashes(self) -> Path:
        return self.evals / "holdout-hashes.txt"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def worktrees(self) -> Path:
        return self.root / "worktrees"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    def ensure(self) -> "DataPaths":
        for d in (*(self.split(s) for s in SPLITS), self.results, self.worktrees,
                  self.models, self.logs, self.tmp):
            d.mkdir(parents=True, exist_ok=True)
        return self
