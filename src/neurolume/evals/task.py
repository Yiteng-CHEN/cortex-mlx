"""Eval task manifests (D14): references only, no copied code."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..paths import SPLITS, DataPaths


@dataclass
class EvalTask:
    id: str
    repo: str
    base_sha: str
    target_sha: str
    source_files: list[str]
    test_files: list[str]
    statement: str
    content_hash: str
    split: str = "candidates"
    stats: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "EvalTask":
        data = json.loads(text)
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def task_path(paths: DataPaths, task: EvalTask) -> Path:
    return paths.split(task.split) / f"{task.id}.json"


def save_task(paths: DataPaths, task: EvalTask) -> Path:
    p = task_path(paths, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(task.to_json() + "\n", encoding="utf-8")
    return p


def load_split(paths: DataPaths, split: str) -> list[EvalTask]:
    d = paths.split(split)
    if not d.is_dir():
        return []
    return [EvalTask.from_json(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))]


def find_task(paths: DataPaths, task_id: str) -> EvalTask | None:
    for split in SPLITS:
        p = paths.split(split) / f"{task_id}.json"
        if p.is_file():
            return EvalTask.from_json(p.read_text(encoding="utf-8"))
    return None


def move_task(paths: DataPaths, task: EvalTask, new_split: str) -> EvalTask:
    old = task_path(paths, task)
    task.split = new_split
    save_task(paths, task)
    if old != task_path(paths, task) and old.exists():
        old.unlink()
    if new_split == "holdout":
        add_holdout_hash(paths, task.content_hash)
    return task


def holdout_hashes(paths: DataPaths) -> set[str]:
    p = paths.holdout_hashes
    if not p.is_file():
        return set()
    return {line.strip() for line in p.read_text().splitlines() if line.strip()}


def add_holdout_hash(paths: DataPaths, h: str) -> None:
    if h in holdout_hashes(paths):
        return
    paths.holdout_hashes.parent.mkdir(parents=True, exist_ok=True)
    with paths.holdout_hashes.open("a", encoding="utf-8") as f:
        f.write(h + "\n")
