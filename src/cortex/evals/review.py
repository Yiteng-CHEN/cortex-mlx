"""Human review of mined candidates: d=dev, h=holdout, r=reject, s=skip, q=quit."""
from __future__ import annotations

from typing import Callable

from ..config import Config
from ..paths import DataPaths
from .task import EvalTask, load_split, move_task

KEYS = {"d": "dev", "h": "holdout", "r": "rejected"}


def describe(task: EvalTask) -> str:
    st = task.stats
    statement = "\n".join("    " + line for line in task.statement.strip().splitlines()[:15])
    return (
        f"── {task.id}  ({task.repo} {task.base_sha[:8]} → {task.target_sha[:8]})\n"
        f"{statement}\n"
        f"  source: {', '.join(task.source_files)}  "
        f"(±{st.get('source_lines_changed', '?')} lines, ~{st.get('prompt_tokens_est', '?')} prompt tokens)\n"
        f"  tests:  {', '.join(task.test_files)}"
    )


def review(cfg: Config, paths: DataPaths, *, decisions: dict[str, str] | None = None,
           ask: Callable[[str], str] = input, log: Callable[[str], None] = print) -> dict[str, int]:
    """Interactive unless `decisions` (task_id -> d/h/r) is given."""
    counts = {"dev": 0, "holdout": 0, "rejected": 0, "skipped": 0}
    for task in load_split(paths, "candidates"):
        if decisions is not None:
            key = decisions.get(task.id, "s")
        else:
            log("\n" + describe(task))
            key = ""
            while key not in ("d", "h", "r", "s", "q"):
                key = ask("  [d]ev / [h]oldout / [r]eject / [s]kip / [q]uit > ").strip().lower()[:1]
            if key == "q":
                break
        if key in KEYS:
            move_task(paths, task, KEYS[key])
            counts[KEYS[key]] += 1
        else:
            counts["skipped"] += 1
    return counts
