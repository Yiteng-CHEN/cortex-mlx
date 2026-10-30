"""Mine eval tasks from git history: commits that change both source and tests (D4)."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Callable

from ..config import EvalSettings, RepoConfig
from ..gitutil import commit_changes, commit_message, show_file, single_parent_commits
from ..paths import SPLITS, DataPaths
from .sandbox import evaluate, reference_solution
from .task import EvalTask, find_task, save_task


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def is_test_file(path: str, patterns: tuple[str, ...]) -> bool:
    name = PurePosixPath(path).name
    for pat in patterns:
        target = path if "/" in pat else name
        if fnmatch.fnmatchcase(target, pat):
            return True
    return False


def is_source_file(path: str, extensions: tuple[str, ...]) -> bool:
    return PurePosixPath(path).suffix in extensions


def content_hash(repo: RepoConfig, target_sha: str, test_files: list[str]) -> str:
    h = hashlib.sha256(target_sha.encode())
    for f in sorted(test_files):
        h.update(f.encode())
        h.update((show_file(repo.path, target_sha, f) or "").encode())
    return h.hexdigest()


@dataclass
class MineReport:
    repo: str
    scanned: int = 0
    accepted: list[str] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)      # cheap static filters
    rejected: dict[str, str] = field(default_factory=dict)  # sha -> reason (after running tests)

    def to_dict(self) -> dict:
        return {"repo": self.repo, "scanned": self.scanned, "accepted": self.accepted,
                "skipped": dict(self.skipped), "rejected": self.rejected}


def candidate_from_commit(repo: RepoConfig, sha: str, parent: str, s: EvalSettings) -> tuple[EvalTask | None, str]:
    """Static checks only (no test runs). Returns (task, '') or (None, skip_reason)."""
    changes = commit_changes(repo.path, parent, sha)
    tests = [c for c in changes if is_test_file(c.path, s.test_patterns)]
    sources = [c for c in changes if not is_test_file(c.path, s.test_patterns)
               and is_source_file(c.path, s.source_extensions)]
    if not tests:
        return None, "no_test_change"
    if not sources:
        return None, "tests_only"
    if any(c.added is None for c in tests + sources):
        return None, "binary"
    if any(c.status not in ("A", "M") for c in sources):
        return None, "source_deleted_or_retyped"
    tests = [c for c in tests if c.status in ("A", "M")]
    if not tests:
        return None, "tests_deleted_only"
    if len(sources) > s.max_source_files:
        return None, "too_many_files"
    lines = sum((c.added or 0) + (c.deleted or 0) for c in sources)
    if lines > s.max_source_lines:
        return None, "too_many_lines"

    source_files = [c.path for c in sources]
    test_files = [c.path for c in tests]
    base_text = "".join(show_file(repo.path, parent, f) or "" for f in source_files)
    target_src = "".join(show_file(repo.path, sha, f) or "" for f in source_files)
    test_text = "".join(show_file(repo.path, sha, f) or "" for f in test_files)
    prompt_tokens = estimate_tokens(base_text + test_text)
    if prompt_tokens > s.context_budget_tokens:
        return None, "over_context_budget"

    message = commit_message(repo.path, sha)
    statement = message + "\n\nMake the following tests pass: " + ", ".join(test_files)
    task = EvalTask(
        id=f"{repo.name}-{sha[:10]}",
        repo=repo.name,
        base_sha=parent,
        target_sha=sha,
        source_files=source_files,
        test_files=test_files,
        statement=statement,
        content_hash=content_hash(repo, sha, test_files),
        stats={
            "source_lines_changed": lines,
            "source_files": len(source_files),
            "prompt_tokens_est": prompt_tokens,
            "solution_tokens_est": estimate_tokens(target_src),
        },
    )
    return task, ""


def mine(paths: DataPaths, repo: RepoConfig, s: EvalSettings, *, limit: int | None = None,
         log: Callable[[str], None] = print) -> MineReport:
    paths.ensure()
    limit = limit or s.max_tasks_per_mine
    report = MineReport(repo=repo.name)
    for sha, parent in single_parent_commits(repo.path, s.max_commits):
        if len(report.accepted) >= limit:
            break
        report.scanned += 1
        if find_task(paths, f"{repo.name}-{sha[:10]}") is not None:
            report.skipped["already_known"] += 1
            continue
        task, reason = candidate_from_commit(repo, sha, parent, s)
        if task is None:
            report.skipped[reason] += 1
            continue

        base = evaluate(paths, repo, task, s, solution=None, runs=s.stability_runs)
        if base.setup is not None and not base.setup.ok:
            report.rejected[sha] = "setup_failed"
            log(f"  reject {sha[:10]}: setup failed")
            continue
        if not base.all_fail:
            report.rejected[sha] = "base_not_failing" if base.all_pass else "flaky_at_base"
            log(f"  reject {sha[:10]}: {report.rejected[sha]}")
            continue
        target = evaluate(paths, repo, task, s, solution=reference_solution(repo, task),
                          runs=s.stability_runs)
        if not target.all_pass:
            report.rejected[sha] = "target_not_passing" if target.all_fail else "flaky_at_target"
            log(f"  reject {sha[:10]}: {report.rejected[sha]}")
            continue

        save_task(paths, task)
        report.accepted.append(task.id)
        log(f"  accept {task.id}: {task.statement.splitlines()[0][:70]}")

    paths.logs.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (paths.logs / f"mine-{repo.name}-{stamp}.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def split_counts(paths: DataPaths) -> dict[str, int]:
    return {s: len(list(paths.split(s).glob("*.json"))) if paths.split(s).is_dir() else 0
            for s in SPLITS}
