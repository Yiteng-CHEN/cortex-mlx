"""Run a task's tests against a candidate solution inside a throwaway git worktree.

SAFETY NOTE: not a security sandbox. Each run gets its own detached worktree, a minimal
environment (no tokens/keys are passed through), a wall-clock timeout that kills the whole
process group, and CPU-time / file-size rlimits. It does not block network or filesystem
access. Only mine repositories you trust.
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..config import EvalSettings, RepoConfig
from ..gitutil import git, show_file
from ..paths import DataPaths
from .task import EvalTask

log = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 6000
ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "USER", "SHELL", "TERM")


class UnsafePathError(ValueError):
    pass


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    timed_out: bool
    seconds: float
    output: str

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0


def _limits(cpu_seconds: int):
    def apply():  # runs in the child (POSIX)
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            fsize = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        except Exception:
            pass
    return apply


def minimal_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


def run_command(cmd: str, cwd: Path, *, timeout_s: int, cpu_seconds: int,
                env: dict[str, str]) -> CommandResult:
    start = time.monotonic()
    proc = subprocess.Popen(
        ["/bin/sh", "-c", cmd], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True, preexec_fn=_limits(cpu_seconds),
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
    return CommandResult(
        returncode=None if timed_out else proc.returncode,
        timed_out=timed_out,
        seconds=round(time.monotonic() - start, 3),
        output=(out or "")[-MAX_OUTPUT_CHARS:],
    )


def safe_target(root: Path, rel: str) -> Path:
    """Resolve rel under root, refusing absolute paths, '..' and symlink escapes."""
    if not rel or os.path.isabs(rel) or rel.startswith("~"):
        raise UnsafePathError(f"absolute or empty path not allowed: {rel!r}")
    parts = Path(rel).parts
    if ".." in parts or ".git" in parts:
        raise UnsafePathError(f"path escapes worktree: {rel!r}")
    root_r = root.resolve()
    target = (root / rel).resolve()
    if root_r != target and root_r not in target.parents:
        raise UnsafePathError(f"path escapes worktree (symlink?): {rel!r}")
    return target


def apply_files(root: Path, allowed: list[str], files: dict[str, str]) -> None:
    allowed_set = set(allowed)
    for rel, content in files.items():
        if rel not in allowed_set:
            raise UnsafePathError(f"file not allowed for this task: {rel!r}")
        target = safe_target(root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.write_text(content, encoding="utf-8")


def reference_solution(repo: RepoConfig, task: EvalTask) -> dict[str, str]:
    out = {}
    for f in task.source_files:
        content = show_file(repo.path, task.target_sha, f)
        if content is None:
            raise ValueError(f"{task.id}: {f} missing at target {task.target_sha[:10]}")
        out[f] = content
    return out


def base_files(repo: RepoConfig, task: EvalTask) -> dict[str, str | None]:
    return {f: show_file(repo.path, task.base_sha, f) for f in task.source_files}


class Worktree:
    """Detached worktree at task.base_sha with the target commit's test files checked out."""

    def __init__(self, paths: DataPaths, repo: RepoConfig, task: EvalTask):
        self.paths, self.repo, self.task = paths, repo, task
        self.path = paths.worktrees / f"{task.id}-{uuid.uuid4().hex[:8]}"

    def __enter__(self) -> "Worktree":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        git(self.repo.path, "worktree", "add", "--detach", "--force", str(self.path), self.task.base_sha)
        if self.task.test_files:
            git(self.path, "checkout", self.task.target_sha, "--", *self.task.test_files)
        return self

    def __exit__(self, *exc) -> None:
        remove_worktree(self.repo.path, self.path)


def remove_worktree(repo_path: Path, wt: Path) -> None:
    git(repo_path, "worktree", "remove", "--force", str(wt), check=False)
    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)
    git(repo_path, "worktree", "prune", check=False)


def format_test_command(template: str, test_files: list[str]) -> str:
    return (template
            .replace("{tests}", " ".join(shlex.quote(t) for t in test_files))
            .replace("{python}", shlex.quote(sys.executable)))


@dataclass(frozen=True)
class EvalOutcome:
    runs: list[CommandResult]
    setup: CommandResult | None

    @property
    def all_pass(self) -> bool:
        return bool(self.runs) and all(r.ok for r in self.runs)

    @property
    def all_fail(self) -> bool:
        return bool(self.runs) and not any(r.ok for r in self.runs)

    @property
    def first(self) -> CommandResult:
        return self.runs[0] if self.runs else self.setup


def evaluate(paths: DataPaths, repo: RepoConfig, task: EvalTask, settings: EvalSettings,
             solution: dict[str, str] | None, runs: int = 1) -> EvalOutcome:
    """Apply `solution` (None = leave base files) and run the tests `runs` times."""
    timeout = repo.timeout_s or settings.timeout_s
    env = minimal_env(repo.env)
    with Worktree(paths, repo, task) as wt:
        if solution:
            apply_files(wt.path, task.source_files, solution)
        setup = None
        if repo.setup:
            setup = run_command(repo.setup, wt.path, timeout_s=timeout,
                                cpu_seconds=settings.cpu_seconds, env=env)
            if not setup.ok:
                return EvalOutcome(runs=[], setup=setup)
        cmd = format_test_command(repo.test, task.test_files)
        results = [run_command(cmd, wt.path, timeout_s=timeout,
                               cpu_seconds=settings.cpu_seconds, env=env) for _ in range(runs)]
        return EvalOutcome(runs=results, setup=setup)


def cleanup_stale(paths: DataPaths, repos: list[RepoConfig], dry_run: bool = True) -> list[Path]:
    """Remove leftover worktrees from crashed runs. Logs every action (dry-run first, D-policy)."""
    stale = sorted(p for p in paths.worktrees.glob("*") if p.is_dir()) if paths.worktrees.is_dir() else []
    paths.logs.mkdir(parents=True, exist_ok=True)
    with (paths.logs / "cleanup.log").open("a", encoding="utf-8") as f:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for p in stale:
            f.write(f"{stamp} {'DRY-RUN would remove' if dry_run else 'remove'} {p}\n")
            if not dry_run:
                shutil.rmtree(p, ignore_errors=True)
    if not dry_run:
        for r in repos:
            if r.path.is_dir():
                git(r.path, "worktree", "prune", check=False)
    return stale
