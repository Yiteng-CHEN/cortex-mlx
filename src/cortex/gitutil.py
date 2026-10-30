"""Thin wrappers around the git CLI."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True, binary: bool = False):
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=not binary,
    )
    if check and proc.returncode != 0:
        err = proc.stderr if not binary else proc.stderr.decode(errors="replace")
        raise GitError(f"git {' '.join(args)} failed: {err.strip()}")
    return proc.stdout


def show_file(repo: Path, sha: str, path: str) -> str | None:
    """File content at a commit, or None if it does not exist there or is not UTF-8 text."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{sha}:{path}"], capture_output=True
    )
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


@dataclass(frozen=True)
class FileChange:
    status: str        # A, M, D, T ...
    path: str
    added: int | None  # None for binary
    deleted: int | None


def commit_changes(repo: Path, parent: str, sha: str) -> list[FileChange]:
    status = {}
    for line in git(repo, "diff", "--no-renames", "--name-status", parent, sha).splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            status[parts[-1]] = parts[0][0]
    out = []
    for line in git(repo, "diff", "--no-renames", "--numstat", parent, sha).splitlines():
        a, d, path = line.split("\t", 2)
        out.append(FileChange(
            status=status.get(path, "M"),
            path=path,
            added=None if a == "-" else int(a),
            deleted=None if d == "-" else int(d),
        ))
    return out


def single_parent_commits(repo: Path, limit: int) -> list[tuple[str, str]]:
    """(sha, parent) for non-merge commits reachable from HEAD, newest first."""
    out = []
    for line in git(repo, "rev-list", "--no-merges", "--parents", f"--max-count={limit}", "HEAD").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def commit_message(repo: Path, sha: str) -> str:
    return git(repo, "log", "-1", "--format=%B", sha).strip()
