"""Prompt construction for eval tasks."""
from __future__ import annotations

from pathlib import PurePosixPath

from ..config import RepoConfig
from ..gitutil import show_file
from .task import EvalTask

LANG = {".py": "python", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".jsx": "jsx",
        ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin", ".swift": "swift",
        ".rb": "ruby", ".c": "c", ".cc": "cpp", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
        ".cs": "csharp", ".php": "php", ".mjs": "javascript"}


def fence(path: str, content: str) -> str:
    lang = LANG.get(PurePosixPath(path).suffix, "")
    body = content if content.endswith("\n") else content + "\n"
    return f"### {path}\n```{lang}\n{body}```"


def build_prompt(repo: RepoConfig, task: EvalTask) -> str:
    parts = [
        f"You are fixing code in the repository `{task.repo}`.",
        "## Task\n" + task.statement.strip(),
        "## Tests that must pass",
    ]
    for f in task.test_files:
        parts.append(fence(f, show_file(repo.path, task.target_sha, f) or ""))
    parts.append("## Current source files")
    for f in task.source_files:
        content = show_file(repo.path, task.base_sha, f)
        if content is None:
            parts.append(f"### {f}\n(this file does not exist yet; create it)")
        else:
            parts.append(fence(f, content))
    allowed = ", ".join(f"`{f}`" for f in task.source_files)
    parts.append(
        "## Output format\n"
        "Return the COMPLETE new contents of every file you change, each as a level-3 heading "
        "with the path followed by one fenced code block:\n\n"
        "### path/to/file.ext\n```\n<full file contents>\n```\n\n"
        f"You may only change these files: {allowed}. Do not output tests, diffs or explanations."
    )
    return "\n\n".join(parts)


def max_tokens_for(task: EvalTask, cap: int) -> int:
    est = int(task.stats.get("solution_tokens_est", 2000))
    return max(1024, min(cap, int(est * 1.5) + 1024))
