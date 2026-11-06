"""Parse model output of the form `### path` + fenced code block (full file contents)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HEADER = re.compile(r"^#{2,4}\s+`?([^`\s]+)`?\s*$")
_FENCE_OPEN = re.compile(r"^\s*(```+|~~~+)")


@dataclass
class ParseResult:
    files: dict[str, str] = field(default_factory=dict)
    ignored: list[str] = field(default_factory=list)   # paths not allowed for the task
    errors: list[str] = field(default_factory=list)

    @property
    def format_error(self) -> bool:
        return not self.files


def _next_nonblank(lines: list[str], i: int) -> int:
    while i < len(lines) and not lines[i].strip():
        i += 1
    return i


def parse_files(text: str, allowed: list[str]) -> ParseResult:
    text = _THINK.sub("", text)
    # Unterminated <think> (generation cut off while thinking): drop everything before </think>
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    lines = text.splitlines()
    allowed_set = set(allowed)
    res = ParseResult()
    i = 0
    while i < len(lines):
        m = _HEADER.match(lines[i])
        if not m:
            i += 1
            continue
        path = m.group(1).removeprefix("./")
        j = _next_nonblank(lines, i + 1)
        fm = _FENCE_OPEN.match(lines[j]) if j < len(lines) else None
        if not fm:
            i += 1
            continue
        marker = fm.group(1)
        body: list[str] = []
        k = j + 1
        closed = False
        while k < len(lines):
            if lines[k].strip() == marker:
                nxt = _next_nonblank(lines, k + 1)
                # A closing fence is followed by EOF or the next file header; otherwise it is
                # a fence inside the file (e.g. a Markdown file or a docstring example).
                if nxt >= len(lines) or _HEADER.match(lines[nxt]):
                    closed = True
                    break
            body.append(lines[k])
            k += 1
        if not closed:
            res.errors.append(f"unterminated code block for {path}")
        if path in allowed_set:
            res.files[path] = "\n".join(body) + "\n"
        else:
            res.ignored.append(path)
        i = k + 1
    if not res.files:
        res.errors.append("no allowed files found in output")
    return res
