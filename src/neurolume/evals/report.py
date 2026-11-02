"""Summaries and candidate comparison tables."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..paths import DataPaths


def summarize(records: list[dict]) -> dict:
    n = len(records)
    count = lambda s: sum(r["status"] == s for r in records)  # noqa: E731
    toks = sum(r.get("completion_tokens") or 0 for r in records)
    secs = sum(r.get("gen_seconds") or 0 for r in records)
    tps = [r["tokens_per_second"] for r in records if r.get("tokens_per_second")]
    mems = [r["peak_memory_gb"] for r in records if r.get("peak_memory_gb")]
    return {
        "runs": n,
        "pass_rate": round(count("pass") / n, 4) if n else 0.0,
        "format_error_rate": round(count("format_error") / n, 4) if n else 0.0,
        "timeouts": count("timeout"),
        "setup_failed": count("setup_failed"),
        "mean_gen_seconds": round(secs / n, 2) if n else 0.0,
        "tokens_per_second": round(sum(tps) / len(tps), 2) if tps else (round(toks / secs, 2) if secs else None),
        "peak_memory_gb": round(max(mems), 2) if mems else None,
    }


def latest_results(paths: DataPaths, split: str | None = None) -> list[dict]:
    """Newest eval result per (candidate, split)."""
    latest: dict[tuple[str, str], tuple[str, dict]] = {}
    for p in sorted(paths.results.glob("eval-*.json")) if paths.results.is_dir() else []:
        data = json.loads(p.read_text(encoding="utf-8"))
        if split and data.get("split") != split:
            continue
        key = (data["candidate"], data["split"])
        if key not in latest or data["created"] >= latest[key][0]:
            latest[key] = (data["created"], data)
    return [v[1] for _, v in sorted(latest.items())]


def probe_results(paths: DataPaths) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(paths.results.glob("probe-*.json")) if paths.results.is_dir() else []:
        data = json.loads(p.read_text(encoding="utf-8"))
        out[data["candidate"]] = data
    return out


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v:.0%}" if pct else str(v)


def comparison_rows(results: list[dict], probes: dict[str, dict]) -> list[tuple[str, ...]]:
    rows = [("candidate", "split", "runs", "pass", "format err", "tok/s", "gen s/task",
             "peak GB (gen)", "LoRA s/step", "peak GB (train)")]
    for r in results:
        s = r["summary"]
        pr = probes.get(r["candidate"], {})
        rows.append((r["candidate"], r["split"], str(s["runs"]), _fmt(s["pass_rate"], True),
                     _fmt(s["format_error_rate"], True), _fmt(s["tokens_per_second"]),
                     _fmt(s["mean_gen_seconds"]), _fmt(s["peak_memory_gb"]),
                     _fmt(pr.get("seconds_per_step")), _fmt(pr.get("peak_memory_gb"))))
    return rows


def as_text(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows)


def as_markdown(rows: list[tuple[str, ...]]) -> str:
    head, *body = rows
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def write_comparison(paths: DataPaths, rows: list[tuple[str, ...]]) -> Path:
    p = paths.results / f"compare-{time.strftime('%Y%m%d-%H%M%S')}.md"
    p.write_text("# Neurolume candidate comparison\n\n" + as_markdown(rows) + "\n", encoding="utf-8")
    return p
