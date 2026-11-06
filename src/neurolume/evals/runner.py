"""Run a backend over eval tasks and score the answers with the hidden tests."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from ..backends.base import Backend
from ..config import Config
from ..paths import DataPaths
from .parse import parse_files
from .prompt import build_prompt, max_tokens_for
from .sandbox import UnsafePathError, evaluate
from .task import EvalTask

STATUSES = ("pass", "fail", "format_error", "timeout", "setup_failed")


def score_one(cfg: Config, paths: DataPaths, backend: Backend, task: EvalTask) -> dict:
    repo = cfg.repo(task.repo)
    gen = backend.generate(build_prompt(repo, task), max_tokens=max_tokens_for(task, cfg.max_tokens_cap),
                           temperature=cfg.temperature, task=task)
    parsed = parse_files(gen.text, task.source_files)
    record = {
        "task_id": task.id,
        "gen_seconds": gen.seconds,
        "prompt_tokens": gen.prompt_tokens,
        "completion_tokens": gen.completion_tokens,
        "tokens_per_second": gen.tokens_per_second,
        "peak_memory_gb": gen.peak_memory_gb,
        "ignored_files": parsed.ignored,
        "parse_errors": parsed.errors,
        "response": gen.text,
    }
    if parsed.format_error:
        record.update(status="format_error", test_output="")
        return record
    try:
        outcome = evaluate(paths, repo, task, cfg.eval, solution=parsed.files, runs=1)
    except UnsafePathError as e:
        record.update(status="format_error", test_output=str(e))
        return record
    if outcome.setup is not None and not outcome.setup.ok:
        record.update(status="setup_failed", test_output=outcome.setup.output)
        return record
    run = outcome.runs[0]
    record.update(status="pass" if run.ok else ("timeout" if run.timed_out else "fail"),
                  test_seconds=run.seconds, test_output=run.output)
    return record


def run_eval(cfg: Config, paths: DataPaths, backend: Backend, tasks: list[EvalTask], *,
             split: str, samples: int = 1, log: Callable[[str], None] = print) -> dict:
    records = []
    for task in tasks:
        for i in range(samples):
            rec = score_one(cfg, paths, backend, task)
            rec["sample"] = i
            records.append(rec)
            log(f"[{backend.name}] {task.id}#{i}: {rec['status']}"
                + (f"  {rec['tokens_per_second']} tok/s" if rec.get("tokens_per_second") else ""))
    from .report import summarize
    return {
        "candidate": backend.name,
        "split": split,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settings": {"temperature": cfg.temperature, "samples": samples,
                     "max_tokens_cap": cfg.max_tokens_cap},
        "summary": summarize(records),
        "records": records,
    }


def save_result(paths: DataPaths, result: dict) -> Path:
    paths.results.mkdir(parents=True, exist_ok=True)
    p = paths.results / f"eval-{result['candidate']}-{result['split']}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
