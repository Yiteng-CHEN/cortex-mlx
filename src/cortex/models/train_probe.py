"""Measure LoRA training cost on a candidate: a few steps of `mlx_lm lora`, timing + peak memory.

UNVERIFIED on hardware: assumes the mlx-lm LoRA CLI accepts a YAML config (-c) with the keys
below and prints lines like "Iter 5: Train loss ..., It/sec 0.41, Tokens/sec 812, ..., Peak mem
31.2 GB". Both `python -m mlx_lm lora` and the older `python -m mlx_lm.lora` are tried.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ..config import Config
from ..evals.prompt import build_prompt, fence
from ..evals.sandbox import reference_solution
from ..evals.task import load_split
from ..paths import DataPaths

_SYNTH_CODE = '''def merge_intervals(intervals):
    """Merge overlapping [start, end] intervals and return them sorted."""
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
'''


def synthetic_examples(n: int, target_tokens: int) -> list[dict]:
    """Code-shaped examples padded to roughly target_tokens, so memory use is realistic."""
    reps = max(1, (target_tokens * 4) // (len(_SYNTH_CODE) * 2))
    out = []
    for i in range(n):
        body = "\n".join(_SYNTH_CODE.replace("merge_intervals", f"merge_intervals_{i}_{k}")
                         for k in range(reps))
        out.append({"messages": [
            {"role": "user", "content": f"Rewrite this module with clearer names (variant {i}):\n\n"
                                        + fence("lib/intervals.py", body)},
            {"role": "assistant", "content": fence("lib/intervals.py", body)},
        ]})
    return out


def task_examples(cfg: Config, paths: DataPaths, n: int) -> list[dict]:
    """Dev tasks with their reference fix (never holdout)."""
    out = []
    for task in load_split(paths, "dev")[:n]:
        repo = cfg.repo(task.repo)
        answer = "\n\n".join(fence(p, c) for p, c in reference_solution(repo, task).items())
        out.append({"messages": [{"role": "user", "content": build_prompt(repo, task)},
                                 {"role": "assistant", "content": answer}]})
    return out


_PATTERNS = {
    "it_per_sec": re.compile(r"It/sec\s+([\d.]+)"),
    "tokens_per_sec": re.compile(r"Tokens/sec\s+([\d.]+)"),
    "peak_memory_gb": re.compile(r"Peak mem\s+([\d.]+)\s*GB"),
}


def parse_lora_output(text: str) -> dict:
    out: dict[str, float | None] = {}
    for key, pat in _PATTERNS.items():
        vals = [float(v) for v in pat.findall(text)]
        out[key] = (max(vals) if key == "peak_memory_gb" else vals[-1]) if vals else None
    out["seconds_per_step"] = round(1 / out["it_per_sec"], 2) if out.get("it_per_sec") else None
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def run_probe(cfg: Config, paths: DataPaths, name: str, *, log=print) -> tuple[dict, Path]:
    c = cfg.candidate(name)
    if c.backend != "mlx":
        raise ValueError(f"train-probe needs an mlx candidate, {name} is {c.backend}")
    p = {"iters": 10, "batch_size": 1, "num_layers": 16, "max_seq_length": 2048, "rank": 16,
         "examples": 20, **cfg.train_probe}

    rows = task_examples(cfg, paths, p["examples"])
    source = "dev_tasks"
    if len(rows) < 4:
        rows = synthetic_examples(p["examples"], int(p["max_seq_length"] * 0.8))
        source = "synthetic"

    work = paths.tmp / f"probe-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    data = work / "data"
    data.mkdir(parents=True)
    _write_jsonl(data / "train.jsonl", rows)
    _write_jsonl(data / "valid.jsonl", rows[:2])
    lora_cfg = {  # JSON is valid YAML
        "model": c.model, "train": True, "data": str(data), "fine_tune_type": "lora",
        "iters": p["iters"], "batch_size": p["batch_size"], "num_layers": p["num_layers"],
        "max_seq_length": p["max_seq_length"], "steps_per_report": 5, "steps_per_eval": 100000,
        "val_batches": 1, "save_every": 100000, "adapter_path": str(work / "adapters"),
        "grad_checkpoint": True,
        "lora_parameters": {"rank": p["rank"], "scale": 20.0, "dropout": 0.0},
    }
    cfg_path = work / "lora.yaml"
    cfg_path.write_text(json.dumps(lora_cfg, indent=2), encoding="utf-8")

    attempts = [[sys.executable, "-m", "mlx_lm", "lora", "-c", str(cfg_path)],
                [sys.executable, "-m", "mlx_lm.lora", "-c", str(cfg_path)]]
    proc = None
    start = time.monotonic()
    for cmd in attempts:
        log("running: " + " ".join(cmd[1:]))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        if proc.returncode == 0 or not re.search(r"invalid choice|No module named|usage:", out):
            break
    wall = time.monotonic() - start
    output = (proc.stdout + proc.stderr) if proc else ""
    (work / "lora.log").write_text(output, encoding="utf-8")

    result = {
        "candidate": name, "model": c.model, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ok": bool(proc and proc.returncode == 0), "returncode": proc.returncode if proc else None,
        "data_source": source, "params": p, "wall_seconds": round(wall, 1),
        **parse_lora_output(output), "log_tail": output[-3000:],
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    out_path = paths.results / f"probe-{name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["ok"]:
        shutil.rmtree(work, ignore_errors=True)   # our own temp dir; keep it on failure for debugging
    return result, out_path
