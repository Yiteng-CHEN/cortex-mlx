"""Measure LoRA training cost on a candidate: a few steps of `mlx_lm lora`, timing + peak memory.

Verified on mlx-lm 0.32.2 (M0 run): `python -m mlx_lm lora -c cfg` works, but it prints a rich
table (iter / train_loss / tok/s / tokens) with no peak memory. So we run it through a tiny
wrapper that prints MLX's peak memory at exit, and parse both the new table and the old
"It/sec ... Peak mem" log lines. On failure (e.g. Metal out of memory) we retry with shorter
sequences and record every attempt.
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


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ROW = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s*[▼▲]?\s+(\d+)\s+([\d.]+)([kM]?)\s*$", re.M)
_OLD = {
    "it_per_sec": re.compile(r"It/sec\s+([\d.]+)"),
    "tokens_per_sec": re.compile(r"Tokens/sec\s+([\d.]+)"),
    "peak_memory_gb": re.compile(r"Peak mem\s+([\d.]+)\s*GB"),
}
_PEAK = re.compile(r"NEUROLUME_PEAK_GB=([\d.]+)")

_RUNNER = """\
import sys
import mlx.core as mx
sys.argv = ["mlx_lm", "lora", "-c", sys.argv[1]]
try:
    from mlx_lm import cli
    cli.main()
finally:
    fn = getattr(mx, "get_peak_memory", None) or getattr(mx.metal, "get_peak_memory", None)
    if fn:
        print(f"NEUROLUME_PEAK_GB={fn() / 1e9:.2f}", flush=True)
"""


def parse_lora_output(text: str) -> dict:
    text = _ANSI.sub("", text)
    out: dict[str, float | None] = {"it_per_sec": None, "tokens_per_sec": None,
                                    "peak_memory_gb": None, "seconds_per_step": None}
    rows = _ROW.findall(text)
    if rows:
        scale = {"": 1, "k": 1e3, "M": 1e6}
        tps = [float(r[2]) for r in rows]
        out["tokens_per_sec"] = round(sum(tps) / len(tps), 1)
        last_iter, tokens = int(rows[-1][0]), float(rows[-1][3]) * scale[rows[-1][4]]
        if last_iter and out["tokens_per_sec"]:
            out["seconds_per_step"] = round(tokens / last_iter / out["tokens_per_sec"], 2)
    else:
        for key, pat in _OLD.items():
            vals = [float(v) for v in pat.findall(text)]
            out[key] = (max(vals) if key == "peak_memory_gb" else vals[-1]) if vals else None
        if out["it_per_sec"]:
            out["seconds_per_step"] = round(1 / out["it_per_sec"], 2)
    peaks = [float(v) for v in _PEAK.findall(text)]
    if peaks:
        out["peak_memory_gb"] = max(peaks)
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _attempt(c, work: Path, rows: list[dict], p: dict, seq: int) -> dict:
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    _write_jsonl(data / "train.jsonl", rows)
    _write_jsonl(data / "valid.jsonl", rows[:2])
    lora_cfg = {  # JSON is valid YAML
        "model": c.model, "train": True, "data": str(data), "fine_tune_type": "lora",
        "iters": p["iters"], "batch_size": p["batch_size"], "num_layers": p["num_layers"],
        "max_seq_length": seq, "steps_per_report": 5, "steps_per_eval": 100000,
        "val_batches": 1, "save_every": 100000, "adapter_path": str(work / "adapters"),
        "grad_checkpoint": True,
        "lora_parameters": {"rank": p["rank"], "scale": 20.0, "dropout": 0.0},
    }
    (work / "lora.yaml").write_text(json.dumps(lora_cfg, indent=2), encoding="utf-8")
    (work / "run_lora.py").write_text(_RUNNER, encoding="utf-8")
    start = time.monotonic()
    proc = subprocess.run([sys.executable, str(work / "run_lora.py"), str(work / "lora.yaml")],
                          capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    (work / f"lora-{seq}.log").write_text(output, encoding="utf-8")
    return {"max_seq_length": seq, "ok": proc.returncode == 0, "returncode": proc.returncode,
            "wall_seconds": round(time.monotonic() - start, 1), **parse_lora_output(output),
            "log_tail": _ANSI.sub("", output)[-2500:]}


def run_probe(cfg: Config, paths: DataPaths, name: str, *, log=print) -> tuple[dict, Path]:
    c = cfg.candidate(name)
    if c.backend != "mlx":
        raise ValueError(f"train-probe needs an mlx candidate, {name} is {c.backend}")
    p = {"iters": 10, "batch_size": 1, "num_layers": 16, "max_seq_length": 2048, "rank": 16,
         "examples": 20, **cfg.train_probe}

    top = int(p["max_seq_length"])
    seqs = [s for s in (top, 1024, 512, 256) if s <= top]
    work = paths.tmp / f"probe-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    attempts: list[dict] = []
    source = "?"
    for seq in dict.fromkeys(seqs):
        rows = task_examples(cfg, paths, p["examples"])
        source = "dev_tasks"
        if len(rows) < 4:
            rows = synthetic_examples(p["examples"], int(seq * 0.8))
            source = "synthetic"
        log(f"{name}: LoRA probe, max_seq_length={seq}, {p['num_layers']} layers, rank {p['rank']} ...")
        a = _attempt(c, work, rows, p, seq)
        attempts.append(a)
        log(f"  -> {'ok' if a['ok'] else 'FAILED'}: {a['seconds_per_step'] or '-'} s/step, "
            f"peak {a['peak_memory_gb'] or '-'} GB")
        if a["ok"]:
            break

    best = attempts[-1]
    result = {
        "candidate": name, "model": c.model, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ok": best["ok"], "data_source": source, "params": p,
        "max_seq_length_reached": best["max_seq_length"] if best["ok"] else None,
        "seconds_per_step": best["seconds_per_step"], "tokens_per_sec": best["tokens_per_sec"],
        "peak_memory_gb": best["peak_memory_gb"], "returncode": best["returncode"],
        "attempts": [{k: v for k, v in a.items() if k != "log_tail"} for a in attempts],
        "log_tail": best["log_tail"],
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    out_path = paths.results / f"probe-{name}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["ok"]:
        shutil.rmtree(work, ignore_errors=True)   # our own temp dir; keep it on failure for debugging
    return result, out_path
