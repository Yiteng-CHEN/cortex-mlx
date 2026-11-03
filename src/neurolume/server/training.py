"""Guided LoRA training: import data -> validate -> train in the background -> adapters.

All of it is driven by the web UI (no commands). Training runs `mlx_lm lora` as a subprocess so
the server stays responsive and a crash cannot take the server down. Real signals produced:
loss / tok/s / peak memory while training, and the adapter's actual weight change (delta W)
sampled at the same (row, col) positions as the weight map.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..models.train_probe import _ANSI, _PEAK, _ROW, _RUNNER

SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}"
                    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[abp]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,})")
KEYS = ["self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"]
MAX_INPUT_CHARS = 50_000_000
PAIRS = (("prompt", "completion"), ("instruction", "output"), ("instruction", "response"),
         ("question", "answer"), ("input", "output"))


# ---------------------------------------------------------------- datasets
def _row_from(o) -> dict | None:
    if not isinstance(o, dict):
        return None
    msgs = o.get("messages")
    if isinstance(msgs, list) and msgs and all(isinstance(m, dict) and isinstance(m.get("content"), str)
                                               and m.get("role") in ("system", "user", "assistant") for m in msgs):
        return {"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]}
    for pk, ck in PAIRS:
        if isinstance(o.get(pk), str) and isinstance(o.get(ck), str):
            return {"messages": [{"role": "user", "content": o[pk]}, {"role": "assistant", "content": o[ck]}]}
    if isinstance(o.get("text"), str):
        return {"text": o["text"]}
    return None


def chunk_text(text: str, limit: int = 1800) -> list[str]:
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if cur and len(cur) + len(para) > limit:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.strip())
    return [c for c in out if len(c) >= 40]


def parse_files(files: list[dict]) -> dict:
    """files: [{"name", "text"}] -> rows + counts + warnings. Rows with secrets are dropped."""
    rows, seen = [], set()
    stats = {"bad_lines": 0, "secrets": 0, "dupes": 0, "files": len(files)}
    warnings: list[str] = []
    total = sum(len(f.get("text", "")) for f in files)
    if total > MAX_INPUT_CHARS:
        raise ValueError("数据太大（超过 50MB），请分批导入")
    for f in files:
        name, text = str(f.get("name", "")), str(f.get("text", ""))
        cand: list[dict] = []
        if name.endswith((".jsonl", ".json")):
            lines = text.splitlines()
            objs = []
            try:
                whole = json.loads(text) if name.endswith(".json") else None
                objs = whole if isinstance(whole, list) else ([whole] if whole is not None else [])
            except ValueError:
                objs = []
            if not objs:
                for ln in lines:
                    if not ln.strip():
                        continue
                    try:
                        objs.append(json.loads(ln))
                    except ValueError:
                        stats["bad_lines"] += 1
            for o in objs:
                r = _row_from(o)
                if r:
                    cand.append(r)
                else:
                    stats["bad_lines"] += 1
        else:
            cand = [{"text": c} for c in chunk_text(text)]
        for r in cand:
            s = json.dumps(r, ensure_ascii=False)
            if SECRET.search(s):
                stats["secrets"] += 1
                continue
            h = hashlib.sha1(s.encode()).hexdigest()
            if h in seen:
                stats["dupes"] += 1
                continue
            seen.add(h)
            rows.append(r)
    if stats["secrets"]:
        warnings.append(f"已剔除 {stats['secrets']} 条疑似含密钥/令牌的样本")
    if stats["bad_lines"]:
        warnings.append(f"{stats['bad_lines']} 行格式无法识别，已跳过")
    if stats["dupes"]:
        warnings.append(f"去掉 {stats['dupes']} 条重复样本")
    if 0 < len(rows) < 20:
        warnings.append("样本少于 20 条，训练效果会很不明显；建议多导入一些")
    n_text = sum(1 for r in rows if "text" in r)
    if rows and n_text / len(rows) > .5 and sum(len(r["text"]) for r in rows if "text" in r) > 200_000:
        warnings.append("这是大段文档内容。想让 AI 记住书本知识，建议放进知识库（检索）而不是微调：微调更适合学习写法、风格和做事套路")
    return {"rows": rows, "stats": stats, "warnings": warnings}


class Datasets:
    def __init__(self, root: Path):
        self.root = Path(root)

    def create(self, name: str, files: list[dict]) -> dict:
        p = parse_files(files)
        rows = p["rows"]
        if not rows:
            raise ValueError("没有读到可用的样本。支持：.jsonl（messages / prompt+completion / instruction+output / text）、.md、.txt 和代码文件")
        did = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha1(json.dumps(rows[:3]).encode()).hexdigest()[:4]
        d = self.root / did
        d.mkdir(parents=True, exist_ok=True)
        valid = [r for i, r in enumerate(rows) if i % 10 == 9] or rows[-1:]
        train = [r for i, r in enumerate(rows) if i % 10 != 9] or rows
        for fn, rs in (("train.jsonl", train), ("valid.jsonl", valid)):
            (d / fn).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
        meta = {"id": did, "name": (name or did)[:60], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_train": len(train), "n_valid": len(valid), "warnings": p["warnings"], "stats": p["stats"],
                "preview": [json.dumps(r, ensure_ascii=False)[:220] for r in rows[:3]]}
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return meta

    def list(self) -> list[dict]:
        out = []
        for m in sorted(self.root.glob("*/meta.json"), reverse=True):
            try:
                out.append(json.loads(m.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return out

    def path(self, did: str) -> Path:
        if not re.fullmatch(r"[\w-]+", did):
            raise ValueError("bad dataset id")
        d = self.root / did
        if not (d / "train.jsonl").exists():
            raise KeyError(did)
        return d


# ---------------------------------------------------------------- training
PRESETS = {"quick": {"label": "快速试训", "max_iters": 100, "per_example": 1.0, "lr": 1e-5},
           "standard": {"label": "标准", "max_iters": 600, "per_example": 3.0, "lr": 1e-5}}


def plan_iters(n_train: int, preset: str) -> int:
    p = PRESETS[preset]
    return int(max(20, min(p["max_iters"], round(n_train * p["per_example"]))))


def parse_line(line: str) -> dict | None:
    line = _ANSI.sub("", line)
    m = _ROW.match(line)
    if m:
        k = {"": 1, "k": 1e3, "M": 1e6}[m.group(5)]
        return {"iter": int(m.group(1)), "loss": float(m.group(2)), "tok_s": float(m.group(3)),
                "tokens": float(m.group(4)) * k}
    v = (re.search(r"[Vv]al(?:idation)? loss\s*[:=]?\s*([\d.]+)", line)
         or re.match(r"\s*\d+\s+val\s+([\d.]+)", line))          # mlx-lm's table UI: "   10    val 1.487    0.23s"
    if v:
        return {"val_loss": float(v.group(1))}
    o = re.search(r"Iter (\d+): Train loss ([\d.]+).*?Tokens/sec ([\d.]+)", line)
    if o:
        return {"iter": int(o.group(1)), "loss": float(o.group(2)), "tok_s": float(o.group(3))}
    return None


class Trainer:
    def __init__(self, data_dir: Path, datasets: Datasets, spawn=None):
        self.root = Path(data_dir) / "adapters"
        self.datasets = datasets
        self._spawn = spawn or self._default_spawn
        self.lock = threading.RLock()
        self.run: dict | None = None
        self.proc = None

    @staticmethod
    def _default_spawn(cmd, cwd):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        return subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)

    def running(self) -> bool:
        return bool(self.run and self.run["state"] == "running")

    def start(self, model_key: str, repo: str, dataset: str, preset: str, tokenizer_config: dict | None = None) -> dict:
        if preset not in PRESETS:
            raise ValueError("unknown preset")
        with self.lock:
            if self.running():
                raise RuntimeError("已有训练在进行")
            d = self.datasets.path(dataset)
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            iters = plan_iters(meta["n_train"], preset)
            rid = time.strftime("%Y%m%d-%H%M%S")
            out = self.root / model_key / rid
            out.mkdir(parents=True, exist_ok=True)
            cfg = {"model": repo, "train": True, "data": str(d), "fine_tune_type": "lora",
                   "iters": iters, "batch_size": 1, "num_layers": 16, "max_seq_length": 2048,
                   "steps_per_report": 2, "steps_per_eval": max(10, iters // 4), "val_batches": 2,
                   "save_every": max(5, iters // 10), "adapter_path": str(out), "grad_checkpoint": True,
                   "learning_rate": PRESETS[preset]["lr"],
                   "lora_parameters": {"rank": 16, "scale": 20.0, "dropout": 0.0, "keys": KEYS}}
            (out / "lora.yaml").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            (out / "run_lora.py").write_text(_RUNNER, encoding="utf-8")
            self.run = {"id": f"{model_key}/{rid}", "model": model_key, "repo": repo, "dataset": dataset,
                        "dataset_name": meta["name"], "preset": preset, "iters": iters, "state": "running",
                        "iter": 0, "loss": None, "val_loss": None, "val_first": None, "tok_s": None, "peak_gb": None,
                        "history": [], "started": time.time(), "log_tail": [], "dir": str(out), "error": None}
            self.proc = self._spawn([sys.executable, str(out / "run_lora.py"), str(out / "lora.yaml")], str(out))
            threading.Thread(target=self._read, args=(self.run, self.proc), daemon=True).start()
            return self.status()

    def _read(self, run: dict, proc) -> None:
        for line in proc.stdout:
            clean = _ANSI.sub("", line).rstrip()
            if clean:
                run["log_tail"] = (run["log_tail"] + [clean])[-30:]
            pk = _PEAK.search(clean)
            if pk:
                run["peak_gb"] = float(pk.group(1))
            r = parse_line(clean)
            if r:
                if "iter" in r:
                    run["iter"] = r["iter"]
                    run["loss"], run["tok_s"] = r["loss"], r["tok_s"]
                    run["history"].append([r["iter"], r["loss"]])
                if "val_loss" in r:
                    if run.get("val_first") is None:      # mlx-lm evaluates once before the first update
                        run["val_first"] = r["val_loss"]
                    run["val_loss"] = r["val_loss"]
        rc = proc.wait()
        adapter = Path(run["dir"]) / "adapters.safetensors"
        final, error = run["state"], None
        if final == "cancelled":
            pass
        elif rc == 0 and adapter.exists():
            final = "done"
        else:
            final = "failed"
            tail = " ".join(run["log_tail"][-6:])
            error = ("内存不足：请关闭其他大型应用，或选择更小的模型/更少的数据" if
                     ("out of memory" in tail.lower() or "metal" in tail.lower() and "memory" in tail.lower())
                     else f"训练异常退出（代码 {rc}）")
        run["finished"] = time.time()
        meta = {k: run.get(k) for k in ("model", "repo", "dataset", "dataset_name", "preset", "iters",
                                        "loss", "val_loss", "val_first", "peak_gb", "started", "finished")}
        meta["state"] = final
        try:   # written before the state flips, so whoever sees "done" also finds the run's metadata
            (Path(run["dir"]) / "neurolume-run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        except OSError:
            pass
        if final == "done":
            run["iter"] = run["iters"]
        run["error"] = error
        run["state"] = final

    def cancel(self) -> None:
        with self.lock:
            if self.running():
                self.run["state"] = "cancelled"
                self.proc.terminate()

    def status(self) -> dict:
        r = self.run
        if not r:
            return {"state": "idle"}
        out = {k: v for k, v in r.items() if k != "dir"}
        out["adapter"] = r["id"] if r["state"] == "done" else None
        out["elapsed"] = round((r.get("finished") or time.time()) - r["started"])
        if r["iter"] and r["state"] == "running":
            out["eta"] = round(out["elapsed"] / r["iter"] * (r["iters"] - r["iter"]))
        out["history"] = r["history"][-400:]
        return out

    def adapters(self) -> list[dict]:
        out = []
        for f in sorted(self.root.glob("*/*/adapters.safetensors"), reverse=True):
            d = f.parent
            meta = {}
            try:
                meta = json.loads((d / "neurolume-run.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            out.append({"id": f"{d.parent.name}/{d.name}", "model": d.parent.name,
                        "dataset": meta.get("dataset_name"), "loss": meta.get("loss"),
                        "val_loss": meta.get("val_loss"), "val_first": meta.get("val_first"),
                        "iters": meta.get("iters"), "repo": meta.get("repo"), "dataset_id": meta.get("dataset"),
                        "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))})
        return out

    def adapter_dir(self, aid: str) -> Path:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", aid):
            raise ValueError("bad adapter id")
        d = (self.root / aid).resolve()
        if self.root.resolve() not in d.parents or not (d / "adapters.safetensors").exists():
            raise KeyError(aid)
        return d


# ---------------------------------------------------------------- real delta-W from an adapter
def sample_idx(n: int, grid: int = 12) -> list[int]:
    return [round(k * (n - 1) / (grid - 1)) for k in range(grid)]


def cells(a_sel: list[list[float]], b_sel: list[list[float]], scale: float) -> list[float]:
    """dW[row i, col j] = scale * sum_k B[i][k] * A[j][k]; flattened as i*G + j (row-major)."""
    return [scale * sum(b * a for a, b in zip(aj, bi)) for bi in b_sel for aj in a_sel]


_KEY = re.compile(r"layers\.(\d+)\.(.+)\.lora_([ab])$")


def adapter_delta(adapter_dir: Path, grid: int = 12) -> dict:
    """Sample delta-W of every LoRA matrix at the weight map's positions (needs MLX)."""
    import mlx.core as mx

    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text(encoding="utf-8")) \
        if (Path(adapter_dir) / "adapter_config.json").exists() else {}
    scale = float((cfg.get("lora_parameters") or {}).get("scale", 20.0))
    tensors = mx.load(str(Path(adapter_dir) / "adapters.safetensors"))
    pairs: dict[tuple[int, str], dict] = {}
    for k, v in tensors.items():
        m = _KEY.search(k)
        if m:
            pairs.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = v
    layers: dict[int, list] = {}
    for (l, name), ab in sorted(pairs.items()):
        if "a" not in ab or "b" not in ab:
            continue
        a, b = ab["a"].astype(mx.float32), ab["b"].astype(mx.float32)     # a: (in, r)  b: (r, out)
        ci, ri = sample_idx(a.shape[0], grid), sample_idx(b.shape[1], grid)
        a_sel, b_sel = a[mx.array(ci)], b[:, mx.array(ri)].T              # (G, r) each
        v = (scale * (b_sel @ a_sel.T)).reshape(-1)
        m1, m2 = a.T @ a, b @ b.T
        norm = scale * mx.sqrt(mx.sum(m1 * m2))
        mx.eval(v, norm)
        layers.setdefault(l, []).append({"name": name, "v": [round(x, 7) for x in v.tolist()],
                                         "norm": round(float(norm), 6)})
    return {"scale": scale, "grid": grid,
            "layers": [{"i": i, "m": m} for i, m in sorted(layers.items())]}
