"""Export a trained adapter so the result runs outside Neurolume.

- "mlx":    fuse the adapter into the base weights (mlx_lm fuse) -> a standalone MLX model folder. LM Studio on a
            Mac can load MLX models directly; we link the folder into ~/.lmstudio/models when LM Studio is installed.
- "ollama": fuse + dequantize to plain safetensors, write a Modelfile and, if the `ollama` command exists, run
            `ollama create` (Ollama quantizes on import). Whether Ollama can import a given architecture is up to
            Ollama; the folder and Modelfile are kept either way.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

OLLAMA_PATHS = ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama", "/Applications/Ollama.app/Contents/Resources/ollama")


def find_ollama() -> str | None:
    return shutil.which("ollama") or next((p for p in OLLAMA_PATHS if os.path.exists(p)), None)


def safe_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9._-]+", "-", s.lower()).strip("-.")
    return s[:60] or "model"


def dir_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size     # follows HF cache symlinks to blobs
            except OSError:
                pass
    return total


class Exporter:
    def __init__(self, data_dir: Path, spawn=None, which_ollama=find_ollama, lmstudio_dir: Path | None = None):
        self.root = Path(data_dir) / "exports"
        self.job: dict = {"state": "idle"}
        self._spawn = spawn or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True))
        self._which = which_ollama
        self.lmstudio = lmstudio_dir or (Path.home() / ".lmstudio" / "models")

    def status(self) -> dict:
        return dict(self.job)

    def list(self) -> list[dict]:
        out = []
        for m in sorted(self.root.glob("*/neurolume-export.json"), reverse=True):
            try:
                out.append(json.loads(m.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return out

    def start(self, kind: str, adapter_id: str, adapter_dir: Path, repo: str, model_bytes: int, name: str = "") -> dict:
        if kind not in ("mlx", "ollama"):
            raise ValueError("unknown export kind")
        if self.job.get("state") == "running":
            raise RuntimeError("已有导出在进行")
        tag = safe_name(name or f"{adapter_id.split('/')[0]}-{adapter_id.split('/')[-1]}")
        out = self.root / f"{tag}-{kind}"
        if out.exists():
            raise ValueError("同名导出已存在，请换个名字或先删除")
        need = model_bytes * (4.2 if kind == "ollama" else 1.2)          # dequantized fp16 is ~4x a 4-bit model
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free
        if model_bytes and free < need:
            raise RuntimeError(f"磁盘空间不够：导出大约需要 {need / 1e9:.0f} GB，可用 {free / 1e9:.0f} GB")
        self.job = {"state": "running", "kind": kind, "name": tag, "step": "fuse", "path": str(out), "error": None,
                    "ollama": None, "lmstudio": None, "started": time.time()}
        job = self.job

        def work():
            cmd = [sys.executable, "-m", "mlx_lm", "fuse", "--model", repo, "--adapter-path", str(adapter_dir),
                   "--save-path", str(out)] + (["--dequantize"] if kind == "ollama" else [])
            r = self._spawn(cmd)
            if r.returncode != 0:
                job["state"], job["error"] = "failed", (r.stderr or r.stdout or "")[-400:]
                return
            if kind == "mlx":
                if self.lmstudio.parent.is_dir():
                    try:
                        dst = self.lmstudio / "neurolume" / tag
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            dst.symlink_to(out, target_is_directory=True)
                        job["lmstudio"] = str(dst)
                    except OSError:
                        pass
            else:
                (out / "Modelfile").write_text("FROM .\n", encoding="utf-8")
                exe = self._which()
                cmd2 = ["ollama", "create", f"neurolume-{tag}", "-f", str(out / "Modelfile"), "--quantize", "q4_K_M"]
                job["ollama_cmd"] = " ".join(cmd2)
                if exe:
                    job["step"] = "ollama"
                    r2 = self._spawn([exe] + cmd2[1:])
                    if r2.returncode == 0:
                        job["ollama"] = f"neurolume-{tag}"
                    else:
                        job["error"] = "Ollama 导入失败（可能还不支持这个模型结构）：" + (r2.stderr or r2.stdout or "")[-300:]
            meta = {k: job.get(k) for k in ("kind", "name", "path", "ollama", "lmstudio", "ollama_cmd")}
            meta.update({"adapter": adapter_id, "base": repo, "created": time.strftime("%Y-%m-%d %H:%M")})
            (out / "neurolume-export.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            job["state"] = "done"
        threading.Thread(target=work, daemon=True).start()
        return self.status()
