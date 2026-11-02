"""Model library: what can be loaded, what is on disk, one-click download / delete.

Entries come from config/neurolume.toml (mlx candidates) plus models the user adds by Hugging Face
repo id (stored in <data_dir>/library.json). Downloads run in a background thread and report
progress by measuring bytes on disk against the repo's total size. Network and huggingface_hub
are injected so tests need neither.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from ..config import Candidate, Config

REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def slug(repo: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", repo.split("/")[-1].lower()).strip("-")


def default_info(repo: str) -> dict:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    files = {s.rfilename: (s.size or 0) for s in info.siblings}
    return {"size": sum(files.values()), "files": list(files)}


def default_downloader(repo: str, cache_dir: Path, should_cancel: Callable[[], bool]) -> None:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm

    class Watch(tqdm):
        def update(self, n=1):
            if should_cancel():
                raise InterruptedError("cancelled")
            return super().update(n)

    try:
        snapshot_download(repo, cache_dir=str(cache_dir), tqdm_class=Watch)
    except TypeError:                       # older huggingface_hub without tqdm_class
        snapshot_download(repo, cache_dir=str(cache_dir))


class Library:
    def __init__(self, cfg: Config, data_dir: Path, hub_cache: Path,
                 info: Callable[[str], dict] = default_info,
                 downloader: Callable = default_downloader):
        self.cfg, self.hub, self._info, self._dl = cfg, Path(hub_cache), info, downloader
        self.file = Path(data_dir) / "library.json"
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self._cancel: set[str] = set()

    # ---- entries
    def _custom(self) -> list[dict]:
        try:
            return json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def entries(self) -> list[dict]:
        out = [{"key": n, "repo": c.model, "name": n, "source": "catalog"}
               for n, c in self.cfg.candidates.items() if c.backend == "mlx"]
        out += [{**e, "source": "custom"} for e in self._custom()]
        return out

    def find(self, key: str) -> dict | None:
        return next((e for e in self.entries() if e["key"] == key), None)

    def candidate(self, key: str) -> Candidate | None:
        if key in self.cfg.candidates:
            return self.cfg.candidates[key]
        e = next((e for e in self._custom() if e["key"] == key), None)
        return Candidate(name=key, backend="mlx", model=e["repo"]) if e else None

    # ---- disk
    def repo_dir(self, repo: str) -> Path:
        return self.hub / ("models--" + repo.replace("/", "--"))

    def disk_bytes(self, repo: str) -> int:
        blobs = self.repo_dir(repo) / "blobs"
        total = 0
        if blobs.is_dir():
            for f in blobs.iterdir():
                try:
                    total += f.stat().st_size          # follows symlinks into the shared store
                except OSError:
                    pass
        return total

    def is_downloaded(self, repo: str) -> bool:
        d = self.repo_dir(repo)
        blobs = d / "blobs"
        if blobs.is_dir() and any(p.name.endswith(".incomplete") for p in blobs.iterdir()):
            return False
        for snap in (d / "snapshots").glob("*") if (d / "snapshots").is_dir() else []:
            if (snap / "config.json").exists() and any(snap.glob("*.safetensors")):
                return True
        return False

    def free_bytes(self) -> int:
        p = self.hub
        while not p.exists() and p != p.parent:
            p = p.parent
        return shutil.disk_usage(p).free

    # ---- status
    def status(self, loaded: str | None = None, weights_dir: Path | None = None) -> dict:
        items = []
        for e in self.entries():
            job = self.jobs.get(e["key"])
            if job and job["state"] == "running":
                job["bytes"] = self.disk_bytes(e["repo"])
            items.append({**e, "downloaded": self.is_downloaded(e["repo"]),
                          "disk_bytes": self.disk_bytes(e["repo"]), "loaded": e["key"] == loaded,
                          "job": job and {k: v for k, v in job.items() if k != "thread"},
                          "weights": any((d / f"weights-{e['key']}.json").exists() for d in
                                                     ((weights_dir if isinstance(weights_dir, (list, tuple)) else [weights_dir]) if weights_dir else []))})
        return {"items": items, "free_bytes": self.free_bytes(), "hub": str(self.hub)}

    # ---- add / remove
    def add(self, repo: str, name: str | None = None) -> dict:
        repo = repo.strip()
        if not REPO_RE.match(repo):
            raise ValueError("请输入 Hugging Face 上的模型名，格式如 mlx-community/某个模型")
        key = slug(repo)
        with self.lock:
            if self.find(key) or any(e["repo"] == repo for e in self.entries()):
                raise ValueError("这个模型已经在列表里了")
            try:
                info = self._info(repo)
            except Exception as e:
                raise ValueError(f"找不到这个模型，或网络不可用（{type(e).__name__}）") from e
            files = info.get("files", [])
            if not any(f.endswith(".safetensors") for f in files) or "config.json" not in files:
                raise ValueError("这个仓库不是可直接加载的模型（需要 config.json 和 .safetensors 权重）")
            entry = {"key": key, "repo": repo, "name": name or key, "size": info.get("size")}
            data = self._custom() + [entry]
            self.file.parent.mkdir(parents=True, exist_ok=True)
            self.file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            return {**entry, "source": "custom"}

    def remove_custom(self, key: str) -> None:
        with self.lock:
            data = [e for e in self._custom() if e["key"] != key]
            self.file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- download / delete
    def download(self, key: str) -> dict:
        e = self.find(key)
        if not e:
            raise KeyError(f"unknown model {key!r}")
        with self.lock:
            job = self.jobs.get(key)
            if job and job["state"] == "running":
                return job
            try:
                total = self._info(e["repo"]).get("size") or None
            except Exception:
                total = None
            if total and total * 1.05 > self.free_bytes() + self.disk_bytes(e["repo"]):
                raise ValueError(f"磁盘空间不够：需要约 {total / 1e9:.1f} GB，可用 {self.free_bytes() / 1e9:.1f} GB")
            self._cancel.discard(key)
            job = {"state": "running", "bytes": self.disk_bytes(e["repo"]), "total": total,
                   "started": time.time(), "error": None}
            self.jobs[key] = job
        t = threading.Thread(target=self._run, args=(key, e["repo"], job), daemon=True)
        t.start()
        return job

    def _run(self, key: str, repo: str, job: dict) -> None:
        try:
            self._dl(repo, self.hub, lambda: key in self._cancel)
            job["state"] = "done" if self.is_downloaded(repo) else "error"
            if job["state"] == "error":
                job["error"] = "下载结束但文件不完整，请重试"
        except Exception as ex:
            cancelled = key in self._cancel
            job["state"] = "cancelled" if cancelled else "error"
            job["error"] = None if cancelled else f"{type(ex).__name__}: {ex}"
        job["bytes"] = self.disk_bytes(repo)

    def cancel(self, key: str) -> None:
        self._cancel.add(key)

    def delete(self, key: str) -> int:
        e = self.find(key)
        if not e:
            raise KeyError(key)
        job = self.jobs.get(key)
        if job and job["state"] == "running":
            raise ValueError("正在下载，请先取消")
        d = self.repo_dir(e["repo"]).resolve()
        if self.hub.resolve() not in d.parents:
            raise ValueError("拒绝删除模型目录之外的路径")
        size = self.disk_bytes(e["repo"])
        if d.exists():
            shutil.rmtree(d)
        self.jobs.pop(key, None)
        return size
