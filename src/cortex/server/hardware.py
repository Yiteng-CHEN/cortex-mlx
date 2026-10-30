"""Live hardware load for the sphere: GPU utilisation, memory, CPU (macOS; degrades to None elsewhere).

UNVERIFIED on hardware: parsers follow the usual `ioreg -c IOAccelerator` ("Device Utilization %")
and `vm_stat` formats. Nothing needs sudo. Anything that cannot be read is returned as None and the
page falls back to the model's own activity.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time


def parse_gpu(text: str) -> float | None:
    """Highest 'Device Utilization %' among accelerators, as 0..1."""
    vals = [int(v) for v in re.findall(r'"Device Utilization %"\s*=\s*(\d+)', text)]
    return min(1.0, max(vals) / 100) if vals else None


def parse_vm_stat(text: str) -> int | None:
    """Used bytes ≈ (active + wired + compressor) pages."""
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m.group(1)) if m else 16384

    def pages(label):
        r = re.search(label + r":\s+(\d+)", text)
        return int(r.group(1)) if r else 0
    used = pages("Pages active") + pages("Pages wired down") + pages("Pages occupied by compressor")
    return used * page if used else None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


class Hardware:
    def __init__(self, run=_run, active_model_bytes=None, ttl: float = 0.9):
        self._run = run
        self._model_bytes = active_model_bytes
        self._ttl = ttl
        self._lock = threading.Lock()
        self._at = 0.0
        self._val: dict = {}
        self._total: int | None = None

    def sample(self) -> dict:
        with self._lock:
            now = time.time()
            if self._val and now - self._at < self._ttl:
                return self._val
            if self._total is None:
                t = self._run(["sysctl", "-n", "hw.memsize"]).strip()
                self._total = int(t) if t.isdigit() else 0
            used = parse_vm_stat(self._run(["vm_stat"]))
            gpu = parse_gpu(self._run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"]))
            try:
                cpu = min(1.0, os.getloadavg()[0] / (os.cpu_count() or 1))
            except OSError:
                cpu = None
            model = None
            if self._model_bytes:
                try:
                    model = self._model_bytes()
                except Exception:
                    model = None
            total = self._total or None
            self._val = {"gpu": gpu, "cpu": cpu,
                         "mem_used_gb": round(used / 1e9, 1) if used else None,
                         "mem_total_gb": round(total / 1e9, 1) if total else None,
                         "mem_pct": round(used / total, 3) if used and total else None,
                         "model_gb": round(model / 1e9, 1) if model else None}
            self._at = now
            return self._val
