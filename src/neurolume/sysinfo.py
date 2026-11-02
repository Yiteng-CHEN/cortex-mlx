"""Host resource checks (memory now; power/thermal come with the M1 scheduler)."""
from __future__ import annotations

import platform
import re
import subprocess
from pathlib import Path


def available_memory_gb() -> float | None:
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        page = int(re.search(r"page size of (\d+) bytes", out).group(1))
        pages = 0
        for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            m = re.search(rf"{label}:\s+(\d+)", out)
            pages += int(m.group(1)) if m else 0
        return round(pages * page / 1e9, 1)
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        m = re.search(r"MemAvailable:\s+(\d+) kB", meminfo.read_text())
        return round(int(m.group(1)) * 1024 / 1e9, 1) if m else None
    return None
