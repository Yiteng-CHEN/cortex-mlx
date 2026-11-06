from __future__ import annotations

import os
import tempfile
from pathlib import Path

from neurolume.config import DEFAULTS, PROJECT_ROOT, load_config
from neurolume.devtools.fixture_repo import TEST_COMMAND, build
from neurolume.paths import DataPaths


class FixtureEnv:
    """Temp data dir + synthetic repo + local.toml pointing at both."""

    def __init__(self, **eval_overrides):
        os.environ.pop("NEUROLUME_DATA_DIR", None)
        self._tmp = tempfile.TemporaryDirectory(prefix="neurolume-test-")
        root = Path(self._tmp.name)
        self.repo = build(root / "repo")
        self.local = root / "local.toml"
        lines = [
            "[paths]", f'data_dir = "{root / "data"}"',
            "[repos.fixture]", f'path = "{self.repo}"', f'test = "{TEST_COMMAND}"', "timeout_s = 60",
        ]
        if eval_overrides:
            lines.append("[eval]")
            lines += [f"{k} = {v!r}" for k, v in eval_overrides.items()]
        self.local.write_text("\n".join(lines) + "\n")
        self.cfg = load_config(DEFAULTS, self.local)
        self.paths = DataPaths(self.cfg.data_dir).ensure()

    def cleanup(self):
        self._tmp.cleanup()
