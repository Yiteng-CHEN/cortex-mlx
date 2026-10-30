import os
import tempfile
import unittest
from pathlib import Path

from cortex.config import PROJECT_ROOT, deep_merge, load_config

MAIN = PROJECT_ROOT / "config" / "cortex.toml"


class TestConfig(unittest.TestCase):
    def setUp(self):
        os.environ.pop("CORTEX_DATA_DIR", None)
        self.tmp = tempfile.TemporaryDirectory()
        self.local = Path(self.tmp.name) / "local.toml"

    def tearDown(self):
        os.environ.pop("CORTEX_DATA_DIR", None)
        self.tmp.cleanup()

    def test_defaults(self):
        cfg = load_config(MAIN, self.local)  # local does not exist
        self.assertTrue(str(cfg.data_dir).endswith("Library/Application Support/Cortex"))
        self.assertIn("qwen3.6-27b", cfg.candidates)
        self.assertEqual(cfg.candidates["qwen3.6-27b"].chat_template_kwargs, {"enable_thinking": False})
        self.assertEqual(cfg.eval.stability_runs, 3)
        self.assertEqual(cfg.repos, {})

    def test_local_overrides_and_repo(self):
        self.local.write_text(
            '[paths]\ndata_dir = "~/cortex-data"\n[eval]\nstability_runs = 1\n'
            '[repos.app]\npath = "~/code/app"\ntest = "pytest {tests}"\n'
            '[candidates."qwen3.6-27b"]\nmin_free_gb = 60\n')
        cfg = load_config(MAIN, self.local)
        self.assertEqual(cfg.data_dir, Path("~/cortex-data").expanduser().resolve())
        self.assertEqual(cfg.eval.stability_runs, 1)
        self.assertEqual(cfg.repo("app").path, Path("~/code/app").expanduser().resolve())
        c = cfg.candidate("qwen3.6-27b")
        self.assertEqual(c.min_free_gb, 60)
        self.assertEqual(c.model, "unsloth/Qwen3.6-27B-MLX-8bit")  # merged, not replaced

    def test_env_data_dir_wins(self):
        os.environ["CORTEX_DATA_DIR"] = self.tmp.name
        self.assertEqual(load_config(MAIN, self.local).data_dir, Path(self.tmp.name).resolve())

    def test_unknown_names(self):
        cfg = load_config(MAIN, self.local)
        with self.assertRaises(KeyError):
            cfg.repo("nope")
        with self.assertRaises(KeyError):
            cfg.candidate("nope")

    def test_repo_requires_test(self):
        self.local.write_text('[repos.app]\npath = "/tmp"\n')
        with self.assertRaises(ValueError):
            load_config(MAIN, self.local)

    def test_deep_merge(self):
        self.assertEqual(deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}), {"a": {"b": 1, "c": 3}})


if __name__ == "__main__":
    unittest.main()
