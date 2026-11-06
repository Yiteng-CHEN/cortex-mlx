import os
import tempfile
import unittest
from pathlib import Path

from neurolume.config import EvalSettings, RepoConfig
from neurolume.evals.sandbox import (UnsafePathError, apply_files, cleanup_stale, evaluate,
                                  minimal_env, run_command)
from neurolume.evals.task import EvalTask
from neurolume.gitutil import git

from .helpers import FixtureEnv


class TestPathSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_allowed_file_written(self):
        apply_files(self.root, ["pkg/a.py"], {"pkg/a.py": "x = 1\n"})
        self.assertEqual((self.root / "pkg/a.py").read_text(), "x = 1\n")

    def test_rejects_not_allowed(self):
        with self.assertRaises(UnsafePathError):
            apply_files(self.root, ["a.py"], {"b.py": ""})

    def test_rejects_traversal_and_absolute(self):
        for bad in ("../evil.py", "/etc/passwd", "a/../../evil.py", ".git/config", "~/x.py"):
            with self.assertRaises(UnsafePathError, msg=bad):
                apply_files(self.root, [bad], {bad: ""})

    def test_rejects_symlink_escape(self):
        outside = tempfile.TemporaryDirectory()
        try:
            os.symlink(outside.name, self.root / "link")
            with self.assertRaises(UnsafePathError):
                apply_files(self.root, ["link/evil.py"], {"link/evil.py": ""})
            self.assertFalse((Path(outside.name) / "evil.py").exists())
        finally:
            outside.cleanup()


class TestRunCommand(unittest.TestCase):
    def test_timeout_kills_process_group(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_command("sleep 30 & sleep 30", Path(d), timeout_s=1, cpu_seconds=10, env=minimal_env())
            self.assertTrue(r.timed_out)
            self.assertFalse(r.ok)
            self.assertLess(r.seconds, 10)

    def test_env_is_minimal(self):
        os.environ["NEUROLUME_FAKE_SECRET_TOKEN"] = "s3cret"
        try:
            with tempfile.TemporaryDirectory() as d:
                r = run_command("env", Path(d), timeout_s=10, cpu_seconds=10, env=minimal_env({"X": "1"}))
            self.assertNotIn("s3cret", r.output)
            self.assertIn("X=1", r.output)
        finally:
            del os.environ["NEUROLUME_FAKE_SECRET_TOKEN"]


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        self.env = FixtureEnv(stability_runs=1)
        self.repo = self.env.cfg.repo("fixture")
        shas = git(self.repo.path, "rev-list", "--reverse", "HEAD").split()
        # commit 2 = "Fix subtract sign bug"
        self.task = EvalTask(id="t", repo="fixture", base_sha=shas[0], target_sha=shas[1],
                             source_files=["calc.py"], test_files=["tests/test_calc.py"],
                             statement="fix", content_hash="h")

    def tearDown(self):
        self.env.cleanup()

    def test_base_fails_reference_passes_and_worktrees_cleaned(self):
        from neurolume.evals.sandbox import reference_solution

        s = self.env.cfg.eval
        base = evaluate(self.env.paths, self.repo, self.task, s, solution=None, runs=2)
        self.assertTrue(base.all_fail)
        good = evaluate(self.env.paths, self.repo, self.task, s,
                        solution=reference_solution(self.repo, self.task))
        self.assertTrue(good.all_pass, good.first.output)
        self.assertEqual(list(self.env.paths.worktrees.iterdir()), [])
        listed = git(self.repo.path, "worktree", "list")
        self.assertEqual(len(listed.strip().splitlines()), 1)

    def test_setup_failure_is_reported(self):
        repo = RepoConfig(name="fixture", path=self.repo.path, test=self.repo.test, setup="exit 3")
        out = evaluate(self.env.paths, repo, self.task, self.env.cfg.eval, solution=None)
        self.assertFalse(out.setup.ok)
        self.assertEqual(out.runs, [])

    def test_cleanup_stale_dry_run_then_apply(self):
        stale = self.env.paths.worktrees / "leftover"
        stale.mkdir()
        found = cleanup_stale(self.env.paths, [self.repo], dry_run=True)
        self.assertEqual(found, [stale])
        self.assertTrue(stale.exists())
        cleanup_stale(self.env.paths, [self.repo], dry_run=False)
        self.assertFalse(stale.exists())
        log = (self.env.paths.logs / "cleanup.log").read_text()
        self.assertIn("DRY-RUN would remove", log)


if __name__ == "__main__":
    unittest.main()
