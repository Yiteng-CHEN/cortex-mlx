"""End to end on the synthetic repo: mine -> review -> run (mock backends) -> compare."""
import io
import json
import unittest
from contextlib import redirect_stdout

from neurolume.backends.factory import make_backend
from neurolume.cli import main
from neurolume.evals.miner import mine, split_counts
from neurolume.evals.report import comparison_rows, latest_results, probe_results
from neurolume.evals.review import review
from neurolume.evals.runner import run_eval, save_result
from neurolume.evals.task import holdout_hashes, load_split

from .helpers import FixtureEnv


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = FixtureEnv()
        cls.report = mine(cls.env.paths, cls.env.cfg.repo("fixture"), cls.env.cfg.eval, log=lambda m: None)

    @classmethod
    def tearDownClass(cls):
        cls.env.cleanup()

    def test_1_mining_results(self):
        r = self.report
        stmts = sorted(t.statement.splitlines()[0] for t in load_split(self.env.paths, "candidates"))
        self.assertEqual(stmts, ["Add multiply", "Fix subtract sign bug"])
        self.assertEqual(sorted(r.rejected.values()), ["base_not_failing", "flaky_at_base"])
        self.assertEqual(r.skipped["tests_only"], 1)
        self.assertEqual(r.skipped["too_many_lines"], 1)
        self.assertEqual(r.skipped["no_test_change"], 1)
        self.assertEqual(r.scanned, 7)

    def test_2_mining_is_idempotent(self):
        again = mine(self.env.paths, self.env.cfg.repo("fixture"), self.env.cfg.eval, log=lambda m: None)
        self.assertEqual(again.accepted, [])
        self.assertEqual(again.skipped["already_known"], 2)

    def test_3_review_run_compare(self):
        paths, cfg = self.env.paths, self.env.cfg
        ids = {t.statement.splitlines()[0]: t.id for t in load_split(paths, "candidates")}
        counts = review(cfg, paths, decisions={ids["Fix subtract sign bug"]: "d",
                                               ids["Add multiply"]: "h"})
        self.assertEqual(counts["dev"], 1)
        self.assertEqual(counts["holdout"], 1)
        self.assertEqual(split_counts(paths)["candidates"], 0)
        holdout = load_split(paths, "holdout")[0]
        self.assertIn(holdout.content_hash, holdout_hashes(paths))

        expected = {"mock-reference": "pass", "mock-noop": "fail", "mock-garbage": "format_error"}
        for name, status in expected.items():
            for split in ("dev", "holdout"):
                res = run_eval(cfg, paths, make_backend(cfg, name), load_split(paths, split),
                               split=split, log=lambda m: None)
                self.assertEqual([r["status"] for r in res["records"]], [status], (name, split))
                save_result(paths, res)

        rows = comparison_rows(latest_results(paths, split="dev"), probe_results(paths))
        table = {r[0]: r for r in rows[1:]}
        self.assertEqual(table["mock-reference"][3], "100%")
        self.assertEqual(table["mock-noop"][3], "0%")
        self.assertEqual(table["mock-garbage"][4], "100%")
        self.assertEqual(list(paths.worktrees.iterdir()), [])

    def test_4_cli_smoke(self):
        argv = ["--local-config", str(self.env.local)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(argv + ["eval", "list"]), 0)
            self.assertEqual(main(argv + ["eval", "run", "--candidate", "mock-reference"]), 0)
            self.assertEqual(main(argv + ["eval", "compare", "--split", "dev"]), 0)
            self.assertEqual(main(argv + ["sys", "config"]), 0)
            self.assertEqual(main(argv + ["eval", "mine", "nope"]), 2)
        out = buf.getvalue()
        self.assertIn("mock-reference", out)
        self.assertIn("compare-", out)


if __name__ == "__main__":
    unittest.main()
