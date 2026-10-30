from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from cortex.config import PROJECT_ROOT, load_config
from cortex.server.library import Library, slug
from cortex.server.training import (Datasets, Trainer, cells, chunk_text, parse_files, parse_line,
                                    plan_iters, sample_idx)


def fake_info(repo):
    if "missing" in repo:
        raise OSError("404")
    if "tokenizer-only" in repo:
        return {"size": 10, "files": ["tokenizer.json"]}
    return {"size": 3000, "files": ["config.json", "model.safetensors"]}


def fake_dl_factory(delay=0.0, fail=False):
    def dl(repo, cache, should_cancel):
        d = Path(cache) / ("models--" + repo.replace("/", "--"))
        (d / "blobs").mkdir(parents=True, exist_ok=True)
        (d / "snapshots" / "rev1").mkdir(parents=True, exist_ok=True)
        for i in range(3):
            if should_cancel():
                raise InterruptedError("cancelled")
            time.sleep(delay)
            (d / "blobs" / f"b{i}").write_bytes(b"x" * 1000)
        if fail:
            raise OSError("network down")
        (d / "snapshots" / "rev1" / "config.json").write_text("{}")
        (d / "snapshots" / "rev1" / "model.safetensors").write_bytes(b"w")
    return dl


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = load_config(PROJECT_ROOT / "config" / "cortex.toml", None)
        self.lib = Library(self.cfg, self.root, self.root / "hub", fake_info, fake_dl_factory())

    def tearDown(self):
        self.tmp.cleanup()

    def wait(self, key, state="done"):
        for _ in range(100):
            if self.lib.jobs[key]["state"] != "running":
                break
            time.sleep(0.02)
        self.assertEqual(self.lib.jobs[key]["state"], state)

    def test_catalog_entries_and_status(self):
        keys = [e["key"] for e in self.lib.entries()]
        self.assertIn("devstral-small-2", keys)
        self.assertNotIn("demo-chat", keys)
        st = self.lib.status()
        self.assertTrue(all(not i["downloaded"] for i in st["items"]))

    def test_add_custom_validates(self):
        e = self.lib.add("someone/Cool-Model-4bit")
        self.assertEqual(e["key"], slug("someone/Cool-Model-4bit"))
        with self.assertRaises(ValueError):
            self.lib.add("someone/Cool-Model-4bit")        # duplicate
        with self.assertRaises(ValueError):
            self.lib.add("not a repo")
        with self.assertRaises(ValueError):
            self.lib.add("x/missing")
        with self.assertRaises(ValueError):
            self.lib.add("x/tokenizer-only")
        self.assertEqual(self.lib.candidate(e["key"]).model, "someone/Cool-Model-4bit")

    def test_download_then_delete(self):
        self.lib.add("a/b-model")
        job = self.lib.download("b-model")
        self.assertEqual(job["state"], "running")
        self.wait("b-model")
        repo = "a/b-model"
        self.assertTrue(self.lib.is_downloaded(repo))
        self.assertEqual(self.lib.disk_bytes(repo), 3000)
        self.assertEqual(self.lib.delete("b-model"), 3000)
        self.assertFalse(self.lib.is_downloaded(repo))

    def test_failed_and_cancelled(self):
        self.lib._dl = fake_dl_factory(fail=True)
        self.lib.add("a/bad-model")
        self.lib.download("bad-model")
        self.wait("bad-model", "error")
        self.assertIn("network down", self.lib.jobs["bad-model"]["error"])
        self.assertFalse(self.lib.is_downloaded("a/bad-model"))
        self.lib._dl = fake_dl_factory(delay=0.05)
        self.lib.add("a/slow-model")
        self.lib.download("slow-model")
        self.lib.cancel("slow-model")
        self.wait("slow-model", "cancelled")

    def test_incomplete_file_means_not_downloaded(self):
        self.lib._dl = fake_dl_factory()
        self.lib.add("a/inc-model")
        self.lib.download("inc-model")
        self.wait("inc-model")
        (self.lib.repo_dir("a/inc-model") / "blobs" / "x.incomplete").write_bytes(b"1")
        self.assertFalse(self.lib.is_downloaded("a/inc-model"))

    def test_disk_space_guard(self):
        self.lib._info = lambda repo: {"size": 10 ** 18, "files": ["config.json", "m.safetensors"]}
        self.lib.add("a/huge-model")
        with self.assertRaises(ValueError):
            self.lib.download("huge-model")


class DatasetTests(unittest.TestCase):
    def test_jsonl_formats_and_secrets(self):
        lines = [
            json.dumps({"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]}),
            json.dumps({"prompt": "q2", "completion": "a2"}),
            json.dumps({"instruction": "q3", "output": "a3"}),
            json.dumps({"text": "plain text sample"}),
            json.dumps({"prompt": "my key sk-" + "a" * 30, "completion": "x"}),
            json.dumps({"prompt": "q2", "completion": "a2"}),     # duplicate
            "not json",
            json.dumps({"weird": 1}),
        ]
        p = parse_files([{"name": "d.jsonl", "text": "\n".join(lines)}])
        self.assertEqual(len(p["rows"]), 4)
        self.assertEqual(p["stats"]["secrets"], 1)
        self.assertEqual(p["stats"]["dupes"], 1)
        self.assertEqual(p["stats"]["bad_lines"], 2)
        self.assertTrue(any("密钥" in w for w in p["warnings"]))
        self.assertTrue(any("20" in w for w in p["warnings"]))

    def test_text_chunking(self):
        text = "\n\n".join(f"Paragraph {i} " + "word " * 120 for i in range(10))
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c) <= 2600 for c in chunks))

    def test_create_split_and_list(self):
        with tempfile.TemporaryDirectory() as d:
            ds = Datasets(Path(d))
            rows = "\n".join(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) for i in range(25))
            meta = ds.create("mine", [{"name": "x.jsonl", "text": rows}])
            self.assertEqual(meta["n_train"] + meta["n_valid"], 25)
            self.assertGreaterEqual(meta["n_valid"], 2)
            self.assertEqual(ds.list()[0]["id"], meta["id"])
            self.assertTrue((ds.path(meta["id"]) / "valid.jsonl").exists())
            with self.assertRaises(ValueError):
                ds.create("empty", [{"name": "e.jsonl", "text": "garbage"}])
            with self.assertRaises(ValueError):
                ds.path("../etc")


class _FakeProc:
    def __init__(self, lines, rc=0, hold=None):
        self.stdout = iter(lines)
        self._rc, self._hold = rc, hold

    def wait(self):
        if self._hold:
            self._hold.wait(2)
        return self._rc

    def terminate(self):
        if self._hold:
            self._hold.set()


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ds = Datasets(self.root / "datasets")
        rows = "\n".join(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) for i in range(30))
        self.meta = self.ds.create("t", [{"name": "x.jsonl", "text": rows}])

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_line(self):
        r = parse_line("       10  1.234  ▼  45  1.2k")
        self.assertEqual((r["iter"], r["loss"], r["tok_s"], r["tokens"]), (10, 1.234, 45.0, 1200.0))
        self.assertEqual(parse_line("Val loss 0.987")["val_loss"], 0.987)
        self.assertIsNone(parse_line("Loading model..."))

    def test_plan_iters(self):
        self.assertEqual(plan_iters(5, "quick"), 20)
        self.assertEqual(plan_iters(30, "quick"), 30)
        self.assertEqual(plan_iters(5000, "quick"), 100)
        self.assertEqual(plan_iters(5000, "standard"), 600)

    def run_with(self, lines, rc, write_adapter=True, hold=None):
        def spawn(cmd, cwd):
            if write_adapter:
                (Path(cmd[1]).parent / "adapters.safetensors").write_bytes(b"x")
            return _FakeProc(lines, rc, hold)
        tr = Trainer(self.root, self.ds, spawn)
        tr.start("m1", "org/model", self.meta["id"], "quick")
        return tr

    def wait_done(self, tr):
        for _ in range(100):
            if not tr.running():
                break
            time.sleep(0.02)

    def test_successful_run_tracks_progress(self):
        tr = self.run_with(["Loading", "   2  2.5  ▼  40  100", "   4  2.1  ▼  42  200", "Val loss 1.9",
                            "CORTEX_PEAK_GB=18.30"], 0)
        self.wait_done(tr)
        st = tr.status()
        self.assertEqual(st["state"], "done")
        self.assertEqual(st["loss"], 2.1)
        self.assertEqual(st["val_loss"], 1.9)
        self.assertEqual(st["peak_gb"], 18.3)
        self.assertEqual(st["history"], [[2, 2.5], [4, 2.1]])
        self.assertTrue(st["adapter"].startswith("m1/"))
        self.assertEqual(tr.adapters()[0]["id"], st["adapter"])
        self.assertTrue(tr.adapter_dir(st["adapter"]).is_dir())
        with self.assertRaises(ValueError):
            tr.adapter_dir("../../etc/passwd")

    def test_failure_is_reported_in_plain_words(self):
        tr = self.run_with(["Error: [metal::malloc] Resource limit out of memory"], 1, write_adapter=False)
        self.wait_done(tr)
        st = tr.status()
        self.assertEqual(st["state"], "failed")
        self.assertIn("内存", st["error"])

    def test_cancel_and_single_run(self):
        hold = threading.Event()
        tr = self.run_with(["   2  2.5  ▼  40  100"], 1, write_adapter=False, hold=hold)
        with self.assertRaises(RuntimeError):
            tr.start("m1", "org/model", self.meta["id"], "quick")
        tr.cancel()
        self.wait_done(tr)
        self.assertEqual(tr.status()["state"], "cancelled")

    def test_config_written_for_lora_on_four_matrices(self):
        tr = self.run_with([], 0)
        self.wait_done(tr)
        cfgs = list((self.root / "adapters").glob("*/*/lora.yaml"))
        c = json.loads(cfgs[0].read_text())
        self.assertEqual(c["lora_parameters"]["keys"], ["self_attn.q_proj", "self_attn.o_proj",
                                                       "mlp.gate_proj", "mlp.down_proj"])
        self.assertTrue(c["train"] and c["grad_checkpoint"])


class DeltaMathTests(unittest.TestCase):
    def test_sample_idx_matches_weight_map(self):
        self.assertEqual(sample_idx(4096)[0], 0)
        self.assertEqual(sample_idx(4096)[-1], 4095)
        self.assertEqual(len(sample_idx(5120)), 12)

    def test_cells_equals_scaled_outer_product(self):
        # A rows (in-positions j) and B cols (out-positions i), rank 2
        a_sel = [[1.0, 2.0], [0.0, 1.0]]            # j = 0, 1
        b_sel = [[3.0, 0.5], [1.0, 1.0]]            # i = 0, 1
        out = cells(a_sel, b_sel, scale=2.0)
        self.assertEqual(out, [2 * (3 * 1 + 0.5 * 2), 2 * (3 * 0 + 0.5 * 1),
                               2 * (1 * 1 + 1 * 2), 2 * (1 * 0 + 1 * 1)])   # index = i*G + j


class ServerFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cortex.server.app import serve
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cfg = load_config(PROJECT_ROOT / "config" / "cortex.toml", None)
        cls.lib = Library(cfg, root, root / "hub", fake_info, fake_dl_factory())
        ds = Datasets(root / "datasets")
        cls.cap = {}

        def spawn(cmd, cwd):
            (Path(cmd[1]).parent / "adapters.safetensors").write_bytes(b"x")
            cls.cap["cmd"] = cmd
            return _FakeProc(["   2  2.0  ▼  40  100"], 0)
        tr = Trainer(root, ds, spawn)
        cls.httpd = serve(cfg, port=0, block=False, library=cls.lib, trainer=tr,
                          loader=lambda n: __import__("cortex.server.chat", fromlist=["DemoChat"]).DemoChat(n, delay=0))
        cls.state = cls.httpd.cortex_state
        cls.state.datasets = ds
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def test_library_flow_over_http(self):
        code, j = self.call("/api/library/add", {"repo": "z/flow-model"})
        self.assertEqual(code, 200)
        code, j = self.call("/api/library/download", {"key": "flow-model"})
        self.assertEqual(code, 200)
        for _ in range(100):
            _, st = self.call("/api/library")
            it = next(i for i in st["items"] if i["key"] == "flow-model")
            if it["downloaded"]:
                break
            time.sleep(0.02)
        self.assertTrue(it["downloaded"])
        code, j = self.call("/api/library/delete", {"key": "flow-model"})
        self.assertEqual((code, j["freed"]), (200, 3000))

    def test_load_requires_download_message(self):
        # default loader is replaced in this fixture, so check the HTTP error mapping directly
        code, j = self.call("/api/select", {"name": "nope"})
        self.assertEqual(code, 404)

    def test_dataset_to_training_over_http(self):
        rows = "\n".join(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) for i in range(25))
        code, meta = self.call("/api/train/dataset", {"name": "d", "files": [{"name": "a.jsonl", "text": rows}]})
        self.assertEqual(code, 200)
        code, j = self.call("/api/train/dataset", {"name": "d", "files": [{"name": "a.jsonl", "text": "junk"}]})
        self.assertEqual(code, 400)
        # model not downloaded -> 409 with a code the UI can act on
        code, j = self.call("/api/train/start", {"model": "devstral-small-2", "dataset": meta["id"], "preset": "quick"})
        self.assertEqual((code, j.get("code")), (409, "not_downloaded"))
        # download a custom model, then train on it
        self.call("/api/library/add", {"repo": "z/train-model"})
        self.call("/api/library/download", {"key": "train-model"})
        for _ in range(100):
            if self.lib.is_downloaded("z/train-model"):
                break
            time.sleep(0.02)
        code, j = self.call("/api/train/start", {"model": "train-model", "dataset": meta["id"], "preset": "quick"})
        self.assertEqual(code, 200)
        for _ in range(100):
            _, st = self.call("/api/train/status")
            if st["state"] != "running":
                break
            time.sleep(0.02)
        self.assertEqual(st["state"], "done")
        _, ad = self.call("/api/train/adapters")
        self.assertEqual(ad["adapters"][0]["id"], st["adapter"])


if __name__ == "__main__":
    unittest.main()
