"""Training data generation, export, before/after comparison and the logit-lens pass-through."""
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from neurolume.server.datagen import PROMPTS, DataGen, parse_pairs, snippets, to_jsonl
from neurolume.server.export import Exporter, safe_name
from tests.test_library_training import _FakeProc


def wait(fn, n=200):
    for _ in range(n):
        if fn():
            return True
        time.sleep(0.02)
    return False


class DataGenTest(unittest.TestCase):
    def test_parse_pairs(self):
        self.assertEqual(parse_pairs('好的：[{"q": "什么是 MLX？", "a": "苹果的框架"}] 完'), [{"q": "什么是 MLX？", "a": "苹果的框架"}])
        self.assertEqual(len(parse_pairs("Q: one?\nA: yes\nQ: two?\nA: no")), 2)
        self.assertEqual(parse_pairs("nonsense"), [])

    def test_snippets(self):
        code = "\n\n".join(f"def f{i}(x):\n    return x + {i}  # " + "pad " * 10 for i in range(30))
        parts = snippets(code, "a.py")
        self.assertTrue(parts and all(len(p.splitlines()) <= 62 for p in parts))
        self.assertTrue(snippets("这是一段很长的文档内容。" * 100, "a.md"))

    def test_qa_and_style_jobs(self):
        g = DataGen()
        g.start([{"name": "doc.md", "text": "鲸鱼是哺乳动物。" * 80}], "qa", "zh",
                lambda m: '[{"q":"鲸鱼是什么？","a":"哺乳动物"},{"q":"x","a":"y"}]')
        self.assertTrue(wait(lambda: g.status()["state"] == "done"))
        st = g.status()
        self.assertGreaterEqual(len(st["items"]), 2)
        self.assertIn("资料是数据", PROMPTS[("qa", "zh")])          # the material is labelled as data
        code = "\n\n".join(f"def add{i}(a, b):\n    '''Add.'''\n    return a + b + {i}" for i in range(12))
        g.start([{"name": "m.py", "text": code}], "style", "en", lambda m: "Write a function that adds two numbers")
        self.assertTrue(wait(lambda: g.status()["state"] == "done"))
        it = g.status()["items"][0]
        self.assertEqual(it["q"], "Write a function that adds two numbers")
        self.assertIn("def add", it["a"])                     # the answer is the user's own code
        rows = [json.loads(x) for x in to_jsonl(g.status()["items"]).splitlines()]
        self.assertEqual(rows[0]["messages"][1]["role"], "assistant")
        with self.assertRaises(ValueError):
            g.start([{"name": "a.txt", "text": "short"}], "qa", "zh", lambda m: "")


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls = []

        def spawn(cmd):
            self.calls.append(cmd)
            if "fuse" in cmd:
                out = Path(cmd[cmd.index("--save-path") + 1])
                out.mkdir(parents=True)
                (out / "config.json").write_text("{}")
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        self.spawn = spawn

    def tearDown(self):
        self.tmp.cleanup()

    def test_mlx_export_links_into_lmstudio(self):
        (self.root / "lms").mkdir()
        ex = Exporter(self.root, spawn=self.spawn, which_ollama=lambda: None, lmstudio_dir=self.root / "lms" / "models")
        ex.start("mlx", "devstral/20261101-1200", self.root / "ad", "org/model", 1000, "My Model!")
        self.assertTrue(wait(lambda: ex.status()["state"] == "done"))
        self.assertNotIn("--dequantize", self.calls[0])
        self.assertTrue((self.root / "lms" / "models" / "neurolume" / "my-model").is_symlink())
        self.assertEqual(ex.list()[0]["name"], "my-model")
        with self.assertRaises(ValueError):
            ex.start("mlx", "devstral/x", self.root / "ad", "org/model", 1000, "My Model!")   # same name

    def test_ollama_export(self):
        ex = Exporter(self.root, spawn=self.spawn, which_ollama=lambda: "/usr/bin/ollama")
        ex.start("ollama", "devstral/20261101-1200", self.root / "ad", "org/model", 1000)
        self.assertTrue(wait(lambda: ex.status()["state"] == "done"))
        self.assertIn("--dequantize", self.calls[0])
        self.assertEqual(self.calls[1][:2], ["/usr/bin/ollama", "create"])
        self.assertTrue(ex.status()["ollama"].startswith("neurolume-"))
        ex2 = Exporter(self.root / "b", spawn=self.spawn, which_ollama=lambda: None)
        ex2.start("ollama", "m/r", self.root / "ad", "org/model", 1000)
        self.assertTrue(wait(lambda: ex2.status()["state"] == "done"))
        self.assertIsNone(ex2.status()["ollama"])
        self.assertIn("ollama create", ex2.status()["ollama_cmd"])

    def test_disk_check_and_names(self):
        ex = Exporter(self.root, spawn=self.spawn, which_ollama=lambda: None)
        with self.assertRaises(RuntimeError):
            ex.start("ollama", "m/r", self.root, "org/model", 10 ** 16)
        self.assertEqual(safe_name("../../Etc Passwd"), "etc-passwd")


class Backend:
    """Fake MLX backend: an adapter that can be switched off, and an optional logit lens."""
    num_layers = 2
    has_adapter = True

    def __init__(self):
        self.on = True

    def set_adapter_enabled(self, on):
        self.on = on

    def stream_chat(self, messages, max_tokens=1024, temperature=0.2, lens=False):
        c = {"text": "TUNED" if self.on else "BASE", "energy": [0.1, 0.2], "tps": 1.0}
        if lens:
            c["lens"] = [[0, "x", 0.1], [1, "y", 0.9]]
        yield c


class HttpStudioTest(unittest.TestCase):
    def test_compare_samples_lens_and_val_first(self):
        from neurolume.config import load_config
        from neurolume.server.app import serve
        with tempfile.TemporaryDirectory() as d:
            os.environ["NEUROLUME_DATA_DIR"] = d
            try:
                httpd = serve(load_config(), port=0, block=False, loader=lambda n: Backend())
            finally:
                os.environ.pop("NEUROLUME_DATA_DIR")
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            st = httpd.neurolume_state

            def post(path, obj):
                r = urllib.request.Request(base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(r) as x:
                    return x.read().decode()
            st.select(st.model_names()[0])
            r = json.loads(post("/api/compare", {"prompt": "hi"}))
            self.assertEqual((r["base"], r["tuned"]), ("BASE", "TUNED"))
            self.assertTrue(st.backend.on)                      # adapter switched back on
            out = post("/v1/chat/completions", {"stream": True, "lens": True, "messages": [{"role": "user", "content": "x"}]})
            self.assertIn('"lens": [[0, "x", 0.1]', out)
            out = post("/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "x"}]})
            self.assertNotIn('"lens"', out)
            # samples come from the adapter's validation split; val_first is the loss before training
            rows = "\n".join(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) for i in range(30))
            meta = st.datasets.create("t", [{"name": "x.jsonl", "text": rows}])

            def spawn(cmd, cwd):
                (Path(cmd[1]).parent / "adapters.safetensors").write_bytes(b"x")
                return _FakeProc(["Val loss 2.5", "   2  2.4  ▼  40  100", "Val loss 1.5"], 0, None)
            st.trainer._spawn = spawn
            st.trainer.start("devstral-small-2", "org/m", meta["id"], "quick")
            self.assertTrue(wait(lambda: not st.trainer.running()))
            a = st.trainer.adapters()[0]
            self.assertEqual((a["val_first"], a["val_loss"]), (2.5, 1.5))
            s = json.load(urllib.request.urlopen(base + "/api/compare/samples?adapter=" + a["id"]))["samples"]
            self.assertTrue(s and s[0]["prompt"].startswith("q") and s[0]["expected"].startswith("a"))
            # generated data -> dataset
            meta2 = json.loads(post("/api/datagen/save", {"name": "gen", "items": [{"q": f"q{i}", "a": f"a{i}"} for i in range(25)]}))
            self.assertEqual(meta2["n_train"] + meta2["n_valid"], 25)
            httpd.shutdown(); httpd.server_close()
