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
        c = {"text": "TUNED" if self.on else "BASE", "energy": [0.1, 0.2], "tps": 1.0,
             "token": "T", "conf": 0.62, "alts": [["T", 0.62], ["U", 0.2]], "gid": 7}
        if lens:
            c["lens"] = [[0, "x", 0.1], [1, "y", 0.9]]
        yield c


    def lens_backfill(self, gid):
        return {"gid": gid, "tokens": ["T"], "lens": [[[0, "x", 0.1], [1, "T", 0.9]]]}


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
            self.assertIn('"conf": 0.62', out)                 # per-token confidence + candidates reach the browser
            self.assertIn('"alts": [["T", 0.62], ["U", 0.2]]', out)
            self.assertIn('"gid": 7', out)                     # answer id, for computing the lens afterwards
            st.samples.add([{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}])
            m = json.loads(post("/api/train/dataset", {"name": "x", "include_samples": True,
                                                       "files": [{"name": "a.jsonl", "text": '{"prompt": "p", "completion": "c"}'}]}))
            self.assertEqual(m["n_train"], 2)                     # file rows + samples saved from chat
            r = json.loads(post("/api/lens/backfill", {"gid": 7}))
            self.assertEqual(r["lens"][0][1], [1, "T", 0.9])
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


class AlignerTest(unittest.TestCase):
    def test_reports_the_pass_that_produced_each_token(self):
        from neurolume.telemetry.probe import Aligner
        a = Aligner()
        # prefill chunk, last prompt step (-> token 0), step(token 0) (-> token 1), then token 0 is yielded
        for s in ("prefill", "P0", "S0"):
            a.push(s)
        self.assertEqual(a.take(), "P0")
        a.push("S1")                      # step(token 1) runs before token 1 is yielded
        self.assertEqual(a.take(), "S0")
        self.assertEqual(a.take(), "S1")  # final token: no further step was started
        self.assertIsNone(a.take())


class WeightsGenerateTest(unittest.TestCase):
    def test_generates_from_loaded_model_without_reloading(self):
        from neurolume.config import load_config
        from neurolume.server.app import serve
        from neurolume.tools import dump_weights
        calls = []

        def fake_dump(model, model_id, key, grid, out_dir, log=print):
            calls.append((model, key))
            out = Path(out_dir) / f"weights-{key}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"layers": []}))
            return out

        class M:
            num_layers = 2
            _model = "MODEL"
            def stream_chat(self, messages, **kw):
                yield {"text": "x"}
        orig = dump_weights.dump_from_model
        dump_weights.dump_from_model = fake_dump
        try:
            with tempfile.TemporaryDirectory() as d:
                os.environ["NEUROLUME_DATA_DIR"] = d
                try:
                    httpd = serve(load_config(), port=0, block=False, loader=lambda n: M())
                finally:
                    os.environ.pop("NEUROLUME_DATA_DIR")
                threading.Thread(target=httpd.serve_forever, daemon=True).start()
                base = f"http://127.0.0.1:{httpd.server_address[1]}"
                st = httpd.neurolume_state
                st.select("devstral-small-2")
                r = urllib.request.Request(base + "/api/weights/generate", b"{}", {"Content-Type": "application/json"})
                with urllib.request.urlopen(r) as x:
                    self.assertTrue(json.load(x)["ok"])
                self.assertEqual(calls, [("MODEL", "devstral-small-2")])
                with urllib.request.urlopen(base + "/api/weights?model=devstral-small-2") as x:
                    self.assertEqual(json.load(x), {"layers": []})
                httpd.shutdown(); httpd.server_close()
        finally:
            dump_weights.dump_from_model = orig


class ConfidenceTest(unittest.TestCase):
    def test_top_candidates_from_logprobs(self):
        try:
            import mlx.core as mx
        except Exception:
            self.skipTest("mlx not installed")
        from neurolume.backends.mlx_backend import MLXBackend
        b = MLXBackend.__new__(MLXBackend)
        b.decode_token = lambda i: "t%d" % i
        lp = mx.log(mx.array([[0.05, 0.6, 0.1, 0.2, 0.05]]))
        p, alts = b._confidence(lp, 3)
        self.assertAlmostEqual(p, 0.2, places=3)
        self.assertEqual([a[0] for a in alts], ["t1", "t3", "t2", "t0"][:3] + [alts[3][0]])


class ProbeCaptureTest(unittest.TestCase):
    def test_keeps_answer_positions_across_prefill_chunks(self):
        from neurolume.telemetry.probe import LayerProbe

        class Seq:                                    # minimal [1, T, 1] array: .shape and [0, a:b, :]
            def __init__(self, vals):
                self.v = list(vals)
                self.shape = (1, len(self.v), 1)

            def __getitem__(self, k):
                return self.v[k[1]]

        blocks = [lambda x: x, lambda x: x]
        p = LayerProbe(blocks)
        p.enabled = True
        p.capture_range(3, 7)                         # positions 3..6 of a 9-token sequence
        for s in range(0, 9, 4):                     # chunked prefill: 4 + 4 + 1
            chunk = Seq(range(s, min(9, s + 4)))
            for b in blocks:
                chunk = b(chunk)
        h = p.end_capture()
        self.assertEqual([v for part in h[1] for v in part], [3, 4, 5, 6])
        self.assertIsNone(p.capture)


class SamplesTest(unittest.TestCase):
    def test_collect_dedupe_remove_and_make_dataset(self):
        from neurolume.server.samples import Samples
        from neurolume.server.training import Datasets
        with tempfile.TemporaryDirectory() as d:
            s = Samples(Path(d) / "samples" / "collected.jsonl")
            qa = [{"role": "user", "content": "起订量多少？"}, {"role": "assistant", "content": "挂耳 500 盒起订。"}]
            r = s.add(qa)
            self.assertEqual(r["n"], 1)
            self.assertTrue(s.add(qa)["duplicate"])
            with self.assertRaises(ValueError):
                s.add([{"role": "user", "content": "只有问题"}])
            s.add([{"role": "user", "content": "能打样吗"}, {"role": "assistant", "content": "可以。"}], edited=True)
            lst = s.list()
            self.assertEqual(lst["n"], 2)
            self.assertTrue(lst["items"][0]["edited"])                 # newest first
            meta = Datasets(Path(d) / "ds").create("t", [{"name": "collected.jsonl", "text": s.as_jsonl()}])
            self.assertEqual(meta["n_train"], 2)            # tiny sets reuse one sample for validation
            s.remove(lst["items"][0]["id"])
            self.assertEqual(s.list()["n"], 1)


class ChatImportTest(unittest.TestCase):
    def test_chatgpt_claude_and_claude_code_exports(self):
        from neurolume.server.training import parse_files
        gpt = [{"title": "t", "current_node": "c", "mapping": {
            "r": {"message": None, "parent": None},
            "a": {"message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["你好"]}}, "parent": "r"},
            "b": {"message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["你好！"]}}, "parent": "a"},
            "x": {"message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["被丢弃的分支"]}}, "parent": "a"},
            "c": {"message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["再见"]}}, "parent": "b"}}}]
        claude = [{"name": "n", "chat_messages": [{"sender": "human", "text": "Q1"}, {"sender": "assistant", "text": "A1"},
                                                 {"sender": "human", "text": "Q2"}, {"sender": "assistant", "text": "",
                                                  "content": [{"type": "text", "text": "A2"}]}]}]
        cc = [{"type": "user", "message": {"role": "user", "content": "修一下这个 bug"}},
              {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "我先看看"},
                                                                                {"type": "tool_use", "name": "Read"}]}},
              {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "..."}]}},
              {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "修好了：少了一个判断"}]}},
              {"type": "user", "message": {"role": "user", "content": "<command-name>/clear</command-name>"}}]
        p = parse_files([{"name": "conversations.json", "text": json.dumps(gpt)},
                         {"name": "claude.json", "text": json.dumps(claude)},
                         {"name": "s.jsonl", "text": "\n".join(json.dumps(o) for o in cc)}])
        pairs = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in p["rows"]]
        self.assertIn(("你好", "你好！"), pairs)                          # kept branch only, unanswered last turn dropped
        self.assertNotIn("被丢弃的分支", json.dumps(pairs, ensure_ascii=False))
        self.assertIn(("Q1", "A1"), pairs)
        self.assertIn(("Q2", "A2"), pairs)
        self.assertIn(("修一下这个 bug", "修好了：少了一个判断"), pairs)   # final answer only, tool turns skipped
        self.assertEqual(len(pairs), 4)
        self.assertTrue(any("Claude Code" in w for w in p["warnings"]))


class DemoDataTest(unittest.TestCase):
    def test_online_demo_recording_is_complete(self):
        web = Path(__file__).resolve().parents[1] / "src" / "neurolume" / "web" / "demo"
        s = json.loads((web / "session.json").read_text(encoding="utf-8"))
        self.assertTrue(s["question"] and s["frames"])
        n = s["layers"]
        for f in s["frames"]:
            self.assertEqual(len(f["e"]), n)
            self.assertEqual(len(f["lens"]), n)
            self.assertIsNotNone(f["p"])
        self.assertEqual("".join(f["t"] for f in s["frames"]).strip(), s["answer"].strip())
        d = json.loads((web / "delta.json").read_text(encoding="utf-8"))
        self.assertTrue(d["delta"]["layers"])
        w = json.loads((web / "weights-devstral-small-2.json").read_text(encoding="utf-8"))
        self.assertEqual(len(w["layers"]), n)
