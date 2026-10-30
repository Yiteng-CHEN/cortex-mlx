from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from cortex.config import PROJECT_ROOT, load_config
from cortex.server.app import serve


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = load_config(PROJECT_ROOT / "config" / "cortex.toml", None)
        cls.httpd = serve(cfg, port=0, block=False)
        cls.port = cls.httpd.server_address[1]
        cls.t = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.t.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return json.load(r)

    def post(self, path, obj):
        req = urllib.request.Request(self.base + path, json.dumps(obj).encode(),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req)

    def test_state_lists_chat_models_only(self):
        names = [m["name"] for m in self.get("/api/state")["models"]]
        self.assertIn("demo-chat", names)
        self.assertIn("devstral-small-2", names)
        self.assertNotIn("mock-reference", names)

    def test_chat_requires_loaded_model(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(cm.exception.code, 409)

    def test_select_and_stream_with_telemetry(self):
        info = json.load(self.post("/api/select", {"name": "demo-chat"}))
        self.assertEqual(info["active"], "demo-chat")
        self.assertEqual(info["layers"], 40)
        self.assertTrue(info["simulated"])
        r = self.post("/v1/chat/completions", {"stream": True, "max_tokens": 6,
                                               "messages": [{"role": "user", "content": "hi"}]})
        lines = [l.decode().strip() for l in r if l.strip()]
        self.assertEqual(lines[-1], "data: [DONE]")
        chunks = [json.loads(l[6:]) for l in lines[:-1]]
        with_energy = [c for c in chunks if c.get("cortex", {}).get("energy")]
        self.assertGreaterEqual(len(with_energy), 3)
        self.assertEqual(len(with_energy[0]["cortex"]["energy"]), 40)
        self.assertTrue(with_energy[0]["cortex"]["simulated"])

    def test_nonstream_openai_shape(self):
        self.post("/api/select", {"name": "demo-chat"})
        out = json.load(self.post("/v1/chat/completions", {"max_tokens": 4,
                                  "messages": [{"role": "user", "content": "hi"}]}))
        self.assertEqual(out["choices"][0]["message"]["role"], "assistant")
        self.assertTrue(out["choices"][0]["message"]["content"])

    def test_rejects_foreign_host_header(self):
        req = urllib.request.Request(self.base + "/api/state", headers={"Host": "evil.example"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 403)

    def test_refuses_non_local_bind(self):
        cfg = load_config(PROJECT_ROOT / "config" / "cortex.toml", None)
        with self.assertRaises(ValueError):
            serve(cfg, host="0.0.0.0", port=0, block=False)

    def test_unknown_model_404(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/api/select", {"name": "nope"})
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
