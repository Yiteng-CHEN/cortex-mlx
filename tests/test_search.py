import json
import threading
import unittest
import urllib.request

from cortex.server import websearch
from cortex.server.search import SearchError, Searcher, check_public_url, page_text, parse_ddg

DDG = """<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&amp;rut=x">Apple <b>MLX</b> docs</a></h2>
<a class="result__snippet" href="x">MLX is an array framework.</a></div>
<div class="result"><a class="result__a" href="https://example.com/b">Second</a><a class="result__snippet">Snippet two</a></div>"""
PAGE = "<html><head><title>T  itle</title><style>x{}</style></head><body><nav>menu</nav><p>Hello</p><script>evil()</script><div>World " + "x" * 300 + "</div></body></html>"


class TestParsing(unittest.TestCase):
    def test_ddg(self):
        r = parse_ddg(DDG)
        self.assertEqual(r[0]["url"], "https://example.org/a")
        self.assertEqual(r[0]["title"], "Apple MLX docs")
        self.assertEqual(r[0]["snippet"], "MLX is an array framework.")
        self.assertEqual(len(r), 2)

    def test_page_text(self):
        title, text = page_text(PAGE)
        self.assertEqual(title, "T itle")
        self.assertIn("Hello", text)
        self.assertNotIn("evil", text)
        self.assertNotIn("menu", text)

    def test_blocks_private_hosts(self):
        for u in ("http://127.0.0.1:8765/", "http://localhost/", "http://192.168.1.2/", "file:///etc/passwd",
                  "http://169.254.169.254/latest/meta-data"):
            with self.assertRaises(SearchError):
                check_public_url(u)


class FakeFetch:
    def __init__(self):
        self.calls = []

    def __call__(self, url, data=None, public_only=True, **kw):
        self.calls.append(url)
        if "duckduckgo" in url:
            return DDG, url
        return PAGE, url


class TestSearcher(unittest.TestCase):
    def test_gather_reads_top_pages(self):
        f = FakeFetch()
        items = Searcher(fetch=f).gather("mlx")
        self.assertEqual(len(items), 2)
        self.assertIn("World", items[0]["text"])
        self.assertEqual(f.calls[0], "https://html.duckduckgo.com/html/")

    def test_searxng(self):
        def fetch(url, **kw):
            self.assertIn("format=json", url)
            return json.dumps({"results": [{"title": "t", "url": "https://a.org", "content": "c"}]}), url
        r = Searcher("searxng", "http://localhost:8080", fetch=fetch).search("q")
        self.assertEqual(r[0]["url"], "https://a.org")


def chat_script(rounds):
    """chat(messages) returning chunks per call from a list of lists of strings."""
    calls = []

    def chat(msgs):
        calls.append(msgs)
        parts = rounds[len(calls) - 1]
        return iter([{"text": p, "energy": [0.5], "tps": 1.0} for p in parts])
    chat.calls = calls
    return chat


class FakeSearcher:
    def gather(self, q):
        return [{"title": "Doc", "url": "https://example.org/a", "text": "Answer is 42."}]


class TestRun(unittest.TestCase):
    def test_search_then_answer_with_split_tag(self):
        chat = chat_script([["让我查一下 <se", "arch>mlx 是什", "么</search>"], ["MLX 是框架 [1]。"]])
        ev = list(websearch.run(chat, [{"role": "user", "content": "mlx?"}], FakeSearcher()))
        text = "".join(e["text"] for e in ev if e["type"] == "text")
        self.assertEqual(text, "让我查一下 MLX 是框架 [1]。")
        self.assertNotIn("<search>", text)
        self.assertEqual(ev[-1], {"type": "sources", "items": [{"title": "Doc", "url": "https://example.org/a"}]})
        self.assertTrue(any(e["type"] == "status" and "mlx 是什么" in e["text"] for e in ev))
        second = chat.calls[1]
        self.assertIn("Answer is 42.", second[-1]["content"])
        self.assertEqual(second[0]["role"], "system")

    def test_no_search_passes_text_and_holds_nothing_back(self):
        chat = chat_script([["价格 <", "5 元"]])
        ev = list(websearch.run(chat, [{"role": "user", "content": "x"}], FakeSearcher()))
        self.assertEqual("".join(e["text"] for e in ev if e["type"] == "text"), "价格 <5 元")

    def test_failed_search_is_reported_to_the_model(self):
        class Bad:
            def gather(self, q):
                raise SearchError("没网")
        chat = chat_script([["<search>x</search>"], ["抱歉无法联网"]])
        ev = list(websearch.run(chat, [{"role": "user", "content": "x"}], Bad()))
        self.assertIn("搜索失败：没网", chat.calls[1][-1]["content"])
        self.assertEqual("".join(e["text"] for e in ev if e["type"] == "text"), "抱歉无法联网")

    def test_search_limit(self):
        chat = chat_script([["<search>a</search>"]] * 3 + [["<search>d</search> 最终"]])
        ev = list(websearch.run(chat, [{"role": "user", "content": "x"}], FakeSearcher(), max_searches=3))
        self.assertIn("最终", "".join(e["text"] for e in ev if e["type"] == "text"))


class TestServerWeb(unittest.TestCase):
    def test_http_flow(self):
        import tempfile, pathlib
        from cortex.config import load_config
        from cortex.server.app import serve

        class B:
            num_layers = 1
            simulated = False
            n = 0

            def stream_chat(self, messages, **kw):
                B.n += 1
                txt = "<search>mlx</search>" if B.n == 1 else "答案 [1]"
                yield {"text": txt, "energy": [0.3], "tps": 2.0}

        cfg = load_config()
        httpd = serve(cfg, port=0, block=False, loader=lambda n: B())
        st = httpd.cortex_state
        st.search_fetch = FakeFetch()
        st._search_file = pathlib.Path(tempfile.mkdtemp()) / "s.json"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def post(path, body):
            r = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            return urllib.request.urlopen(r)
        name = st.model_names()[0]
        post("/api/select", {"name": name})
        out = json.load(post("/v1/chat/completions", {"model": name, "web": True, "messages": [{"role": "user", "content": "q"}]}))
        m = out["choices"][0]["message"]
        self.assertEqual(m["content"], "答案 [1]")
        self.assertEqual(m["sources"][0]["url"], "https://example.org/a")
        B.n = 0
        raw = post("/v1/chat/completions", {"model": name, "web": True, "stream": True,
                                           "messages": [{"role": "user", "content": "q"}]}).read().decode()
        self.assertIn('"status": "正在搜索：mlx"', raw)
        self.assertIn('"sources"', raw)
        self.assertEqual(json.load(post("/api/search", {"provider": "searxng", "searxng_url": "http://localhost:8080"}))["provider"], "searxng")
        with self.assertRaises(Exception):
            post("/api/search", {"provider": "bing"})
        httpd.shutdown()


if __name__ == "__main__":
    unittest.main()


class TestMCP(unittest.TestCase):
    def test_protocol(self):
        from cortex.mcp_search import handle
        sf = lambda: Searcher(fetch=FakeFetch())
        r = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}, sf)
        self.assertEqual(r["result"]["serverInfo"]["name"], "cortex-search")
        self.assertIsNone(handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, sf))
        names = [t["name"] for t in handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, sf)["result"]["tools"]]
        self.assertEqual(names, ["web_search", "web_fetch"])
        r = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "web_search", "arguments": {"query": "mlx"}}}, sf)
        self.assertIn("https://example.org/a", r["result"]["content"][0]["text"])
        r = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "web_fetch", "arguments": {"url": "http://127.0.0.1/"}}},
                   lambda: Searcher())
        self.assertTrue(r["result"]["isError"])
