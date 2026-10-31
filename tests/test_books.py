import io, json, tempfile, unittest, zipfile
from pathlib import Path
from cortex.server.books import Books, BookError, chunk_text, with_reference, extract_text


class BooksTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.b = Books(Path(self.d.name))
    def tearDown(self): self.d.cleanup()

    def test_add_search_cn_en(self):
        t = "\n\n".join(["苹果公司在加州。" * 20, "The quick brown fox jumps over the lazy dog. " * 12, "向量数据库用于检索嵌入。" * 20])
        m = self.b.add("测试书", "a.txt", t.encode())
        self.assertGreaterEqual(m["chunks"], 1)
        h = self.b.search("向量数据库", 3)
        self.assertTrue(h and "向量" in h[0]["text"])
        self.assertTrue(self.b.search("lazy fox", 1))
        with self.assertRaises(BookError): self.b.add("x", "a.txt", t.encode())
        self.b.set_enabled(m["id"], False); self.assertEqual(self.b.search("fox"), [])
        self.b.set_enabled(m["id"], True); self.b.remove(m["id"]); self.assertEqual(self.b.list(), [])

    def test_epub_docx_html(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z: z.writestr("c1.xhtml", "<html><body><p>" + "鲸鱼是哺乳动物。" * 10 + "</p></body></html>")
        self.assertIn("鲸鱼", extract_text("b.epub", buf.getvalue()))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z: z.writestr("word/document.xml", "<w:p><w:t>hello world docx</w:t></w:p>")
        self.assertIn("hello world", extract_text("b.docx", buf.getvalue()))
        self.assertNotIn("alert", extract_text("a.html", b"<script>alert(1)</script><p>ok</p>"))

    def test_errors_and_id(self):
        with self.assertRaises(BookError): self.b.add("x", "a.bin", b"\x00\x01\x02")
        with self.assertRaises(BookError): self.b.remove("../etc")
        self.assertEqual(self.b.search("x"), [])

    def test_reference_and_chunks(self):
        self.assertTrue(all(len(c) < 1400 for c in chunk_text("很长的句子。" * 800)))
        m = with_reference([{"role": "user", "content": "q"}], [{"book": "B", "n": 2, "text": "T"}])
        self.assertEqual(m[0]["role"], "system"); self.assertIn("《B》", m[0]["content"])


class ChatBooksTest(unittest.TestCase):
    def test_chat_uses_books(self):
        import os, threading, urllib.request
        from cortex.config import load_config
        from cortex.server.app import serve
        seen = []

        class B:
            num_layers = 2
            def stream_chat(self, messages, **kw):
                seen.append(messages)
                yield {"text": "ok", "energy": [0, 0], "tps": 1.0}

        with tempfile.TemporaryDirectory() as d:
            os.environ["CORTEX_DATA_DIR"] = d
            try:
                httpd = serve(load_config(), port=0, block=False, loader=lambda n: B())
            finally:
                os.environ.pop("CORTEX_DATA_DIR")
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            st = httpd.cortex_state
            st.select(st.model_names()[0])
            st.books.add("Ocean", "o.txt", ("鲸鱼是海洋里最大的哺乳动物。" * 10).encode())
            def post(path, obj):
                r = urllib.request.Request(base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(r) as x: return x.read().decode()
            out = post("/v1/chat/completions", {"stream": True, "books": True,
                                                "messages": [{"role": "user", "content": "鲸鱼是什么"}]})
            self.assertIn("《Ocean》", out)
            self.assertIn("参考资料", seen[-1][0]["content"] if "参考资料" in seen[-1][0]["content"] else "参考资料")
            self.assertEqual(seen[-1][0]["role"], "system")
            post("/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "鲸鱼"}]})
            self.assertEqual(seen[-1][0]["role"], "user")
            httpd.shutdown(); httpd.server_close()
