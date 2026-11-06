import json, os, tempfile, threading, time, unittest, urllib.request
from pathlib import Path
from neurolume.server import agent as A
from neurolume.server.agent import AgentError, Proposals, Workspace


def mkws():
    d = tempfile.TemporaryDirectory()
    r = Path(d.name).resolve() / "proj"
    (r / "src").mkdir(parents=True)
    (r / "src" / "a.py").write_text("def add(a, b):\n    return a - b\n")
    (r / ".env").write_text("SECRET=sk-" + "a" * 30)
    (r / ".git").mkdir()
    (r / ".git" / "config").write_text("x")
    (r / "notes.txt").write_text("hello\nkey=sk-" + "b" * 30 + "\n")
    return d, Workspace(r)


class WorkspaceTest(unittest.TestCase):
    def test_path_safety(self):
        d, ws = mkws()
        with d:
            for bad in ["../x", "/etc/passwd", "src/../../x", ".env", ".git/config", "src/../.env"]:
                with self.assertRaises(AgentError, msg=bad): ws.resolve(bad)
            os.symlink("/etc", ws.root / "lnk")
            with self.assertRaises(AgentError): ws.resolve("lnk/passwd")
            self.assertEqual(ws.resolve("src/a.py").name, "a.py")
            self.assertEqual(ws.resolve(str(ws.root / "src" / "a.py")).name, "a.py")

    def test_check_workspace(self):
        for bad in ["", "/", str(Path.home()), "/nonexistent-dir-xyz", "/usr"]:
            with self.assertRaises(AgentError, msg=bad): A.check_workspace(bad)
        with tempfile.TemporaryDirectory() as t: self.assertEqual(A.check_workspace(t), Path(t).resolve())

    def test_read_search_list(self):
        d, ws = mkws()
        with d:
            out = ws.read_file("src/a.py"); self.assertIn("    2|     return a - b", out)
            self.assertIn("已隐藏密钥", ws.read_file("notes.txt")); self.assertNotIn("bbbb", ws.read_file("notes.txt"))
            self.assertIn("src/a.py:2", ws.search("a - b")); self.assertIn("没有匹配", ws.search("zzzz"))
            self.assertNotIn(".env", ws.list_dir(".", 2)); self.assertNotIn(".git", ws.list_dir(".", 2))
            self.assertNotIn("SECRET", ws.search("SECRET"))

    def test_propose_edit_rules(self):
        d, ws = mkws()
        with d:
            with self.assertRaises(AgentError): ws.propose_edit("src/a.py", "nope", "x")
            with self.assertRaises(AgentError): ws.propose_edit("src/a.py", "a", "x")        # not unique
            with self.assertRaises(AgentError): ws.propose_edit("src/a.py", "", "x")
            p, b, a = ws.propose_edit("src/a.py", "a - b", "a + b"); self.assertIn("a + b", a)
            with self.assertRaises(AgentError): ws.propose_write(".env", "x")


class ProposalsTest(unittest.TestCase):
    def test_apply_undo(self):
        d, ws = mkws()
        with d, tempfile.TemporaryDirectory() as ld:
            pr = Proposals(Path(ld))
            p, b, a = ws.propose_edit("src/a.py", "a - b", "a + b")
            it = pr.create(ws.root, "src/a.py", b, a)
            self.assertIn("-    return a - b", it["diff"])
            self.assertEqual((ws.root / "src/a.py").read_text(), b)          # nothing written yet
            self.assertEqual(pr.decide(it["id"], True)["status"], "applied")
            self.assertIn("a + b", (ws.root / "src/a.py").read_text())
            with self.assertRaises(AgentError): pr.decide(it["id"], True)
            self.assertEqual(len(pr.changes(ws.root)), 1)
            pr.undo(it["id"]); self.assertEqual((ws.root / "src/a.py").read_text(), b)
            with self.assertRaises(AgentError): pr.undo(it["id"])
            # new file + undo deletes it
            p, b2, a2 = ws.propose_write("new/x.txt", "hi\n"); it2 = pr.create(ws.root, "new/x.txt", b2, a2)
            pr.decide(it2["id"], True); self.assertTrue((ws.root / "new/x.txt").exists())
            pr.undo(it2["id"]); self.assertFalse((ws.root / "new/x.txt").exists())

    def test_reject_and_conflict(self):
        d, ws = mkws()
        with d, tempfile.TemporaryDirectory() as ld:
            pr = Proposals(Path(ld))
            p, b, a = ws.propose_edit("src/a.py", "a - b", "a + b")
            it = pr.create(ws.root, "src/a.py", b, a)
            self.assertEqual(pr.decide(it["id"], False)["status"], "rejected")
            self.assertEqual((ws.root / "src/a.py").read_text(), b)
            it = pr.create(ws.root, "src/a.py", b, a)
            (ws.root / "src/a.py").write_text("changed by user\n")             # file changed after the proposal
            self.assertEqual(pr.decide(it["id"], True)["status"], "failed")
            self.assertEqual((ws.root / "src/a.py").read_text(), "changed by user\n")


def scripted(replies):
    it = iter(replies); seen = []
    def chat(msgs):
        seen.append(msgs)
        r = next(it)
        for i in range(0, len(r), 7): yield {"text": r[i:i + 7], "energy": [0.1], "tps": 1.0}
    return chat, seen


class LoopTest(unittest.TestCase):
    def test_read_then_edit_then_answer(self):
        d, ws = mkws()
        with d, tempfile.TemporaryDirectory() as ld:
            pr = Proposals(Path(ld))
            chat, seen = scripted([
                '我先看看。<tool>{"name":"read_file","args":{"path":"src/a.py"}}</tool>',
                '<tool>{"name":"edit_file","args":{"path":"src/a.py","old":"a - b","new":"a + b"}}</tool>',
                "改好了：add 现在返回 a + b。"])
            out, kinds = "", []
            for e in A.run(chat, [{"role": "user", "content": "修复 add"}], ws, pr):
                if e["type"] == "text": out += e["text"]
                elif e["type"] == "agent":
                    kinds.append(e["kind"])
                    if e["kind"] == "proposal":
                        threading.Timer(0.2, lambda i=e["proposal"]["id"]: pr.decide(i, True)).start()
                elif e["type"] == "ping": pass
            self.assertIn("改好了", out); self.assertNotIn("<tool>", out)
            self.assertEqual(kinds, ["tool", "proposal", "result"])
            self.assertIn("a + b", (ws.root / "src/a.py").read_text())
            self.assertIn("tool_result", seen[1][-1]["content"]); self.assertIn("已批准", seen[2][-1]["content"])
            self.assertEqual([m["role"] for m in seen[2]], ["system", "user", "assistant", "user", "assistant", "user"])

    def test_errors_repeat_and_unknown(self):
        d, ws = mkws()
        with d, tempfile.TemporaryDirectory() as ld:
            pr = Proposals(Path(ld))
            call = '<tool>{"name":"read_file","args":{"path":"../etc/passwd"}}</tool>'
            chat, seen = scripted([call, '<tool>{bad json}</tool>', '<tool>{"name":"rm","args":{}}</tool>', "好的"])
            ev = list(A.run(chat, [{"role": "user", "content": "x"}], ws, pr))
            self.assertTrue(any("不在项目文件夹内" in m[-1]["content"] for m in seen[1:2]))
            self.assertTrue(any("JSON" in m[-1]["content"] for m in seen[2:3]))
            self.assertTrue(any("没有这个工具" in m[-1]["content"] for m in seen[3:4]))
            chat, _ = scripted([call] * 5)
            ev = list(A.run(chat, [{"role": "user", "content": "x"}], ws, pr))
            self.assertIn("重复", "".join(e.get("text", "") for e in ev))

    def test_disconnect_cancels_pending(self):
        d, ws = mkws()
        with d, tempfile.TemporaryDirectory() as ld:
            pr = Proposals(Path(ld))
            chat, _ = scripted(['<tool>{"name":"write_file","args":{"path":"n.txt","content":"x"}}</tool>'])
            g = A.run(chat, [{"role": "user", "content": "x"}], ws, pr)
            for e in g:
                if e["type"] == "agent" and e["kind"] == "proposal": pid = e["proposal"]["id"]; break
            g.close()
            self.assertEqual(pr.get(pid)["status"], "cancelled")
            with self.assertRaises(AgentError): pr.decide(pid, True)


class HttpAgentTest(unittest.TestCase):
    def test_stream_with_approval(self):
        from neurolume.config import load_config
        from neurolume.server.app import serve
        d, ws = mkws()
        replies = iter(['<tool>{"name":"edit_file","args":{"path":"src/a.py","old":"a - b","new":"a + b"}}</tool>', "完成"])

        class B:
            num_layers = 2
            def stream_chat(self, messages, **kw):
                yield {"text": next(replies), "energy": [0, 0], "tps": 1.0}
        with d, tempfile.TemporaryDirectory() as data:
            os.environ["NEUROLUME_DATA_DIR"] = data
            try: httpd = serve(load_config(), port=0, block=False, loader=lambda n: B())
            finally: os.environ.pop("NEUROLUME_DATA_DIR")
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            def post(path, obj):
                r = urllib.request.Request(base + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(r) as x: return x.status, x.read().decode()
                except urllib.error.HTTPError as e: return e.code, e.read().decode()
            st = httpd.neurolume_state; st.select(st.model_names()[0])
            self.assertEqual(post("/api/agent/workspace", {"path": "/"})[0], 400)
            self.assertEqual(post("/api/agent/workspace", {"path": str(ws.root)})[0], 200)
            def approver():
                for _ in range(100):
                    time.sleep(.1)
                    pend = [i for i in st.proposals._items.values() if i["status"] == "pending"]
                    if pend: post("/api/agent/decide", {"id": pend[0]["id"], "approve": True}); return
            threading.Thread(target=approver, daemon=True).start()
            code, out = post("/v1/chat/completions", {"stream": True, "agent": True, "messages": [{"role": "user", "content": "fix"}]})
            self.assertIn('"kind": "proposal"', out); self.assertIn('"applied"', out.replace("已应用", "applied")); self.assertIn("完成", out)
            self.assertIn("a + b", (ws.root / "src/a.py").read_text())
            info = json.load(urllib.request.urlopen(base + "/api/agent"))
            self.assertEqual(len(info["changes"]), 1)
            self.assertEqual(post("/api/agent/undo", {"id": info["changes"][0]["id"]})[0], 200)
            self.assertIn("a - b", (ws.root / "src/a.py").read_text())
            httpd.shutdown(); httpd.server_close()
