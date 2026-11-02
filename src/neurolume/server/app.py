"""Local Neurolume server: web UI + OpenAI-compatible chat API + live layer telemetry +
model library (one-click download) + guided LoRA training.

    neurolume serve --open          # http://127.0.0.1:8765

Binds to localhost only and rejects requests whose Host header is not localhost (DNS rebinding).
One model is loaded at a time. Training and chat never run together: starting a training run
unloads the model first. The chat stream is standard OpenAI SSE; each chunk carries an extra
"neurolume" object with per-layer energy for the UI.
"""
from __future__ import annotations

import gc
import inspect
import json
import subprocess
import sys
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..config import PROJECT_ROOT, WEB_DIR, Config, hf_cache_dir
from .chat import DemoChat
from .library import Library
from . import websearch
from .books import Books, BookError, with_reference
from . import agent as agentmod
from .agent import AgentError, Proposals, Workspace
from .datagen import DataGen, to_jsonl
from .export import Exporter, dir_size
from .search import SearchError, Searcher
from .hardware import Hardware
from .training import Datasets, Trainer, adapter_delta

BIG_BODY = {"/api/train/dataset": 60_000_000, "/api/books/add": 80_000_000, "/api/datagen/start": 40_000_000,
            "/api/datagen/save": 40_000_000}


STATIC_TYPES = {"css": "text/css; charset=utf-8", "js": "text/javascript; charset=utf-8"}


def _accepts(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class NotDownloaded(Exception):
    pass


class MLXWorker:
    """One dedicated thread for everything that touches MLX.

    MLX streams are thread-local: a model loaded on one HTTP-handler thread cannot be used from another
    ("There is no Stream(gpu, N) in current thread"). Loading, generating and unloading all run here.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._t = threading.Thread(target=self._loop, name="mlx-worker", daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            fn = self._q.get()
            try:
                fn()
            except BaseException:      # each job reports its own errors
                pass

    def call(self, fn, *args, **kw):
        if threading.current_thread() is self._t:
            return fn(*args, **kw)
        box: dict = {}
        done = threading.Event()

        def job():
            try:
                box["v"] = fn(*args, **kw)
            except BaseException as e:
                box["e"] = e
            finally:
                done.set()
        self._q.put(job)
        done.wait()
        if "e" in box:
            raise box["e"]
        return box.get("v")

    def stream(self, gen_factory):
        """Drive a generator on the worker thread; yield its items here. Closing this closes the source."""
        out: queue.Queue = queue.Queue(maxsize=8)
        stop = threading.Event()
        END = object()

        def job():
            gen = None
            try:
                gen = gen_factory()
                for item in gen:
                    if stop.is_set():
                        break
                    out.put(("item", item))
                out.put((END, None))
            except BaseException as e:
                out.put(("error", e))
            finally:
                if gen is not None:
                    try:
                        gen.close()
                    except Exception:
                        pass
        self._q.put(job)
        try:
            while True:
                kind, val = out.get()
                if kind == "item":
                    yield val
                elif kind == "error":
                    raise val
                else:
                    return
        finally:
            stop.set()
            try:                         # unblock a producer waiting on a full queue
                while True:
                    out.get_nowait()
            except queue.Empty:
                pass


class State:
    def __init__(self, cfg: Config, loader=None, library: Library | None = None,
                 trainer: Trainer | None = None, spawn=None, delta_fn=None, demo: bool = False):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.worker = MLXWorker()
        self.demo = demo or loader is not None      # 模拟后端只在测试/显式演示时出现在选择列表里
        self.backend: Any = None
        self.active: str | None = None
        self.adapter: str | None = None
        self.loading: str | None = None
        self._custom_loader = loader
        self.library = library or Library(cfg, cfg.data_dir, hf_cache_dir(cfg))
        self.datasets = Datasets(cfg.data_dir / "datasets")
        self.books = Books(cfg.data_dir / "books")
        self.weights_dir = cfg.data_dir / "weights"
        self.datagen = DataGen()
        self.exporter = Exporter(cfg.data_dir)
        if self.weights_dir not in _WEIGHTS_DIRS:
            _WEIGHTS_DIRS.insert(0, self.weights_dir)
        self.proposals = Proposals(cfg.data_dir / "agent" / "changes")
        self._ws_file = cfg.data_dir / "agent" / "workspace.txt"
        self.workspace: Workspace | None = None
        try:
            self.workspace = Workspace(agentmod.check_workspace(self._ws_file.read_text(encoding="utf-8")))
        except (OSError, AgentError):
            self.workspace = None

        self.trainer = trainer or Trainer(cfg.data_dir, self.datasets, spawn)
        self.delta_fn = delta_fn or adapter_delta
        self.dumps: dict[str, dict] = {}
        self.hardware = Hardware(active_model_bytes=self._mlx_active_bytes)
        self.search_fetch = None                      # tests inject a fake fetcher
        self._search_file = cfg.data_dir / "search.json"

    def _mlx_active_bytes(self):
        if self.backend is None or getattr(self.backend, "simulated", False) or self._custom_loader:
            return None
        from ..backends import mlxutil
        return mlxutil.active_memory_bytes()

    # ---- web search (opt-in per request)
    # ---- generation helpers used by the studio features
    def chat_text(self, messages: list[dict], max_tokens: int = 1200, temperature: float = 0.4) -> str:
        b = self.backend
        if b is None:
            raise RuntimeError("还没有加载模型，请先在上方选择一个")
        with self.lock:
            return "".join(c.get("text") or "" for c in self.worker.stream(
                lambda: b.stream_chat(messages, max_tokens=max_tokens, temperature=temperature)))

    def compare(self, prompt: str, max_tokens: int = 500) -> dict:
        """The same question answered by the base model and by the loaded adapter (switched off/on in place)."""
        b = self.backend
        if b is None or not getattr(b, "has_adapter", False):
            raise RuntimeError("请先在上方选择一个适配器（训练结果），才能和原模型对比")
        if not prompt.strip():
            raise ValueError("问题是空的")
        msgs = [{"role": "user", "content": prompt}]
        with self.lock:                                   # no chat may run while the adapter is switched off
            try:
                b.set_adapter_enabled(False)
                base = self.chat_text(msgs, max_tokens, 0.0)
                b.set_adapter_enabled(True)
                tuned = self.chat_text(msgs, max_tokens, 0.0)
            finally:
                b.set_adapter_enabled(True)
        return {"prompt": prompt, "base": base, "tuned": tuned, "adapter": self.adapter}

    def compare_samples(self, adapter_id: str, k: int = 5) -> list[dict]:
        """A few held-out questions (and the expected answers) from the adapter's validation split."""
        meta = next((a for a in self.trainer.adapters() if a["id"] == adapter_id), None)
        if not meta or not meta.get("dataset_id"):
            return []
        try:
            f = self.datasets.path(meta["dataset_id"]) / "valid.jsonl"
            rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        except (KeyError, ValueError, OSError):
            return []
        out = []
        for r in rows:
            m = r.get("messages") or []
            u = next((x["content"] for x in m if x.get("role") == "user"), None)
            a = next((x["content"] for x in m if x.get("role") == "assistant"), None)
            if u:
                out.append({"prompt": u[:2000], "expected": (a or "")[:2000]})
            if len(out) >= k:
                break
        return out

    def generate_weights(self) -> dict:
        """Real weight map from the model that is already loaded (seconds; no second load)."""
        b = self.backend
        if b is None or getattr(b, "simulated", False) or not hasattr(b, "_model"):
            raise RuntimeError("需要先加载一个真实模型")
        from ..tools import dump_weights
        c = self.candidate(self.active)
        with self.lock:
            out = self.worker.call(dump_weights.dump_from_model, b._model, c.model, self.active, 12,
                                   self.weights_dir, lambda m: None)
        return {"ok": True, "model": self.active, "path": str(out)}

    def export(self, kind: str, adapter_id: str, name: str = "") -> dict:
        if self.trainer.running():
            raise RuntimeError("正在训练，请等训练结束再导出")
        meta = next((a for a in self.trainer.adapters() if a["id"] == adapter_id), None)
        if not meta:
            raise KeyError(adapter_id)
        repo = meta.get("repo") or self.candidate(meta["model"]).model
        cache = hf_cache_dir(self.cfg) / ("models--" + repo.replace("/", "--"))
        self.unload()                                     # fusing loads the model again in a separate process
        return self.exporter.start(kind, adapter_id, self.trainer.adapter_dir(adapter_id), repo,
                                   dir_size(cache) if cache.exists() else 0, name)

    def start_datagen(self, files: list[dict], mode: str, lang: str) -> dict:
        if self.backend is None or getattr(self.backend, "simulated", False):
            raise RuntimeError("生成训练数据需要先加载一个真实模型")
        return self.datagen.start(files, mode, lang, self.chat_text)

    def set_workspace(self, path: str) -> dict:
        ws = Workspace(agentmod.check_workspace(path))
        self.workspace = ws
        self._ws_file.parent.mkdir(parents=True, exist_ok=True)
        self._ws_file.write_text(str(ws.root), encoding="utf-8")
        return self.agent_info()

    def clear_workspace(self) -> dict:
        self.workspace = None
        try:
            self._ws_file.unlink()
        except OSError:
            pass
        return self.agent_info()

    def agent_info(self) -> dict:
        ws = self.workspace
        return {"workspace": str(ws.root) if ws else None, "name": ws.name if ws else None,
                "changes": self.proposals.changes(ws.root if ws else None)}

    def pick_folder(self) -> dict:
        if sys.platform != "darwin":
            raise RuntimeError("自动选择文件夹目前只支持 macOS，请直接粘贴文件夹路径")
        r = subprocess.run(["osascript", "-e", 'POSIX path of (choose folder with prompt "选择项目文件夹 / Choose a project folder")'],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return {"path": None}
        return {"path": r.stdout.strip().rstrip("/") or None}

    def search_config(self) -> dict:
        try:
            j = json.loads(self._search_file.read_text())
        except (OSError, ValueError):
            j = {}
        prov = j.get("provider") if j.get("provider") in ("duckduckgo", "searxng") else "duckduckgo"
        return {"provider": prov, "searxng_url": str(j.get("searxng_url", ""))}

    def set_search_config(self, provider: str, searxng_url: str) -> dict:
        if provider not in ("duckduckgo", "searxng"):
            raise ValueError("未知的搜索来源")
        url = searxng_url.strip()
        if url and not re.match(r"^https?://[^\s]+$", url):
            raise ValueError("SearXNG 地址要以 http:// 或 https:// 开头")
        self._search_file.parent.mkdir(parents=True, exist_ok=True)
        self._search_file.write_text(json.dumps({"provider": provider, "searxng_url": url}))
        return self.search_config()

    def searcher(self) -> Searcher:
        c = self.search_config()
        kw = {"fetch": self.search_fetch} if self.search_fetch else {}
        return Searcher(c["provider"], c["searxng_url"], **kw)

    # ---- models
    def model_names(self) -> list[str]:
        names = [e["key"] for e in self.library.entries()]
        if not self.demo:
            return names
        return names + [n for n, c in self.cfg.candidates.items() if c.backend == "mock" and c.mode == "chat"]

    def candidate(self, name: str):
        c = self.library.candidate(name) or self.cfg.candidates.get(name)
        if c is None:
            raise KeyError(f"unknown chat model {name!r}")
        return c

    def _load(self, name: str, adapter_dir: str | None):
        if self._custom_loader:
            return self._custom_loader(name)
        c = self.candidate(name)
        if c.backend == "mock":
            return DemoChat(name)
        if not self.library.is_downloaded(c.model):
            raise NotDownloaded(name)
        from ..backends.mlx_backend import MLXBackend
        return MLXBackend(c.model, name=name, chat_template_kwargs=c.chat_template_kwargs,
                          tokenizer_config=c.tokenizer_config, probe=True, adapter_path=adapter_dir)

    def unload(self) -> None:
        with self.lock:
            self.backend, self.active, self.adapter = None, None, None
            self.worker.call(self._clear_cache)

    @staticmethod
    def _clear_cache() -> None:
        gc.collect()
        try:
            import mlx.core as mx
            (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
        except Exception:
            pass

    def select(self, name: str, adapter: str | None = None) -> None:
        if name not in self.model_names():
            raise KeyError(f"unknown chat model {name!r}")
        if self.trainer.running():
            raise RuntimeError("正在训练，训练期间无法加载模型")
        adapter_dir = str(self.trainer.adapter_dir(adapter)) if adapter else None
        with self.lock:
            if self.active == name and self.adapter == adapter and self.backend is not None:
                return
            self.loading = name
            print(f"[neurolume] 正在加载 {name} …", file=sys.stderr, flush=True)
            try:
                self.unload()
                self.backend = self.worker.call(self._load, name, adapter_dir)
                self.active, self.adapter = name, adapter
                print(f"[neurolume] 已加载 {name}，层数 {getattr(self.backend, 'num_layers', None)}",
                      file=sys.stderr, flush=True)
            finally:
                self.loading = None

    def info(self) -> dict:
        models = []
        for n in self.model_names():
            c = self.candidate(n)
            models.append({"name": n, "model": c.model or "demo", "simulated": c.backend == "mock",
                           "loaded": n == self.active,
                           "downloaded": True if c.backend == "mock" else self.library.is_downloaded(c.model)})
        b = self.backend
        return {"models": models, "active": self.active, "adapter": self.adapter, "loading": self.loading,
                "training": self.trainer.running(),
                "layers": getattr(b, "num_layers", None) if b else None,
                "simulated": bool(getattr(b, "simulated", False)) if b else None}

    # ---- weights map
    def dump_weights(self, key: str) -> dict:
        e = self.library.find(key)
        if not e:
            raise KeyError(key)
        if not self.library.is_downloaded(e["repo"]):
            raise NotDownloaded(key)
        with self.lock:
            j = self.dumps.get(key)
            if j and j["state"] == "running":
                return j
            j = {"state": "running", "error": None, "log": ""}
            self.dumps[key] = j
        cmd = [sys.executable, "-m", "neurolume.tools.dump_weights", "--out", str(self.weights_dir)]
        cmd += ["--candidate", key] if e["source"] == "catalog" else ["--repo", e["repo"], "--key", key]

        def run():
            try:
                p = subprocess.run(cmd, capture_output=True, text=True)
                j["state"] = "done" if p.returncode == 0 else "error"
                j["error"] = None if p.returncode == 0 else (p.stderr or p.stdout)[-300:]
            except Exception as ex:
                j["state"], j["error"] = "error", str(ex)
        threading.Thread(target=run, daemon=True).start()
        return j


_WEIGHTS_DIRS: list[Path] = []


def weights_path(name: str) -> Path:
    """Weight-map dumps live in <data_dir>/weights; a checkout's old data/ folder is still read."""
    for d in _WEIGHTS_DIRS + [PROJECT_ROOT / "data"]:
        f = d / f"weights-{name}.json"
        if f.exists():
            return f
    return (_WEIGHTS_DIRS[0] if _WEIGHTS_DIRS else PROJECT_ROOT / "data") / f"weights-{name}.json"


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        server_version = "Neurolume"

        def log_message(self, fmt, *args):
            pass

        def _ok_host(self) -> bool:
            port = self.server.server_address[1]
            if self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return True
            self._json({"error": "bad host"}, 403)
            return False

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self, path: str) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > BIG_BODY.get(path, 2_000_000):
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def _search_test(self, q: str) -> dict:
            try:
                res = state.searcher().search(q or "Apple Silicon MLX", 3)
            except SearchError as e:
                return {"ok": False, "error": str(e)}
            return {"ok": True, "results": res}

        def _guard(self, fn):
            try:
                return fn()
            except NotDownloaded:
                self._json({"error": "这个模型还没下载，请先在“模型库”里下载", "code": "not_downloaded"}, 409)
            except KeyError as e:
                self._json({"error": f"没有找到：{e.args[0] if e.args else ''}"}, 404)
            except (ValueError, RuntimeError) as e:
                self._json({"error": str(e)}, 400 if isinstance(e, ValueError) else 409)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        # ---------------- GET
        def do_GET(self):
            if not self._ok_host():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            p = u.path
            if p in ("/", "/index.html"):
                f = WEB_DIR / "index.html"
                if not f.exists():
                    return self._json({"error": "web/index.html missing"}, 404)
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif p.startswith(("/css/", "/js/")) and p.rsplit(".", 1)[-1] in STATIC_TYPES:
                f = (WEB_DIR / p.lstrip("/")).resolve()
                if not f.is_file() or WEB_DIR.resolve() not in f.parents:
                    return self._json({"error": "not found"}, 404)
                data = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", STATIC_TYPES[p.rsplit(".", 1)[-1]])
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            elif p == "/api/state":
                self._guard(lambda: self._json(state.info()))
            elif p == "/api/library":
                self._guard(lambda: self._json({**state.library.status(state.active, [state.weights_dir, PROJECT_ROOT / "data"]),
                                                "dumps": state.dumps}))
            elif p == "/api/weights":
                name = (q.get("model") or [state.active or ""])[0]
                f = weights_path(name) if name.replace("-", "").replace(".", "").isalnum() else None
                if f and f.exists():
                    data = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._json({"error": "no weight sample"}, 404)
            elif p == "/api/search":
                self._guard(lambda: self._json(state.search_config()))
            elif p == "/api/compare/samples":
                self._guard(lambda: self._json({"samples": state.compare_samples((q.get("adapter") or [""])[0])}))
            elif p == "/api/export":
                self._guard(lambda: self._json({"job": state.exporter.status(), "exports": state.exporter.list()}))
            elif p == "/api/datagen":
                self._guard(lambda: self._json(state.datagen.status()))
            elif p == "/api/agent":
                self._guard(lambda: self._json(state.agent_info()))
            elif p == "/api/books":
                self._guard(lambda: self._json({"books": state.books.list()}))
            elif p == "/api/hw":
                self._guard(lambda: self._json(state.hardware.sample()))
            elif p == "/api/train/datasets":
                self._guard(lambda: self._json({"datasets": state.datasets.list()}))
            elif p == "/api/train/status":
                self._guard(lambda: self._json(state.trainer.status()))
            elif p == "/api/train/adapters":
                self._guard(lambda: self._json({"adapters": state.trainer.adapters()}))
            elif p == "/api/train/delta":
                aid = (q.get("adapter") or [""])[0]
                self._guard(lambda: self._json(state.worker.call(state.delta_fn, state.trainer.adapter_dir(aid))))
            elif p == "/v1/models":
                self._json({"object": "list", "data": [{"id": n, "object": "model"} for n in state.model_names()]})
            else:
                self._json({"error": "not found"}, 404)

        # ---------------- POST
        def do_POST(self):
            if not self._ok_host():
                return
            path = urlparse(self.path).path
            try:
                body = self._body(path)
            except Exception as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            lib, tr = state.library, state.trainer
            routes = {
                "/api/select": lambda: (state.select(str(body.get("name", "")), body.get("adapter") or None),
                                        self._json(state.info())),
                "/api/unload": lambda: (state.unload(), self._json(state.info())),
                "/api/library/add": lambda: self._json(lib.add(str(body.get("repo", "")), body.get("name"))),
                "/api/library/remove": lambda: (lib.remove_custom(str(body.get("key", ""))), self._json({"ok": True})),
                "/api/library/download": lambda: self._json(self._download(str(body.get("key", "")))),
                "/api/library/cancel": lambda: (lib.cancel(str(body.get("key", ""))), self._json({"ok": True})),
                "/api/library/delete": lambda: self._json({"freed": self._delete(str(body.get("key", "")))}),
                "/api/search": lambda: self._json(state.set_search_config(
                    str(body.get("provider", "duckduckgo")), str(body.get("searxng_url", "")))),
                "/api/agent/workspace": lambda: self._json(state.set_workspace(str(body.get("path", ""))) if body.get("path")
                                                           else state.clear_workspace()),
                "/api/agent/pick": lambda: self._json(state.pick_folder()),
                "/api/agent/decide": lambda: self._json(state.proposals.decide(str(body.get("id", "")), bool(body.get("approve")))),
                "/api/agent/undo": lambda: self._json(state.proposals.undo(str(body.get("id", "")))),
                "/api/weights/generate": lambda: self._json(state.generate_weights()),
                "/api/compare": lambda: self._json(state.compare(str(body.get("prompt", "")))),
                "/api/export": lambda: self._json(state.export(str(body.get("kind", "")), str(body.get("adapter", "")),
                                                               str(body.get("name", "")))),
                "/api/datagen/start": lambda: self._json(state.start_datagen(body.get("files") or [], str(body.get("mode", "qa")),
                                                                             str(body.get("lang", "zh")))),
                "/api/datagen/cancel": lambda: (state.datagen.cancel(), self._json(state.datagen.status())),
                "/api/datagen/save": lambda: self._json(state.datasets.create(
                    str(body.get("name", "")) or "generated", [{"name": "generated.jsonl", "text": to_jsonl(body.get("items") or [])}])),
                "/api/books/add": lambda: self._json(state.books.add_b64(
                    str(body.get("name", "")), str(body.get("filename", "")), str(body.get("data", "")))),
                "/api/books/remove": lambda: (state.books.remove(str(body.get("id", ""))), self._json({"ok": True})),
                "/api/books/enable": lambda: (state.books.set_enabled(str(body.get("id", "")), bool(body.get("on"))),
                                              self._json({"ok": True})),
                "/api/books/search": lambda: self._json({"hits": state.books.search(str(body.get("q", "")), 5)}),
                "/api/search/test": lambda: self._json(self._search_test(str(body.get("q", "")))),
                "/api/library/weights": lambda: self._json(state.dump_weights(str(body.get("key", "")))),
                "/api/train/dataset": lambda: self._json(state.datasets.create(str(body.get("name", "")), body.get("files") or [])),
                "/api/train/start": lambda: self._json(self._train(body)),
                "/api/train/cancel": lambda: (tr.cancel(), self._json(tr.status())),
            }
            if path == "/v1/chat/completions":
                return self._chat(body)
            fn = routes.get(path)
            if not fn:
                return self._json({"error": "not found"}, 404)
            self._guard(fn)

        def _download(self, key: str) -> dict:
            if state.trainer.running():
                raise RuntimeError("正在训练，请等训练结束再下载")
            return state.library.download(key)

        def _delete(self, key: str) -> int:
            if key == state.active:
                raise ValueError("这个模型正在使用，请先卸载")
            return state.library.delete(key)

        def _train(self, body: dict) -> dict:
            key = str(body.get("model", ""))
            c = state.candidate(key)
            if c.backend != "mlx":
                raise ValueError("请选择真实模型来训练")
            if not state.library.is_downloaded(c.model):
                raise NotDownloaded(key)
            state.unload()                                   # training needs the memory
            return state.trainer.start(key, c.model, str(body.get("dataset", "")),
                                       str(body.get("preset", "quick")), c.tokenizer_config)

        # ---------------- chat
        def _chat(self, body: dict) -> None:
            def prepare():
                model = body.get("model")
                if model and (model != state.active or body.get("adapter", state.adapter) != state.adapter):
                    state.select(model, body.get("adapter", state.adapter))
                if state.trainer.running():
                    raise RuntimeError("正在训练，训练期间无法聊天")
                if state.backend is None:
                    raise RuntimeError("还没有加载模型，请先在上方选择一个")
                if not (body.get("messages") or []):
                    raise ValueError("messages required")
            ok = []
            self._guard(lambda: (prepare(), ok.append(1)))
            if not ok:
                return
            backend = state.backend
            messages = body["messages"]
            max_tokens = max(1, min(int(body.get("max_tokens") or 1024), 4096))
            temperature = float(body.get("temperature", 0.2))
            simulated = bool(getattr(backend, "simulated", False))
            cid = f"chatcmpl-{int(time.time() * 1000)}"

            def chunk(delta: dict, extra: dict | None = None, finish=None) -> bytes:
                obj = {"id": cid, "object": "chat.completion.chunk", "model": state.active,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                if extra:
                    obj["neurolume"] = extra
                return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"

            use_web = bool(body.get("web")) and not simulated

            want_lens = bool(body.get("lens")) and not simulated and _accepts(backend.stream_chat, "lens")

            def chat(msgs):
                kw = {"lens": True} if want_lens else {}
                return state.worker.stream(lambda: backend.stream_chat(
                    msgs, max_tokens=max_tokens, temperature=temperature, **kw))

            lang = "en" if body.get("lang") == "en" else "zh"
            use_books = bool(body.get("books")) and not simulated and state.books.has_any() and not body.get("agent")
            msgs_in = messages
            book_src: list[dict] = []
            if use_books:
                q = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), "")
                hits = state.books.search(q if isinstance(q, str) else "", 5)
                if hits:
                    msgs_in = with_reference(messages, hits, lang)
                    book_src = [{"title": (f"《{h['book']}》第 {h['n']} 段" if lang != "en" else f"\"{h['book']}\", passage {h['n']}"), "url": ""} for h in hits]

            use_agent = bool(body.get("agent")) and bool(body.get("stream")) and not simulated and state.workspace is not None

            def events():
                if use_agent:
                    yield from agentmod.run(chat, messages, state.workspace, state.proposals, lang=lang,
                                            books=state.books if body.get("books") else None)
                    return
                if book_src:
                    yield {"type": "sources", "items": book_src}
                if use_web:
                    yield from websearch.run(chat, msgs_in, state.searcher(), lang=lang)
                else:
                    for c in chat(msgs_in):
                        yield {"type": "text", "text": c["text"], "energy": c.get("energy"), "tps": c.get("tps"), "lens": c.get("lens"), "token": c.get("token")}

            with state.lock:
                if not body.get("stream"):
                    ev = list(events())
                    text = "".join(e["text"] for e in ev if e["type"] == "text")
                    msg = {"role": "assistant", "content": text}
                    src = next((e["items"] for e in ev if e["type"] == "sources"), None)
                    if src:
                        msg["sources"] = src
                    return self._json({"id": cid, "object": "chat.completion", "model": state.active,
                                       "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(chunk({"role": "assistant"}))
                    seen_energy = 0
                    gen = events()
                    try:
                        for e in gen:
                            if e["type"] == "text":
                                extra = {"energy": e.get("energy"), "tps": e.get("tps"), "simulated": simulated}
                                if e.get("lens"):
                                    extra["lens"] = e["lens"]
                                if e.get("token") is not None:
                                    extra["token"] = e["token"]
                                seen_energy += 1 if e.get("energy") else 0
                                self.wfile.write(chunk({"content": e["text"]} if e["text"] else {}, extra))
                            elif e["type"] == "status":
                                self.wfile.write(chunk({}, {"status": e["text"], "simulated": simulated}))
                            elif e["type"] == "ping":
                                self.wfile.write(b": ping\n\n")
                            elif e["type"] == "agent":
                                self.wfile.write(chunk({}, {"agent": {k: v for k, v in e.items() if k != "type"}, "simulated": simulated}))
                            else:
                                self.wfile.write(chunk({}, {"sources": e["items"], "simulated": simulated}))
                            self.wfile.flush()
                    finally:
                        gen.close()
                        if not simulated and seen_energy == 0:
                            print(f"[neurolume] 警告：模型 {state.active} 本次回复没有产生逐层信号（探针层数="
                                  f"{getattr(backend, 'num_layers', None)}）", file=sys.stderr, flush=True)
                    self.wfile.write(chunk({}, finish="stop"))
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as e:
                    try:
                        self.wfile.write(b"data: " + json.dumps({"error": str(e)}).encode() + b"\n\n")
                    except Exception:
                        pass

    return H


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8765, model: str | None = None,
          loader=None, block: bool = True, open_browser: bool = False, **state_kw) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Neurolume only binds to localhost")
    state = State(cfg, loader, **state_kw)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.neurolume_state = state          # type: ignore[attr-defined]
    if model:
        state.select(model)
    if block:
        url = f"http://{host}:{httpd.server_address[1]}"
        print(f"Neurolume at {url}   (Ctrl+C to stop)")
        if open_browser:
            import webbrowser
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
    return httpd
