"""Local Cortex server: web UI + OpenAI-compatible chat API + live layer telemetry +
model library (one-click download) + guided LoRA training.

    cortex serve --open          # http://127.0.0.1:8765

Binds to localhost only and rejects requests whose Host header is not localhost (DNS rebinding).
One model is loaded at a time. Training and chat never run together: starting a training run
unloads the model first. The chat stream is standard OpenAI SSE; each chunk carries an extra
"cortex" object with per-layer energy for the UI.
"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..config import PROJECT_ROOT, Config, hf_cache_dir
from .chat import DemoChat
from .library import Library
from .training import Datasets, Trainer, adapter_delta

WEB_DIR = PROJECT_ROOT / "web"
BIG_BODY = {"/api/train/dataset": 60_000_000}


class NotDownloaded(Exception):
    pass


class State:
    def __init__(self, cfg: Config, loader=None, library: Library | None = None,
                 trainer: Trainer | None = None, spawn=None, delta_fn=None):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.backend: Any = None
        self.active: str | None = None
        self.adapter: str | None = None
        self.loading: str | None = None
        self._custom_loader = loader
        self.library = library or Library(cfg, cfg.data_dir, hf_cache_dir(cfg))
        self.datasets = Datasets(cfg.data_dir / "datasets")
        self.trainer = trainer or Trainer(cfg.data_dir, self.datasets, spawn)
        self.delta_fn = delta_fn or adapter_delta
        self.dumps: dict[str, dict] = {}

    # ---- models
    def model_names(self) -> list[str]:
        names = [e["key"] for e in self.library.entries()]
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
            try:
                self.unload()
                self.backend = self._load(name, adapter_dir)
                self.active, self.adapter = name, adapter
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
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "dump_weights.py")]
        cmd += ["--candidate", key] if e["source"] == "catalog" else ["--repo", e["repo"], "--key", key]

        def run():
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
                j["state"] = "done" if p.returncode == 0 else "error"
                j["error"] = None if p.returncode == 0 else (p.stderr or p.stdout)[-300:]
            except Exception as ex:
                j["state"], j["error"] = "error", str(ex)
        threading.Thread(target=run, daemon=True).start()
        return j


def weights_path(name: str) -> Path:
    return PROJECT_ROOT / "data" / f"weights-{name}.json"


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        server_version = "Cortex"

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
            elif p == "/api/state":
                self._guard(lambda: self._json(state.info()))
            elif p == "/api/library":
                self._guard(lambda: self._json({**state.library.status(state.active, PROJECT_ROOT / "data"),
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
            elif p == "/api/train/datasets":
                self._guard(lambda: self._json({"datasets": state.datasets.list()}))
            elif p == "/api/train/status":
                self._guard(lambda: self._json(state.trainer.status()))
            elif p == "/api/train/adapters":
                self._guard(lambda: self._json({"adapters": state.trainer.adapters()}))
            elif p == "/api/train/delta":
                aid = (q.get("adapter") or [""])[0]
                self._guard(lambda: self._json(state.delta_fn(state.trainer.adapter_dir(aid))))
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
                    obj["cortex"] = extra
                return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n"

            with state.lock:
                if not body.get("stream"):
                    text = "".join(c["text"] for c in backend.stream_chat(
                        messages, max_tokens=max_tokens, temperature=temperature))
                    return self._json({"id": cid, "object": "chat.completion", "model": state.active,
                                       "choices": [{"index": 0, "finish_reason": "stop",
                                                    "message": {"role": "assistant", "content": text}}]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(chunk({"role": "assistant"}))
                    gen = backend.stream_chat(messages, max_tokens=max_tokens, temperature=temperature)
                    try:
                        for c in gen:
                            extra = {"energy": c.get("energy"), "tps": c.get("tps"), "simulated": simulated}
                            self.wfile.write(chunk({"content": c["text"]}, extra))
                            self.wfile.flush()
                    finally:
                        gen.close()
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
        raise ValueError("Cortex only binds to localhost")
    state = State(cfg, loader, **state_kw)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.cortex_state = state          # type: ignore[attr-defined]
    if model:
        state.select(model)
    if block:
        url = f"http://{host}:{httpd.server_address[1]}"
        print(f"Cortex at {url}   (Ctrl+C to stop)")
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
