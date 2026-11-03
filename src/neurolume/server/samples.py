"""Training samples collected while chatting.

Under any answer in the chat the user can press “save as training sample” (optionally after editing the
question or the answer into what they actually wanted). Samples are kept in one JSONL file in the data
directory; the training wizard turns them into a dataset with one click.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

MAX_CHARS = 20_000          # one sample (question + answer); longer ones would be cut by training anyway


def _clean(messages) -> list[dict]:
    if not isinstance(messages, list):
        raise ValueError("样本格式不对")
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            raise ValueError("样本格式不对")
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            raise ValueError("问题和回答都不能是空的")
        out.append({"role": m["role"], "content": c.strip()})
    roles = [m["role"] for m in out if m["role"] != "system"]
    if len(roles) < 2 or roles[0] != "user" or roles[-1] != "assistant":
        raise ValueError("样本需要一个问题和一个回答")
    if sum(len(m["content"]) for m in out) > MAX_CHARS:
        raise ValueError("这条样本太长了（超过 2 万字），请删掉一部分再存")
    return out


class Samples:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(ln))
            except ValueError:
                pass
        return rows

    def _write(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, messages, edited: bool = False, model: str | None = None) -> dict:
        msgs = _clean(messages)
        row = {"id": uuid.uuid4().hex[:10], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "edited": bool(edited), "model": model or "", "messages": msgs}
        with self.lock:
            rows = self._read()
            key = json.dumps(msgs, ensure_ascii=False)
            if any(json.dumps(r.get("messages"), ensure_ascii=False) == key for r in rows):
                return {"id": None, "n": len(rows), "duplicate": True}
            rows.append(row)
            self._write(rows)
            return {"id": row["id"], "n": len(rows), "duplicate": False}

    def remove(self, sid: str) -> dict:
        with self.lock:
            rows = [r for r in self._read() if r.get("id") != sid]
            self._write(rows)
            return {"n": len(rows)}

    def list(self, limit: int = 500) -> dict:
        rows = self._read()
        items = []
        for r in reversed(rows[-limit:]):
            msgs = r.get("messages") or []
            q = next((m["content"] for m in msgs if m["role"] == "user"), "")
            a = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
            items.append({"id": r.get("id"), "created": r.get("created"), "edited": r.get("edited"),
                          "q": q[:160], "a": a[:220]})
        return {"n": len(rows), "items": items}

    def as_jsonl(self) -> str:
        return "".join(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n" for r in self._read()
                       if r.get("messages"))
