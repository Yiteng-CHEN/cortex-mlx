"""Turn chat histories exported from other AI tools into user → assistant training samples.

Recognised automatically by the training wizard's import:
- ChatGPT data export: conversations.json (a list of conversations with a "mapping" tree)
- Claude.ai data export: conversations.json (a list of conversations with "chat_messages")
- Claude Code session logs: ~/.claude/projects/<project>/<session>.jsonl (one event per line)

Every user message followed by an assistant reply becomes one sample. Tool calls, tool results,
images and empty turns are skipped. Nothing is sent anywhere; this only reshapes the user's own files.
"""
from __future__ import annotations

MAX_TURN_CHARS = 12_000


def _pair_rows(turns: list[tuple[str, str]], last_only: bool = False) -> list[dict]:
    """[(role, text)...] in order -> one {"messages": [user, assistant]} per user turn that got a reply.
    last_only: keep only the final assistant text of a turn (agent logs narrate between tool calls)."""
    rows, q, a = [], None, []

    def flush():
        if q and a:
            ans = (a[-1] if last_only else "\n\n".join(a)).strip()
            if ans and len(q) + len(ans) <= MAX_TURN_CHARS:
                rows.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]})

    for role, text in turns:
        text = (text or "").strip()
        if not text:
            continue
        if role == "user":
            flush()
            q, a = text, []
        elif role == "assistant" and q:
            a.append(text)
    flush()
    return rows


def _chatgpt(conv: dict) -> list[dict]:
    mapping = conv.get("mapping") or {}
    node = conv.get("current_node")
    chain = []
    seen = set()
    while node and node in mapping and node not in seen:          # follow the branch that was kept
        seen.add(node)
        chain.append(mapping[node])
        node = mapping[node].get("parent")
    if not chain:                                                   # older exports: no current_node
        chain = list(mapping.values())[::-1]
    turns = []
    for n in reversed(chain):
        m = n.get("message") or {}
        role = (m.get("author") or {}).get("role")
        c = m.get("content") or {}
        if role not in ("user", "assistant") or c.get("content_type") not in ("text", None):
            continue
        if (m.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
            continue
        parts = [p for p in (c.get("parts") or []) if isinstance(p, str)]
        turns.append((role, "\n".join(parts)))
    return _pair_rows(turns)


def _claude_ai(conv: dict) -> list[dict]:
    turns = []
    for m in conv.get("chat_messages") or []:
        role = {"human": "user", "assistant": "assistant"}.get(m.get("sender"))
        if not role:
            continue
        text = m.get("text")
        if not text and isinstance(m.get("content"), list):
            text = "\n".join(b.get("text", "") for b in m["content"] if isinstance(b, dict) and b.get("type") == "text")
        turns.append((role, text or ""))
    return _pair_rows(turns)


def from_export(obj) -> tuple[str, list[dict]] | None:
    """A parsed .json file -> (source name, rows), or None if it is not a known chat export."""
    if not isinstance(obj, list) or not obj or not all(isinstance(c, dict) for c in obj[:5]):
        return None
    head = obj[0]
    if "mapping" in head:
        return "ChatGPT", [r for c in obj for r in _chatgpt(c)]
    if "chat_messages" in head:
        return "Claude.ai", [r for c in obj for r in _claude_ai(c)]
    return None


def _cc_text(content, role: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    if role == "user" and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return ""                                                   # a tool result, not something the user typed
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def is_claude_code(objs: list) -> bool:
    return sum(1 for o in objs[:50] if isinstance(o, dict) and o.get("type") in ("user", "assistant")
               and isinstance(o.get("message"), dict)) >= 2


def from_claude_code(objs: list) -> list[dict]:
    turns = []
    for o in objs:
        if not isinstance(o, dict) or o.get("type") not in ("user", "assistant") or o.get("isSidechain"):
            continue
        if o.get("isMeta") or o.get("isCompactSummary"):
            continue
        m = o.get("message") or {}
        role = m.get("role") or o["type"]
        text = _cc_text(m.get("content"), role)
        if role == "user" and (text.startswith("<command-") or text.startswith("<local-command")
                               or text.startswith("Caveat:") or "<system-reminder>" in text[:40]):
            continue                                                # slash commands and harness notes
        turns.append((role, text))
    return _pair_rows(turns, last_only=True)
