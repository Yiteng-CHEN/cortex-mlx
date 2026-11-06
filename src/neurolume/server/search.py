"""Web search + page reading for local models (no API keys needed).

Providers: DuckDuckGo (HTML endpoint, keyless) or a SearXNG instance the user runs.
Safety: page fetches only reach PUBLIC hosts (no localhost / private ranges, also after redirects),
responses are size-capped, and results are plain text. The DuckDuckGo markup parser is UNVERIFIED
against the live site (written without internet access) — if it breaks, SearXNG is the stable fallback.
"""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15 Neurolume/0.1"
MAX_BYTES = 1_500_000


class SearchError(Exception):
    pass


def host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            return False
    return bool(infos)


def check_public_url(url: str) -> str:
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SearchError("只支持 http/https 网址")
    if not host_is_public(u.hostname):
        raise SearchError("出于安全考虑，不读取本机或内网地址")
    return url


class _Redirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, public_only: bool):
        self.public_only = public_only

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if self.public_only:
            check_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_get(url: str, *, data: bytes | None = None, timeout: float = 12, public_only: bool = True,
             headers: dict | None = None) -> tuple[str, str]:
    """Returns (text, final_url). Raises SearchError."""
    if public_only:
        check_public_url(url)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                                                          **(headers or {})})
    opener = urllib.request.build_opener(_Redirect(public_only))
    try:
        with opener.open(req, timeout=timeout) as r:
            raw = r.read(MAX_BYTES + 1)[:MAX_BYTES]
            ctype = r.headers.get("Content-Type", "")
            if ctype and not re.search(r"text|html|json|xml", ctype, re.I):
                raise SearchError(f"不支持的内容类型：{ctype.split(';')[0]}")
            m = re.search(r"charset=([\w-]+)", ctype, re.I)
            enc = m.group(1) if m else "utf-8"
            try:
                return raw.decode(enc, "replace"), r.geturl()
            except LookupError:
                return raw.decode("utf-8", "replace"), r.geturl()
    except SearchError:
        raise
    except urllib.error.HTTPError as e:
        raise SearchError(f"网站返回 {e.code}")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise SearchError(f"连接失败：{getattr(e, 'reason', e)}")


# ---------------- parsing
class _DDG(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._cur: dict | None = None
        self._mode: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "a" and "result__a" in cls.split():
            self._cur = {"title": "", "url": _unwrap(a.get("href", "")), "snippet": ""}
            self.results.append(self._cur)
            self._mode = "title"
        elif tag in ("a", "div", "td") and "result__snippet" in cls.split() and self._cur is not None:
            self._mode = "snippet"

    def handle_endtag(self, tag):
        if tag == "a" and self._mode == "title":
            self._mode = None
        elif tag in ("a", "div", "td") and self._mode == "snippet":
            self._mode = None

    def handle_data(self, data):
        if self._cur is None or not self._mode:
            return
        self._cur[self._mode] += data


def _unwrap(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    u = urllib.parse.urlparse(href)
    if u.netloc.endswith("duckduckgo.com") and u.path.startswith("/l/"):
        q = urllib.parse.parse_qs(u.query).get("uddg")
        if q:
            return q[0]
    return href


def parse_ddg(html: str) -> list[dict]:
    p = _DDG()
    p.feed(html)
    out = []
    for r in p.results:
        r = {k: re.sub(r"\s+", " ", v).strip() for k, v in r.items()}
        if r["url"].startswith("http") and r["title"]:
            out.append(r)
    return out


class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form", "iframe", "template"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def page_text(html: str, limit: int = 6000) -> tuple[str, str]:
    p = _Text()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return re.sub(r"\s+", " ", p.title).strip(), text[:limit]


# ---------------- searcher
class Searcher:
    def __init__(self, provider: str = "duckduckgo", searxng_url: str = "", fetch=http_get):
        self.provider = provider
        self.searxng_url = searxng_url.rstrip("/")
        self.fetch = fetch

    def search(self, query: str, n: int = 5) -> list[dict]:
        query = query.strip()
        if not query:
            raise SearchError("搜索词是空的")
        if self.provider == "searxng":
            if not self.searxng_url:
                raise SearchError("还没有填写 SearXNG 地址")
            url = f"{self.searxng_url}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})
            text, _ = self.fetch(url, public_only=False)
            try:
                items = json.loads(text).get("results", [])
            except ValueError:
                raise SearchError("SearXNG 没有返回 JSON（需要在它的设置里启用 json 格式）")
            res = [{"title": i.get("title", ""), "url": i.get("url", ""), "snippet": i.get("content", "")} for i in items]
        else:
            text, _ = self.fetch("https://html.duckduckgo.com/html/", data=urllib.parse.urlencode({"q": query}).encode())
            res = parse_ddg(text)
            if not res:
                raise SearchError("没有解析到搜索结果（DuckDuckGo 页面可能改版或要求验证）")
        return [r for r in res if r["url"].startswith("http")][:n]

    def read(self, url: str, limit: int = 6000) -> dict:
        html, final = self.fetch(url)
        title, text = page_text(html, limit)
        return {"url": final, "title": title, "text": text}

    def gather(self, query: str, n: int = 4, read_top: int = 2, per_page: int = 2500) -> list[dict]:
        """Search, then read the top pages. Each item: title, url, text (page text or the search snippet)."""
        items = []
        for i, r in enumerate(self.search(query, n)):
            text = r["snippet"]
            if i < read_top:
                try:
                    page = self.read(r["url"], per_page)
                    if len(page["text"]) > 200:
                        text = page["text"]
                except SearchError:
                    pass
            items.append({"title": r["title"] or r["url"], "url": r["url"], "text": text})
        return items
