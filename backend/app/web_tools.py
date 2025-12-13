from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


class WebToolError(RuntimeError):
    pass


_URL_RE = re.compile(r"https?://[^\s)<>\"]+", re.IGNORECASE)
_SEARCH_PREFIX_RE = re.compile(r"(?is)^\s*(?:web|ddg|search)\s*:\s*(.+?)\s*$")

_TLD_ALLOWLIST = {
    "com",
    "net",
    "org",
    "io",
    "ai",
    "app",
    "dev",
    "cloud",
    "info",
    "biz",
    "tv",
    "me",
    "us",
    "ca",
    "eu",
    "uk",
    "de",
    "fr",
    "it",
    "es",
    "nl",
    "se",
    "no",
    "fi",
    "pl",
    "cz",
    "ch",
    "at",
    "au",
    "nz",
    "br",
    "in",
    "sg",
    "hk",
    "jp",
    "cn",
    "kr",
    "tw",
    "tr",
    "gr",
    "pt",
    "ru",
    "ua",
}


def normalize_url(candidate: str) -> str | None:
    c = str(candidate or "").strip()
    if not c:
        return None

    c = c.strip("()[]{}<>\"'")
    c = c.rstrip(".,);]}>\"'?!")
    if not c:
        return None

    low = c.lower()
    if low.startswith(("http://", "https://")):
        return c
    if low.startswith("www."):
        return f"https://{c}"

    # Bare domain/path (only if TLD looks like a real web TLD).
    if "." not in c:
        return None
    head = re.split(r"[/?#]", c, maxsplit=1)[0]
    head = head.split(":", 1)[0]
    if not head or not re.fullmatch(r"[a-z0-9.-]+", head, flags=re.IGNORECASE):
        return None
    tld = head.rsplit(".", 1)[-1].lower()
    if tld not in _TLD_ALLOWLIST:
        return None
    return f"https://{c}"


def extract_urls(text: str, max_urls: int) -> list[str]:
    if max_urls <= 0:
        return []

    raw: list[str] = []
    raw.extend(_URL_RE.findall(text or ""))
    # Also consider whitespace-separated candidates like "www.example.com" or "example.com/path".
    raw.extend(re.split(r"\s+", (text or "").strip()))
    raw = [r for r in raw if r]

    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for u in raw:
        u = normalize_url(u)
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= max_urls:
            break
    return out


def parse_search_query(text: str) -> str | None:
    m = _SEARCH_PREFIX_RE.match(text or "")
    if not m:
        return None
    q = str(m.group(1) or "").strip()
    return q or None


def _is_http_url(url: str) -> bool:
    try:
        u = urlparse(url)
    except Exception:
        return False
    return u.scheme in ("http", "https")


async def _read_limited(resp: httpx.Response, max_bytes: int) -> tuple[bytes, bool]:
    if max_bytes <= 0:
        raise WebToolError("max_bytes must be > 0")
    buf = bytearray()
    truncated = False
    async for chunk in resp.aiter_bytes():
        if not chunk:
            continue
        remaining = max_bytes - len(buf)
        if remaining <= 0:
            truncated = True
            break
        if len(chunk) > remaining:
            buf.extend(chunk[:remaining])
            truncated = True
            break
        buf.extend(chunk)
    return bytes(buf), truncated


def _html_to_text(html: str) -> tuple[str | None, str]:
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    title = None
    if soup.title and soup.title.string:
        title = unescape(str(soup.title.string)).strip() or None

    root = soup.body or soup
    text = root.get_text("\n", strip=True)
    text = unescape(text)
    # collapse excessive blank lines
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    content_type: str
    title: str | None
    text: str
    bytes: int
    truncated: bool


async def fetch_url(
    url: str,
    *,
    timeout_s: float = 15.0,
    max_bytes: int = 1_000_000,
    max_chars: int = 20_000,
) -> FetchResult:
    if not isinstance(url, str) or not url.strip() or not _is_http_url(url):
        raise WebToolError("url must be a valid http/https URL")
    if max_chars <= 0:
        raise WebToolError("max_chars must be > 0")

    headers = {
        "user-agent": "luxriot-mcp/0.1 (+local)",
        "accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(timeout_s)) as client:
        try:
            async with client.stream("GET", url, headers=headers) as resp:
                status = int(resp.status_code)
                content_type = str(resp.headers.get("content-type", "") or "")
                raw_type = content_type.split(";", 1)[0].strip().lower()

                if status >= 400:
                    body, _ = await _read_limited(resp, min(max_bytes, 200_000))
                    msg = body.decode("utf-8", errors="replace").strip()
                    raise WebToolError(f"Fetch failed ({status}): {msg[:400]}")

                is_text = raw_type.startswith("text/") or ("json" in raw_type) or not raw_type
                if not is_text:
                    raise WebToolError(f"Unsupported content-type: {raw_type or '(unknown)'}")

                data, truncated_bytes = await _read_limited(resp, max_bytes=max_bytes)
                bytes_read = len(data)
                encoding = resp.encoding or "utf-8"
                raw_text = data.decode(encoding, errors="replace")

                title = None
                text_out = raw_text
                if "html" in raw_type:
                    title, text_out = _html_to_text(raw_text)

                truncated_chars = len(text_out) > max_chars
                if truncated_chars:
                    text_out = text_out[:max_chars]

                return FetchResult(
                    url=url,
                    final_url=str(resp.url),
                    status=status,
                    content_type=raw_type or content_type,
                    title=title,
                    text=text_out.strip(),
                    bytes=bytes_read,
                    truncated=bool(truncated_bytes or truncated_chars),
                )
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            raise WebToolError(f"Fetch failed: {type(e).__name__}: {e}") from e


def _normalize_ddg_href(href: str) -> str:
    if not href:
        return ""
    href = href.strip()
    # DuckDuckGo HTML sometimes returns relative /l/?uddg=... links
    href = urljoin("https://duckduckgo.com", href)
    try:
        u = urlparse(href)
    except Exception:
        return href
    qs = parse_qs(u.query or "")
    uddg = qs.get("uddg", [None])[0]
    if isinstance(uddg, str) and uddg:
        try:
            return unquote(unescape(uddg))
        except Exception:
            return uddg
    return href


async def duckduckgo_search(
    query: str,
    *,
    k: int = 5,
    timeout_s: float = 15.0,
    max_bytes: int = 1_000_000,
) -> list[dict[str, Any]]:
    q = str(query or "").strip()
    if not q:
        raise WebToolError("query must be a non-empty string")
    if k < 1 or k > 10:
        raise WebToolError("k must be in range 1..10")

    url = "https://html.duckduckgo.com/html/?" + str(httpx.QueryParams({"q": q}))
    headers = {"user-agent": "luxriot-mcp/0.1 (+local)", "accept": "text/html"}

    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(timeout_s)) as client:
        try:
            async with client.stream("GET", url, headers=headers) as resp:
                status = int(resp.status_code)
                if status >= 400:
                    body, _ = await _read_limited(resp, min(max_bytes, 200_000))
                    msg = body.decode("utf-8", errors="replace").strip()
                    raise WebToolError(f"DuckDuckGo failed ({status}): {msg[:400]}")
                data, truncated = await _read_limited(resp, max_bytes=max_bytes)
                html = data.decode(resp.encoding or "utf-8", errors="replace")
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            raise WebToolError(f"DuckDuckGo failed: {type(e).__name__}: {e}") from e

    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for a in soup.select("a.result__a"):
        href = _normalize_ddg_href(str(a.get("href") or ""))
        title = a.get_text(" ", strip=True)
        if not href or not title:
            continue
        if href in seen:
            continue
        seen.add(href)

        snippet = ""
        container = a.find_parent(class_=re.compile(r"\bresult\b"))
        if container is not None:
            sn = container.select_one(".result__snippet")
            if sn is not None:
                snippet = sn.get_text(" ", strip=True)

        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= k:
            break

    # Expose whether results may be incomplete due to truncation.
    if truncated:
        for r in results:
            r["truncated_html"] = True

    return results
