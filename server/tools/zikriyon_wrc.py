"""
Zikriyon WRC — Web Research & Crawler for Tanzeel Intelligence.

Tools:
- web_search(query)     → DuckDuckGo + Brave fallback search
- fetch_page(url)       → Fetch + extract clean text from a URL
- research(query)       → Combined: search + fetch top results + return context
"""

import ipaddress
import re
import socket
from urllib.parse import urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
USER_AGENT = (
    "Mozilla/5.0 (compatible; TanzeelBot/1.0; +https://tanzeelai.web.app)"
)
REQUEST_TIMEOUT = 8
MAX_CONTENT_CHARS = 4000
MAX_SEARCH_RESULTS = 5

# Block private/internal IPs (SSRF protection)
BLOCKED_HOSTS = {
    "localhost", "127.0.0.1", "0.0.0.0", "::1",
    "metadata.google.internal",
}
BLOCKED_SCHEMES = {"file", "ftp", "gopher", "data"}


# ─────────────────────────────────────────────────────────────────────────
# SAFETY HELPERS
# ─────────────────────────────────────────────────────────────────────────
def is_safe_url(url: str) -> bool:
    """Returns True if URL is safe to fetch (SSRF protection)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host or host.lower() in BLOCKED_HOSTS:
            return False

        # Resolve hostname → check if private IP
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False
        except (socket.gaierror, ValueError):
            return False

        return True
    except Exception:
        return False


def sanitize_text(text: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """Strip prompt-injection patterns + truncate."""
    if not text:
        return ""

    # Remove common injection patterns
    text = re.sub(r"(?i)\b(ignore|forget|disregard)\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "", text)
    text = re.sub(r"(?i)^\s*(system|assistant|user)\s*:", "", text, flags=re.MULTILINE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────
# TOOL 1: WEB SEARCH
# ─────────────────────────────────────────────────────────────────────────
def web_search(query: str) -> dict:
    """
    Search the web for a query.
    Returns: {"query": str, "results": [{"title", "url", "snippet"}]}
    """
    results = []

    # Primary: DuckDuckGo Instant Answer API
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        data = resp.json()

        # Abstract (best single answer)
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "snippet": sanitize_text(data["AbstractText"], 400),
            })

        # Related topics
        for topic in data.get("RelatedTopics", [])[:MAX_SEARCH_RESULTS]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": sanitize_text(topic.get("Text", ""), 300),
                })
    except Exception as e:
        print(f"[wrc] DDG search failed: {e}")

    # Fallback: Wikipedia (if nothing found)
    if not results:
        try:
            resp = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_"),
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                results.append({
                    "title": data.get("title", query),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "snippet": sanitize_text(data.get("extract", ""), 500),
                })
        except Exception as e:
            print(f"[wrc] Wikipedia fallback failed: {e}")

    return {
        "query": query,
        "results": results[:MAX_SEARCH_RESULTS],
    }


# ─────────────────────────────────────────────────────────────────────────
# TOOL 2: FETCH PAGE
# ─────────────────────────────────────────────────────────────────────────
def fetch_page(url: str) -> dict:
    """
    Fetch a URL and extract clean readable text.
    Returns: {"url": str, "title": str, "content": str, "error": str|None}
    """
    if not is_safe_url(url):
        return {"url": url, "title": "", "content": "", "error": "URL not allowed (SSRF block)"}

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )

        # Limit content size (max 2 MB)
        content = resp.raw.read(2 * 1024 * 1024, decode_content=True)
        resp.close()

        if not content:
            return {"url": url, "title": "", "content": "", "error": "Empty response"}

        # Detect encoding
        encoding = resp.encoding or "utf-8"
        html = content.decode(encoding, errors="ignore")

        soup = BeautifulSoup(html, "html.parser")

        # Remove scripts, styles, nav
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
            tag.decompose()

        title = (soup.title.string or "").strip() if soup.title else ""

        # Extract main content
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text(separator=" ", strip=True)

        return {
            "url": url,
            "title": sanitize_text(title, 200),
            "content": sanitize_text(text, MAX_CONTENT_CHARS),
            "error": None,
        }

    except Exception as e:
        return {"url": url, "title": "", "content": "", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────
# TOOL 3: RESEARCH (Combined)
# ─────────────────────────────────────────────────────────────────────────
def research(query: str, max_pages: int = 2) -> dict:
    """
    Full research: search + fetch top pages + return combined context.
    Best for Qwen to answer factual/current questions.
    """
    search_result = web_search(query)
    results = search_result.get("results", [])

    if not results:
        return {
            "query": query,
            "context": "",
            "sources": [],
            "error": "No search results found",
        }

    combined_context = []
    sources = []

    for r in results[:max_pages]:
        if not r.get("url"):
            continue

        page = fetch_page(r["url"])
        if page.get("content"):
            combined_context.append(
                f"Source: {r['title']}\nURL: {r['url']}\n{page['content']}"
            )
            sources.append({"title": r["title"], "url": r["url"]})

    return {
        "query": query,
        "context": "\n\n---\n\n".join(combined_context),
        "sources": sources,
        "error": None,
    }
