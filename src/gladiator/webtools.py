from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from gladiator.runtime.context import ObservationLimiter


class WebTools:
    def __init__(
        self,
        *,
        searxng_url: str | None,
        storage_dir: Path,
        char_limit: int = 12_000,
        search_result_limit: int = 5,
    ):
        self.searxng_url = searxng_url.rstrip("/") if searxng_url else None
        self.storage_dir = storage_dir
        self.search_result_limit = max(1, min(search_result_limit, 20))
        self.limiter = ObservationLimiter(char_limit, storage_dir)

    def search(self, query: str) -> str:
        if not self.searxng_url:
            return "Web search is not configured. Run `gladiator setup` and enable SearXNG."
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json"},
            )
            response.raise_for_status()
            payload = response.json()
        results = payload.get("results") or []
        lines: list[str] = []
        for index, result in enumerate(results[: self.search_result_limit], 1):
            title = str(result.get("title") or "(untitled)").strip()
            url = str(result.get("url") or "").strip()
            snippet = " ".join(str(result.get("content") or "").split())[:700]
            lines.append(f"[{index}] {title}\n{url}\n{snippet}".strip())
        return "\n\n".join(lines) if lines else "No search results."

    def fetch(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "Only http:// and https:// URLs are supported."
        headers = {"User-Agent": "GladiatorAgent/0.1 (+local research fetcher)"}
        with httpx.Client(timeout=45.0, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type and "json" not in content_type:
            return f"Unsupported content type for text extraction: {content_type or 'unknown'}"
        text = self._extract_text(response.text, content_type)
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        result = self.limiter.limit(text, label=f"web-{digest}")
        prefix = f"Source: {response.url}\n"
        return prefix + result.text

    @staticmethod
    def _extract_text(raw: str, content_type: str) -> str:
        if "html" not in content_type:
            return raw
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
            tag.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup
        lines = [" ".join(piece.split()) for piece in root.stripped_strings]
        return "\n".join(line for line in lines if line)
