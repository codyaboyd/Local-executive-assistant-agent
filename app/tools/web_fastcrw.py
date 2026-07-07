"""Self-hosted FastCRW web research tools.

This module talks only to a caller-configured FastCRW base URL. Defaults point to
localhost so the assistant never uses hosted FastCRW unless the user explicitly
sets a different server.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import socket

from exec_agent.config import get_settings
from app.memory.vector_store import VectorStore

PROVIDER = "fastcrw_self_hosted"
SOURCE_TYPE = "web"


class FastCRWError(RuntimeError):
    """Base FastCRW integration error."""


class FastCRWServerOfflineError(FastCRWError):
    """Raised when the configured FastCRW server cannot be reached."""


class FastCRWInvalidAPIKeyError(FastCRWError):
    """Raised when FastCRW rejects the configured API key."""


class FastCRWTimeoutError(FastCRWError):
    """Raised when FastCRW does not respond within the configured timeout."""


class FastCRWEmptyResultsError(FastCRWError):
    """Raised when a search completes but returns no results."""


class FastCRWScrapeBlockedError(FastCRWError):
    """Raised when FastCRW reports that scraping was blocked."""


class FastCRWCrawlLimitExceededError(FastCRWError):
    """Raised when a crawl request exceeds the allowed page limit."""


@dataclass(frozen=True)
class WebPage:
    """Normalized scraped/crawled page content."""

    url: str
    title: str
    content: str
    fetched_at: str

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "fetched_at": self.fetched_at,
            "source": self.url,
            "source_type": SOURCE_TYPE,
            "provider": PROVIDER,
        }


def _settings():
    return get_settings()


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    api_key = _settings().fastcrw_api_key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    return headers


def _request(path: str, payload: dict[str, Any] | None = None, *, method: str | None = None) -> dict[str, Any]:
    settings = _settings()
    base_url = str(settings.fastcrw_base_url).rstrip("/")
    api_prefix = str(settings.fastcrw_api_prefix).strip().rstrip("/")
    if api_prefix and not api_prefix.startswith("/"):
        api_prefix = f"/{api_prefix}"
    url = f"{base_url}{api_prefix}{path}" if path != "/health" else f"{base_url}{path}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=_headers(), method=method or ("POST" if payload is not None else "GET"))
    try:
        with urlopen(request, timeout=settings.fastcrw_timeout_seconds) as response:  # noqa: S310 - user-configured self-hosted URL
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise FastCRWInvalidAPIKeyError("FastCRW rejected the configured API key.") from exc
        if exc.code in {408, 504}:
            raise FastCRWTimeoutError(f"FastCRW request timed out after {settings.fastcrw_timeout_seconds} seconds.") from exc
        detail = exc.read().decode("utf-8", errors="ignore")
        if exc.code in {423, 429, 451} or "blocked" in detail.lower():
            raise FastCRWScrapeBlockedError("FastCRW reported that scraping was blocked.") from exc
        if "limit" in detail.lower():
            raise FastCRWCrawlLimitExceededError("FastCRW crawl limit exceeded.") from exc
        raise FastCRWError(f"FastCRW request failed with HTTP {exc.code}: {detail or exc.reason}") from exc
    except socket.timeout as exc:
        raise FastCRWTimeoutError(f"FastCRW request timed out after {settings.fastcrw_timeout_seconds} seconds.") from exc
    except URLError as exc:
        if isinstance(exc.reason, socket.timeout):
            raise FastCRWTimeoutError(f"FastCRW request timed out after {settings.fastcrw_timeout_seconds} seconds.") from exc
        raise FastCRWServerOfflineError(f"FastCRW server appears offline at {base_url}.") from exc

    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FastCRWError("FastCRW returned invalid JSON.") from exc


def _items(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = data.get("results", data.get("data", data.get("pages", [])))
    if isinstance(value, dict):
        value = value.get("results", value.get("pages", []))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _page(item: dict[str, Any]) -> WebPage:
    fetched_at = item.get("fetched_at") or item.get("fetchedAt") or datetime.now(timezone.utc).isoformat()
    return WebPage(
        url=str(item.get("url") or item.get("source") or ""),
        title=str(item.get("title") or "Untitled"),
        content=str(item.get("content") or item.get("markdown") or item.get("text") or item.get("description") or ""),
        fetched_at=str(fetched_at),
    )


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Search the self-hosted FastCRW server."""

    if max_results <= 0:
        return []
    data = _request("/search", {"query": query, "limit": max_results, "max_results": max_results})
    results = _items(data)
    if not results:
        raise FastCRWEmptyResultsError(f"FastCRW returned no search results for {query!r}.")
    return results[:max_results]


def scrape_url(url: str) -> WebPage:
    """Scrape a URL using self-hosted FastCRW and store it in vector DB."""

    if not _settings().fastcrw_enable_scrape:
        raise FastCRWScrapeBlockedError("FastCRW scraping is disabled by FASTCRW_ENABLE_SCRAPE.")
    data = _request("/scrape", {"url": url, "formats": ["markdown"]})
    item = data.get("data", data)
    if not isinstance(item, dict) or item.get("blocked"):
        raise FastCRWScrapeBlockedError(f"FastCRW could not scrape {url}.")
    page = _page(item | {"url": item.get("url", url)})
    _store_pages([page])
    return page


def crawl_url(url: str, limit: int = 10) -> list[WebPage]:
    """Crawl a URL using self-hosted FastCRW and store pages in vector DB."""

    if limit <= 0:
        raise FastCRWCrawlLimitExceededError("FastCRW crawl limit must be greater than zero.")
    if not _settings().fastcrw_enable_crawl:
        raise FastCRWScrapeBlockedError("FastCRW crawling is disabled by FASTCRW_ENABLE_CRAWL.")
    data = _request("/crawl", {"url": url, "limit": limit, "maxPages": limit})
    pages = [_page(item) for item in _items(data)[:limit]]
    if not pages:
        raise FastCRWEmptyResultsError(f"FastCRW crawled no pages for {url}.")
    _store_pages(pages)
    return pages


def health_check() -> dict[str, Any]:
    """Return FastCRW health information."""

    data = _request("/health", None, method="GET")
    return data or {"status": "ok"}


def target_domain(url: str) -> str:
    """Return a URL's target domain for approval prompts."""

    return urlparse(url).netloc or url


def _store_pages(pages: list[WebPage]) -> list[str]:
    chunks = [page.content for page in pages if page.content.strip()]
    metadata = [page.metadata for page in pages if page.content.strip()]
    if not chunks:
        return []
    return VectorStore().add_documents(chunks, metadata)
