import json
from urllib.error import HTTPError, URLError

import pytest

from app.tools import web_fastcrw
from exec_agent.config import get_settings


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def setup_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EXEC_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FASTCRW_BASE_URL", "http://localhost:3002")
    monkeypatch.setenv("FASTCRW_MAX_RESULTS", "5")
    monkeypatch.setenv("FASTCRW_ENABLE_SCRAPE", "true")
    monkeypatch.setenv("FASTCRW_ENABLE_CRAWL", "true")
    get_settings.cache_clear()


def test_search_web_uses_self_hosted_fastcrw(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse({"results": [{"title": "One", "url": "https://example.com"}]})

    monkeypatch.setattr(web_fastcrw, "urlopen", fake_urlopen)

    results = web_fastcrw.search_web("local news", max_results=1)

    assert captured == {"url": "http://localhost:3002/search", "body": {"query": "local news", "max_results": 1}, "timeout": 30}
    assert results[0]["title"] == "One"


def test_search_web_empty_results(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)
    monkeypatch.setattr(web_fastcrw, "urlopen", lambda request, timeout: FakeResponse({"results": []}))

    with pytest.raises(web_fastcrw.FastCRWEmptyResultsError):
        web_fastcrw.search_web("missing")


def test_scrape_url_stores_web_metadata(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)
    stored = {}

    class FakeVectorStore:
        def add_documents(self, chunks, metadata=None):
            stored["chunks"] = chunks
            stored["metadata"] = metadata
            return ["web-1"]

    monkeypatch.setattr(web_fastcrw, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(
        web_fastcrw,
        "urlopen",
        lambda request, timeout: FakeResponse({"url": "https://example.com", "title": "Example", "content": "Page content"}),
    )

    page = web_fastcrw.scrape_url("https://example.com")

    assert page.title == "Example"
    assert stored["chunks"] == ["Page content"]
    assert stored["metadata"][0]["url"] == "https://example.com"
    assert stored["metadata"][0]["source_type"] == "web"
    assert stored["metadata"][0]["provider"] == "fastcrw_self_hosted"


def test_crawl_limit_exceeded(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)

    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 400, "Bad Request", {}, None)

    class ErrorBody:
        def read(self):
            return b"crawl limit exceeded"

        def close(self):
            return None

    def fake_urlopen_with_limit(request, timeout):
        raise HTTPError(request.full_url, 400, "Bad Request", {}, ErrorBody())

    monkeypatch.setattr(web_fastcrw, "urlopen", fake_urlopen_with_limit)

    with pytest.raises(web_fastcrw.FastCRWCrawlLimitExceededError):
        web_fastcrw.crawl_url("https://example.com", limit=10)


def test_invalid_api_key_error(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)

    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(web_fastcrw, "urlopen", fake_urlopen)

    with pytest.raises(web_fastcrw.FastCRWInvalidAPIKeyError):
        web_fastcrw.health_check()


def test_server_offline_error(monkeypatch, tmp_path):
    setup_env(monkeypatch, tmp_path)
    monkeypatch.setattr(web_fastcrw, "urlopen", lambda request, timeout: (_ for _ in ()).throw(URLError("offline")))

    with pytest.raises(web_fastcrw.FastCRWServerOfflineError):
        web_fastcrw.health_check()
