import re

from fastapi.testclient import TestClient

from exec_agent.config import get_settings
from exec_agent.web import create_app, hash_password


def _client(monkeypatch):
    monkeypatch.setenv("EXEC_AGENT_RUNTIME_PROFILE", "research-online")
    monkeypatch.setenv("EXEC_AGENT_WEB_PASSWORD_HASH", hash_password("secret"))
    monkeypatch.setenv("EXEC_AGENT_WEB_SESSION_SECRET", "test-session-secret")
    get_settings.cache_clear()
    return TestClient(create_app())


def _csrf(response):
    return re.search(r"name='csrf_token' value='([^']+)'", response.text).group(1)


def test_web_routes_redirect_to_login(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert response.headers["x-frame-options"] == "DENY"


def test_login_sets_signed_session_and_logout(monkeypatch):
    client = _client(monkeypatch)
    login = client.get("/login")

    response = client.post(
        "/login",
        data={"password": "secret", "csrf_token": _csrf(login)},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "session=" in response.headers["set-cookie"]
    home = client.get("/")
    assert home.status_code == 200
    assert "Logout" in home.text

    token = re.search(r"name='csrf_token' value='([^']+)'", home.text).group(1)
    logout = client.post("/logout", data={"csrf_token": token}, follow_redirects=False)
    assert logout.status_code == 303
    assert logout.headers["location"] == "/login"


def test_state_changing_requests_require_csrf(monkeypatch):
    client = _client(monkeypatch)
    login = client.get("/login")
    client.post("/login", data={"password": "secret", "csrf_token": _csrf(login)})

    response = client.post("/memory", data={"content": "blocked"}, follow_redirects=False)

    assert response.status_code == 403


def test_login_rate_limiting(monkeypatch):
    client = _client(monkeypatch)

    for _ in range(5):
        login = client.get("/login")
        response = client.post("/login", data={"password": "bad", "csrf_token": _csrf(login)}, follow_redirects=False)
        assert response.status_code == 303
    login = client.get("/login")
    response = client.post("/login", data={"password": "bad", "csrf_token": _csrf(login)}, follow_redirects=False)

    assert response.status_code == 429
