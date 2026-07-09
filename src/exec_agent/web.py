"""FastAPI Bootstrap web UI for the local executive assistant."""

from __future__ import annotations

import asyncio
import html
import json
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.memory.long_term import LongTermMemoryStore
from app.models.registry import REGISTRY, benchmark_selection
from app.tools.web_fastcrw import FastCRWError
from exec_agent.services import get_backend
from exec_agent.chat import ChatSession, default_streamer
from exec_agent.config import RUNTIME_PROFILES, get_settings
from exec_agent.safety import UserFacingError
from exec_agent.tasks import TaskStore

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = get_settings().expanded_data_dir / "web_uploads"

TASK_EVENTS: dict[str, asyncio.Queue[str]] = {}
PENDING_APPROVALS: dict[str, dict[str, Any]] = {}
CHAT_SESSION = ChatSession()
PASSWORD_HASHER = PasswordHasher()
LOGIN_ATTEMPTS: dict[str, list[float]] = {}


def create_app() -> FastAPI:
    """Create the FastAPI application with HTML pages and JSON/SSE APIs."""

    settings = get_settings()
    app = FastAPI(title=f"{settings.app_name} Web UI")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        try:
            _enforce_authentication(request)
            if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path != "/login":
                await _enforce_csrf(request)
        except HTTPException as exc:
            if exc.status_code == 303:
                response: Response = RedirectResponse(exc.headers.get("Location", "/login"), status_code=303)
            else:
                response = HTMLResponse(html.escape(str(exc.detail)), status_code=exc.status_code)
            _add_security_headers(response)
            return response
        response = await call_next(request)
        _add_security_headers(response)
        return response

    app.add_middleware(
        SessionMiddleware,
        secret_key=_session_secret(),
        max_age=settings.web_session_timeout_minutes * 60,
        same_site="lax",
        https_only=settings.web_cookie_secure,
    )

    @app.get("/login", response_class=HTMLResponse)
    async def login(request: Request) -> HTMLResponse:
        return _page("Login", _login_form(request), request, public=True)

    @app.post("/login")
    async def do_login(request: Request, password: str = Form(""), csrf_token: str = Form("")) -> RedirectResponse:
        token = request.session.get("csrf_token")
        if not token or not secrets.compare_digest(str(token), csrf_token):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
        client = _client_id(request)
        if _login_rate_limited(client):
            raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")
        if not _verify_password(password):
            _record_login_failure(client)
            return RedirectResponse("/login?error=1", status_code=303)
        LOGIN_ATTEMPTS.pop(client, None)
        request.session.clear()
        request.session["authenticated"] = True
        request.session["authenticated_at"] = int(time.time())
        request.session["csrf_token"] = secrets.token_urlsafe(32)
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    async def logout(request: Request) -> RedirectResponse:
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        _require_login(request)
        store = TaskStore()
        cards = "".join(_task_card(task) for task in store.list(5)) or '<p class="text-body-secondary">No tasks yet.</p>'
        body = f"""
        <div class='row g-3'>
          <div class='col-lg-8'><div class='card'><div class='card-body'><h1>Executive Assistant</h1><p class='lead'>Remote Bootstrap control center for chat, autonomous tasks, memory, files, web research, models, and shell tools.</p><div class='d-flex gap-2 flex-wrap'>{_quick_links()}</div></div></div></div>
          <div class='col-lg-4'><div class='card'><div class='card-header'>Recent tasks</div><div class='card-body'>{cards}</div></div></div>
        </div>"""
        return _page("Dashboard", body, request)

    @app.get("/chat", response_class=HTMLResponse)
    async def chat(request: Request) -> HTMLResponse:
        _require_login(request)
        return _page("Chat", _chat_ui(), request)

    @app.get("/api/chat/stream")
    async def chat_stream(request: Request, message: str) -> StreamingResponse:
        _require_login(request)
        async def events():
            CHAT_SESSION.add("user", message)
            prompt = CHAT_SESSION.render_prompt()
            parts: list[str] = []
            for chunk in default_streamer(prompt):
                parts.append(chunk)
                yield _sse("token", {"chunk": chunk})
                await asyncio.sleep(0)
            CHAT_SESSION.add("assistant", "".join(parts))
            yield _sse("done", {"message": "complete"})
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/tasks", response_class=HTMLResponse)
    async def tasks(request: Request) -> HTMLResponse:
        _require_login(request)
        rows = "".join(_task_row(t, request) for t in TaskStore().list(50))
        return _page("Tasks", _tasks_ui(rows, request), request)

    @app.post("/tasks")
    async def create_task(background: BackgroundTasks, request: Request, description: str = Form(...), autonomy_level: str = Form("off")) -> RedirectResponse:
        _require_login(request)
        queue: asyncio.Queue[str] = asyncio.Queue()
        created: dict[str, str] = {}
        def progress(msg: str) -> None:
            if created.get("id"):
                TASK_EVENTS.setdefault(created["id"], queue)
            queue.put_nowait(msg)
        def run() -> None:
            task = get_backend().run_task(description, autonomy_level=autonomy_level, progress=progress)  # type: ignore[arg-type]
            created["id"] = task.task_id
            TASK_EVENTS[task.task_id] = queue
            queue.put_nowait(f"Task {task.task_id} {task.status}")
        background.add_task(run)
        return RedirectResponse("/tasks", status_code=303)

    @app.get("/tasks/{task_id}", response_class=HTMLResponse)
    async def task_detail(request: Request, task_id: str) -> HTMLResponse:
        _require_login(request)
        task = TaskStore().get(task_id)
        if not task:
            raise HTTPException(404)
        steps = "".join(f"<li class='list-group-item'><strong>{s.phase}</strong>: {html.escape(s.action)}<pre>{html.escape(s.result or s.error)}</pre></li>" for s in TaskStore().steps(task_id))
        approvals = "".join(_approval_card(k, v) for k, v in PENDING_APPROVALS.items()) or '<p class="text-body-secondary">No pending approvals.</p>'
        stop = f"<form method='post' action='/tasks/{task.task_id}/stop' class='d-inline'>{_csrf_input(request)}<button class='btn btn-danger'>Emergency stop</button></form>" if task.status == "running" else ""
        report = f"<a class='btn btn-outline-secondary' href='/tasks/{task.task_id}/report'>Export task report</a>"
        safety = _task_start_preview(_task_start_summary(task.description, task.autonomy_level))
        return _page("Task detail", f"<h1>Task {task.task_id}</h1><div class='d-flex gap-2 mb-3'>{stop}{report}</div>{_task_card(task)}<div class='card card-body mb-3'>{safety}</div><h2>Progress</h2><ul id='task-log' class='list-group mb-3'>{steps}</ul><script>streamTask('{task.task_id}')</script><h2>HITL approvals</h2>{approvals}", request)

    @app.get("/api/tasks/{task_id}/events")
    async def task_events(request: Request, task_id: str) -> StreamingResponse:
        _require_login(request)
        async def events():
            queue = TASK_EVENTS.setdefault(task_id, asyncio.Queue())
            while True:
                yield _sse("progress", {"message": await queue.get()})
        return StreamingResponse(events(), media_type="text/event-stream")


    @app.post("/tasks/{task_id}/stop")
    async def stop_task(request: Request, task_id: str) -> RedirectResponse:
        _require_login(request)
        if not TaskStore().cancel(task_id):
            raise HTTPException(404)
        queue = TASK_EVENTS.setdefault(task_id, asyncio.Queue())
        queue.put_nowait("Emergency stop requested by user.")
        return RedirectResponse(f"/tasks/{task_id}", status_code=303)

    @app.get("/tasks/{task_id}/report")
    async def task_report(request: Request, task_id: str) -> JSONResponse:
        _require_login(request)
        task = TaskStore().get(task_id)
        if not task:
            raise HTTPException(404)
        steps = TaskStore().steps(task_id)
        settings = get_settings()
        payload = {
            "task": task.__dict__,
            "safety": _task_start_summary(task.description, task.autonomy_level),
            "steps": [step.__dict__ for step in steps],
            "report_generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_profile": _model_profile(settings),
        }
        return JSONResponse(payload, headers={"Content-Disposition": f'attachment; filename="task-{task_id}-report.json"'})

    @app.get("/audit", response_class=HTMLResponse)
    async def audit_log(request: Request) -> HTMLResponse:
        _require_login(request)
        store = TaskStore()
        rows = []
        for task in store.list(100):
            rows.append(_audit_task_row(task))
            rows.extend(_audit_step_row(task, step) for step in store.steps(task.task_id))
        body = "".join(rows) or "<tr><td colspan='6' class='text-body-secondary'>No audit events yet.</td></tr>"
        return _page("Audit log", _audit_ui(body), request)

    @app.post("/api/approvals/{approval_id}/{decision}")
    async def approval(request: Request, approval_id: str, decision: str) -> dict[str, str]:
        _require_login(request)
        if decision not in {"approve", "reject", "edit"}:
            raise HTTPException(400)
        PENDING_APPROVALS.pop(approval_id, None)
        return {"status": decision}

    @app.get("/files", response_class=HTMLResponse)
    async def files(request: Request, path: str = ".") -> HTMLResponse:
        _require_login(request)
        try:
            entries = get_backend().list_files(path)
            listing = "".join(f"<li class='list-group-item'>{html.escape(e)}</li>" for e in entries)
        except UserFacingError as exc:
            listing = f"<div class='alert alert-danger'>{html.escape(str(exc))}</div>"
        return _page("Files", _files_ui(path, listing, request), request)

    @app.post("/files/upload")
    async def upload(request: Request, upload: UploadFile = File(...)) -> RedirectResponse:
        _require_login(request)
        _validate_upload(upload.filename or "")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / Path(upload.filename or "upload.bin").name
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        return RedirectResponse(f"/ingest?path={dest}", status_code=303)

    @app.get("/ingest", response_class=HTMLResponse)
    async def ingest_page(request: Request, path: str = "") -> HTMLResponse:
        _require_login(request)
        return _page("Ingest", _ingest_ui(path, request), request)

    @app.post("/ingest")
    async def ingest(request: Request, path: str = Form(...)) -> HTMLResponse:
        _require_login(request)
        count = _ingest_path(Path(path))
        return _page("Ingest", f"<div class='alert alert-success'>Ingested {count} chunks from {html.escape(path)}.</div>" + _ingest_ui(path, request), request)

    @app.get("/memory", response_class=HTMLResponse)
    async def memory(request: Request, q: str = "") -> HTMLResponse:
        _require_login(request)
        store = LongTermMemoryStore()
        memories = store.search(q) if q else store.list()
        rows = "".join(f"<tr><td>{m.id}</td><td>{html.escape(m.content)}</td><td>{html.escape(', '.join(m.tags))}</td><td>{html.escape(m.source)}</td></tr>" for m in memories)
        return _page("Memory", _memory_ui(q, rows, request), request)

    @app.post("/memory")
    async def add_memory(request: Request, content: str = Form(...), tags: str = Form("")) -> RedirectResponse:
        _require_login(request)
        LongTermMemoryStore().add(content, [t.strip() for t in tags.split(",")], "web")
        return RedirectResponse("/memory", status_code=303)

    @app.get("/models", response_class=HTMLResponse)
    async def models(request: Request) -> HTMLResponse:
        _require_login(request)
        selected = benchmark_selection()
        rows = "".join(f"<tr><td>{s.role.value}</td><td>{html.escape(s.display_name)}</td><td>{html.escape(s.backend)}</td><td>{s.recommended_vram_gb:g} GB</td></tr>" for s in REGISTRY)
        active = "".join(f"<li>{x['role']}: <code>{x['model_id']}</code></li>" for x in selected)
        return _page("Models", f"<h1>Model registry/status</h1><h2>Active selections</h2><ul>{active}</ul><table class='table table-sm'><thead><tr><th>Role</th><th>Model</th><th>Backend</th><th>VRAM</th></tr></thead><tbody>{rows}</tbody></table>", request)

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> HTMLResponse:
        _require_login(request)
        s = get_settings()
        opts = "".join(f"<option {'selected' if p == s.runtime_profile else ''}>{p}</option>" for p in RUNTIME_PROFILES)
        return _page("Settings", f"<h1>Settings</h1><form method='post'>{_csrf_input(request)}<label class='form-label'>Active profile</label><select name='profile' class='form-select'>{opts}</select><button class='btn btn-primary mt-3'>Switch active profile</button></form><pre>{html.escape(json.dumps(s.model_dump(mode='json'), indent=2, default=str))}</pre>", request)

    @app.post("/settings")
    async def switch_profile(request: Request, profile: str = Form(...)) -> RedirectResponse:
        _require_login(request)
        request.session["profile_notice"] = f"Set EXEC_AGENT_RUNTIME_PROFILE={profile} and restart to persist this profile."
        return RedirectResponse("/settings", status_code=303)

    @app.get("/web", response_class=HTMLResponse)
    async def web_page(request: Request, q: str = "") -> HTMLResponse:
        _require_login(request)
        body = _web_ui(q, "")
        if q:
            try:
                results = get_backend().search_web(q, get_settings().fastcrw_max_results)
                items = "".join(f"<li class='list-group-item'><strong>{html.escape(str(r.get('title','Untitled')))}</strong><br><code>{html.escape(str(r.get('url','')))}</code><p>{html.escape(str(r.get('description', r.get('content','')))[:500])}</p></li>" for r in results)
                body = _web_ui(q, items)
            except FastCRWError as exc:
                body = _web_ui(q, f"<div class='alert alert-warning'>{html.escape(str(exc))}</div>")
        return _page("Web", body, request)

    @app.get("/shell", response_class=HTMLResponse)
    async def shell(request: Request) -> HTMLResponse:
        _require_login(request)
        return _page("Shell", _shell_ui("", _history_rows(), request), request)

    @app.post("/shell", response_class=HTMLResponse)
    async def run_shell(request: Request, command: str = Form(...), cwd: str = Form("")) -> HTMLResponse:
        _require_login(request)
        try:
            result = get_backend().run_shell(command, cwd or None)
            output = f"<pre class='terminal'>{html.escape(result.stdout + result.stderr)}</pre>"
        except UserFacingError as exc:
            output = f"<div class='alert alert-danger'>{html.escape(str(exc))}</div>"
        return _page("Shell", _shell_ui(output, _history_rows(), request), request)

    return app



def _session_secret() -> str:
    configured = get_settings().web_session_secret
    if configured:
        return configured
    return str(get_settings().expanded_data_dir / "web-session-secret")


def hash_password(password: str) -> str:
    """Return an Argon2 hash for a plaintext password."""

    return PASSWORD_HASHER.hash(password)


def _configured_password_hash() -> str:
    return get_settings().web_password_hash.strip()


def _verify_password(password: str) -> bool:
    password_hash = _configured_password_hash()
    if not password_hash:
        return False
    try:
        return PASSWORD_HASHER.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def _require_login(request: Request) -> None:
    _enforce_authentication(request)


def _is_public_path(path: str) -> bool:
    return path == "/login" or path.startswith("/static/")


def _enforce_authentication(request: Request) -> None:
    if _is_public_path(request.url.path):
        return
    if not _configured_password_hash():
        raise HTTPException(status_code=503, detail="Web UI password is not configured. Run `exec-agent web set-password`.")
    if not request.session.get("authenticated"):
        raise HTTPException(status_code=303, headers={"Location": "/login"})


async def _enforce_csrf(request: Request) -> None:
    token = request.session.get("csrf_token")
    provided = request.headers.get("x-csrf-token")
    if provided is None:
        form = await request.form()
        provided = str(form.get("csrf_token", ""))
    if not token or not secrets.compare_digest(str(token), str(provided)):
        raise HTTPException(status_code=403, detail="Invalid CSRF token")


def _csrf_input(request: Request) -> str:
    token = request.session.setdefault("csrf_token", secrets.token_urlsafe(32))
    return f"<input type='hidden' name='csrf_token' value='{html.escape(str(token))}'>"


def _client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def _login_rate_limited(client: str) -> bool:
    cutoff = time.time() - 300
    attempts = [ts for ts in LOGIN_ATTEMPTS.get(client, []) if ts >= cutoff]
    LOGIN_ATTEMPTS[client] = attempts
    return len(attempts) >= 5


def _record_login_failure(client: str) -> None:
    LOGIN_ATTEMPTS.setdefault(client, []).append(time.time())


def _add_security_headers(response: Response) -> None:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Content-Security-Policy", "default-src 'self' https://cdn.jsdelivr.net; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'")


def _page(title: str, body: str, request: Request, *, public: bool = False) -> HTMLResponse:
    csrf_meta = html.escape(str(request.session.get("csrf_token", "")))
    nav = ""
    if not public:
        links = "".join(
            f"<a class='nav-link' href='{href}'>{label}</a>"
            for href, label in [
                ("/chat", "Chat"),
                ("/tasks", "Tasks"),
                ("/audit", "Audit log"),
                ("/files", "Files"),
                ("/memory", "Memory"),
                ("/models", "Models"),
                ("/settings", "Settings"),
                ("/web", "Web"),
                ("/shell", "Shell"),
                ("/ingest", "Ingest"),
            ]
        )
        nav = f"""
        <nav class='navbar navbar-expand-lg border-bottom sticky-top bg-body'>
          <div class='container-fluid'>
            <a class='navbar-brand' href='/'>Exec Agent</a>
            <button class='navbar-toggler' data-bs-toggle='collapse' data-bs-target='#nav'><span class='navbar-toggler-icon'></span></button>
            <div id='nav' class='collapse navbar-collapse'>
              <div class='navbar-nav'>{links}</div>
              <div class='ms-auto d-flex gap-2'>
                <button class='btn btn-outline-secondary btn-sm' onclick='toggleTheme()'>Dark mode</button>
                <form method='post' action='/logout' class='m-0'>{_csrf_input(request)}<button class='btn btn-outline-danger btn-sm'>Logout</button></form>
              </div>
            </div>
          </div>
        </nav>"""
    return HTMLResponse(f"""<!doctype html><html lang='en' data-bs-theme='auto'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title><meta name='csrf-token' content='{csrf_meta}'><link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css' rel='stylesheet'><link href='/static/web.css' rel='stylesheet'></head><body>{nav}<main class='container-fluid py-3'>{body}</main><script src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js'></script><script src='/static/web.js'></script></body></html>""")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _quick_links() -> str:
    return "".join(f"<a class='btn btn-primary' href='{h}'>{l}</a>" for h,l in [('/chat','Start chat'),('/tasks','Run task'),('/files','Browse files'),('/memory','Search memory'),('/web','FastCRW search'),('/shell','Shell')])


def _login_form(request: Request) -> str:
    error = "<div class='alert alert-danger'>Invalid password.</div>" if request.query_params.get("error") else ""
    return f"""<div class='row justify-content-center'><div class='col-sm-10 col-md-6 col-lg-4'><div class='card mt-5'><div class='card-body'><h1 class='h3'>Login</h1>{error}<form method='post'>{_csrf_input(request)}<label class='form-label'>Web password</label><input name='password' type='password' class='form-control' autofocus><button class='btn btn-primary w-100 mt-3'>Login</button><p class='text-body-secondary mt-3'>Run <code>exec-agent web set-password</code> to configure an Argon2 password hash. Plaintext passwords are never stored.</p></form></div></div></div></div>"""


def _chat_ui() -> str:
    return """<h1>Chat</h1><div id='chat-output' class='rich-box mb-3'></div><form onsubmit='sendChat(event)' class='input-group'><textarea id='chat-message' class='form-control' rows='2' placeholder='Ask the assistant...'></textarea><button class='btn btn-primary'>Stream</button></form>"""


def _task_card(task: Any) -> str:
    status_class = "text-bg-success" if task.status == "completed" else "text-bg-danger" if task.status in {"failed", "cancelled"} else "text-bg-warning" if task.status == "blocked" else "text-bg-secondary"
    return f"<div class='mb-2'><a href='/tasks/{task.task_id}'><strong>{task.task_id}</strong></a> <span class='badge {status_class}'>{task.status}</span><br><span>{html.escape(task.description)}</span></div>"


def _task_row(task: Any, request: Request) -> str:
    actions = f"<a class='btn btn-sm btn-outline-secondary' href='/tasks/{task.task_id}/report'>Export report</a>"
    if task.status == "running":
        actions = f"<form method='post' action='/tasks/{task.task_id}/stop' class='d-inline'>{_csrf_input(request)}<button class='btn btn-sm btn-danger'>Emergency stop</button></form> " + actions
    return f"<tr><td><a href='/tasks/{task.task_id}'>{task.task_id}</a></td><td>{html.escape(task.description)}</td><td>{_autonomy_label(task.autonomy_level)}</td><td>{task.status}</td><td>{task.updated_at}</td><td>{actions}</td></tr>"


def _tasks_ui(rows: str, request: Request) -> str:
    settings = get_settings()
    controls = _safety_controls_panel(settings)
    options = "".join(f"<option value='{value}' {'selected' if value == settings.autonomy_level else ''}>{label}</option>" for value, label in _autonomy_options())
    summary = _task_start_summary("", settings.autonomy_level)
    return f"""<h1>Autonomous tasks</h1>{controls}<form method='post' class='card card-body mb-3' oninput='updateTaskPreview()'>{_csrf_input(request)}<label class='form-label'>Task goal</label><textarea id='task-goal' name='description' class='form-control' required></textarea><label class='form-label mt-2'>Autonomy level</label><select id='autonomy-level' name='autonomy_level' class='form-select'>{options}</select><div id='task-preview' class='alert alert-info mt-3'>{_task_start_preview(summary)}</div><button class='btn btn-primary mt-2'>Run task</button></form><h2>Task history</h2><table class='table table-responsive'><thead><tr><th>ID</th><th>Description</th><th>Autonomy</th><th>Status</th><th>Updated</th><th>Actions</th></tr></thead><tbody>{rows}</tbody></table>"""


def _approval_card(key: str, data: dict[str, Any]) -> str:
    proposed = html.escape(str(data.get("proposed_action", data.get("action", key))))
    reason = html.escape(str(data.get("reason", "Human approval is required before this side effect.")))
    affected = html.escape(json.dumps(data.get("affected_files_commands", data.get("payload", data)), indent=2))
    risk = html.escape(str(data.get("risk_level", "medium")))
    return f"""<div class='card mb-2 approval-card'><div class='card-body'><div class='d-flex justify-content-between'><strong>{proposed}</strong><span class='badge text-bg-warning'>Risk: {risk}</span></div><p><strong>Reason:</strong> {reason}</p><p><strong>Affected files/commands:</strong></p><pre>{affected}</pre><button class='btn btn-success' onclick=decideApproval('{key}','approve')>Approve</button> <button class='btn btn-outline-primary' onclick=editApproval('{key}')>Edit</button> <button class='btn btn-danger' onclick=decideApproval('{key}','reject')>Reject</button></div></div>"""



def _autonomy_options() -> list[tuple[str, str]]:
    return [
        ("off", "off"),
        ("suggest_only", "suggest only"),
        ("human_approved", "require approval"),
        ("autonomous_limited", "autonomous limited"),
        ("autonomous_full", "autonomous full"),
    ]


def _autonomy_label(value: str) -> str:
    return dict(_autonomy_options()).get(value, value)


def _csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _model_profile(settings: Any) -> dict[str, Any]:
    return {
        "runtime_profile": settings.runtime_profile,
        "model_preset": settings.model_preset,
        "primary_model": settings.model_id,
        "device": settings.device,
        "general_model": settings.general_model_id,
        "coding_model": settings.coding_model_id,
        "research_model": settings.research_model_id,
        "tool_model": settings.tool_model_id,
    }


def _allowed_tools(settings: Any) -> list[str]:
    tools = ["planner", "reflect", "filesystem"]
    if settings.shell_enabled:
        tools.append("shell")
    if settings.web_enabled and settings.fastcrw_enabled:
        tools.append("web_fastcrw")
    return tools


def _task_start_summary(goal: str, autonomy_level: str) -> dict[str, Any]:
    settings = get_settings()
    return {
        "task_goal": goal or "Enter a goal before starting.",
        "autonomy_level": _autonomy_label(autonomy_level),
        "max_steps": settings.max_autonomous_steps,
        "allowed_tools": _allowed_tools(settings),
        "allowed_directories": _csv_items(settings.allowed_dirs),
    }


def _task_start_preview(summary: dict[str, Any]) -> str:
    tools = "".join(f"<span class='badge text-bg-secondary me-1'>{html.escape(tool)}</span>" for tool in summary["allowed_tools"])
    dirs = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in summary["allowed_directories"])
    return f"""<h2 class='h5'>Before this autonomous task begins</h2><dl class='row mb-2'><dt class='col-sm-3'>Task goal</dt><dd class='col-sm-9' data-preview='goal'>{html.escape(str(summary['task_goal']))}</dd><dt class='col-sm-3'>Autonomy level</dt><dd class='col-sm-9' data-preview='autonomy'>{html.escape(str(summary['autonomy_level']))}</dd><dt class='col-sm-3'>Max steps</dt><dd class='col-sm-9'>{summary['max_steps']}</dd><dt class='col-sm-3'>Allowed tools</dt><dd class='col-sm-9'>{tools}</dd><dt class='col-sm-3'>Allowed directories</dt><dd class='col-sm-9'><ul class='mb-0'>{dirs}</ul></dd></dl>"""


def _safety_controls_panel(settings: Any) -> str:
    allowed_dirs = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in _csv_items(settings.allowed_dirs))
    allowlist = "".join(f"<span class='badge text-bg-success me-1 mb-1'>{html.escape(cmd)}</span>" for cmd in _csv_items(settings.shell_allowlist))
    denylist = "".join(f"<span class='badge text-bg-danger me-1 mb-1'>{html.escape(cmd)}</span>" for cmd in _csv_items(settings.shell_denylist))
    model = _model_profile(settings)
    model_rows = "".join(f"<tr><th>{html.escape(k.replace('_', ' ').title())}</th><td><code>{html.escape(str(v))}</code></td></tr>" for k, v in model.items())
    return f"""<div class='row g-3 mb-3'><div class='col-lg-4'><div class='card h-100'><div class='card-header'>Active allowed directories</div><div class='card-body'><ul class='mb-0'>{allowed_dirs}</ul></div></div></div><div class='col-lg-4'><div class='card h-100'><div class='card-header'>Shell allowlist / denylist</div><div class='card-body'><h2 class='h6'>Allowlist</h2>{allowlist}<h2 class='h6 mt-3'>Denylist</h2>{denylist}</div></div></div><div class='col-lg-4'><div class='card h-100'><div class='card-header'>Current model profile</div><div class='card-body p-0'><table class='table table-sm mb-0'>{model_rows}</table></div></div></div></div>"""


def _audit_task_row(task: Any) -> str:
    return f"<tr><td>{html.escape(task.updated_at)}</td><td>task</td><td><a href='/tasks/{task.task_id}'>{task.task_id}</a></td><td>{html.escape(task.status)}</td><td>{html.escape(task.autonomy_level)}</td><td>{html.escape(task.description)}</td></tr>"


def _audit_step_row(task: Any, step: Any) -> str:
    detail = step.error or step.result
    return f"<tr><td>{html.escape(step.created_at)}</td><td>step</td><td><a href='/tasks/{task.task_id}'>{task.task_id}</a></td><td>{html.escape(step.phase)}</td><td>{html.escape(step.tool_name)}</td><td>{html.escape(step.action)}<pre>{html.escape(detail[:1000])}</pre></td></tr>"


def _audit_ui(rows: str) -> str:
    return f"""<h1>Audit log</h1><p class='text-body-secondary'>Chronological record of autonomous tasks, steps, decisions, and tool outputs persisted by the task store.</p><table class='table table-sm table-responsive'><thead><tr><th>Time</th><th>Type</th><th>Task</th><th>Status/phase</th><th>Autonomy/tool</th><th>Details</th></tr></thead><tbody>{rows}</tbody></table>"""

def _files_ui(path: str, listing: str, request: Request) -> str:
    return f"""<h1>Files</h1><form class='input-group mb-3'><input name='path' value='{html.escape(path)}' class='form-control'><button class='btn btn-outline-primary'>Browse allowed directory</button></form><ul class='list-group mb-3'>{listing}</ul><form action='/files/upload' method='post' enctype='multipart/form-data' class='card card-body'>{_csrf_input(request)}<label class='form-label'>Upload PDF/DOCX/image/text</label><input type='file' name='upload' class='form-control'><button class='btn btn-primary mt-3'>Upload</button></form>"""


def _ingest_ui(path: str, request: Request) -> str:
    return f"""<h1>Ingest files into vector DB</h1><form method='post' class='card card-body'>{_csrf_input(request)}<label class='form-label'>File path</label><input name='path' value='{html.escape(path)}' class='form-control'><button class='btn btn-primary mt-3'>Ingest</button></form>"""


def _memory_ui(q: str, rows: str, request: Request) -> str:
    return f"""<h1>Memory</h1><form class='input-group mb-3'><input name='q' value='{html.escape(q)}' class='form-control' placeholder='Search memory'><button class='btn btn-outline-primary'>Search</button></form><form method='post' class='card card-body mb-3'>{_csrf_input(request)}<textarea name='content' class='form-control' placeholder='New long-term memory'></textarea><input name='tags' class='form-control mt-2' placeholder='tags, comma-separated'><button class='btn btn-primary mt-2'>Add memory</button></form><table class='table'><thead><tr><th>ID</th><th>Content</th><th>Tags</th><th>Source</th></tr></thead><tbody>{rows}</tbody></table>"""


def _web_ui(q: str, results: str) -> str:
    status = ""
    try:
        status = f"<span class='badge text-bg-success'>{html.escape(str(get_backend().web_health().get('status', 'ok')))}</span>"
    except Exception as exc:  # noqa: BLE001
        status = f"<span class='badge text-bg-warning'>{html.escape(str(exc))}</span>"
    return f"<h1>FastCRW web search {status}</h1><form class='input-group mb-3'><input name='q' value='{html.escape(q)}' class='form-control' placeholder='Search the web'><button class='btn btn-primary'>Search</button></form><ul class='list-group'>{results}</ul>"


def _shell_ui(output: str, rows: str, request: Request) -> str:
    return f"""<h1>Allowed shell commands</h1><form method='post' class='card card-body mb-3'>{_csrf_input(request)}<input name='command' class='form-control font-monospace' placeholder='python --version'><input name='cwd' class='form-control mt-2' placeholder='Working directory (inside shell workspace)'><button class='btn btn-primary mt-2'>Run</button></form>{output}<h2>Command history</h2><table class='table table-sm'><tbody>{rows}</tbody></table>"""


def _history_rows() -> str:
    return "".join(f"<tr><td>{r.id}</td><td><code>{html.escape(r.command)}</code></td><td>{r.exit_code}</td><td>{r.started_at}</td></tr>" for r in get_backend().shell_history(25))


def _validate_upload(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    allowed = {x.strip() for x in get_settings().allowed_upload_extensions.split(",") if x.strip()}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported upload type: {suffix}")


def _ingest_path(path: Path) -> int:
    try:
        return get_backend().ingest_path(path)
    except UserFacingError as exc:
        raise HTTPException(400, str(exc)) from exc


app = create_app()
