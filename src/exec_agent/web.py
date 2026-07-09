"""FastAPI Bootstrap web UI for the local executive assistant."""

from __future__ import annotations

import asyncio
import html
import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.memory.long_term import LongTermMemoryStore
from app.memory.vector_store import VectorStore
from app.models.registry import REGISTRY, benchmark_selection
from app.tools import filesystem
from app.tools.docx import ingest_docx
from app.tools.pdf import chunk_text, ingest_pdf
from app.tools.shell import history as shell_history, run_command
from app.tools.web_fastcrw import FastCRWError, health_check, search_web
from exec_agent.chat import ChatSession, default_streamer
from exec_agent.config import RUNTIME_PROFILES, get_settings
from exec_agent.safety import UserFacingError
from exec_agent.tasks import AutonomousTaskRunner, TaskStore

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
UPLOAD_DIR = get_settings().expanded_data_dir / "web_uploads"

TASK_EVENTS: dict[str, asyncio.Queue[str]] = {}
PENDING_APPROVALS: dict[str, dict[str, Any]] = {}
CHAT_SESSION = ChatSession()


def create_app() -> FastAPI:
    """Create the FastAPI application with HTML pages and JSON/SSE APIs."""

    app = FastAPI(title=f"{get_settings().app_name} Web UI")
    app.add_middleware(SessionMiddleware, secret_key=_session_secret())
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/login", response_class=HTMLResponse)
    async def login(request: Request) -> HTMLResponse:
        return _page("Login", _login_form(), request, public=True)

    @app.post("/login")
    async def do_login(request: Request, password: str = Form("")) -> RedirectResponse:
        expected = _web_password()
        if expected and password != expected:
            return RedirectResponse("/login?error=1", status_code=303)
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=303)

    @app.get("/logout")
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
        rows = "".join(_task_row(t) for t in TaskStore().list(50))
        return _page("Tasks", _tasks_ui(rows), request)

    @app.post("/tasks")
    async def create_task(background: BackgroundTasks, request: Request, description: str = Form(...), autonomy_level: str = Form("human_approved")) -> RedirectResponse:
        _require_login(request)
        queue: asyncio.Queue[str] = asyncio.Queue()
        created: dict[str, str] = {}
        def progress(msg: str) -> None:
            if created.get("id"):
                TASK_EVENTS.setdefault(created["id"], queue)
            queue.put_nowait(msg)
        def run() -> None:
            task = AutonomousTaskRunner(progress=progress).run(description, autonomy_level=autonomy_level)  # type: ignore[arg-type]
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
        return _page("Task detail", f"<h1>Task {task.task_id}</h1>{_task_card(task)}<h2>Progress</h2><ul id='task-log' class='list-group mb-3'>{steps}</ul><script>streamTask('{task.task_id}')</script><h2>HITL approvals</h2>{approvals}", request)

    @app.get("/api/tasks/{task_id}/events")
    async def task_events(request: Request, task_id: str) -> StreamingResponse:
        _require_login(request)
        async def events():
            queue = TASK_EVENTS.setdefault(task_id, asyncio.Queue())
            while True:
                yield _sse("progress", {"message": await queue.get()})
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/approvals/{approval_id}/{decision}")
    async def approval(request: Request, approval_id: str, decision: str) -> dict[str, str]:
        _require_login(request)
        if decision not in {"approve", "reject"}:
            raise HTTPException(400)
        PENDING_APPROVALS.pop(approval_id, None)
        return {"status": decision}

    @app.get("/files", response_class=HTMLResponse)
    async def files(request: Request, path: str = ".") -> HTMLResponse:
        _require_login(request)
        try:
            entries = filesystem.list_dir(path)
            listing = "".join(f"<li class='list-group-item'>{html.escape(e)}</li>" for e in entries)
        except UserFacingError as exc:
            listing = f"<div class='alert alert-danger'>{html.escape(str(exc))}</div>"
        return _page("Files", _files_ui(path, listing), request)

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
        return _page("Ingest", _ingest_ui(path), request)

    @app.post("/ingest")
    async def ingest(request: Request, path: str = Form(...)) -> HTMLResponse:
        _require_login(request)
        count = _ingest_path(Path(path))
        return _page("Ingest", f"<div class='alert alert-success'>Ingested {count} chunks from {html.escape(path)}.</div>" + _ingest_ui(path), request)

    @app.get("/memory", response_class=HTMLResponse)
    async def memory(request: Request, q: str = "") -> HTMLResponse:
        _require_login(request)
        store = LongTermMemoryStore()
        memories = store.search(q) if q else store.list()
        rows = "".join(f"<tr><td>{m.id}</td><td>{html.escape(m.content)}</td><td>{html.escape(', '.join(m.tags))}</td><td>{html.escape(m.source)}</td></tr>" for m in memories)
        return _page("Memory", _memory_ui(q, rows), request)

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
        return _page("Settings", f"<h1>Settings</h1><form method='post'><label class='form-label'>Active profile</label><select name='profile' class='form-select'>{opts}</select><button class='btn btn-primary mt-3'>Switch active profile</button></form><pre>{html.escape(json.dumps(s.model_dump(mode='json'), indent=2, default=str))}</pre>", request)

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
                results = search_web(q, get_settings().fastcrw_max_results)
                items = "".join(f"<li class='list-group-item'><strong>{html.escape(str(r.get('title','Untitled')))}</strong><br><code>{html.escape(str(r.get('url','')))}</code><p>{html.escape(str(r.get('description', r.get('content','')))[:500])}</p></li>" for r in results)
                body = _web_ui(q, items)
            except FastCRWError as exc:
                body = _web_ui(q, f"<div class='alert alert-warning'>{html.escape(str(exc))}</div>")
        return _page("Web", body, request)

    @app.get("/shell", response_class=HTMLResponse)
    async def shell(request: Request) -> HTMLResponse:
        _require_login(request)
        return _page("Shell", _shell_ui("", _history_rows()), request)

    @app.post("/shell", response_class=HTMLResponse)
    async def run_shell(request: Request, command: str = Form(...), cwd: str = Form("")) -> HTMLResponse:
        _require_login(request)
        try:
            result = run_command(command, cwd or None)
            output = f"<pre class='terminal'>{html.escape(result.stdout + result.stderr)}</pre>"
        except UserFacingError as exc:
            output = f"<div class='alert alert-danger'>{html.escape(str(exc))}</div>"
        return _page("Shell", _shell_ui(output, _history_rows()), request)

    return app



def _session_secret() -> str:
    return str(get_settings().expanded_data_dir / "web-session-secret")


def _web_password() -> str:
    return __import__("os").environ.get("EXEC_AGENT_WEB_PASSWORD", "")


def _require_login(request: Request) -> None:
    if _web_password() and not request.session.get("authenticated"):
        raise HTTPException(status_code=303, headers={"Location": "/login"})


def _page(title: str, body: str, request: Request, *, public: bool = False) -> HTMLResponse:
    nav = "" if public else """<nav class='navbar navbar-expand-lg border-bottom sticky-top bg-body'><div class='container-fluid'><a class='navbar-brand' href='/'>Exec Agent</a><button class='navbar-toggler' data-bs-toggle='collapse' data-bs-target='#nav'><span class='navbar-toggler-icon'></span></button><div id='nav' class='collapse navbar-collapse'><div class='navbar-nav'>""" + "".join(f"<a class='nav-link' href='{href}'>{label}</a>" for href,label in [('/chat','Chat'),('/tasks','Tasks'),('/files','Files'),('/memory','Memory'),('/models','Models'),('/settings','Settings'),('/web','Web'),('/shell','Shell'),('/ingest','Ingest')]) + """</div><div class='ms-auto d-flex gap-2'><button class='btn btn-outline-secondary btn-sm' onclick='toggleTheme()'>Dark mode</button><a class='btn btn-outline-danger btn-sm' href='/logout'>Logout</a></div></div></div></nav>"""
    return HTMLResponse(f"""<!doctype html><html lang='en' data-bs-theme='auto'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title><link href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css' rel='stylesheet'><link href='/static/web.css' rel='stylesheet'></head><body>{nav}<main class='container-fluid py-3'>{body}</main><script src='https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js'></script><script src='/static/web.js'></script></body></html>""")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _quick_links() -> str:
    return "".join(f"<a class='btn btn-primary' href='{h}'>{l}</a>" for h,l in [('/chat','Start chat'),('/tasks','Run task'),('/files','Browse files'),('/memory','Search memory'),('/web','FastCRW search'),('/shell','Shell')])


def _login_form() -> str:
    return """<div class='row justify-content-center'><div class='col-sm-10 col-md-6 col-lg-4'><div class='card mt-5'><div class='card-body'><h1 class='h3'>Login</h1><form method='post'><label class='form-label'>Web password</label><input name='password' type='password' class='form-control' autofocus><button class='btn btn-primary w-100 mt-3'>Login</button><p class='text-body-secondary mt-3'>Set EXEC_AGENT_WEB_PASSWORD to require authentication. With no password configured, submit once to continue.</p></form></div></div></div></div>"""


def _chat_ui() -> str:
    return """<h1>Chat</h1><div id='chat-output' class='rich-box mb-3'></div><form onsubmit='sendChat(event)' class='input-group'><textarea id='chat-message' class='form-control' rows='2' placeholder='Ask the assistant...'></textarea><button class='btn btn-primary'>Stream</button></form>"""


def _task_card(task: Any) -> str:
    return f"<div class='mb-2'><a href='/tasks/{task.task_id}'><strong>{task.task_id}</strong></a> <span class='badge text-bg-secondary'>{task.status}</span><br><span>{html.escape(task.description)}</span></div>"


def _task_row(task: Any) -> str:
    return f"<tr><td><a href='/tasks/{task.task_id}'>{task.task_id}</a></td><td>{html.escape(task.description)}</td><td>{task.autonomy_level}</td><td>{task.status}</td><td>{task.updated_at}</td></tr>"


def _tasks_ui(rows: str) -> str:
    return f"""<h1>Autonomous tasks</h1><form method='post' class='card card-body mb-3'><label class='form-label'>Goal</label><textarea name='description' class='form-control' required></textarea><label class='form-label mt-2'>Autonomy</label><select name='autonomy_level' class='form-select'><option>human_approved</option><option>suggest_only</option><option>autonomous_limited</option><option>autonomous_full</option></select><button class='btn btn-primary mt-3'>Run task</button></form><table class='table table-responsive'><thead><tr><th>ID</th><th>Description</th><th>Autonomy</th><th>Status</th><th>Updated</th></tr></thead><tbody>{rows}</tbody></table>"""


def _approval_card(key: str, data: dict[str, Any]) -> str:
    return f"<div class='card mb-2'><div class='card-body'><strong>{html.escape(key)}</strong><pre>{html.escape(json.dumps(data, indent=2))}</pre><button class='btn btn-success' onclick=decideApproval('{key}','approve')>Approve</button> <button class='btn btn-danger' onclick=decideApproval('{key}','reject')>Reject</button></div></div>"


def _files_ui(path: str, listing: str) -> str:
    return f"""<h1>Files</h1><form class='input-group mb-3'><input name='path' value='{html.escape(path)}' class='form-control'><button class='btn btn-outline-primary'>Browse allowed directory</button></form><ul class='list-group mb-3'>{listing}</ul><form action='/files/upload' method='post' enctype='multipart/form-data' class='card card-body'><label class='form-label'>Upload PDF/DOCX/image/text</label><input type='file' name='upload' class='form-control'><button class='btn btn-primary mt-3'>Upload</button></form>"""


def _ingest_ui(path: str) -> str:
    return f"""<h1>Ingest files into vector DB</h1><form method='post' class='card card-body'><label class='form-label'>File path</label><input name='path' value='{html.escape(path)}' class='form-control'><button class='btn btn-primary mt-3'>Ingest</button></form>"""


def _memory_ui(q: str, rows: str) -> str:
    return f"""<h1>Memory</h1><form class='input-group mb-3'><input name='q' value='{html.escape(q)}' class='form-control' placeholder='Search memory'><button class='btn btn-outline-primary'>Search</button></form><form method='post' class='card card-body mb-3'><textarea name='content' class='form-control' placeholder='New long-term memory'></textarea><input name='tags' class='form-control mt-2' placeholder='tags, comma-separated'><button class='btn btn-primary mt-2'>Add memory</button></form><table class='table'><thead><tr><th>ID</th><th>Content</th><th>Tags</th><th>Source</th></tr></thead><tbody>{rows}</tbody></table>"""


def _web_ui(q: str, results: str) -> str:
    status = ""
    try:
        status = f"<span class='badge text-bg-success'>{html.escape(str(health_check().get('status', 'ok')))}</span>"
    except Exception as exc:  # noqa: BLE001
        status = f"<span class='badge text-bg-warning'>{html.escape(str(exc))}</span>"
    return f"<h1>FastCRW web search {status}</h1><form class='input-group mb-3'><input name='q' value='{html.escape(q)}' class='form-control' placeholder='Search the web'><button class='btn btn-primary'>Search</button></form><ul class='list-group'>{results}</ul>"


def _shell_ui(output: str, rows: str) -> str:
    return f"""<h1>Allowed shell commands</h1><form method='post' class='card card-body mb-3'><input name='command' class='form-control font-monospace' placeholder='python --version'><input name='cwd' class='form-control mt-2' placeholder='Working directory (inside shell workspace)'><button class='btn btn-primary mt-2'>Run</button></form>{output}<h2>Command history</h2><table class='table table-sm'><tbody>{rows}</tbody></table>"""


def _history_rows() -> str:
    return "".join(f"<tr><td>{r.id}</td><td><code>{html.escape(r.command)}</code></td><td>{r.exit_code}</td><td>{r.started_at}</td></tr>" for r in shell_history(25))


def _validate_upload(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    allowed = {x.strip() for x in get_settings().allowed_upload_extensions.split(",") if x.strip()}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported upload type: {suffix}")


def _ingest_path(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(path)
    if suffix == ".docx":
        return ingest_docx(path)
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        VectorStore().add_documents(chunks, {"source": path.name, "file_type": suffix.lstrip(".")})
        return len(chunks)
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        VectorStore().add_documents([f"Uploaded image: {path.name}"], {"source": path.name, "file_type": suffix.lstrip(".")})
        return 1
    raise HTTPException(400, f"Unsupported ingest type: {suffix}")


app = create_app()
