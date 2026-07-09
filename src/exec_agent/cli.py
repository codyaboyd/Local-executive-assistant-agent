"""Command-line interface for the executive assistant."""

from pathlib import Path
import inspect

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.models.registry import ModelRole, REGISTRY, benchmark_selection, pull_model, select_model
from exec_agent.config import RUNTIME_PROFILES, get_settings
from exec_agent.logging import configure_logging
from exec_agent.safety import UserFacingError, validate_local_file
from exec_agent.chat import TerminalChat
from exec_agent.models.llm import generate_text
from exec_agent.sessions import ChatSessionStore, PersistedChatSession
from exec_agent.tasks import AutonomousTaskRunner, TaskStore
from app.tools.pdf import ingest_pdf as ingest_pdf_file
from app.tools.docx import ingest_docx as ingest_docx_file
from app.tools.image import ask_image as ask_image_file
from app.tools.image import describe_image as describe_image_file
from app.tools import web_fastcrw
from app.tools import filesystem as fs_tools
from app.tools import shell as shell_tools
from app.memory.long_term import LongTermMemory, LongTermMemoryStore
from app.memory.vector_store import VectorSearchResult, VectorStore
from app.evals import render_results_table, run_evals

app = typer.Typer(
    name="exec-agent",
    help="A local-first terminal AI executive assistant for Linux.",
    no_args_is_help=True,
)
console = Console(width=140)
configure_logging(get_settings().log_level, structured=get_settings().structured_logging)
memory_app = typer.Typer(help="Manage SQLite-backed long-term memories.")
rag_app = typer.Typer(help="Search local vector RAG context.")
ingest_app = typer.Typer(help="Ingest documents into local vector RAG context.")
image_app = typer.Typer(help="Analyze images with local Hugging Face vision-language models.")
web_app = typer.Typer(help="Use self-hosted FastCRW for web research.")
sessions_app = typer.Typer(help="Manage SQLite-backed persistent chat sessions.")
task_app = typer.Typer(help="Run executive-assistant task workflows without modifying external systems.")
profile_app = typer.Typer(help="List and activate runtime profiles.")
eval_app = typer.Typer(help="Run offline testing and evaluation tasks.")
models_app = typer.Typer(help="Inspect, pull, benchmark, and pin role-specific models.")
fs_app = typer.Typer(help="Controlled filesystem access within configured allowed directories.")
shell_app = typer.Typer(help="Run safe allowlisted shell commands in the configured workspace.")
app.add_typer(memory_app, name="memory")
app.add_typer(rag_app, name="rag")
app.add_typer(ingest_app, name="ingest")
app.add_typer(image_app, name="image")
app.add_typer(web_app, name="web")
app.add_typer(sessions_app, name="sessions")
app.add_typer(task_app, name="task")
app.add_typer(profile_app, name="profile")
app.add_typer(eval_app, name="eval")
app.add_typer(models_app, name="models")
app.add_typer(fs_app, name="fs")
app.add_typer(shell_app, name="shell")


@shell_app.command("run")
def shell_run(
    command: str = typer.Argument(..., help="Command to run, quoted as one string."),
    cwd: str | None = typer.Option(None, "--cwd", help="Working directory inside EXEC_AGENT_SHELL_WORKDIR."),
    timeout: float | None = typer.Option(None, "--timeout", help="Override timeout in seconds."),
) -> None:
    """Run a safe command and persist its result in shell history."""

    try:
        result = shell_tools.run_command(command, cwd=cwd, timeout=timeout)
    except (UserFacingError, OSError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[bold]Exit code:[/bold] {result.exit_code}")
    console.print(f"[bold]Duration:[/bold] {result.duration_seconds:.2f}s")
    console.print(f"[bold]Working directory:[/bold] {result.cwd}")
    raise typer.Exit(code=0 if result.exit_code == 0 else result.exit_code)


@shell_app.command("history")
def shell_history(limit: int = typer.Option(50, "--limit", "-n", min=1, help="Number of commands to show.")) -> None:
    """Show recent shell command history."""

    table = Table(title="Shell Command History")
    for column in ["ID", "Exit", "Duration", "CWD", "Command", "Finished"]:
        table.add_column(column)
    for item in shell_tools.history(limit=limit):
        table.add_row(
            str(item.id),
            str(item.exit_code),
            f"{item.duration_seconds:.2f}s",
            item.cwd,
            item.command,
            item.finished_at[:19],
        )
    console.print(table)


@fs_app.command("list")
def fs_list(path: str = typer.Argument("./workspace", help="Allowed directory to list.")) -> None:
    """List files in an allowed directory."""

    try:
        for entry in fs_tools.list_dir(path):
            console.print(entry)
    except (FileNotFoundError, UserFacingError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@fs_app.command("read")
def fs_read(path: str = typer.Argument(..., help="Allowed file to read.")) -> None:
    """Read a file from an allowed directory."""

    try:
        console.print(fs_tools.read_file(path))
    except (FileNotFoundError, UserFacingError, OSError, UnicodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@fs_app.command("search")
def fs_search(query: str, root: str = typer.Argument("./workspace", help="Allowed root to search.")) -> None:
    """Search files under an allowed directory for a keyword."""

    try:
        for match in fs_tools.search_files(query, root):
            console.print(match)
    except (FileNotFoundError, UserFacingError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@models_app.command("list")
def models_list() -> None:
    """List curated open-source models by role."""

    settings = get_settings()
    table = Table(title="Curated Model Registry")
    for column in ["Selected", "Role", "Model", "VRAM", "CPU", "Quant", "Preset", "Strengths"]:
        table.add_column(column)
    selected = {role: select_model(role, settings).model_id for role in ModelRole}
    for spec in REGISTRY:
        table.add_row("*" if selected.get(spec.role) == spec.model_id else "", spec.role.value, spec.model_id, f"{spec.recommended_vram_gb:g}GB", str(spec.cpu_friendly), spec.quantization, ",".join(spec.default_presets), ", ".join(spec.strengths))
    console.print(table)


@models_app.command("status")
def models_status() -> None:
    """Show effective role-to-model mapping and preset budget."""

    settings = get_settings()
    table = Table(title="Model Status")
    table.add_column("Role", style="cyan")
    table.add_column("Model")
    table.add_column("Backend")
    table.add_column("Budget")
    table.add_column("Auto Pull")
    for role in ModelRole:
        spec = select_model(role, settings)
        table.add_row(role.value, spec.model_id, spec.backend, f"preset={settings.model_preset}, max_vram={settings.max_vram_gb}GB", str(settings.model_auto_pull))
    console.print(table)


def _warn_if_oversized(spec) -> None:
    if spec.recommended_vram_gb > get_settings().max_vram_gb:
        console.print(f"[yellow]Warning: {spec.model_id} recommends {spec.recommended_vram_gb:g}GB VRAM, above configured budget {get_settings().max_vram_gb}GB.[/yellow]")


@models_app.command("pull-defaults")
def models_pull_defaults() -> None:
    """Pull recommended default models for the active preset."""

    seen: set[str] = set()
    for role in ModelRole:
        spec = select_model(role)
        if spec.model_id in seen:
            continue
        seen.add(spec.model_id)
        _warn_if_oversized(spec)
        console.print(f"Pulling {role.value}: {spec.model_id}")
        console.print(pull_model(spec.model_id))


@models_app.command("pull")
def models_pull(role: str = typer.Option(..., "--role", help="Model role to pull.")) -> None:
    """Pull the selected model for a role."""

    spec = select_model(ModelRole(role))
    _warn_if_oversized(spec)
    console.print(pull_model(spec.model_id))


@models_app.command("set-role")
def models_set_role(role: str, model_id: str) -> None:
    """Persist a role-specific model override in .env."""

    env_names = {
        ModelRole.GENERAL_REASONING: "EXEC_AGENT_GENERAL_MODEL_ID",
        ModelRole.CODING: "EXEC_AGENT_CODING_MODEL_ID",
        ModelRole.SUMMARIZATION: "EXEC_AGENT_SUMMARY_MODEL_ID",
        ModelRole.DOCUMENT_QA: "EXEC_AGENT_DOCQA_MODEL_ID",
        ModelRole.WEB_RESEARCH: "EXEC_AGENT_RESEARCH_MODEL_ID",
        ModelRole.TOOL_CALLING: "EXEC_AGENT_TOOL_MODEL_ID",
        ModelRole.EMBEDDINGS: "EXEC_AGENT_EMBEDDING_MODEL_ID",
        ModelRole.VISION: "EXEC_AGENT_VISION_MODEL_ID",
    }
    key = env_names[ModelRole(role)]
    env_path = _env_file_path()
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    rendered = [line for line in lines if not line.startswith(f"{key}=")]
    rendered.append(f"{key}={model_id}")
    env_path.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8")
    get_settings.cache_clear()
    console.print(f"[green]Set {role} to {model_id}.[/green]")


@models_app.command("benchmark")
def models_benchmark() -> None:
    """Benchmark role model selection without heavyweight generation."""

    table = Table(title="Model Selection Benchmark")
    table.add_column("Role")
    table.add_column("Model")
    table.add_column("Selection ms")
    for row in benchmark_selection():
        table.add_row(row["role"], row["model_id"], row["selection_ms"])
    console.print(table)


@eval_app.command("run")
def eval_run() -> None:
    """Run CI-safe offline evals with mocked tools."""

    results = run_evals()
    console.print(render_results_table(results))
    if not all(result.passed for result in results):
        raise typer.Exit(code=1)


def _env_file_path() -> Path:
    return Path(".env")


def _write_runtime_profile_to_env(profile: str) -> None:
    env_path = _env_file_path()
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated = False
    rendered: list[str] = []
    for line in lines:
        if line.startswith("EXEC_AGENT_RUNTIME_PROFILE="):
            rendered.append(f"EXEC_AGENT_RUNTIME_PROFILE={profile}")
            updated = True
        else:
            rendered.append(line)
    if not updated:
        rendered.append(f"EXEC_AGENT_RUNTIME_PROFILE={profile}")
    env_path.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8")
    get_settings.cache_clear()


@profile_app.command("list")
def profile_list() -> None:
    """List available runtime profiles and their controls."""

    active_profile = get_settings().runtime_profile
    table = Table(title="Runtime Profiles")
    table.add_column("Active", style="green", no_wrap=True)
    table.add_column("Profile", style="cyan", no_wrap=True)
    table.add_column("Model", style="white")
    table.add_column("Device", style="magenta", no_wrap=True)
    table.add_column("Web", style="yellow", no_wrap=True)
    table.add_column("HITL", style="yellow", no_wrap=True)
    table.add_column("Vector DB", style="blue")
    table.add_column("Log", style="dim", no_wrap=True)
    for name, profile in RUNTIME_PROFILES.items():
        vector_path = get_settings().expanded_data_dir / profile.vector_db_subdir
        table.add_row(
            "*" if name == active_profile else "",
            name,
            profile.model_id,
            profile.device,
            str(profile.web_enabled),
            str(profile.hitl),
            str(vector_path),
            profile.log_level,
        )
    console.print(table)


@profile_app.command("use")
def profile_use(profile: str = typer.Argument(..., help="Runtime profile to activate.")) -> None:
    """Persist the active runtime profile in the local .env file."""

    if profile not in RUNTIME_PROFILES:
        valid = ", ".join(RUNTIME_PROFILES)
        console.print(f"[red]Unknown profile {profile!r}. Choose one of: {valid}[/red]")
        raise typer.Exit(code=1)
    _write_runtime_profile_to_env(profile)
    settings = get_settings()
    console.print(f"[green]Activated runtime profile:[/green] {settings.runtime_profile}")
    console.print(f"Model: {settings.model_id}; device: {settings.device}; web: {settings.web_enabled}; HITL: {settings.hitl}")
    console.print(f"Vector DB: {settings.expanded_vector_db_path}; log level: {settings.log_level}")


def _read_task_input(text: str | None = None, path: str | None = None) -> str:
    """Read workflow input from an inline string, a file path, or stdin."""

    if text and path:
        raise typer.BadParameter("Use either --text or --file, not both.")
    if path:
        try:
            return validate_local_file(path, allowed_extensions={".txt", ".md"}, purpose="input file").read_text(encoding="utf-8")
        except (FileNotFoundError, UserFacingError, OSError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    if text:
        return text
    stdin_text = typer.get_text_stream("stdin").read()
    if stdin_text.strip():
        return stdin_text
    console.print("[red]Provide input with --text, --file, or stdin.[/red]")
    raise typer.Exit(code=1)


def _render_workflow_result(title: str, body: str) -> None:
    console.print(Panel(body.strip() or "[dim](no output)[/dim]", title=title, border_style="cyan", expand=False))


def _generate_text_for_role(prompt: str, role: str) -> str:
    """Call generate_text with role support while preserving test doubles with the old signature."""

    if "role" in inspect.signature(generate_text).parameters:
        return generate_text(prompt, role=role)
    return generate_text(prompt)


def _generate_workflow(title: str, prompt: str, role: str = ModelRole.GENERAL_REASONING.value) -> None:
    try:
        output = _generate_text_for_role(prompt, role).strip()
    except Exception as exc:  # noqa: BLE001 - CLI boundary returns clear user-facing errors.
        console.print(f"[red]Could not generate workflow output safely: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    _render_workflow_result(title, output)


def _web_research_context(topic: str, max_results: int) -> str:
    try:
        results = web_fastcrw.search_web(topic, max_results=max_results)
    except web_fastcrw.FastCRWError as exc:
        return f"Web search unavailable: {exc}"
    lines = []
    for index, result in enumerate(results, start=1):
        title = str(result.get("title", "Untitled"))
        url = str(result.get("url", ""))
        snippet = str(result.get("snippet") or result.get("description") or result.get("content") or "")
        lines.append(f"[{index}] {title}\nURL: {url}\nSnippet: {snippet}".strip())
    return "\n\n".join(lines)


def _rag_context(query: str, k: int) -> str:
    results = VectorStore().similarity_search(query, k=k)
    if not results:
        return "No local document context found."
    lines = []
    for index, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown")
        lines.append(f"[{index}] Source: {source}\n{result.content}")
    return "\n\n".join(lines)


@task_app.command("run")
def task_run(
    description: str = typer.Argument(..., help="Task description to execute."),
    autonomous: bool = typer.Option(False, "--autonomous", help="Run with autonomous_full for this task."),
) -> None:
    """Run a persisted autonomous task loop with progress streaming."""

    settings = get_settings()
    level = "autonomous_full" if autonomous else settings.autonomy_level
    runner = AutonomousTaskRunner(progress=lambda message: console.print(f"[cyan]{message}[/cyan]"))
    task = runner.run(description, autonomy_level=level)
    console.print(f"Task ID: {task.task_id}")
    console.print(f"Status: {task.status}")
    if task.final_summary:
        console.print(Panel(task.final_summary, title="Final Summary", border_style="green"))
    if task.error:
        console.print(f"[yellow]Blocked/Error:[/yellow] {task.error}")


@task_app.command("status")
def task_status(task_id: str | None = typer.Argument(None, help="Task ID; defaults to latest task.")) -> None:
    """Show task status and recorded steps."""

    store = TaskStore()
    task = store.get(task_id) if task_id else store.latest()
    if task is None:
        console.print("[yellow]No task found.[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[bold]Task:[/bold] {task.task_id} ({task.status})")
    console.print(f"[bold]Goal:[/bold] {task.description}")
    for step in store.steps(task.task_id):
        console.print(f"{step.step_number}. {step.phase} / {step.tool_name}: {step.result or step.error}")


@task_app.command("cancel")
def task_cancel(task_id: str) -> None:
    """Cancel a persisted task by ID."""

    if not TaskStore().cancel(task_id):
        console.print(f"[red]Task {task_id!r} not found.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Cancelled task {task_id}.[/green]")


@task_app.command("history")
def task_history() -> None:
    """List recent autonomous task runs."""

    table = Table(title="Task History")
    for column in ["Task ID", "Status", "Autonomy", "Description", "Updated"]:
        table.add_column(column)
    for task in TaskStore().list():
        table.add_row(task.task_id, task.status, task.autonomy_level, task.description, task.updated_at[:19])
    console.print(table)


@task_app.command("summarize-notes")
def task_summarize_notes(
    text: str | None = typer.Option(None, "--text", help="Notes text to summarize."),
    file: str | None = typer.Option(None, "--file", "-f", help="Path to a notes file to summarize."),
) -> None:
    """Summarize notes into concise executive-ready bullets."""

    notes = _read_task_input(text, file)
    _generate_workflow(
        "Notes Summary",
        "Summarize these notes for an executive. Return clean terminal output with sections: Key Points, Decisions, Risks, and Next Steps.\n\n"
        f"Notes:\n{notes}\n",
        role=ModelRole.SUMMARIZATION.value,
    )


@task_app.command("draft-email")
def task_draft_email(
    prompt: str = typer.Argument(..., help="What the email should accomplish."),
    tone: str = typer.Option("professional", "--tone", help="Desired email tone."),
    context: str | None = typer.Option(None, "--context", help="Additional context for the draft."),
) -> None:
    """Draft email text only; does not send email or touch external systems."""

    _generate_workflow(
        "Email Draft (not sent)",
        "Draft email text only. Do not imply the email was sent. Return Subject and Body. "
        f"Tone: {tone}. Goal: {prompt}. Context: {context or 'None'}",
    )


@task_app.command("meeting-brief")
def task_meeting_brief(
    topic: str = typer.Argument(..., help="Meeting topic or purpose."),
    attendees: list[str] = typer.Option(None, "--attendee", "-a", help="Attendee name; may be used multiple times."),
    context_file: str | None = typer.Option(None, "--file", "-f", help="Optional context file."),
) -> None:
    """Create a meeting brief from supplied context."""

    context = _read_task_input(path=context_file) if context_file else ""
    attendee_text = ", ".join(attendees or []) or "Not specified"
    _generate_workflow(
        "Meeting Brief",
        "Create an executive meeting brief with sections: Objective, Attendees, Background, Suggested Agenda, Questions to Ask, and Prep Checklist.\n"
        f"Topic: {topic}\nAttendees: {attendee_text}\nContext:\n{context}",
    )


@task_app.command("research-topic")
def task_research_topic(topic: str, max_results: int = typer.Option(5, "--max-results", "-n", min=1)) -> None:
    """Research a topic using configured web search without scraping or modifying external systems."""

    web_context = _web_research_context(topic, max_results)
    _generate_workflow(
        "Topic Research",
        "Prepare a concise executive research memo using the web search results below. Include Overview, Key Findings, Watchouts, and Sources.\n\n"
        f"Topic: {topic}\n\nWeb results:\n{web_context}",
        role=ModelRole.WEB_RESEARCH.value,
    )


@task_app.command("action-items")
def task_action_items(
    text: str | None = typer.Option(None, "--text", help="Document text to inspect."),
    file: str | None = typer.Option(None, "--file", "-f", help="Path to a document text file."),
) -> None:
    """Extract action items from document text."""

    document = _read_task_input(text, file)
    _generate_workflow(
        "Action Items",
        "Extract action items from this document. Return a terminal-friendly table-like list with Owner, Action, Due Date, Priority, and Evidence. Use 'Unassigned' or 'Not specified' where missing.\n\n"
        f"Document:\n{document}",
        role=ModelRole.DOCUMENT_QA.value,
    )


@task_app.command("daily-briefing")
def task_daily_briefing(
    focus: str = typer.Option("today", "--focus", help="Briefing focus or query."),
    k: int = typer.Option(5, "--k", min=1, help="Number of local document chunks to include."),
    max_results: int = typer.Option(5, "--max-results", "-n", min=1, help="Number of web search results to include."),
) -> None:
    """Create a daily briefing from local memory, local docs, and configured web search."""

    memory_store = LongTermMemoryStore()
    memories = memory_store.search(focus, limit=10) or memory_store.list()[:10]
    memory_context = "\n".join(f"- {memory.content} (source={memory.source})" for memory in memories) or "No memory context found."
    docs_context = _rag_context(focus, k)
    web_context = _web_research_context(focus, max_results)
    _generate_workflow(
        "Daily Briefing",
        "Create a daily executive briefing from memory, local documents, and web results. Include Priorities, Schedule/Context, Decisions Needed, Risks, Opportunities, and Source Notes. Do not claim to update calendars, send messages, or modify external systems.\n\n"
        f"Focus: {focus}\n\nMemory:\n{memory_context}\n\nDocuments:\n{docs_context}\n\nWeb:\n{web_context}",
        role=ModelRole.WEB_RESEARCH.value,
    )


@app.command()
def chat(
    hitl: bool = typer.Option(False, "--hitl", help="Require human approval for tool calls and memory writes."),
    debug: bool = typer.Option(False, "--debug", help="Show graph node names and state transitions while streaming."),
    session: str | None = typer.Option(None, "--session", help="Name of a persistent SQLite-backed chat session."),
) -> None:
    """Start the interactive terminal chat interface."""

    store = ChatSessionStore() if session else None
    loaded_session = None
    summary = ""
    if store is not None and session is not None:
        loaded_session, summary = store.load_chat_session(session)
    TerminalChat(
        console=console,
        session=loaded_session,
        session_name=session,
        session_summary=summary,
        session_store=store,
        hitl=hitl or get_settings().actions_hitl,
        debug=debug,
    ).run()


def _sessions_store() -> ChatSessionStore:
    return ChatSessionStore()


def _render_sessions_table(sessions: list[PersistedChatSession]) -> Table:
    table = Table(title="Chat Sessions")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Messages", style="white", justify="right")
    table.add_column("Summary", style="green")
    table.add_column("Created", style="dim", width=20)
    table.add_column("Updated", style="dim", width=20)
    for session in sessions:
        summary = session.summary.replace("\n", " ")
        if len(summary) > 80:
            summary = f"{summary[:77]}..."
        table.add_row(session.name, str(len(session.messages)), summary, session.created_at[:19], session.updated_at[:19])
    return table


@sessions_app.command("list")
def sessions_list() -> None:
    """List persistent chat sessions."""

    console.print(_render_sessions_table(_sessions_store().list()))


@sessions_app.command("show")
def sessions_show(name: str) -> None:
    """Show a persistent chat session transcript and summary."""

    session = _sessions_store().get(name)
    if session is None:
        console.print(f"[yellow]Session {name!r} not found.[/yellow]")
        raise typer.Exit(code=1)
    console.print(f"[bold]Session:[/bold] {session.name}")
    console.print(f"[bold]Created:[/bold] {session.created_at}")
    console.print(f"[bold]Updated:[/bold] {session.updated_at}")
    console.print("[bold]Summary:[/bold]")
    console.print(session.summary or "[dim](empty)[/dim]")
    console.print("[bold]Messages:[/bold]")
    for message in session.messages:
        console.print(f"[cyan]{message.role.title()}[/cyan]: {message.content}")


@sessions_app.command("delete")
def sessions_delete(name: str) -> None:
    """Delete a persistent chat session."""

    if _sessions_store().delete(name):
        console.print(f"[green]Deleted session {name}.[/green]")
    else:
        console.print(f"[yellow]Session {name!r} not found.[/yellow]")
        raise typer.Exit(code=1)


@app.command()
def config() -> None:
    """Show effective configuration values."""

    settings = get_settings()
    table = Table(title="Executive Assistant Configuration")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("app_name", settings.app_name)
    table.add_row("environment", settings.environment)
    table.add_row("log_level", settings.log_level)
    table.add_row("data_dir", str(settings.expanded_data_dir))
    table.add_row("vector_db_path", str(settings.expanded_vector_db_path))
    table.add_row("model_id", settings.model_id)
    table.add_row("model_preset", settings.model_preset)
    table.add_row("model_auto_pull", str(settings.model_auto_pull))
    table.add_row("max_vram_gb", str(settings.max_vram_gb))
    table.add_row("general_model_id", settings.general_model_id)
    table.add_row("coding_model_id", settings.coding_model_id)
    table.add_row("summary_model_id", settings.summary_model_id)
    table.add_row("docqa_model_id", settings.docqa_model_id)
    table.add_row("research_model_id", settings.research_model_id)
    table.add_row("tool_model_id", settings.tool_model_id)
    table.add_row("embedding_model_id", settings.embedding_model_id)
    table.add_row("vision_model_id", settings.vision_model_id)
    table.add_row("image_caption_model_id", settings.image_caption_model_id)
    table.add_row("image_qa_model_id", settings.image_qa_model_id)
    table.add_row("device", settings.device)
    table.add_row("max_tokens", str(settings.max_tokens))
    table.add_row("temperature", str(settings.temperature))
    table.add_row("runtime_profile", settings.runtime_profile)
    table.add_row("hitl", str(settings.hitl))
    table.add_row("actions_hitl", str(settings.actions_hitl))
    table.add_row("autonomy_level", settings.autonomy_level)
    table.add_row("max_autonomous_steps", str(settings.max_autonomous_steps))
    table.add_row("require_approval_for_dangerous_commands", str(settings.require_approval_for_dangerous_commands))
    table.add_row("task_timeout_seconds", str(settings.task_timeout_seconds))
    table.add_row("web_enabled", str(settings.web_enabled))
    table.add_row("local_only", str(settings.local_only))
    table.add_row("fastcrw_enabled", str(settings.fastcrw_enabled))
    table.add_row("fastcrw_crawl_requires_approval", str(settings.fastcrw_crawl_requires_approval))
    table.add_row("fastcrw_base_url", settings.fastcrw_base_url)
    table.add_row("fastcrw_api_prefix", settings.fastcrw_api_prefix)
    table.add_row("fastcrw_api_key", "set" if settings.fastcrw_api_key else "not set")
    table.add_row("fastcrw_timeout_seconds", str(settings.fastcrw_timeout_seconds))
    table.add_row("model_timeout_seconds", str(settings.model_timeout_seconds))
    table.add_row("max_upload_bytes", str(settings.max_upload_bytes))
    table.add_row("allowed_upload_extensions", settings.allowed_upload_extensions)
    table.add_row("structured_logging", str(settings.structured_logging))
    table.add_row("fastcrw_max_results", str(settings.fastcrw_max_results))
    table.add_row("fastcrw_enable_scrape", str(settings.fastcrw_enable_scrape))
    table.add_row("fastcrw_enable_crawl", str(settings.fastcrw_enable_crawl))
    table.add_row("allowed_dirs", settings.allowed_dirs)
    table.add_row("readonly_dirs", settings.readonly_dirs)
    table.add_row("blocked_paths", settings.blocked_paths)
    table.add_row("max_file_size_mb", str(settings.max_file_size_mb))
    console.print(table)


@app.command("model-test")
def model_test(prompt: str) -> None:
    """Generate a sample response with the configured local model."""

    try:
        console.print(generate_text(prompt))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Model test failed safely: {exc}[/red]")
        raise typer.Exit(code=1) from exc


def _memory_store() -> LongTermMemoryStore:
    return LongTermMemoryStore()


def _confirm_memory_write(action: str, preview: str) -> None:
    if get_settings().hitl and not typer.confirm(f"Approve long-term memory {action}: {preview!r}?", default=False):
        console.print(f"[yellow]Memory {action} rejected.[/yellow]")
        raise typer.Exit(code=1)


def _render_memory_table(memories: list[LongTermMemory], *, title: str) -> Table:
    table = Table(title=title)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Content", style="white")
    table.add_column("Tags", style="green")
    table.add_column("Source", style="magenta")
    table.add_column("Updated", style="dim", width=20)
    for memory in memories:
        table.add_row(str(memory.id), memory.content, ", ".join(memory.tags), memory.source, memory.updated_at[:19])
    return table


@memory_app.command("add")
def memory_add(
    content: str,
    tags: list[str] = typer.Option(None, "--tag", "-t", help="Tag to attach to the memory. May be used multiple times."),
    source: str = typer.Option("manual", "--source", help="Where this memory came from."),
) -> None:
    """Add a long-term memory."""

    normalized_tags = tags or []
    _confirm_memory_write("write", content)

    memory = _memory_store().add(content, normalized_tags, source)
    console.print(f"[green]Added memory {memory.id}.[/green]")


@memory_app.command("list")
def memory_list() -> None:
    """List all long-term memories."""

    memories = _memory_store().list()
    console.print(_render_memory_table(memories, title="Long-Term Memories"))


@memory_app.command("search")
def memory_search(query: str) -> None:
    """Search long-term memories by content, tag, or source."""

    memories = _memory_store().search(query)
    console.print(_render_memory_table(memories, title=f"Long-Term Memory Search: {query}"))


@memory_app.command("delete")
def memory_delete(memory_id: int) -> None:
    """Delete a long-term memory by id."""

    _confirm_memory_write("delete", str(memory_id))
    deleted = _memory_store().delete(memory_id)
    if not deleted:
        console.print(f"[red]Memory {memory_id} not found.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Deleted memory {memory_id}.[/green]")


def _render_vector_table(results: list[VectorSearchResult], *, title: str) -> Table:
    table = Table(title=title)
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Content", style="white")
    table.add_column("Source", style="magenta")
    table.add_column("Distance", style="dim", no_wrap=True)
    for index, result in enumerate(results, start=1):
        distance = "" if result.distance is None else f"{result.distance:.4f}"
        table.add_row(str(index), result.content, str(result.metadata.get("source", "")), distance)
    return table


@rag_app.command("search")
def rag_search(query: str, k: int = typer.Option(5, "--k", "-k", min=1, help="Number of similar chunks to return.")) -> None:
    """Search local vector RAG context by semantic similarity."""

    results = VectorStore().similarity_search(query, k=k)
    console.print(_render_vector_table(results, title=f"RAG Search: {query}"))


@ingest_app.command("pdf")
def ingest_pdf(path: str) -> None:
    """Extract, chunk, and store a PDF in the local vector database."""

    try:
        chunk_count = ingest_pdf_file(path)
    except (FileNotFoundError, ValueError, RuntimeError, UserFacingError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Ingested {chunk_count} PDF chunks from {path}.[/green]")


@ingest_app.command("docx")
def ingest_docx(path: str) -> None:
    """Extract, chunk, and store a DOCX in the local vector database."""

    try:
        chunk_count = ingest_docx_file(path)
    except (FileNotFoundError, ValueError, RuntimeError, UserFacingError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Ingested {chunk_count} DOCX chunks from {path}.[/green]")


@image_app.command("describe")
def image_describe(
    path: str,
    model_id: str | None = typer.Option(None, "--model", help="Hugging Face image-to-text model to run locally."),
    device: str | None = typer.Option(None, "--device", help="Device for inference: auto, cpu, or cuda."),
) -> None:
    """Describe an image and store the description as searchable RAG context."""

    try:
        result = describe_image_file(path, model_id=model_id, device=_normalize_device_option(device))
    except (FileNotFoundError, ValueError, RuntimeError, UserFacingError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(result.text)
    console.print(f"[green]Stored image description in vector context for {path}.[/green]")


@image_app.command("ask")
def image_ask(
    path: str,
    question: str,
    model_id: str | None = typer.Option(None, "--model", help="Hugging Face visual-question-answering model to run locally."),
    device: str | None = typer.Option(None, "--device", help="Device for inference: auto, cpu, or cuda."),
) -> None:
    """Ask a question about an image and store the answer as searchable RAG context."""

    try:
        result = ask_image_file(path, question, model_id=model_id, device=_normalize_device_option(device))
    except (FileNotFoundError, ValueError, RuntimeError, UserFacingError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(result.text)
    console.print(f"[green]Stored image answer in vector context for {path}.[/green]")



def _handle_fastcrw_error(exc: Exception) -> None:
    console.print(f"[red]{exc}[/red]")
    raise typer.Exit(code=1) from exc


@web_app.command("health")
def web_health() -> None:
    """Check the configured self-hosted FastCRW server."""

    try:
        result = web_fastcrw.health_check()
    except web_fastcrw.FastCRWError as exc:
        _handle_fastcrw_error(exc)
    console.print(result)


@web_app.command("search")
def web_search(query: str, max_results: int = typer.Option(None, "--max-results", "-n", min=1, help="Maximum results to return.")) -> None:
    """Search with the configured self-hosted FastCRW server."""

    try:
        results = web_fastcrw.search_web(query, max_results=max_results or get_settings().fastcrw_max_results)
    except web_fastcrw.FastCRWError as exc:
        _handle_fastcrw_error(exc)
    table = Table(title=f"FastCRW Search: {query}")
    table.add_column("Rank", style="cyan", no_wrap=True)
    table.add_column("Title", style="white")
    table.add_column("URL", style="magenta")
    for index, result in enumerate(results, start=1):
        table.add_row(str(index), str(result.get("title", "Untitled")), str(result.get("url", "")))
    console.print(table)


@web_app.command("scrape")
def web_scrape(url: str) -> None:
    """Scrape a URL with self-hosted FastCRW and store page content in vector DB."""

    try:
        page = web_fastcrw.scrape_url(url)
    except web_fastcrw.FastCRWError as exc:
        _handle_fastcrw_error(exc)
    console.print(f"[green]Scraped and stored:[/green] {page.title} ({page.url})")


@web_app.command("crawl")
def web_crawl(url: str, limit: int = typer.Option(10, "--limit", min=1, help="Maximum pages to crawl.")) -> None:
    """Crawl a URL with self-hosted FastCRW and store page content in vector DB."""

    settings = get_settings()
    if settings.hitl or settings.fastcrw_crawl_requires_approval:
        domain = web_fastcrw.target_domain(url)
        edited_limit = typer.prompt("Max page limit", default=str(limit))
        limit = int(edited_limit)
        if not typer.confirm(f"Approve FastCRW crawl of {domain} with max {limit} pages?", default=False):
            console.print("[yellow]Crawl rejected.[/yellow]")
            raise typer.Exit(code=1)
    try:
        pages = web_fastcrw.crawl_url(url, limit=limit)
    except web_fastcrw.FastCRWError as exc:
        _handle_fastcrw_error(exc)
    console.print(f"[green]Crawled and stored {len(pages)} pages from {url}.[/green]")

def _normalize_device_option(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = device.lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise typer.BadParameter("device must be one of: auto, cpu, cuda")
    return normalized


def _format_references(results: list[VectorSearchResult]) -> str:
    refs: list[str] = []
    seen: set[tuple[str, object]] = set()
    for result in results:
        source = str(result.metadata.get("source", "unknown"))
        page = result.metadata.get("page")
        section_heading = result.metadata.get("section_heading")
        key = (source, page or section_heading)
        if key in seen:
            continue
        seen.add(key)
        if page:
            refs.append(f"{source} p. {page}")
        elif section_heading:
            refs.append(f"{source} section: {section_heading}")
        else:
            refs.append(source)
    return ", ".join(refs)


def _build_document_qa_prompt(question: str, results: list[VectorSearchResult]) -> str:
    context_lines: list[str] = []
    for index, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown")
        file_type = result.metadata.get("file_type", "document")
        page = result.metadata.get("page")
        section_heading = result.metadata.get("section_heading")
        location = f"page {page}" if page else f"section {section_heading}" if section_heading else "unknown location"
        context_lines.append(f"[{index}] Source: {source}, type {file_type}, {location}\n{result.content}")
    context = "\n\n".join(context_lines)
    return (
        "Answer the user's question using only the document context below. "
        "When the context supports an answer, include source filename and page or section references. "
        "If the context is insufficient, say you do not know based on the uploaded documents.\n\n"
        f"Document context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


@app.command("ask")
def ask(question: str, k: int = typer.Option(5, "--k", "-k", min=1, help="Number of document chunks to retrieve.")) -> None:
    """Ask a question about ingested PDF and DOCX documents."""

    results = [
        result
        for result in VectorStore().similarity_search(question, k=k)
        if result.metadata.get("file_type") in {"pdf", "docx"}
    ]
    if not results:
        console.print("[yellow]No relevant document context found. Ingest documents with: python -m exec_agent ingest pdf ./file.pdf or python -m exec_agent ingest docx ./file.docx[/yellow]")
        return

    answer = _generate_text_for_role(_build_document_qa_prompt(question, results), ModelRole.DOCUMENT_QA.value).strip()
    console.print(answer)
    references = _format_references(results)
    if references:
        console.print(f"[dim]References: {references}[/dim]")
