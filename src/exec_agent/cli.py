"""Command-line interface for the executive assistant scaffold."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from exec_agent.config import RUNTIME_PROFILES, get_settings
from exec_agent.chat import TerminalChat
from exec_agent.models.llm import generate_text
from exec_agent.sessions import ChatSessionStore, PersistedChatSession
from app.tools.pdf import ingest_pdf as ingest_pdf_file
from app.tools.docx import ingest_docx as ingest_docx_file
from app.tools.image import ask_image as ask_image_file
from app.tools.image import describe_image as describe_image_file
from app.tools import web_fastcrw
from app.memory.long_term import LongTermMemory, LongTermMemoryStore
from app.memory.vector_store import VectorSearchResult, VectorStore

app = typer.Typer(
    name="exec-agent",
    help="A local-first terminal AI executive assistant scaffold.",
    no_args_is_help=True,
)
console = Console(width=140)
memory_app = typer.Typer(help="Manage SQLite-backed long-term memories.")
rag_app = typer.Typer(help="Search local vector RAG context.")
ingest_app = typer.Typer(help="Ingest documents into local vector RAG context.")
image_app = typer.Typer(help="Analyze images with local Hugging Face vision-language models.")
web_app = typer.Typer(help="Use self-hosted FastCRW for web research.")
sessions_app = typer.Typer(help="Manage SQLite-backed persistent chat sessions.")
task_app = typer.Typer(help="Run executive-assistant task workflows without modifying external systems.")
profile_app = typer.Typer(help="List and activate runtime profiles.")
app.add_typer(memory_app, name="memory")
app.add_typer(rag_app, name="rag")
app.add_typer(ingest_app, name="ingest")
app.add_typer(image_app, name="image")
app.add_typer(web_app, name="web")
app.add_typer(sessions_app, name="sessions")
app.add_typer(task_app, name="task")
app.add_typer(profile_app, name="profile")


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
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            console.print(f"[red]File not found: {path}[/red]")
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


def _generate_workflow(title: str, prompt: str) -> None:
    _render_workflow_result(title, generate_text(prompt).strip())


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
        hitl=hitl or get_settings().hitl,
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
    table.add_row("image_caption_model_id", settings.image_caption_model_id)
    table.add_row("image_qa_model_id", settings.image_qa_model_id)
    table.add_row("device", settings.device)
    table.add_row("max_tokens", str(settings.max_tokens))
    table.add_row("temperature", str(settings.temperature))
    table.add_row("runtime_profile", settings.runtime_profile)
    table.add_row("hitl", str(settings.hitl))
    table.add_row("web_enabled", str(settings.web_enabled))
    table.add_row("fastcrw_enabled", str(settings.fastcrw_enabled))
    table.add_row("fastcrw_crawl_requires_approval", str(settings.fastcrw_crawl_requires_approval))
    table.add_row("fastcrw_base_url", settings.fastcrw_base_url)
    table.add_row("fastcrw_api_prefix", settings.fastcrw_api_prefix)
    table.add_row("fastcrw_api_key", "set" if settings.fastcrw_api_key else "not set")
    table.add_row("fastcrw_timeout_seconds", str(settings.fastcrw_timeout_seconds))
    table.add_row("fastcrw_max_results", str(settings.fastcrw_max_results))
    table.add_row("fastcrw_enable_scrape", str(settings.fastcrw_enable_scrape))
    table.add_row("fastcrw_enable_crawl", str(settings.fastcrw_enable_crawl))
    console.print(table)


@app.command("model-test")
def model_test(prompt: str) -> None:
    """Generate a sample response with the configured local model."""

    console.print(generate_text(prompt))


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
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Ingested {chunk_count} PDF chunks from {path}.[/green]")


@ingest_app.command("docx")
def ingest_docx(path: str) -> None:
    """Extract, chunk, and store a DOCX in the local vector database."""

    try:
        chunk_count = ingest_docx_file(path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
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
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
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
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
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

    answer = generate_text(_build_document_qa_prompt(question, results)).strip()
    console.print(answer)
    references = _format_references(results)
    if references:
        console.print(f"[dim]References: {references}[/dim]")
