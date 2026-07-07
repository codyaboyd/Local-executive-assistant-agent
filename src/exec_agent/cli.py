"""Command-line interface for the executive assistant scaffold."""

import typer
from rich.console import Console
from rich.table import Table

from exec_agent.config import get_settings
from exec_agent.chat import TerminalChat
from exec_agent.models.llm import generate_text
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
app.add_typer(memory_app, name="memory")
app.add_typer(rag_app, name="rag")
app.add_typer(ingest_app, name="ingest")
app.add_typer(image_app, name="image")
app.add_typer(web_app, name="web")


@app.command()
def chat(
    hitl: bool = typer.Option(False, "--hitl", help="Require human approval for tool calls and memory writes."),
    debug: bool = typer.Option(False, "--debug", help="Show graph node names and state transitions while streaming."),
) -> None:
    """Start the interactive terminal chat interface."""

    TerminalChat(console=console, hitl=hitl or get_settings().hitl, debug=debug).run()


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
