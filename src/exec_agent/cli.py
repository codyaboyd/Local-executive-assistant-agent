"""Command-line interface for the executive assistant scaffold."""

import typer
from rich.console import Console
from rich.table import Table

from exec_agent.config import get_settings
from exec_agent.chat import TerminalChat
from exec_agent.models.llm import generate_text
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
app.add_typer(memory_app, name="memory")
app.add_typer(rag_app, name="rag")


@app.command()
def chat(
    hitl: bool = typer.Option(False, "--hitl", help="Require human approval for tool calls and memory writes."),
) -> None:
    """Start the interactive terminal chat interface."""

    TerminalChat(console=console, hitl=hitl or get_settings().hitl).run()


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
    table.add_row("device", settings.device)
    table.add_row("max_tokens", str(settings.max_tokens))
    table.add_row("temperature", str(settings.temperature))
    table.add_row("hitl", str(settings.hitl))
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
