"""Command-line interface for the executive assistant scaffold."""

import typer
from rich.console import Console
from rich.table import Table

from exec_agent.config import get_settings
from exec_agent.chat import TerminalChat
from exec_agent.models.llm import generate_text

app = typer.Typer(
    name="exec-agent",
    help="A local-first terminal AI executive assistant scaffold.",
    no_args_is_help=True,
)
console = Console()


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
