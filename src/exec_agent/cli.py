"""Command-line interface for the executive assistant scaffold."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from exec_agent.config import get_settings

app = typer.Typer(
    name="exec-agent",
    help="A local-first terminal AI executive assistant scaffold.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def chat() -> None:
    """Start the placeholder chat interface."""

    settings = get_settings()
    console.print(
        Panel.fit(
            "[bold green]Executive assistant scaffold is ready.[/bold green]\n\n"
            "Full agent capabilities are not implemented yet.",
            title=settings.app_name,
            border_style="green",
        )
    )


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
    console.print(table)
