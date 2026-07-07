"""Terminal chat interface primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from exec_agent.config import get_settings
from app.graph.builder import build_graph
from app.graph.state import AssistantState
from exec_agent.models.llm import generate_text, stream_text


class ChatAction(str, Enum):
    """Supported slash-command actions in the terminal chat."""

    EXIT = "exit"
    HELP = "help"
    CLEAR = "clear"


@dataclass(frozen=True)
class ParsedInput:
    """Parsed user input for the chat loop."""

    text: str
    action: ChatAction | None = None


@dataclass
class ChatMessage:
    """A single in-memory chat message."""

    role: str
    content: str


@dataclass
class ChatSession:
    """In-memory chat session state."""

    messages: list[ChatMessage] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        """Append a message to the current session."""

        self.messages.append(ChatMessage(role=role, content=content))

    def clear(self) -> None:
        """Remove all messages from the current session."""

        self.messages.clear()

    def render_prompt(self) -> str:
        """Render the current message list as a simple transcript prompt."""

        transcript = [f"{message.role.title()}: {message.content}" for message in self.messages]
        transcript.append("Assistant:")
        return "\n".join(transcript)


class TextStreamer(Protocol):
    """Callable protocol for model streaming backends."""

    def __call__(self, prompt: str) -> Iterable[str]:
        """Yield generated text chunks for a prompt."""


def parse_chat_input(raw_input: str) -> ParsedInput:
    """Parse terminal chat input into either plain text or a slash command."""

    text = raw_input.strip()
    command = text.lower()
    if command in {"/exit", "/quit"}:
        return ParsedInput(text=text, action=ChatAction.EXIT)
    if command == "/help":
        return ParsedInput(text=text, action=ChatAction.HELP)
    if command == "/clear":
        return ParsedInput(text=text, action=ChatAction.CLEAR)
    return ParsedInput(text=raw_input)


def default_streamer(prompt: str) -> Iterable[str]:
    """Stream model output, falling back to a single generated response if needed."""

    try:
        yield from stream_text(prompt)
    except TypeError:
        # Some test doubles or alternate backends may only support non-streaming calls.
        yield generate_text(prompt)


class TerminalChat:
    """Rich-powered terminal chat loop kept separate from the Typer command."""

    def __init__(
        self,
        *,
        console: Console | None = None,
        session: ChatSession | None = None,
        streamer: TextStreamer = default_streamer,
        input_reader: Callable[[str], str] | None = None,
    ) -> None:
        self.console = console or Console()
        self.session = session or ChatSession()
        self.streamer = streamer
        self.graph = build_graph()
        self.input_reader = input_reader or (lambda prompt: Prompt.ask(prompt, console=self.console))

    def run(self) -> None:
        """Run the interactive terminal chat until the user exits or input closes."""

        settings = get_settings()
        self.console.print(
            Panel.fit(
                "[bold green]Executive assistant chat[/bold green]\n"
                "Type [cyan]/help[/cyan] for commands or [cyan]/exit[/cyan] to quit.",
                title=settings.app_name,
                border_style="green",
            )
        )

        while True:
            try:
                raw_input = self.input_reader("[bold cyan]You[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Goodbye.[/dim]")
                break

            parsed = parse_chat_input(raw_input)
            if not parsed.text.strip():
                continue
            if parsed.action is ChatAction.EXIT:
                self.console.print("[dim]Goodbye.[/dim]")
                break
            if parsed.action is ChatAction.HELP:
                self._print_help()
                continue
            if parsed.action is ChatAction.CLEAR:
                self.session.clear()
                self.console.clear()
                self.console.print("[dim]Session cleared.[/dim]")
                continue

            self._handle_user_message(raw_input)

    def _handle_user_message(self, user_text: str) -> None:
        response_chunks: list[str] = []

        def streaming_printer(prompt: str) -> Iterable[str]:
            for chunk in self.streamer(prompt):
                response_chunks.append(chunk)
                self.console.print(chunk, end="", soft_wrap=True)
                yield chunk

        graph_state: AssistantState = {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self.session.messages
            ],
            "user_input": user_text,
            "streamer": streaming_printer,
        }

        self.console.print("[bold magenta]Assistant[/bold magenta]: ", end="")
        result = self.graph.invoke(graph_state)
        self.console.print()

        self.session.messages = [
            ChatMessage(role=message["role"], content=message["content"])
            for message in result.get("messages", [])
        ]

    def _print_help(self) -> None:
        self.console.print(
            Panel(
                Markdown(
                    """
**Commands**

- `/help` - Show this help message.
- `/clear` - Clear the in-memory session transcript.
- `/exit` or `/quit` - Exit the chat.
                    """.strip()
                ),
                title="Chat help",
                border_style="cyan",
            )
        )
