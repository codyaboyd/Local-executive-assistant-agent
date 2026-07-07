"""Terminal chat interface primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.prompt import Prompt

from exec_agent.config import get_settings
from app.graph.builder import build_graph
from app.graph.state import ApprovalDecision, AssistantState, ProposedAction
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
        hitl: bool | None = None,
        debug: bool = False,
    ) -> None:
        self.console = console or Console()
        self.session = session or ChatSession()
        self.streamer = streamer
        self.graph = build_graph()
        self.input_reader = input_reader or (lambda prompt: Prompt.ask(prompt, console=self.console))
        self.hitl = get_settings().hitl if hitl is None else hitl
        self.debug = debug
        self._progress_status: Status | None = None
        self._last_progress_node: str | None = None

    def run(self) -> None:
        """Run the interactive terminal chat until the user exits or input closes."""

        settings = get_settings()
        self.console.print(
            Panel.fit(
                "[bold green]Executive assistant chat[/bold green]\n"
                "Type [cyan]/help[/cyan] for commands or [cyan]/exit[/cyan] to quit."
                + ("\n[bold yellow]Human-in-the-loop approvals enabled.[/bold yellow]" if self.hitl else "")
                + ("\n[dim]Debug graph progress enabled.[/dim]" if self.debug else ""),
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

    def _handle_progress_event(self, event: dict[str, object]) -> None:
        """Render graph progress using Rich status spinners and optional debug panels."""

        message = str(event.get("message", "Working"))
        node = event.get("node")
        debug_suffix = f" [dim]({node})[/dim]" if self.debug and node else ""
        status_message = f"[bold blue]{message}[/bold blue]{debug_suffix}"
        if self._progress_status is None:
            self._progress_status = self.console.status(status_message, spinner="dots")
            self._progress_status.start()
        else:
            self._progress_status.update(status_message)
        if self.debug and node and node != self._last_progress_node:
            self.console.print(
                Panel.fit(
                    f"[cyan]stage[/cyan]: {event.get('stage', 'unknown')}\n[cyan]node[/cyan]: {node}",
                    title="Graph transition",
                    border_style="blue",
                )
            )
            self._last_progress_node = str(node)

    def _stop_progress(self) -> None:
        """Stop any active Rich progress spinner before streaming visible output or prompting."""

        if self._progress_status is not None:
            self._progress_status.stop()
            self._progress_status = None

    def _handle_user_message(self, user_text: str) -> None:
        response_chunks: list[str] = []

        def streaming_printer(prompt: str) -> Iterable[str]:
            for chunk in self.streamer(prompt):
                response_chunks.append(chunk)
                self._stop_progress()
                self.console.print(chunk, end="", soft_wrap=True)
                yield chunk

        settings = get_settings()
        graph_state: AssistantState = {
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self.session.messages
            ],
            "user_input": user_text,
            "streamer": streaming_printer,
            "hitl_enabled": self.hitl,
            "approval_handler": self._request_human_approval,
            "web_access_enabled": settings.web_enabled,
            "fastcrw_enabled": settings.fastcrw_enabled,
            "active_profile_allows_online_research": settings.runtime_profile in {"research-online", "test-hitl"},
            "debug": self.debug,
            "progress_callback": self._handle_progress_event,
        }

        self.console.print("[bold magenta]Assistant[/bold magenta]: ", end="")
        try:
            result = self.graph.invoke(graph_state)
        finally:
            self._stop_progress()
        self.console.print()

        self.session.messages = [
            ChatMessage(role=message["role"], content=message["content"])
            for message in result.get("messages", [])
        ]


    def _request_human_approval(self, action: ProposedAction) -> ApprovalDecision:
        """Prompt the user to approve, reject, or edit a proposed side effect."""

        payload = action.get("payload", {})
        preview = str(payload)
        if len(preview) > 800:
            preview = f"{preview[:800]}..."
        self._stop_progress()
        self.console.print(
            Panel(
                f"[bold]Proposed action:[/bold] {action.get('name', 'unknown')}\n"
                f"[bold]Reason:[/bold] {action.get('reason', '')}\n"
                f"[bold]Payload preview:[/bold] {preview}",
                title="Human approval required",
                border_style="yellow",
            )
        )
        choice = self.input_reader("Approve, reject, or edit? [a/r/e]").strip().lower()
        if choice in {"a", "approve", "approved", ""}:
            return {"status": "approved", "payload": payload}
        if choice in {"e", "edit", "edited"}:
            return self._edit_approval_payload(action)
        return {"status": "rejected", "payload": payload}

    def _edit_approval_payload(self, action: ProposedAction) -> ApprovalDecision:
        """Collect a small safe edit for the supported HITL action payloads."""

        payload = dict(action.get("payload", {}))
        if action.get("name") == "local_llm.generate":
            payload["prompt"] = self.input_reader("Edited prompt")
        elif action.get("name") == "short_term_memory.write":
            payload["response"] = self.input_reader("Edited assistant response")
        return {"status": "edited", "payload": payload}

    def _print_help(self) -> None:
        self.console.print(
            Panel(
                Markdown(
                    """
**Commands**

- `/help` - Show this help message.
- `/clear` - Clear the in-memory session transcript.
- `/exit` or `/quit` - Exit the chat.
- `--hitl` - Start chat with approvals for tool calls and memory writes.
                    """.strip()
                ),
                title="Chat help",
                border_style="cyan",
            )
        )
