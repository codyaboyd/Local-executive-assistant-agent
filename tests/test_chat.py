from exec_agent.chat import ChatAction, ChatSession, parse_chat_input


def test_parse_exit_command() -> None:
    parsed = parse_chat_input(" /exit ")

    assert parsed.action is ChatAction.EXIT


def test_parse_quit_alias() -> None:
    parsed = parse_chat_input("/QUIT")

    assert parsed.action is ChatAction.EXIT


def test_parse_help_command() -> None:
    parsed = parse_chat_input("/help")

    assert parsed.action is ChatAction.HELP


def test_parse_clear_command() -> None:
    parsed = parse_chat_input("/clear")

    assert parsed.action is ChatAction.CLEAR


def test_parse_plain_message_preserves_text() -> None:
    parsed = parse_chat_input("  hello assistant  ")

    assert parsed.action is None
    assert parsed.text == "  hello assistant  "


def test_session_render_prompt_includes_transcript_and_assistant_cue() -> None:
    session = ChatSession()
    session.add("user", "hello")
    session.add("assistant", "hi")

    assert session.render_prompt() == "User: hello\nAssistant: hi\nAssistant:"


def test_terminal_chat_uses_graph_and_preserves_memory() -> None:
    from io import StringIO
    from rich.console import Console

    from exec_agent.chat import TerminalChat

    prompts: list[str] = []

    def fake_streamer(prompt: str):
        prompts.append(prompt)
        yield "hello"
        yield "!"

    chat = TerminalChat(console=Console(file=StringIO()), streamer=fake_streamer)
    chat._handle_user_message("hi")

    assert prompts == ["User: hi\nAssistant:"]
    assert [(message.role, message.content) for message in chat.session.messages] == [
        ("user", "hi"),
        ("assistant", "hello!"),
    ]


def test_terminal_chat_emits_progress_and_debug_transitions() -> None:
    from io import StringIO
    from rich.console import Console

    from exec_agent.chat import TerminalChat

    output = StringIO()
    chat = TerminalChat(console=Console(file=output, force_terminal=False), streamer=lambda prompt: ["ok"], debug=True)
    chat._handle_user_message("hi")

    rendered = output.getvalue()
    assert "Graph transition" in rendered
    assert "load_context" in rendered
    assert "call_llm" in rendered
    assert "ok" in rendered
