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
