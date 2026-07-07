from app.graph.builder import build_graph
from app.graph.nodes import call_llm, load_context, save_context


def test_graph_shape_and_state_memory() -> None:
    graph = build_graph()

    def fake_streamer(prompt: str):
        assert prompt == "User: hello\nAssistant:"
        yield "hi"

    result = graph.invoke(
        {"messages": [], "user_input": "hello", "streamer": fake_streamer},
    )

    assert result["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    assert result["response_chunks"] == ["hi"]


def test_nodes_log(caplog) -> None:
    caplog.set_level("INFO")

    loaded = load_context({"messages": [], "user_input": "hello"})
    called = call_llm({**loaded, "streamer": lambda prompt: ["hi"]})
    save_context({**loaded, **called})

    assert "Running graph node: load_context" in caplog.text
    assert "Running graph node: call_llm" in caplog.text
    assert "Running graph node: save_context" in caplog.text


def test_hitl_rejects_llm_call_safely() -> None:
    graph = build_graph()
    prompts: list[str] = []

    def fake_streamer(prompt: str):
        prompts.append(prompt)
        yield "unsafe"

    def reject(action):
        assert action["name"] == "local_llm.generate"
        return {"status": "rejected", "payload": action["payload"]}

    result = graph.invoke(
        {"messages": [], "user_input": "hello", "streamer": fake_streamer, "hitl_enabled": True, "approval_handler": reject},
    )

    assert prompts == []
    assert result["messages"] == [{"role": "user", "content": "hello"}]


def test_hitl_edits_llm_prompt_and_memory_write() -> None:
    graph = build_graph()
    prompts: list[str] = []

    def fake_streamer(prompt: str):
        prompts.append(prompt)
        yield "draft"

    def approve_or_edit(action):
        if action["name"] == "local_llm.generate":
            return {"status": "edited", "payload": {"prompt": "edited prompt"}}
        return {"status": "edited", "payload": {"role": "assistant", "response": "edited response"}}

    result = graph.invoke(
        {"messages": [], "user_input": "hello", "streamer": fake_streamer, "hitl_enabled": True, "approval_handler": approve_or_edit},
    )

    assert prompts == ["edited prompt"]
    assert result["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "edited response"},
    ]
