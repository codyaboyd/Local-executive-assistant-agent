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
