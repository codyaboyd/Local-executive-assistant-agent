from typer.testing import CliRunner

from exec_agent.cli import app

runner = CliRunner()


def test_chat_command_runs() -> None:
    result = runner.invoke(app, ["chat"])

    assert result.exit_code == 0
    assert "Executive assistant chat" in result.output


def test_config_command_runs() -> None:
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "Executive Assistant Configuration" in result.output


def test_config_command_shows_hitl() -> None:
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "hitl" in result.output


def test_chat_command_accepts_hitl_flag() -> None:
    result = runner.invoke(app, ["chat", "--hitl"])

    assert result.exit_code == 0
    assert "Human-in-the-loop approvals enabled" in result.output


def test_memory_cli_add_list_search_delete(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXEC_AGENT_DATA_DIR", str(tmp_path))
    from exec_agent.config import get_settings

    get_settings.cache_clear()
    add_result = runner.invoke(app, ["memory", "add", "User prefers concise answers", "--tag", "preference"])
    assert add_result.exit_code == 0
    assert "Added memory 1" in add_result.output

    list_result = runner.invoke(app, ["memory", "list"])
    assert list_result.exit_code == 0
    assert "User prefers concise answers" in list_result.output
    assert "preference" in list_result.output

    search_result = runner.invoke(app, ["memory", "search", "concise"])
    assert search_result.exit_code == 0
    assert "User prefers concise answers" in search_result.output

    delete_result = runner.invoke(app, ["memory", "delete", "1"])
    assert delete_result.exit_code == 0
    assert "Deleted memory 1" in delete_result.output
    get_settings.cache_clear()


def test_rag_search_cli(monkeypatch) -> None:
    from app.memory.vector_store import VectorSearchResult

    class FakeVectorStore:
        def similarity_search(self, query: str, k: int = 5):
            assert query == "policy"
            assert k == 1
            return [VectorSearchResult("Travel policy requires receipts", {"source": "handbook.md"}, "doc-1", 0.2)]

    monkeypatch.setattr("exec_agent.cli.VectorStore", FakeVectorStore)

    result = runner.invoke(app, ["rag", "search", "policy", "--k", "1"])

    assert result.exit_code == 0
    assert "RAG Search: policy" in result.output
    assert "Travel policy requires receipts" in result.output
    assert "handbook.md" in result.output


def test_ingest_pdf_cli(monkeypatch) -> None:
    monkeypatch.setattr("exec_agent.cli.ingest_pdf_file", lambda path: 3)

    result = runner.invoke(app, ["ingest", "pdf", "./file.pdf"])

    assert result.exit_code == 0
    assert "Ingested 3 PDF chunks from ./file.pdf" in result.output


def test_ask_cli_uses_pdf_context_and_prints_references(monkeypatch) -> None:
    from app.memory.vector_store import VectorSearchResult

    captured = {}

    class FakeVectorStore:
        def similarity_search(self, query: str, k: int = 5):
            assert query == "What is the travel policy?"
            assert k == 2
            return [
                VectorSearchResult(
                    "Travel policy requires receipts.",
                    {"source": "handbook.pdf", "page": 7, "file_type": "pdf"},
                    "doc-1",
                    0.1,
                ),
                VectorSearchResult("Ignore non-pdf", {"source": "notes.md", "file_type": "md"}, "doc-2", 0.2),
            ]

    def fake_generate_text(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Receipts are required (handbook.pdf p. 7)."

    monkeypatch.setattr("exec_agent.cli.VectorStore", FakeVectorStore)
    monkeypatch.setattr("exec_agent.cli.generate_text", fake_generate_text)

    result = runner.invoke(app, ["ask", "What is the travel policy?", "--k", "2"])

    assert result.exit_code == 0
    assert "Receipts are required" in result.output
    assert "References: handbook.pdf p. 7" in result.output
    assert "Source: handbook.pdf, page 7" in captured["prompt"]
