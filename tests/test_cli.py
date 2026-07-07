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


def test_ingest_docx_cli(monkeypatch) -> None:
    monkeypatch.setattr("exec_agent.cli.ingest_docx_file", lambda path: 4)

    result = runner.invoke(app, ["ingest", "docx", "./file.docx"])

    assert result.exit_code == 0
    assert "Ingested 4 DOCX chunks from ./file.docx" in result.output


def test_ask_cli_uses_pdf_and_docx_context_and_prints_references(monkeypatch) -> None:
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
                VectorSearchResult(
                    "DOCX policy names approvals.",
                    {"source": "handbook.docx", "section_heading": "Approvals", "file_type": "docx"},
                    "doc-2",
                    0.2,
                ),
                VectorSearchResult("Ignore non-document", {"source": "notes.md", "file_type": "md"}, "doc-3", 0.3),
            ]

    def fake_generate_text(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Receipts are required (handbook.pdf p. 7)."

    monkeypatch.setattr("exec_agent.cli.VectorStore", FakeVectorStore)
    monkeypatch.setattr("exec_agent.cli.generate_text", fake_generate_text)

    result = runner.invoke(app, ["ask", "What is the travel policy?", "--k", "2"])

    assert result.exit_code == 0
    assert "Receipts are required" in result.output
    assert "References: handbook.pdf p. 7, handbook.docx section: Approvals" in result.output
    assert "Source: handbook.pdf, type pdf, page 7" in captured["prompt"]
    assert "Source: handbook.docx, type docx, section Approvals" in captured["prompt"]


def test_image_describe_cli(monkeypatch) -> None:
    from app.tools.image import ImageAnalysisResult

    def fake_describe(path, model_id=None, device=None):
        assert path == "./image.png"
        assert model_id == "caption-model"
        assert device == "cpu"
        return ImageAnalysisResult("a conference room", {"source": "image.png"})

    monkeypatch.setattr("exec_agent.cli.describe_image_file", fake_describe)

    result = runner.invoke(app, ["image", "describe", "./image.png", "--model", "caption-model", "--device", "cpu"])

    assert result.exit_code == 0
    assert "a conference room" in result.output
    assert "Stored image description" in result.output


def test_image_ask_cli(monkeypatch) -> None:
    from app.tools.image import ImageAnalysisResult

    def fake_ask(path, question, model_id=None, device=None):
        assert path == "./image.png"
        assert question == "what is shown here?"
        assert model_id == "vqa-model"
        assert device == "auto"
        return ImageAnalysisResult("Question: what is shown here?\nAnswer: a chart", {"source": "image.png"})

    monkeypatch.setattr("exec_agent.cli.ask_image_file", fake_ask)

    result = runner.invoke(app, ["image", "ask", "./image.png", "what is shown here?", "--model", "vqa-model", "--device", "auto"])

    assert result.exit_code == 0
    assert "Answer: a chart" in result.output
    assert "Stored image answer" in result.output
