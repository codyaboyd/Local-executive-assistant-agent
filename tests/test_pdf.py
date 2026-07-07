from pathlib import Path

import pytest

from app.tools import pdf


def test_chunk_text_preserves_boundaries_and_overlap() -> None:
    text = "First paragraph has the policy.\n\nSecond paragraph explains receipts.\n\nThird paragraph adds approvals."

    chunks = pdf.chunk_text(text, chunk_size=65, overlap=12)

    assert len(chunks) >= 2
    assert "First paragraph" in chunks[0]
    assert any("Second paragraph" in chunk for chunk in chunks)


def test_build_pdf_chunks_includes_required_metadata(monkeypatch) -> None:
    def fake_extract(path: Path):
        return [pdf.PDFPageText(1, "Travel policy requires receipts."), pdf.PDFPageText(2, "Approvals are due Friday.")]

    monkeypatch.setattr(pdf, "extract_pdf_pages", fake_extract)

    chunks = pdf.build_pdf_chunks("/tmp/handbook.pdf")

    assert [chunk.text for chunk in chunks] == ["Travel policy requires receipts.", "Approvals are due Friday."]
    assert chunks[0].metadata == {"source": "handbook.pdf", "page": 1, "file_type": "pdf", "chunk": 1}
    assert chunks[1].metadata == {"source": "handbook.pdf", "page": 2, "file_type": "pdf", "chunk": 1}


def test_ingest_pdf_stores_chunks_with_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        pdf,
        "build_pdf_chunks",
        lambda path: [pdf.PDFChunk("Travel policy requires receipts.", {"source": "handbook.pdf", "page": 1, "file_type": "pdf"})],
    )

    class FakeVectorStore:
        def __init__(self) -> None:
            self.added = None

        def add_documents(self, chunks, metadata):
            self.added = (chunks, metadata)

    store = FakeVectorStore()

    assert pdf.ingest_pdf("handbook.pdf", store) == 1
    assert store.added == (["Travel policy requires receipts."], [{"source": "handbook.pdf", "page": 1, "file_type": "pdf"}])


def test_extract_pdf_pages_rejects_non_pdf(tmp_path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not a pdf")

    with pytest.raises(ValueError):
        pdf.extract_pdf_pages(path)
