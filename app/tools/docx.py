"""DOCX ingestion helpers for retrieval-augmented question answering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.memory.vector_store import VectorStore
from exec_agent.safety import validate_local_file
from app.tools.pdf import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_text


@dataclass(frozen=True)
class DOCXSection:
    """Text extracted from one logical DOCX section."""

    heading: str | None
    text: str


@dataclass(frozen=True)
class DOCXChunk:
    """A chunk of DOCX text with retrieval metadata."""

    text: str
    metadata: dict[str, Any]


def extract_docx_sections(path: str | Path) -> list[DOCXSection]:
    """Extract paragraphs, headings, and tables from a DOCX grouped by heading section."""

    docx_path = validate_local_file(path, allowed_extensions={".docx"}, purpose="DOCX")

    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - dependency should be installed
        raise RuntimeError("python-docx is required for DOCX ingestion. Install project dependencies, then retry.") from exc

    sections: list[DOCXSection] = []
    current_heading: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        text = _normalize_text("\n\n".join(current_parts))
        if text:
            sections.append(DOCXSection(heading=current_heading, text=text))
        current_parts = []

    document = Document(str(docx_path))
    for block in _iter_block_items(document):
        if _is_paragraph(block):
            text = _normalize_text(block.text)
            if not text:
                continue
            if _is_heading(block):
                flush()
                current_heading = text
                current_parts.append(text)
            else:
                current_parts.append(text)
        else:
            table_text = _table_to_text(block)
            if table_text:
                current_parts.append(table_text)

    flush()
    return sections


def build_docx_chunks(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[DOCXChunk]:
    """Extract and chunk a DOCX, preserving source filename, section heading, and file type metadata."""

    docx_path = Path(path).expanduser().resolve()
    chunks: list[DOCXChunk] = []
    for section_index, section in enumerate(extract_docx_sections(docx_path), start=1):
        for chunk_index, text in enumerate(chunk_text(section.text, chunk_size=chunk_size, overlap=overlap), start=1):
            chunks.append(
                DOCXChunk(
                    text=text,
                    metadata={
                        "source": docx_path.name,
                        "section_heading": section.heading or "",
                        "file_type": "docx",
                        "section": section_index,
                        "chunk": chunk_index,
                    },
                )
            )
    return chunks


def ingest_docx(path: str | Path, vector_store: VectorStore | None = None) -> int:
    """Ingest a DOCX into the vector store and return the number of stored chunks."""

    chunks = build_docx_chunks(path)
    if not chunks:
        return 0
    store = vector_store or VectorStore()
    store.add_documents([chunk.text for chunk in chunks], [chunk.metadata for chunk in chunks])
    return len(chunks)


def _iter_block_items(document: Any):
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _is_paragraph(block: Any) -> bool:
    return hasattr(block, "style") and hasattr(block, "text")


def _is_heading(paragraph: Any) -> bool:
    style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
    return style_name.lower().startswith("heading")


def _table_to_text(table: Any) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [_normalize_text(cell.text) for cell in row.cells]
        row_text = " | ".join(cell for cell in cells if cell)
        if row_text:
            rows.append(row_text)
    return _normalize_text("\n".join(rows))


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()
