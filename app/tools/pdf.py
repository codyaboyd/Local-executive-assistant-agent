"""PDF ingestion helpers for retrieval-augmented question answering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from app.memory.vector_store import VectorStore

DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class PDFPageText:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class PDFChunk:
    """A chunk of PDF text with retrieval metadata."""

    text: str
    metadata: dict[str, Any]


def extract_pdf_pages(path: str | Path) -> list[PDFPageText]:
    """Extract text from a PDF page by page using pypdf."""

    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path}")

    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency should be installed
        raise RuntimeError("pypdf is required for PDF ingestion. Install project dependencies, then retry.") from exc

    reader = PdfReader(str(pdf_path))
    pages: list[PDFPageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(PDFPageText(page_number=index, text=_normalize_text(text)))
    return pages


def chunk_text(text: str, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into paragraph/sentence-aware chunks with character overlap."""

    normalized = _normalize_text(text)
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    units = _split_text_units(normalized)
    chunks: list[str] = []
    current = ""

    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_oversized_unit(unit, chunk_size=chunk_size, overlap=overlap))
            continue

        candidate = unit if not current else f"{current}\n\n{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current.strip())
        prefix = _overlap_prefix(current, overlap)
        current = f"{prefix}\n\n{unit}".strip() if prefix else unit

    if current:
        chunks.append(current.strip())
    return chunks


def build_pdf_chunks(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[PDFChunk]:
    """Extract and chunk a PDF, preserving source filename, page number, and file type metadata."""

    pdf_path = Path(path).expanduser().resolve()
    chunks: list[PDFChunk] = []
    for page in extract_pdf_pages(pdf_path):
        for chunk_index, text in enumerate(chunk_text(page.text, chunk_size=chunk_size, overlap=overlap), start=1):
            chunks.append(
                PDFChunk(
                    text=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page.page_number,
                        "file_type": "pdf",
                        "chunk": chunk_index,
                    },
                )
            )
    return chunks


def ingest_pdf(path: str | Path, vector_store: VectorStore | None = None) -> int:
    """Ingest a PDF into the vector store and return the number of stored chunks."""

    chunks = build_pdf_chunks(path)
    if not chunks:
        return 0
    store = vector_store or VectorStore()
    store.add_documents([chunk.text for chunk in chunks], [chunk.metadata for chunk in chunks])
    return len(chunks)


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


def _split_text_units(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= DEFAULT_CHUNK_SIZE:
            units.append(paragraph)
        else:
            units.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part.strip())
    return units or [text]


def _split_oversized_unit(unit: str, *, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(unit):
        end = min(start + chunk_size, len(unit))
        if end < len(unit):
            breakpoint = max(unit.rfind(" ", start, end), unit.rfind("\n", start, end))
            if breakpoint > start + chunk_size // 2:
                end = breakpoint
        chunks.append(unit[start:end].strip())
        if end >= len(unit):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _overlap_prefix(text: str, overlap: int) -> str:
    if overlap == 0:
        return ""
    prefix = text[-overlap:].strip()
    first_space = prefix.find(" ")
    return prefix[first_space + 1 :].strip() if first_space > 0 else prefix
