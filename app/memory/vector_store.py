"""ChromaDB-backed vector store for local RAG retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from exec_agent.config import get_settings

DEFAULT_COLLECTION_NAME = "exec_agent_rag"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class VectorSearchResult:
    """A document returned from vector similarity search."""

    content: str
    metadata: dict[str, Any]
    id: str
    distance: float | None = None


def default_vector_store_path() -> Path:
    """Return the default Chroma persistence directory."""

    return get_settings().expanded_vector_db_path


class SentenceTransformerEmbeddingFunction:
    """Chroma embedding function powered by sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma expects this parameter name.
        embeddings = self.model.encode(input, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()


class VectorStore:
    """Local ChromaDB vector store with sentence-transformer embeddings."""

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_function: Any | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory) if persist_directory is not None else default_vector_store_path()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or SentenceTransformerEmbeddingFunction()
        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        if self._collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        return self._collection

    def add_documents(self, chunks: Sequence[str], metadata: dict[str, Any] | Sequence[dict[str, Any]] | None = None) -> list[str]:
        """Embed and add text chunks with optional metadata, returning document IDs."""

        documents = [chunk for chunk in chunks if chunk.strip()]
        if not documents:
            return []
        metadatas = _normalize_metadata(metadata, len(documents))
        ids = [str(uuid4()) for _ in documents]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return ids

    def similarity_search(self, query: str, k: int = 5) -> list[VectorSearchResult]:
        """Return the top-k chunks most similar to the query."""

        if not query.strip() or k <= 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(documents)
        return [
            VectorSearchResult(content=document, metadata=dict(metadata or {}), id=document_id, distance=distance)
            for document, metadata, document_id, distance in zip(documents, metadatas, ids, distances, strict=False)
        ]

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks whose metadata source matches the supplied source."""

        self.collection.delete(where={"source": source})


def format_vector_results_for_prompt(results: Sequence[VectorSearchResult]) -> str:
    """Render vector results as compact prompt context."""

    lines: list[str] = []
    for result in results:
        source = result.metadata.get("source", "unknown")
        lines.append(f"- [{source}] {result.content}")
    return "\n".join(lines)


def _normalize_metadata(metadata: dict[str, Any] | Sequence[dict[str, Any]] | None, count: int) -> list[dict[str, Any]]:
    if metadata is None:
        return [{} for _ in range(count)]
    if isinstance(metadata, dict):
        return [dict(metadata) for _ in range(count)]
    metadatas = [dict(item) for item in metadata]
    if len(metadatas) != count:
        raise ValueError("metadata sequence length must match chunks length")
    return metadatas
