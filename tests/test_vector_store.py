from app.memory.vector_store import VectorSearchResult, VectorStore, format_vector_results_for_prompt


class FakeCollection:
    def __init__(self) -> None:
        self.added = None
        self.deleted_where = None

    def add(self, **kwargs):
        self.added = kwargs

    def query(self, **kwargs):
        assert kwargs == {"query_texts": ["policy"], "n_results": 2}
        return {
            "documents": [["Travel policy requires receipts"]],
            "metadatas": [[{"source": "handbook.md"}]],
            "ids": [["doc-1"]],
            "distances": [[0.25]],
        }

    def delete(self, **kwargs):
        self.deleted_where = kwargs["where"]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.fake_collection = FakeCollection()
        super().__init__(persist_directory="/tmp/unused", embedding_function=lambda input: [[0.0] for _ in input])

    @property
    def collection(self):
        return self.fake_collection


def test_vector_store_add_search_and_delete() -> None:
    store = FakeVectorStore()

    ids = store.add_documents(["Travel policy requires receipts"], {"source": "handbook.md"})
    assert len(ids) == 1
    assert store.fake_collection.added["documents"] == ["Travel policy requires receipts"]
    assert store.fake_collection.added["metadatas"] == [{"source": "handbook.md"}]

    results = store.similarity_search("policy", k=2)
    assert results == [VectorSearchResult("Travel policy requires receipts", {"source": "handbook.md"}, "doc-1", 0.25)]

    store.delete_by_source("handbook.md")
    assert store.fake_collection.deleted_where == {"source": "handbook.md"}


def test_format_vector_results_for_prompt() -> None:
    rendered = format_vector_results_for_prompt([VectorSearchResult("Bring receipts", {"source": "handbook.md"}, "doc-1")])

    assert rendered == "- [handbook.md] Bring receipts"
