from pathlib import Path

import pytest

from app.tools import image


def test_validate_image_path_accepts_supported_extensions(tmp_path: Path) -> None:
    path = tmp_path / "photo.webp"
    path.write_bytes(b"fake")

    assert image.validate_image_path(path) == path.resolve()


@pytest.mark.parametrize("suffix", [".gif", ".txt", ".pdf"])
def test_validate_image_path_rejects_unsupported_extensions(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"file{suffix}"
    path.write_bytes(b"fake")

    with pytest.raises(ValueError, match="Unsupported image type"):
        image.validate_image_path(path)


def test_describe_image_stores_description(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"fake")
    stored = {}

    class FakeStore:
        def add_documents(self, chunks, metadata):
            stored["chunks"] = chunks
            stored["metadata"] = metadata

    monkeypatch.setattr(image, "_run_image_to_text", lambda image_path, model_id, device: "a desk with a laptop")

    result = image.describe_image(path, model_id="caption-model", device="cpu", vector_store=FakeStore())

    assert result.text == "a desk with a laptop"
    assert result.metadata["file_type"] == "image"
    assert result.metadata["task"] == "describe"
    assert result.metadata["model_id"] == "caption-model"
    assert stored["chunks"] == ["a desk with a laptop"]
    assert stored["metadata"][0]["source"] == "image.png"


def test_ask_image_stores_question_and_answer(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "image.jpg"
    path.write_bytes(b"fake")
    stored = {}

    class FakeStore:
        def add_documents(self, chunks, metadata):
            stored["chunks"] = chunks
            stored["metadata"] = metadata

    monkeypatch.setattr(image, "_run_visual_question_answering", lambda image_path, question, model_id, device: "a cat")

    result = image.ask_image(path, "What is shown?", model_id="vqa-model", device="cuda", vector_store=FakeStore())

    assert result.text == "Question: What is shown?\nAnswer: a cat"
    assert result.metadata["task"] == "ask"
    assert result.metadata["question"] == "What is shown?"
    assert stored["chunks"] == ["Question: What is shown?\nAnswer: a cat"]
    assert stored["metadata"][0]["device"] == "cuda"


def test_load_pipeline_wraps_model_runtime_errors(monkeypatch) -> None:
    def fake_pipeline(*args, **kwargs):
        raise OSError("missing weights")

    monkeypatch.setattr("transformers.pipeline", fake_pipeline)

    with pytest.raises(RuntimeError, match="cannot run locally"):
        image._load_pipeline("image-to-text", "bad-model", "cpu")
