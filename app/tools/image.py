"""Image analysis helpers backed by local Hugging Face vision-language models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.memory.vector_store import VectorStore
from exec_agent.config import get_settings
from exec_agent.safety import validate_local_file

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_IMAGE_QA_MODEL = "dandelin/vilt-b32-finetuned-vqa"


@dataclass(frozen=True)
class ImageAnalysisResult:
    """Result from describing or asking about an image."""

    text: str
    metadata: dict[str, Any]


def validate_image_path(path: str | Path) -> Path:
    """Return a resolved image path, validating existence and supported extension."""

    try:
        return validate_local_file(path, allowed_extensions=SUPPORTED_IMAGE_EXTENSIONS, purpose="image")
    except ValueError as exc:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported image type {Path(path).suffix!r}. Supported image types: {supported}") from exc


def describe_image(
    path: str | Path,
    *,
    model_id: str | None = None,
    device: Literal["cpu", "cuda", "auto"] | None = None,
    vector_store: VectorStore | None = None,
    store_context: bool = True,
) -> ImageAnalysisResult:
    """Generate an image description and optionally store it in the vector database."""

    image_path = validate_image_path(path)
    selected_model = model_id or get_settings().image_caption_model_id
    selected_device = device or get_settings().device
    description = _run_image_to_text(image_path, selected_model, selected_device)
    result = ImageAnalysisResult(
        text=description,
        metadata=_metadata_for_image(image_path, task="describe", model_id=selected_model, device=selected_device),
    )
    if store_context:
        _store_image_context(result, vector_store)
    return result


def ask_image(
    path: str | Path,
    question: str,
    *,
    model_id: str | None = None,
    device: Literal["cpu", "cuda", "auto"] | None = None,
    vector_store: VectorStore | None = None,
    store_context: bool = True,
) -> ImageAnalysisResult:
    """Answer a question about an image and optionally store the answer in the vector database."""

    if not question.strip():
        raise ValueError("Image question must not be empty.")
    image_path = validate_image_path(path)
    selected_model = model_id or get_settings().image_qa_model_id
    selected_device = device or get_settings().device
    answer = _run_visual_question_answering(image_path, question, selected_model, selected_device)
    text = f"Question: {question.strip()}\nAnswer: {answer}"
    result = ImageAnalysisResult(
        text=text,
        metadata={
            **_metadata_for_image(image_path, task="ask", model_id=selected_model, device=selected_device),
            "question": question.strip(),
        },
    )
    if store_context:
        _store_image_context(result, vector_store)
    return result


def _run_image_to_text(image_path: Path, model_id: str, device: Literal["cpu", "cuda", "auto"]) -> str:
    pipeline = _load_pipeline("image-to-text", model_id, device)
    output = pipeline(str(image_path))
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            return str(first.get("generated_text") or first.get("caption") or first).strip()
    return str(output).strip()


def _run_visual_question_answering(image_path: Path, question: str, model_id: str, device: Literal["cpu", "cuda", "auto"]) -> str:
    pipeline = _load_pipeline("visual-question-answering", model_id, device)
    output = pipeline(image=str(image_path), question=question.strip())
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            return str(first.get("answer") or first).strip()
    if isinstance(output, dict):
        return str(output.get("answer") or output).strip()
    return str(output).strip()


def _load_pipeline(task: str, model_id: str, device: Literal["cpu", "cuda", "auto"]) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - dependency should be installed
        raise RuntimeError("transformers is required for local image analysis. Install project dependencies, then retry.") from exc

    device_index = _transformers_device(device)
    try:
        return pipeline(task, model=model_id, device=device_index)
    except Exception as exc:  # noqa: BLE001 - convert many HF/runtime failures into actionable CLI guidance.
        raise RuntimeError(
            "The selected vision model cannot run locally. "
            f"Model: {model_id}; task: {task}; device: {device}. "
            "Try a smaller Hugging Face vision-language model, switch to --device cpu, "
            "or install the model's required dependencies and weights. "
            f"Original error: {exc}"
        ) from exc


def _transformers_device(device: Literal["cpu", "cuda", "auto"]) -> int:
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def _metadata_for_image(image_path: Path, *, task: str, model_id: str, device: str) -> dict[str, Any]:
    return {
        "source": image_path.name,
        "path": str(image_path),
        "file_type": "image",
        "image_format": image_path.suffix.lower().lstrip("."),
        "task": task,
        "model_id": model_id,
        "device": device,
    }


def _store_image_context(result: ImageAnalysisResult, vector_store: VectorStore | None = None) -> None:
    store = vector_store or VectorStore()
    store.add_documents([result.text], [result.metadata])
