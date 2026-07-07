"""Local Hugging Face Transformers LLM abstraction."""

from __future__ import annotations

from functools import lru_cache
from threading import Thread
from typing import Iterator, Literal
import warnings

from exec_agent.config import get_settings

ResolvedDevice = Literal["cpu", "cuda"]


def _resolve_device(requested_device: str) -> ResolvedDevice:
    """Resolve cpu/cuda/auto into an available Transformers device."""

    normalized = requested_device.lower()
    if normalized == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if normalized == "cuda":
            warnings.warn(
                "CUDA was requested but PyTorch is not installed; falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
        return "cpu"

    if normalized == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        warnings.warn(
            "CUDA was requested but is unavailable; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    raise ValueError(f"Unsupported EXEC_AGENT_DEVICE value: {requested_device!r}")


def _pipeline_device(device: ResolvedDevice) -> int:
    """Return the device value expected by transformers.pipeline."""

    return 0 if device == "cuda" else -1


@lru_cache(maxsize=1)
def load_llm():
    """Load and cache the configured local text-generation pipeline."""

    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "Hugging Face Transformers is required for local LLM support. "
            "Install the project dependencies, then retry."
        ) from exc

    settings = get_settings()
    device = _resolve_device(settings.device)
    return pipeline(
        "text-generation",
        model=settings.model_id,
        device=_pipeline_device(device),
    )


def _generation_kwargs() -> dict[str, float | int | bool]:
    """Build generation kwargs from settings."""

    settings = get_settings()
    temperature = settings.temperature
    return {
        "max_new_tokens": settings.max_tokens,
        "temperature": temperature if temperature > 0 else None,
        "do_sample": temperature > 0,
        "return_full_text": False,
    }


def generate_text(prompt: str) -> str:
    """Generate text for a prompt using the configured local model."""

    generator = load_llm()
    result = generator(prompt, **_generation_kwargs())
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"])
    return str(result)


def stream_text(prompt: str) -> Iterator[str]:
    """Stream generated text chunks for a prompt."""

    try:
        from transformers import TextIteratorStreamer
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "Hugging Face Transformers is required for local LLM support. "
            "Install the project dependencies, then retry."
        ) from exc

    generator = load_llm()
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None:
        yield generate_text(prompt)
        return

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = _generation_kwargs()
    kwargs.pop("return_full_text", None)
    thread = Thread(
        target=generator,
        kwargs={"text_inputs": prompt, "streamer": streamer, **kwargs},
        daemon=True,
    )
    thread.start()
    yield from streamer
    thread.join()
