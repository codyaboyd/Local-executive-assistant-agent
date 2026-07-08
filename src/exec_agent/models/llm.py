"""Local Hugging Face Transformers LLM abstraction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import lru_cache
from queue import Queue
from threading import Thread
from typing import Iterator, Literal
import warnings

from app.models.registry import ModelRole, fallback_chain, pull_model, should_auto_pull
from exec_agent.config import get_settings
from exec_agent.safety import UserFacingError

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


@lru_cache(maxsize=16)
def load_llm(role: str = ModelRole.GENERAL_REASONING.value, model_id: str | None = None):
    """Load and cache a role-specific local text-generation pipeline with fallbacks."""

    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "Hugging Face Transformers is required for local LLM support. "
            "Install the project dependencies, then retry."
        ) from exc

    settings = get_settings()
    device = _resolve_device(settings.device)
    errors: list[str] = []
    for spec in fallback_chain(role, settings, override=model_id):
        try:
            if should_auto_pull(model_id, settings):
                if spec.recommended_vram_gb > settings.max_vram_gb:
                    warnings.warn(
                        f"Skipping auto-pull for {spec.model_id}: recommends {spec.recommended_vram_gb:g}GB VRAM "
                        f"above configured budget {settings.max_vram_gb}GB.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    pull_model(spec.model_id)
            return pipeline(
                "text-generation",
                model=spec.model_id,
                device=_pipeline_device(device),
            )
        except Exception as exc:  # noqa: BLE001 - try smaller/CPU-safe models before surfacing.
            errors.append(f"{spec.model_id}: {exc}")
            if device == "cuda":
                try:
                    warnings.warn(f"Could not load {spec.model_id} on CUDA; retrying on CPU.", RuntimeWarning, stacklevel=2)
                    return pipeline("text-generation", model=spec.model_id, device=-1)
                except Exception as cpu_exc:  # noqa: BLE001
                    errors.append(f"{spec.model_id} cpu: {cpu_exc}")
    raise RuntimeError("No configured or fallback model could be loaded. " + " | ".join(errors))


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


def generate_text(prompt: str, role: str = ModelRole.GENERAL_REASONING.value, model_id: str | None = None) -> str:
    """Generate text for a prompt using the best role-specific local model."""

    try:
        generator = load_llm(role, model_id)
    except TypeError:
        generator = load_llm()
    timeout = get_settings().model_timeout_seconds
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(generator, prompt, **_generation_kwargs())
        try:
            result = future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise UserFacingError(f"Model generation timed out after {timeout} seconds. Try a shorter prompt or increase EXEC_AGENT_MODEL_TIMEOUT_SECONDS.") from exc
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "generated_text" in first:
            return str(first["generated_text"])
    return str(result)


def stream_text(prompt: str, role: str = ModelRole.GENERAL_REASONING.value, model_id: str | None = None) -> Iterator[str]:
    """Stream generated text chunks for a role-specific prompt."""

    try:
        from transformers import TextIteratorStreamer
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError(
            "Hugging Face Transformers is required for local LLM support. "
            "Install the project dependencies, then retry."
        ) from exc

    try:
        generator = load_llm(role, model_id)
    except TypeError:
        generator = load_llm()
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is None:
        yield generate_text(prompt)
        return

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=1.0)
    kwargs = _generation_kwargs()
    kwargs.pop("return_full_text", None)
    errors: Queue[BaseException] = Queue(maxsize=1)

    def run_generator() -> None:
        try:
            generator(text_inputs=prompt, streamer=streamer, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - surface model failures after streaming stops.
            errors.put(exc)

    thread = Thread(target=run_generator, daemon=True)
    thread.start()
    timeout = get_settings().model_timeout_seconds
    thread.join(timeout=0)
    import time
    deadline = time.monotonic() + timeout
    while thread.is_alive() or not errors.empty():
        if time.monotonic() > deadline:
            raise UserFacingError(f"Model streaming timed out after {timeout} seconds. Try a shorter prompt or increase EXEC_AGENT_MODEL_TIMEOUT_SECONDS.")
        if not errors.empty():
            raise errors.get()
        try:
            yield next(streamer)
        except StopIteration:
            break
    thread.join(timeout=1)
