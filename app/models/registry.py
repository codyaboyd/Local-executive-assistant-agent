"""Curated local model registry and selection helpers.

The registry intentionally favors small, permissive, open-source models that can run
on consumer GPUs (<=16GB VRAM) or CPU-only machines.  Model IDs are Hugging Face
repository names; quantized GGUF IDs are preferred when VRAM is constrained even
when the default Transformers backend may require users to install an appropriate
runtime for those artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Iterable, Literal

Preset = Literal["default", "low_vram", "cpu_only", "quality", "coding", "research"]


class ModelRole(StrEnum):
    GENERAL_REASONING = "general_reasoning"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    DOCUMENT_QA = "document_qa"
    WEB_RESEARCH = "web_research"
    TOOL_CALLING = "tool_calling"
    EMBEDDINGS = "embeddings"
    VISION = "vision"


@dataclass(frozen=True)
class ModelSpec:
    role: ModelRole
    model_id: str
    display_name: str
    min_vram_gb: float
    recommended_vram_gb: float
    cpu_friendly: bool
    quantization: str
    backend: str
    strengths: tuple[str, ...]
    size_gb: float
    default_presets: tuple[Preset, ...] = ("default",)

    @property
    def is_quantized(self) -> bool:
        return self.quantization.lower() not in {"none", "fp32", "fp16", "bf16"}


REGISTRY: tuple[ModelSpec, ...] = (
    # General reasoning / tool-oriented chat.
    ModelSpec(
        ModelRole.GENERAL_REASONING,
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct (4-bit)",
        10,
        12,
        False,
        "4-bit",
        "transformers",
        ("quality reasoning", "instruction-following", "tool-use prompts"),
        7.5,
        ("default", "quality", "research"),
    ),
    ModelSpec(
        ModelRole.GENERAL_REASONING,
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral 7B Instruct (quantized)",
        14,
        16,
        False,
        "4-bit/GGUF-ready",
        "transformers",
        ("instruction-following", "reasoning", "chat"),
        8.0,
        ("quality",),
    ),
    ModelSpec(
        ModelRole.GENERAL_REASONING,
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5 3B Instruct",
        6,
        8,
        True,
        "4-bit-ready",
        "transformers",
        ("8GB reasoning", "instruction-following", "tool-use prompts"),
        6.2,
        ("default", "low_vram", "research"),
    ),
    ModelSpec(
        ModelRole.GENERAL_REASONING,
        "microsoft/Phi-3.5-mini-instruct",
        "Phi-3.5 Mini Instruct",
        6,
        8,
        True,
        "4-bit-ready",
        "transformers",
        ("8GB reasoning", "longer context", "instruction-following"),
        7.6,
        ("default", "research"),
    ),
    ModelSpec(
        ModelRole.GENERAL_REASONING,
        "google/gemma-2-2b-it",
        "Gemma 2 2B IT",
        4,
        8,
        True,
        "4-bit-ready",
        "transformers",
        ("8GB lightweight reasoning", "instruction-following"),
        5.0,
        ("low_vram",),
    ),
    ModelSpec(
        ModelRole.GENERAL_REASONING,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5 1.5B Instruct",
        2,
        4,
        True,
        "4-bit-ready",
        "transformers",
        ("CPU fallback", "low-vram reasoning", "instruction-following"),
        3.1,
        ("cpu_only",),
    ),
    ModelSpec(
        ModelRole.GENERAL_REASONING,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "TinyLlama 1.1B Chat",
        1,
        2,
        True,
        "4-bit-ready",
        "transformers",
        ("tiny CPU fallback", "basic chat"),
        2.2,
        ("cpu_only",),
    ),
    # Coding.
    ModelSpec(
        ModelRole.CODING,
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen2.5 Coder 7B Instruct (quantized)",
        10,
        16,
        False,
        "4-bit/GGUF-ready",
        "transformers",
        ("quality coding", "structured edits", "instruction-following"),
        8.0,
        ("quality", "coding"),
    ),
    ModelSpec(
        ModelRole.CODING,
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "Qwen2.5 Coder 3B Instruct",
        6,
        8,
        True,
        "4-bit-ready",
        "transformers",
        ("8GB/12GB coding", "structured edits", "instruction-following"),
        6.2,
        ("default", "low_vram", "coding"),
    ),
    ModelSpec(
        ModelRole.CODING,
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen2.5 Coder 1.5B Instruct",
        2,
        4,
        True,
        "4-bit-ready",
        "transformers",
        ("CPU fallback coding", "low-vram coding"),
        3.1,
        ("cpu_only",),
    ),
    # Summarization/document/research/tool roles default to the active general reasoning tier unless overridden.
    ModelSpec(
        ModelRole.SUMMARIZATION,
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct (4-bit)",
        10,
        12,
        False,
        "4-bit",
        "transformers",
        ("higher-quality summaries", "document synthesis"),
        7.5,
        ("default", "quality", "research"),
    ),
    ModelSpec(
        ModelRole.SUMMARIZATION,
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5 3B Instruct",
        6,
        8,
        True,
        "4-bit-ready",
        "transformers",
        ("8GB summarization", "concise instructions"),
        6.2,
        ("default", "low_vram"),
    ),
    ModelSpec(
        ModelRole.SUMMARIZATION,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5 1.5B Instruct",
        2,
        4,
        True,
        "4-bit-ready",
        "transformers",
        ("CPU fallback summarization",),
        3.1,
        ("cpu_only",),
    ),
    ModelSpec(
        ModelRole.DOCUMENT_QA,
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct (4-bit)",
        10,
        12,
        False,
        "4-bit",
        "transformers",
        ("document QA", "grounded reasoning"),
        7.5,
        ("default", "quality", "research"),
    ),
    ModelSpec(
        ModelRole.DOCUMENT_QA,
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5 3B Instruct",
        6,
        8,
        True,
        "4-bit-ready",
        "transformers",
        ("8GB document QA",),
        6.2,
        ("default", "low_vram"),
    ),
    ModelSpec(
        ModelRole.DOCUMENT_QA,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5 1.5B Instruct",
        2,
        4,
        True,
        "4-bit-ready",
        "transformers",
        ("CPU fallback document QA",),
        3.1,
        ("cpu_only",),
    ),
    ModelSpec(
        ModelRole.WEB_RESEARCH,
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct (4-bit)",
        10,
        12,
        False,
        "4-bit",
        "transformers",
        ("source-grounded synthesis", "research memos"),
        7.5,
        ("default", "quality", "research"),
    ),
    ModelSpec(
        ModelRole.WEB_RESEARCH,
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5 3B Instruct",
        6,
        8,
        True,
        "4-bit-ready",
        "transformers",
        ("8GB research",),
        6.2,
        ("default", "low_vram"),
    ),
    ModelSpec(
        ModelRole.WEB_RESEARCH,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5 1.5B Instruct",
        2,
        4,
        True,
        "4-bit-ready",
        "transformers",
        ("CPU fallback research",),
        3.1,
        ("cpu_only",),
    ),
    ModelSpec(
        ModelRole.TOOL_CALLING,
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 7B Instruct (4-bit)",
        10,
        12,
        False,
        "4-bit",
        "transformers",
        ("strong instruction following", "tool-use prompts", "JSON-style responses"),
        7.5,
        ("default", "quality", "research"),
    ),
    ModelSpec(
        ModelRole.TOOL_CALLING,
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5 3B Instruct",
        6,
        8,
        True,
        "4-bit-ready",
        "transformers",
        ("8GB tool prompts", "instruction-following"),
        6.2,
        ("default", "low_vram"),
    ),
    ModelSpec(
        ModelRole.TOOL_CALLING,
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5 1.5B Instruct",
        2,
        4,
        True,
        "4-bit-ready",
        "transformers",
        ("CPU fallback tool prompts",),
        3.1,
        ("cpu_only",),
    ),
    ModelSpec(
        ModelRole.EMBEDDINGS,
        "BAAI/bge-base-en-v1.5",
        "BGE Base EN v1.5",
        1,
        2,
        True,
        "none",
        "sentence-transformers",
        ("quality embeddings", "retrieval quality"),
        0.4,
        ("quality", "research"),
    ),
    ModelSpec(
        ModelRole.EMBEDDINGS,
        "BAAI/bge-small-en-v1.5",
        "BGE Small EN v1.5",
        0.5,
        1,
        True,
        "none",
        "sentence-transformers",
        ("default embeddings", "balanced retrieval"),
        0.13,
        ("default", "coding"),
    ),
    ModelSpec(
        ModelRole.EMBEDDINGS,
        "sentence-transformers/all-MiniLM-L6-v2",
        "all-MiniLM-L6-v2",
        0.5,
        1,
        True,
        "none",
        "sentence-transformers",
        ("low-resource embeddings", "CPU friendly"),
        0.1,
        ("low_vram", "cpu_only"),
    ),
    ModelSpec(
        ModelRole.VISION,
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen2-VL 2B Instruct",
        6,
        8,
        False,
        "4-bit-ready",
        "transformers",
        ("vision instruction following", "visual question answering"),
        5.0,
        ("default", "quality", "research"),
    ),
    ModelSpec(
        ModelRole.VISION,
        "Salesforce/blip-image-captioning-base",
        "BLIP Image Captioning Base",
        2,
        4,
        True,
        "none",
        "transformers",
        ("low-resource image captioning", "CPU capable"),
        1.0,
        ("low_vram", "cpu_only"),
    ),
)

_ROLE_ENV = {
    ModelRole.GENERAL_REASONING: "general_model_id",
    ModelRole.CODING: "coding_model_id",
    ModelRole.SUMMARIZATION: "summary_model_id",
    ModelRole.DOCUMENT_QA: "docqa_model_id",
    ModelRole.WEB_RESEARCH: "research_model_id",
    ModelRole.TOOL_CALLING: "tool_model_id",
    ModelRole.EMBEDDINGS: "embedding_model_id",
    ModelRole.VISION: "vision_model_id",
}


def specs_for_role(role: ModelRole | str) -> list[ModelSpec]:
    role = ModelRole(role)
    return [spec for spec in REGISTRY if spec.role == role]


def select_model(
    role: ModelRole | str, settings=None, *, override: str | None = None
) -> ModelSpec:
    """Select the best configured or curated model for a role."""

    from exec_agent.config import get_settings

    role = ModelRole(role)
    settings = settings or get_settings()
    configured = override or getattr(settings, _ROLE_ENV[role])
    if configured not in {None, "", "auto", "default"}:
        return ModelSpec(
            role,
            configured,
            configured,
            0,
            settings.max_vram_gb,
            True,
            "custom",
            "transformers",
            ("user override",),
            0,
            (settings.model_preset,),
        )

    candidates = specs_for_role(role)
    preset_matches = [
        spec for spec in candidates if settings.model_preset in spec.default_presets
    ]
    if preset_matches:
        candidates = preset_matches
    if settings.model_preset == "cpu_only" or settings.device == "cpu":
        candidates = [spec for spec in candidates if spec.cpu_friendly] or candidates
    budget = max(0.1, float(settings.max_vram_gb))
    within_budget = [spec for spec in candidates if spec.recommended_vram_gb <= budget]
    candidates = within_budget or sorted(
        candidates, key=lambda spec: spec.recommended_vram_gb
    )
    if (
        settings.model_preset in {"cpu_only", "low_vram"}
        or settings.device == "cpu"
        or budget <= 4
    ):
        candidates = sorted(
            candidates,
            key=lambda spec: (
                not spec.cpu_friendly,
                spec.recommended_vram_gb,
                spec.size_gb,
            ),
        )
    else:
        candidates = sorted(
            candidates,
            key=lambda spec: (not spec.is_quantized, -spec.recommended_vram_gb),
        )
    return candidates[0]


def fallback_chain(
    role: ModelRole | str, settings=None, *, override: str | None = None
) -> list[ModelSpec]:
    """Return selected model followed by smaller/CPU fallback candidates."""

    selected = select_model(role, settings, override=override)
    settings = (
        settings
        or __import__("exec_agent.config", fromlist=["get_settings"]).get_settings()
    )
    fallbacks = sorted(
        specs_for_role(ModelRole(role)),
        key=lambda spec: (spec.recommended_vram_gb, spec.size_gb),
    )
    chain = [selected]
    chain.extend(
        spec
        for spec in fallbacks
        if spec.model_id != selected.model_id
        and spec.recommended_vram_gb <= settings.max_vram_gb
    )
    chain.extend(
        spec
        for spec in fallbacks
        if spec.model_id not in {item.model_id for item in chain} and spec.cpu_friendly
    )
    return chain


def pull_model(model_id: str, *, local_dir: Path | None = None) -> str:
    """Download a model snapshot with huggingface_hub when available."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install huggingface_hub to pull models.") from exc
    return snapshot_download(
        repo_id=model_id, local_dir=str(local_dir) if local_dir else None
    )


def should_auto_pull(model_id: str | None, settings) -> bool:
    """Return true when a role model configured as auto/default should be pulled."""

    return bool(settings.model_auto_pull and model_id in {None, "", "auto", "default"})


def benchmark_selection(
    roles: Iterable[ModelRole] | None = None, settings=None
) -> list[dict[str, str]]:
    start = perf_counter()
    selected = [select_model(role, settings) for role in (roles or list(ModelRole))]
    elapsed_ms = (perf_counter() - start) * 1000
    return [
        {
            "role": spec.role.value,
            "model_id": spec.model_id,
            "selection_ms": f"{elapsed_ms:.2f}",
        }
        for spec in selected
    ]
