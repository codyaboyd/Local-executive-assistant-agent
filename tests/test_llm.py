import pytest

from app.models.registry import ModelRole, select_model
from exec_agent.config import Settings
from exec_agent.models import llm


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class Torch:
        cuda = Cuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", Torch())

    assert llm._resolve_device("cuda") == "cpu"


def test_generate_text_uses_pipeline_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_llm():
        def generator(prompt: str, **kwargs):
            assert prompt == "hello"
            assert kwargs["max_new_tokens"] == 5
            return [{"generated_text": "world"}]

        return generator

    monkeypatch.setattr(llm, "load_llm", fake_load_llm)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: Settings(EXEC_AGENT_MAX_TOKENS=5, EXEC_AGENT_TEMPERATURE=0),
    )

    assert llm.generate_text("hello") == "world"


def test_registry_selects_recommended_general_models_by_vram() -> None:
    assert (
        select_model(
            ModelRole.GENERAL_REASONING, Settings(EXEC_AGENT_MAX_VRAM_GB=16)
        ).model_id
        == "Qwen/Qwen2.5-7B-Instruct"
    )
    assert (
        select_model(
            ModelRole.GENERAL_REASONING, Settings(EXEC_AGENT_MAX_VRAM_GB=12)
        ).model_id
        == "Qwen/Qwen2.5-7B-Instruct"
    )
    assert (
        select_model(
            ModelRole.GENERAL_REASONING, Settings(EXEC_AGENT_MAX_VRAM_GB=8)
        ).model_id
        == "Qwen/Qwen2.5-3B-Instruct"
    )
    assert (
        select_model(
            ModelRole.GENERAL_REASONING,
            Settings(EXEC_AGENT_MODEL_PRESET="cpu_only", EXEC_AGENT_MAX_VRAM_GB=2),
        ).model_id
        == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )


def test_registry_selects_recommended_specialist_models() -> None:
    assert (
        select_model(
            ModelRole.CODING,
            Settings(EXEC_AGENT_MODEL_PRESET="coding", EXEC_AGENT_MAX_VRAM_GB=16),
        ).model_id
        == "Qwen/Qwen2.5-Coder-7B-Instruct"
    )
    assert (
        select_model(ModelRole.CODING, Settings(EXEC_AGENT_MAX_VRAM_GB=8)).model_id
        == "Qwen/Qwen2.5-Coder-3B-Instruct"
    )
    assert (
        select_model(ModelRole.EMBEDDINGS, Settings()).model_id
        == "BAAI/bge-small-en-v1.5"
    )
    assert (
        select_model(
            ModelRole.EMBEDDINGS, Settings(EXEC_AGENT_MODEL_PRESET="quality")
        ).model_id
        == "BAAI/bge-base-en-v1.5"
    )
    assert (
        select_model(ModelRole.VISION, Settings(EXEC_AGENT_MAX_VRAM_GB=8)).model_id
        == "Qwen/Qwen2-VL-2B-Instruct"
    )
    assert (
        select_model(
            ModelRole.VISION,
            Settings(EXEC_AGENT_MODEL_PRESET="cpu_only", EXEC_AGENT_MAX_VRAM_GB=4),
        ).model_id
        == "Salesforce/blip-image-captioning-base"
    )
