import pytest

from exec_agent.config import Settings
from exec_agent.models import llm


def test_resolve_device_falls_back_to_cpu_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
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
