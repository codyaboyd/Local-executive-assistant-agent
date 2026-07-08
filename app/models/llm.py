"""Compatibility wrapper for the local LLM abstraction."""

from exec_agent.models.llm import generate_text, load_llm, stream_text
from app.models.registry import ModelRole, select_model

__all__ = ["load_llm", "generate_text", "stream_text", "ModelRole", "select_model"]
