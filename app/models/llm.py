"""Compatibility wrapper for the local LLM abstraction."""

from exec_agent.models.llm import generate_text, load_llm, stream_text

__all__ = ["load_llm", "generate_text", "stream_text"]
