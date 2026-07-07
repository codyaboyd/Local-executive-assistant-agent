"""LangGraph node implementations for the assistant runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from exec_agent.models.llm import generate_text, stream_text

from app.graph.state import AssistantMessage, AssistantState

logger = logging.getLogger(__name__)

TextStreamer = Callable[[str], Iterable[str]]


def _messages_from_state(state: AssistantState) -> list[AssistantMessage]:
    """Return a copy of the short-term-memory messages in graph state."""

    return list(state.get("messages", []))


def _render_prompt(messages: list[AssistantMessage]) -> str:
    """Render graph short-term memory as the prompt expected by the local LLM."""

    transcript = [f"{message['role'].title()}: {message['content']}" for message in messages]
    transcript.append("Assistant:")
    return "\n".join(transcript)


def _get_streamer(state: AssistantState) -> TextStreamer:
    streamer = state.get("streamer")
    if streamer is not None:
        return streamer

    def default_streamer(prompt: str) -> Iterable[str]:
        try:
            yield from stream_text(prompt)
        except TypeError:
            yield generate_text(prompt)

    return default_streamer


def load_context(state: AssistantState) -> AssistantState:
    """Load short-term memory and prepare the prompt for the LLM call."""

    logger.info("Running graph node: load_context")
    messages = _messages_from_state(state)
    user_input = state.get("user_input", "")
    if user_input:
        messages.append({"role": "user", "content": user_input})
    return {"messages": messages, "prompt": _render_prompt(messages)}


def call_llm(state: AssistantState) -> AssistantState:
    """Call the local LLM and keep streamed chunks in graph state."""

    logger.info("Running graph node: call_llm")
    prompt = state.get("prompt", "")
    response_chunks = list(_get_streamer(state)(prompt))
    return {"response_chunks": response_chunks, "response": "".join(response_chunks)}


def save_context(state: AssistantState) -> AssistantState:
    """Save the assistant response back into short-term memory."""

    logger.info("Running graph node: save_context")
    messages = _messages_from_state(state)
    response = state.get("response", "")
    if response:
        messages.append({"role": "assistant", "content": response})
    return {"messages": messages}
