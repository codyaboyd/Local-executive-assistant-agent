"""LangGraph state definitions for the assistant runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypedDict


class AssistantMessage(TypedDict):
    """A short-term-memory message stored in graph state."""

    role: str
    content: str


class AssistantState(TypedDict, total=False):
    """State passed between assistant graph nodes.

    The state intentionally keeps short-term memory in the graph so each turn can
    load the existing transcript, call the model, and save the assistant reply
    back into the same state object shape.
    """

    messages: list[AssistantMessage]
    user_input: str
    prompt: str
    response: str
    response_chunks: list[str]
    streamer: Callable[[str], Iterable[str]]
