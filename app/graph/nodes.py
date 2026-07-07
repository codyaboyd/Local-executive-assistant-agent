"""LangGraph node implementations for the assistant runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any

from exec_agent.models.llm import generate_text, stream_text

from app.graph.state import ApprovalDecision, AssistantMessage, AssistantState, ProposedAction
from app.memory.long_term import LongTermMemoryStore, format_memories_for_prompt

logger = logging.getLogger(__name__)

TextStreamer = Callable[[str], Iterable[str]]


def _messages_from_state(state: AssistantState) -> list[AssistantMessage]:
    """Return a copy of the short-term-memory messages in graph state."""

    return list(state.get("messages", []))


def _render_prompt(messages: list[AssistantMessage], long_term_context: str = "") -> str:
    """Render graph memory context as the prompt expected by the local LLM."""

    transcript: list[str] = []
    if long_term_context:
        transcript.extend([
            "Relevant long-term memories:",
            long_term_context,
            "",
        ])
    transcript.extend(f"{message['role'].title()}: {message['content']}" for message in messages)
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


def _default_approval_handler(action: ProposedAction) -> ApprovalDecision:
    """Approve actions by default when no interactive handler is provided."""

    return {"status": "approved", "payload": action.get("payload", {})}


def _approval_action(kind: str, name: str, reason: str, payload: dict[str, Any]) -> ProposedAction:
    return {"kind": kind, "name": name, "reason": reason, "payload": payload}  # type: ignore[typeddict-item]


def load_context(state: AssistantState) -> AssistantState:
    """Load short-term memory and prepare the prompt for the LLM call."""

    logger.info("Running graph node: load_context")
    messages = _messages_from_state(state)
    user_input = state.get("user_input", "")
    if user_input:
        messages.append({"role": "user", "content": user_input})
    memories = LongTermMemoryStore(state.get("memory_db_path")).search(user_input) if user_input else []
    long_term_context = format_memories_for_prompt(memories)
    prompt = _render_prompt(messages, long_term_context)
    return {
        "messages": messages,
        "prompt": prompt,
        "long_term_memories": [memory.__dict__ for memory in memories],
        "pending_action": _approval_action(
            "tool_call",
            "local_llm.generate",
            "Generate an assistant response from the current conversation.",
            {"prompt": prompt},
        ),
    }


def human_approval(state: AssistantState) -> AssistantState:
    """Pause for human approval before side-effecting tool calls or memory writes."""

    logger.info("Running graph node: human_approval")
    pending_action = state.get("pending_action")
    if not state.get("hitl_enabled", False) or pending_action is None:
        return {"last_approval": {"status": "approved", "payload": {}}}

    decision = state.get("approval_handler", _default_approval_handler)(pending_action)
    status = decision.get("status", "rejected")
    payload = decision.get("payload", pending_action.get("payload", {}))
    updates: AssistantState = {"last_approval": {"status": status, "payload": payload}}

    if status in {"approved", "edited"}:
        if pending_action.get("name") == "local_llm.generate":
            updates["prompt"] = str(payload.get("prompt", state.get("prompt", "")))
        if pending_action.get("name") == "short_term_memory.write":
            response = payload.get("response")
            if response is not None:
                updates["response"] = str(response)
    return updates


def call_llm(state: AssistantState) -> AssistantState:
    """Call the local LLM and keep streamed chunks in graph state."""

    logger.info("Running graph node: call_llm")
    prompt = state.get("prompt", "")
    response_chunks = list(_get_streamer(state)(prompt))
    response = "".join(response_chunks)
    return {
        "response_chunks": response_chunks,
        "response": response,
        "pending_action": _approval_action(
            "memory_write",
            "short_term_memory.write",
            "Persist the assistant response in the in-memory chat transcript.",
            {"role": "assistant", "response": response},
        ),
    }


def save_context(state: AssistantState) -> AssistantState:
    """Save the assistant response back into short-term memory."""

    logger.info("Running graph node: save_context")
    messages = _messages_from_state(state)
    response = state.get("response", "")
    if response:
        messages.append({"role": "assistant", "content": response})
    return {"messages": messages}


def route_after_approval(state: AssistantState) -> str:
    """Route safely after HITL approval for the pending action."""

    if state.get("last_approval", {}).get("status") == "rejected":
        return "end"
    pending_action = state.get("pending_action", {})
    if pending_action.get("name") == "short_term_memory.write":
        return "save_context"
    return "call_llm"
