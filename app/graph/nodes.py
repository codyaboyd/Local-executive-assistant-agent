"""LangGraph node implementations for the assistant runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any

from exec_agent.models.llm import generate_text, stream_text

from app.graph.state import ApprovalDecision, AssistantMessage, AssistantState, ProposedAction
from app.memory.long_term import LongTermMemoryStore, format_memories_for_prompt
from app.memory.vector_store import VectorStore, format_vector_results_for_prompt
from app.tools import web_fastcrw

logger = logging.getLogger(__name__)

TextStreamer = Callable[[str], Iterable[str]]


def _messages_from_state(state: AssistantState) -> list[AssistantMessage]:
    """Return a copy of the short-term-memory messages in graph state."""

    return list(state.get("messages", []))


def _render_prompt(messages: list[AssistantMessage], long_term_context: str = "", vector_context: str = "") -> str:
    """Render graph memory and RAG context as the prompt expected by the local LLM."""

    transcript: list[str] = []
    if vector_context:
        transcript.extend([
            "Relevant vector context:",
            vector_context,
            "",
        ])
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
    vector_results = _search_vector_context(state, user_input) if user_input else []
    web_results = _maybe_search_web_context(state, user_input) if user_input else []
    vector_results = [*web_results, *vector_results]
    long_term_context = format_memories_for_prompt(memories)
    vector_context = format_vector_results_for_prompt(vector_results)
    prompt = _render_prompt(messages, long_term_context, vector_context)
    return {
        "messages": messages,
        "prompt": prompt,
        "long_term_memories": [memory.__dict__ for memory in memories],
        "vector_context": [result.__dict__ for result in vector_results],
        "pending_action": _approval_action(
            "tool_call",
            "local_llm.generate",
            "Generate an assistant response from the current conversation.",
            {"prompt": prompt},
        ),
    }



def _search_vector_context(state: AssistantState, query: str):
    """Retrieve relevant vector context without blocking graph execution when the store is unavailable."""

    try:
        store = VectorStore(state.get("vector_store_path"))
        return store.similarity_search(query, k=int(state.get("vector_search_k", 5)))
    except Exception as exc:  # pragma: no cover - depends on optional local vector runtime state
        logger.warning("Vector context retrieval skipped: %s", exc)
        return []


def _explicitly_requests_web_research(query: str) -> bool:
    lowered = query.lower()
    phrases = ("web research", "search the web", "look up", "online research", "browse", "internet", "latest", "current")
    return any(phrase in lowered for phrase in phrases)


def _maybe_search_web_context(state: AssistantState, query: str):
    """Use FastCRW only when web access is enabled and web research is allowed/requested."""

    if not state.get("web_access_enabled", False) or not state.get("fastcrw_enabled", state.get("web_access_enabled", False)):
        return []
    if not (_explicitly_requests_web_research(query) or state.get("active_profile_allows_online_research", False)):
        return []
    try:
        results = web_fastcrw.search_web(query, max_results=int(state.get("vector_search_k", 5)))
        pages = []
        for result in results:
            url = result.get("url")
            if not url:
                continue
            if state.get("hitl_enabled", False):
                action = _approval_action(
                    "tool_call",
                    "fastcrw.scrape",
                    "Scrape a web search result with self-hosted FastCRW.",
                    {"url": url, "domain": web_fastcrw.target_domain(str(url))},
                )
                decision = state.get("approval_handler", _default_approval_handler)(action)
                if decision.get("status") == "rejected":
                    continue
                url = decision.get("payload", {}).get("url", url)
            pages.append(web_fastcrw.scrape_url(str(url)))
        return [
            type("WebVectorResult", (), {"content": page.content, "metadata": page.metadata, "id": page.url, "distance": None})()
            for page in pages
        ]
    except web_fastcrw.FastCRWError as exc:
        logger.warning("FastCRW web context skipped: %s", exc)
        return []

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
