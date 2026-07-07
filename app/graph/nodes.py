"""LangGraph node implementations for the assistant runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any

from exec_agent.models.llm import generate_text, stream_text

from app.graph.state import ApprovalDecision, AssistantMessage, AssistantState, IntentName, ProposedAction
from app.memory.long_term import LongTermMemoryStore, format_memories_for_prompt
from app.memory.vector_store import VectorStore, format_vector_results_for_prompt
from app.tools import web_fastcrw

logger = logging.getLogger(__name__)

TextStreamer = Callable[[str], Iterable[str]]


def _emit_progress(state: AssistantState, stage: str, message: str, *, node: str | None = None) -> None:
    """Emit terminal progress events without coupling graph nodes to Rich."""

    callback = state.get("progress_callback")
    if callback is None:
        return
    event: dict[str, Any] = {"stage": stage, "message": message}
    if node is not None:
        event["node"] = node
    try:
        callback(event)
    except Exception as exc:  # pragma: no cover - progress UI must never break graph execution
        logger.debug("Progress callback failed: %s", exc)


def _messages_from_state(state: AssistantState) -> list[AssistantMessage]:
    """Return a copy of the short-term-memory messages in graph state."""

    return list(state.get("messages", []))


def _render_prompt(
    messages: list[AssistantMessage],
    long_term_context: str = "",
    vector_context: str = "",
    conversation_summary: str = "",
) -> str:
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
    if conversation_summary:
        transcript.extend([
            "Conversation summary for continuity:",
            conversation_summary,
            "",
        ])
    transcript.extend(f"{message['role'].title()}: {message['content']}" for message in messages)
    transcript.append("Assistant:")
    return "\n".join(transcript)


def _record_tool_call(
    state: AssistantState,
    tool_name: str,
    intent: str,
    details: dict[str, Any] | None = None,
) -> AssistantState:
    """Record and print tool calls so they remain observable in terminal output."""

    entry = {"tool": tool_name, "intent": intent, "details": details or {}}
    logger.info("TOOL CALL: %s intent=%s details=%s", tool_name, intent, entry["details"])
    print(f"TOOL CALL: {tool_name} intent={intent} details={entry['details']}")
    return {"tool_name": tool_name, "tool_call_log": [*state.get("tool_call_log", []), entry]}


def classify_intent(state: AssistantState) -> AssistantState:
    """Classify user intent before routing to the matching tool node."""

    logger.info("Running graph node: classify_intent")
    _emit_progress(state, "node", "Classifying intent", node="classify_intent")
    user_input = state.get("user_input", "")
    lowered = user_input.lower()
    intent: IntentName = "general_chat"
    confidence = 0.75
    reason = "Defaulted to general conversation."

    if not lowered.strip():
        intent, confidence, reason = "uncertain", 0.0, "No user input was provided."
    elif any(
        term in lowered for term in ("remember", "save this", "store this", "update memory", "forget")
    ):
        intent, confidence, reason = "memory_update", 0.86, "User asked to add, update, or remove memory."
    elif any(
        term in lowered for term in ("plan", "schedule", "itinerary", "todo", "task list", "next steps")
    ):
        intent, confidence, reason = "task_planning", 0.84, "User asked for planning or task decomposition."
    elif any(
        term in lowered for term in ("image", "photo", "picture", "screenshot", "diagram", "chart")
    ):
        intent, confidence, reason = "image_question", 0.84, "User referred to visual content."
    elif _explicitly_requests_web_research(user_input):
        intent, confidence, reason = "web_research", 0.88, "User explicitly requested current or web-backed research."
    elif any(
        term in lowered for term in ("document", "pdf", "docx", "file", "handbook", "policy", "uploaded")
    ):
        intent, confidence, reason = "document_question", 0.82, "User asked about local document context."
    elif (
        any(term in lowered for term in ("maybe", "not sure", "could be", "unclear"))
        and len(lowered.split()) < 6
    ):
        intent, confidence, reason = "uncertain", 0.35, "Input is too ambiguous to choose a specialized tool."

    return {"intent": intent, "intent_confidence": confidence, "intent_reason": reason}


def _tool_response_prompt(state: AssistantState, tool_label: str, extra_context: str = "") -> str:
    messages = _messages_from_state(state)
    sections = [f"Tool route: {tool_label}"]
    if extra_context:
        sections.extend(["Tool context:", extra_context])
    sections.extend([state.get("prompt", _render_prompt(messages))])
    return "\n".join(sections)


def general_chat_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: general_chat_tool")
    _emit_progress(state, "tool", "Using tool: general chat", node="general_chat_tool")
    updates = _record_tool_call(state, "general_chat.respond", "general_chat")
    return updates


def document_question_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: document_question_tool")
    _emit_progress(state, "tool", "Using tool: document vector search", node="document_question_tool")
    updates = _record_tool_call(
        state, "documents.vector_search", "document_question", {"k": state.get("vector_search_k", 5)}
    )
    user_input = state.get("user_input", "")
    vector_results = _search_vector_context(state, user_input) if user_input else []
    if vector_results:
        updates["vector_context"] = [result.__dict__ for result in vector_results]
        updates["prompt"] = _tool_response_prompt(
            state, "document_question", format_vector_results_for_prompt(vector_results)
        )
    else:
        updates["prompt"] = _tool_response_prompt(
            state,
            "document_question",
            "No matching document context was found; answer with that limitation.",
        )
    return updates


def web_research_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: web_research_tool")
    _emit_progress(state, "tool", "Using tool: web research", node="web_research_tool")
    updates = _record_tool_call(
        state,
        "fastcrw.web_research",
        "web_research",
        {"enabled": state.get("web_access_enabled", False)},
    )
    web_results = _maybe_search_web_context(state, state.get("user_input", ""))
    if web_results:
        updates["vector_context"] = [*state.get("vector_context", []), *[result.__dict__ for result in web_results]]
        updates["prompt"] = _tool_response_prompt(
            state, "web_research", format_vector_results_for_prompt(web_results)
        )
    else:
        updates["prompt"] = _tool_response_prompt(
            state,
            "web_research",
            "Web research returned no context or is disabled; state this limitation if relevant.",
        )
    return updates


def image_question_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: image_question_tool")
    _emit_progress(state, "tool", "Using tool: image analysis", node="image_question_tool")
    updates = _record_tool_call(state, "image.analyze", "image_question")
    updates["prompt"] = _tool_response_prompt(
        state,
        "image_question",
        "Use available image-analysis context if it has been supplied; otherwise ask for the image.",
    )
    return updates


def memory_update_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: memory_update_tool")
    _emit_progress(state, "tool", "Using tool: memory update", node="memory_update_tool")
    updates = _record_tool_call(state, "memory.update", "memory_update")
    updates["prompt"] = _tool_response_prompt(
        state,
        "memory_update",
        "Identify the requested memory update and confirm what should be stored or changed.",
    )
    return updates


def task_planning_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: task_planning_tool")
    _emit_progress(state, "tool", "Using tool: task planner", node="task_planning_tool")
    updates = _record_tool_call(state, "planner.create_plan", "task_planning")
    updates["prompt"] = _tool_response_prompt(
        state, "task_planning", "Create an actionable plan with clear next steps and assumptions."
    )
    return updates


def fallback_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: fallback_tool")
    _emit_progress(state, "tool", "Using tool: fallback chat", node="fallback_tool")
    updates = _record_tool_call(
        state, "fallback.general_chat", "uncertain", {"reason": state.get("intent_reason", "uncertain")}
    )
    updates["intent"] = "general_chat"
    updates["prompt"] = _tool_response_prompt(
        state,
        "fallback",
        "Intent was uncertain; answer generally and ask a clarifying question if needed.",
    )
    return updates

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
    _emit_progress(state, "memory", "Loading memory", node="load_context")
    messages = _messages_from_state(state)
    user_input = state.get("user_input", "")
    if user_input:
        messages.append({"role": "user", "content": user_input})
    memories = LongTermMemoryStore(state.get("memory_db_path")).search(user_input) if user_input else []
    vector_results = _search_vector_context(state, user_input) if user_input else []
    long_term_context = format_memories_for_prompt(memories)
    vector_context = format_vector_results_for_prompt(vector_results)
    prompt = _render_prompt(messages, long_term_context, vector_context, state.get("conversation_summary", ""))
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

    _emit_progress(state, "vector", "Searching vector DB")
    try:
        store = VectorStore(state.get("vector_store_path"))
        return store.similarity_search(query, k=int(state.get("vector_search_k", 5)))
    except Exception as exc:  # pragma: no cover - depends on optional local vector runtime state
        logger.warning("Vector context retrieval skipped: %s", exc)
        return []


def _explicitly_requests_web_research(query: str) -> bool:
    lowered = query.lower()
    phrases = (
        "web research",
        "search the web",
        "look up",
        "online research",
        "browse",
        "internet",
        "latest",
        "current",
    )
    return any(phrase in lowered for phrase in phrases)


def _maybe_search_web_context(state: AssistantState, query: str):
    """Use FastCRW only when web access is enabled and web research is allowed/requested."""

    if not state.get("web_access_enabled", False) or not state.get(
        "fastcrw_enabled", state.get("web_access_enabled", False)
    ):
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
            type(
                "WebVectorResult",
                (),
                {"content": page.content, "metadata": page.metadata, "id": page.url, "distance": None},
            )()
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

    _emit_progress(state, "approval", "Waiting for approval", node="human_approval")
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
    _emit_progress(state, "generation", "Generating answer", node="call_llm")
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
    """Save the assistant response back into short-term memory and refresh its summary."""

    logger.info("Running graph node: save_context")
    _emit_progress(state, "memory", "Saving response to memory", node="save_context")
    messages = _messages_from_state(state)
    response = state.get("response", "")
    if response:
        messages.append({"role": "assistant", "content": response})
    summary = _summarize_messages(messages)
    return {"messages": messages, "conversation_summary": summary}


def route_after_approval(state: AssistantState) -> str:
    """Route safely after HITL approval for the pending action."""

    if state.get("last_approval", {}).get("status") == "rejected":
        return "end"
    pending_action = state.get("pending_action", {})
    if pending_action.get("name") == "short_term_memory.write":
        return "save_context"
    return "call_llm"


def route_by_intent(state: AssistantState) -> str:
    """Return the graph edge name for the classified intent, with safe fallback."""

    intent = state.get("intent", "uncertain")
    confidence = float(state.get("intent_confidence", 0.0))
    if intent == "uncertain" or confidence < 0.5:
        return "fallback"
    return str(intent)


def _summarize_messages(messages: list[AssistantMessage], *, max_chars: int = 1200) -> str:
    lines = [f"{message['role'].title()}: {message['content']}" for message in messages]
    summary = "\n".join(lines).strip()
    if len(summary) <= max_chars:
        return summary
    return "…" + summary[-max_chars:]
