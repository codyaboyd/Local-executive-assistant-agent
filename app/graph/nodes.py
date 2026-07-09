"""LangGraph node implementations for the assistant runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any

from app.models.registry import ModelRole
from exec_agent.models.llm import generate_text, stream_text
from exec_agent.safety import UserFacingError

from app.graph.state import ApprovalDecision, AssistantMessage, AssistantState, IntentName, ProposedAction
from app.memory.long_term import LongTermMemoryStore, format_memories_for_prompt
from app.memory.vector_store import VectorStore, format_vector_results_for_prompt
from app.tools import web_fastcrw

logger = logging.getLogger(__name__)

TextStreamer = Callable[..., Iterable[str]]


def _safe_tool_error(tool_name: str, exc: Exception) -> AssistantState:
    """Return a user-facing tool failure instead of crashing the graph."""

    logger.exception("Tool failed", extra={"tool": tool_name, "event": "tool_error"})
    message = str(exc) if isinstance(exc, UserFacingError) else (
        f"{tool_name} failed safely. No external action was completed. Details: {exc}"
    )
    return {
        "prompt": f"Tool route: {tool_name}\nTool error: {message}\nExplain this limitation clearly to the user and suggest a safe next step.",
        "tool_call_log": [{"tool": tool_name, "intent": "error", "details": {"error": message}}],
    }


def _guard_tool(tool_name: str, fn: Callable[[AssistantState], AssistantState]) -> Callable[[AssistantState], AssistantState]:
    def wrapped(state: AssistantState) -> AssistantState:
        try:
            return fn(state)
        except Exception as exc:  # noqa: BLE001 - tool boundary must convert failures to safe messages.
            return _safe_tool_error(tool_name, exc)
    return wrapped


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
    logger.info("Tool call", extra={"tool": tool_name, "intent": intent, "event": "tool_call"})
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
    elif _looks_like_coding_question(lowered):
        intent, confidence, reason = "coding_question", 0.86, "User asked for coding, debugging, or software implementation help."
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
    elif _looks_like_summary_request(lowered):
        intent, confidence, reason = "summarization", 0.84, "User asked for a summary or synthesis."
    elif (
        any(term in lowered for term in ("maybe", "not sure", "could be", "unclear"))
        and len(lowered.split()) < 6
    ):
        intent, confidence, reason = "uncertain", 0.35, "Input is too ambiguous to choose a specialized tool."

    role_updates = _model_role_updates(intent, state.get("model_role"))
    return {"intent": intent, "intent_confidence": confidence, "intent_reason": reason, **role_updates}


def _looks_like_coding_question(lowered: str) -> bool:
    coding_terms = (
        "code", "coding", "debug", "bug", "function", "class", "api", "stack trace",
        "python", "javascript", "typescript", "java", "rust", "go ", "sql", "html", "css",
        "implement", "refactor", "unit test", "pytest", "compiler", "exception",
    )
    return any(term in lowered for term in coding_terms)


def _looks_like_summary_request(lowered: str) -> bool:
    summary_terms = ("summarize", "summary", "tl;dr", "recap", "brief me", "briefing", "digest")
    return any(term in lowered for term in summary_terms)


def _model_role_updates(intent: IntentName, previous_role: str | None = None) -> AssistantState:
    role_by_intent = {
        "general_chat": ModelRole.GENERAL_REASONING.value,
        "coding_question": ModelRole.CODING.value,
        "summarization": ModelRole.SUMMARIZATION.value,
        "document_question": ModelRole.DOCUMENT_QA.value,
        "web_research": ModelRole.WEB_RESEARCH.value,
        "image_question": ModelRole.VISION.value,
        "memory_update": ModelRole.TOOL_CALLING.value,
        "task_planning": ModelRole.TOOL_CALLING.value,
        "uncertain": ModelRole.GENERAL_REASONING.value,
    }
    model_role = role_by_intent.get(intent, ModelRole.GENERAL_REASONING.value)
    updates: AssistantState = {"model_role": model_role}
    if intent == "image_question":
        updates["secondary_model_roles"] = [ModelRole.GENERAL_REASONING.value]
    if previous_role and previous_role != model_role:
        logger.debug(
            "Switching model role",
            extra={"event": "model_role_switch", "from_role": previous_role, "to_role": model_role, "intent": intent},
        )
    else:
        logger.debug(
            "Selected model role",
            extra={"event": "model_role_selected", "to_role": model_role, "intent": intent},
        )
    return updates


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


def coding_question_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: coding_question_tool")
    _emit_progress(state, "tool", "Using tool: coding assistant", node="coding_question_tool")
    return _record_tool_call(state, "coding.assistant", "coding_question")


def summarization_tool(state: AssistantState) -> AssistantState:
    logger.info("Running graph node: summarization_tool")
    _emit_progress(state, "tool", "Using tool: summarizer", node="summarization_tool")
    updates = _record_tool_call(state, "summary.create", "summarization")
    updates["prompt"] = _tool_response_prompt(
        state, "summarization", "Summarize the provided conversation or context accurately and concisely."
    )
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
    updates["model_role"] = ModelRole.GENERAL_REASONING.value
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

    def default_streamer(prompt: str, role: str = ModelRole.GENERAL_REASONING.value) -> Iterable[str]:
        try:
            yield from stream_text(prompt, role=role)
        except TypeError:
            yield generate_text(prompt, role=role)

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
    model_role = state.get("model_role", ModelRole.GENERAL_REASONING.value)
    logger.debug("Calling LLM with model role", extra={"event": "model_role_call", "model_role": model_role})
    try:
        streamer = _get_streamer(state)
        try:
            response_chunks = list(streamer(prompt, model_role))
        except TypeError:
            response_chunks = list(streamer(prompt))
        response = "".join(response_chunks)
    except Exception as exc:  # noqa: BLE001 - model boundary must produce a clear user-facing error.
        logger.exception("Model generation failed", extra={"event": "model_error", "node": "call_llm"})
        detail = str(exc) if isinstance(exc, UserFacingError) else f"Model generation failed: {exc}"
        response_chunks = [f"I could not generate a response safely. {detail}"]
        response = response_chunks[0]
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
    final_summary = {
        "what_was_done": response.strip()[:500],
        "files_changed": state.get("files_changed", []),
        "commands_run": state.get("commands_run", []),
        "unresolved_issues": state.get("unresolved_issues", []),
        "task_trace": state.get("task_trace", []),
    }
    return {"messages": messages, "conversation_summary": summary, "final_summary": final_summary}


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


# Ensure every graph tool node has a protective error boundary.
general_chat_tool = _guard_tool("general_chat.respond", general_chat_tool)
coding_question_tool = _guard_tool("coding.assistant", coding_question_tool)
summarization_tool = _guard_tool("summary.create", summarization_tool)
document_question_tool = _guard_tool("documents.vector_search", document_question_tool)
web_research_tool = _guard_tool("fastcrw.web_research", web_research_tool)
image_question_tool = _guard_tool("image.analyze", image_question_tool)
memory_update_tool = _guard_tool("memory.update", memory_update_tool)
task_planning_tool = _guard_tool("planner.create_plan", task_planning_tool)
fallback_tool = _guard_tool("fallback.general_chat", fallback_tool)

AVAILABLE_AUTONOMOUS_TOOLS = [
    "filesystem",
    "shell",
    "vector_search",
    "memory",
    "pdf",
    "docx",
    "image",
    "fastcrw_web",
]

_DECISION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("inspect_files", ("file", "repo", "code", "read", "inspect", "pdf", "docx", "image")),
    ("run_commands", ("run", "test", "command", "shell", "execute", "pytest", "make")),
    ("search_docs", ("document", "docs", "policy", "pdf", "docx", "vector")),
    ("search_web", ("web", "internet", "latest", "current", "browse", "online")),
    ("write_edit_files", ("write", "edit", "change", "implement", "fix", "create", "update")),
    ("ask_user_approval", ("approval", "permission", "confirm", "delete", "destructive", "credential")),
)


def _trace(state: AssistantState, node: str, action: str, details: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    entry = {
        "step": int(state.get("step_count", 0)),
        "node": node,
        "action": action,
        "details": details or {},
    }
    return [*state.get("task_trace", []), entry]


def task_planner_node(state: AssistantState) -> AssistantState:
    """Create or refresh an autonomous task plan from the current goal and context."""

    logger.info("Running graph node: task_planner_node")
    _emit_progress(state, "autonomy", "Planning autonomous task", node="task_planner_node")
    user_input = state.get("user_input", "")
    tool_name = state.get("tool_name") or "general_chat.respond"
    plan = [
        {"id": 1, "description": "Understand the user goal and loaded context.", "status": "completed"},
        {"id": 2, "description": f"Select tools from {', '.join(AVAILABLE_AUTONOMOUS_TOOLS)}.", "status": "completed"},
        {"id": 3, "description": "Execute the next safe action or prepare a response.", "status": "pending"},
        {"id": 4, "description": "Review output and decide whether to iterate, recover, ask approval, or stop.", "status": "pending"},
    ]
    return {
        "available_tools": AVAILABLE_AUTONOMOUS_TOOLS,
        "task_plan": plan,
        "task_trace": _trace(state, "task_planner", "planned", {"goal": user_input, "initial_tool": tool_name}),
    }


def executor_node(state: AssistantState) -> AssistantState:
    """Choose the next autonomous action while keeping side effects behind existing tools/HITL."""

    logger.info("Running graph node: executor_node")
    _emit_progress(state, "autonomy", "Choosing next execution action", node="executor_node")
    step_count = int(state.get("step_count", 0)) + 1
    max_steps = int(state.get("max_steps", 8))
    lowered = state.get("user_input", "").lower()
    decisions = [name for name, keywords in _DECISION_KEYWORDS if any(keyword in lowered for keyword in keywords)]
    if not decisions:
        decisions = ["stop"]
    if step_count >= max_steps:
        decisions = ["stop"]
    tool_map = {
        "inspect_files": "filesystem",
        "run_commands": "shell",
        "search_docs": "vector_search",
        "search_web": "fastcrw_web",
        "write_edit_files": "filesystem",
        "ask_user_approval": "human_approval",
        "stop": "local_llm",
    }
    decision = {
        "step": step_count,
        "actions": decisions,
        "tools": sorted({tool_map[action] for action in decisions}),
        "reason": "Keyword and context based autonomous router decision.",
    }
    return {
        "step_count": step_count,
        "executor_decision": decision,
        "task_trace": _trace({**state, "step_count": step_count}, "executor", "decided", decision),
    }


def critic_reviewer_node(state: AssistantState) -> AssistantState:
    """Review the planned/executed action for safety, usefulness, and missing context."""

    logger.info("Running graph node: critic_reviewer_node")
    _emit_progress(state, "autonomy", "Reviewing execution decision", node="critic_reviewer_node")
    decision = state.get("executor_decision", {})
    issues: list[str] = []
    if "search_web" in decision.get("actions", []) and not state.get("web_access_enabled", False):
        issues.append("Web search was requested but web access is disabled.")
    if "run_commands" in decision.get("actions", []) and state.get("hitl_enabled", False):
        issues.append("Command execution may require human approval.")
    approved = not issues
    review = {"approved": approved, "issues": issues, "decision": decision}
    return {
        "review": review,
        "unresolved_issues": [*state.get("unresolved_issues", []), *issues],
        "task_trace": _trace(state, "critic_reviewer", "reviewed", review),
    }


def completion_checker_node(state: AssistantState) -> AssistantState:
    """Decide whether the autonomous loop can stop or should recover/iterate."""

    logger.info("Running graph node: completion_checker_node")
    _emit_progress(state, "autonomy", "Checking completion", node="completion_checker_node")
    step_count = int(state.get("step_count", 0))
    max_steps = int(state.get("max_steps", 8))
    review = state.get("review", {})
    autonomous = state.get("autonomous_mode", False)
    needs_recovery = bool(review.get("issues"))
    complete = (not autonomous) or step_count >= max_steps or not needs_recovery
    status = {
        "complete": complete,
        "needs_recovery": needs_recovery and step_count < max_steps,
        "max_steps_reached": step_count >= max_steps,
        "step_count": step_count,
        "max_steps": max_steps,
    }
    return {"completion_status": status, "task_trace": _trace(state, "completion_checker", "checked", status)}


def failure_recovery_node(state: AssistantState) -> AssistantState:
    """Record a recovery action and switch to a safe response path after failed review."""

    logger.info("Running graph node: failure_recovery_node")
    _emit_progress(state, "autonomy", "Recovering from blocked action", node="failure_recovery_node")
    failure_count = int(state.get("failure_count", 0)) + 1
    issues = state.get("review", {}).get("issues", [])
    prompt = _tool_response_prompt(
        state,
        "failure_recovery",
        "Some autonomous actions were blocked or unavailable. Explain the limitation and continue with the best safe answer. "
        f"Issues: {issues}",
    )
    return {
        "failure_count": failure_count,
        "prompt": prompt,
        "task_trace": _trace(state, "failure_recovery", "recovered", {"failure_count": failure_count, "issues": issues}),
    }


def route_after_completion_check(state: AssistantState) -> str:
    status = state.get("completion_status", {})
    if status.get("needs_recovery"):
        return "failure_recovery"
    return "human_approval"


def route_after_failure_recovery(state: AssistantState) -> str:
    """Continue autonomous recovery loops until completion or max-step enforcement stops them."""

    if not state.get("autonomous_mode", False):
        return "human_approval"
    if int(state.get("step_count", 0)) >= int(state.get("max_steps", 8)):
        return "human_approval"
    return "executor"
