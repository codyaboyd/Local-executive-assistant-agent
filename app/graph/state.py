"""LangGraph state definitions for the assistant runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, TypedDict


class AssistantMessage(TypedDict):
    """A short-term-memory message stored in graph state."""

    role: str
    content: str


IntentName = Literal[
    "general_chat",
    "coding_question",
    "summarization",
    "document_question",
    "web_research",
    "image_question",
    "memory_update",
    "task_planning",
    "uncertain",
]


class ProposedAction(TypedDict, total=False):
    """A side-effecting action that may require human approval."""

    kind: Literal["tool_call", "memory_write"]
    name: str
    reason: str
    payload: dict[str, Any]


class ApprovalDecision(TypedDict, total=False):
    """Human approval result for a proposed action."""

    status: Literal["approved", "rejected", "edited"]
    payload: dict[str, Any]


class AssistantState(TypedDict, total=False):
    """State passed between assistant graph nodes.

    The state intentionally keeps short-term memory in the graph so each turn can
    load the existing transcript, call the model, and save the assistant reply
    back into the same state object shape.
    """

    messages: list[AssistantMessage]
    user_input: str
    prompt: str
    conversation_summary: str
    response: str
    response_chunks: list[str]
    long_term_memories: list[dict[str, Any]]
    vector_context: list[dict[str, Any]]
    memory_db_path: str
    vector_store_path: str
    vector_search_k: int
    streamer: Callable[[str], Iterable[str]]
    hitl_enabled: bool
    pending_action: ProposedAction
    last_approval: ApprovalDecision
    approval_handler: Callable[[ProposedAction], ApprovalDecision]
    web_access_enabled: bool
    fastcrw_enabled: bool
    active_profile_allows_online_research: bool
    intent: IntentName
    intent_confidence: float
    intent_reason: str
    tool_name: str
    tool_call_log: list[dict[str, Any]]
    model_role: str
    secondary_model_roles: list[str]
    debug: bool
    progress_callback: Callable[[dict[str, Any]], None]
    available_tools: list[str]
    task_plan: list[dict[str, Any]]
    task_trace: list[dict[str, Any]]
    autonomous_mode: bool
    max_steps: int
    step_count: int
    executor_decision: dict[str, Any]
    review: dict[str, Any]
    completion_status: dict[str, Any]
    failure_count: int
    unresolved_issues: list[str]
    files_changed: list[str]
    commands_run: list[str]
    final_summary: dict[str, Any]
