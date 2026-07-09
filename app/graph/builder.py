"""Builder utilities for the LangGraph assistant runtime."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    call_llm,
    completion_checker_node,
    classify_intent,
    coding_question_tool,
    document_question_tool,
    executor_node,
    failure_recovery_node,
    fallback_tool,
    general_chat_tool,
    human_approval,
    critic_reviewer_node,
    image_question_tool,
    load_context,
    memory_update_tool,
    route_after_approval,
    route_after_completion_check,
    route_after_failure_recovery,
    route_by_intent,
    save_context,
    summarization_tool,
    task_planner_node,
    task_planning_tool,
    web_research_tool,
)
from app.graph.state import AssistantState


def build_graph():
    """Build and compile the assistant graph.

    Graph shape:
        START -> load_context -> classify_intent -> intent tool route
        -> task_planner -> executor -> critic_reviewer -> completion_checker
        -> optional failure_recovery/executor loop -> human_approval -> call_llm -> human_approval -> save_context -> END

    The classifier routes each turn to a dedicated observable tool node. Low
    confidence or unknown intent routes to a safe general-chat fallback before
    the normal approval and response-generation path continues.
    """

    graph = StateGraph(AssistantState)
    graph.add_node("load_context", load_context)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("general_chat_tool", general_chat_tool)
    graph.add_node("coding_question_tool", coding_question_tool)
    graph.add_node("summarization_tool", summarization_tool)
    graph.add_node("document_question_tool", document_question_tool)
    graph.add_node("web_research_tool", web_research_tool)
    graph.add_node("image_question_tool", image_question_tool)
    graph.add_node("memory_update_tool", memory_update_tool)
    graph.add_node("task_planning_tool", task_planning_tool)
    graph.add_node("fallback_tool", fallback_tool)
    graph.add_node("task_planner", task_planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("critic_reviewer", critic_reviewer_node)
    graph.add_node("completion_checker", completion_checker_node)
    graph.add_node("failure_recovery", failure_recovery_node)
    graph.add_node("human_approval", human_approval)
    graph.add_node("call_llm", call_llm)
    graph.add_node("save_context", save_context)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "general_chat": "general_chat_tool",
            "coding_question": "coding_question_tool",
            "summarization": "summarization_tool",
            "document_question": "document_question_tool",
            "web_research": "web_research_tool",
            "image_question": "image_question_tool",
            "memory_update": "memory_update_tool",
            "task_planning": "task_planning_tool",
            "fallback": "fallback_tool",
        },
    )
    for node_name in (
        "general_chat_tool",
        "coding_question_tool",
        "summarization_tool",
        "document_question_tool",
        "web_research_tool",
        "image_question_tool",
        "memory_update_tool",
        "task_planning_tool",
        "fallback_tool",
    ):
        graph.add_edge(node_name, "task_planner")
    graph.add_edge("task_planner", "executor")
    graph.add_edge("executor", "critic_reviewer")
    graph.add_edge("critic_reviewer", "completion_checker")
    graph.add_conditional_edges(
        "completion_checker",
        route_after_completion_check,
        {"failure_recovery": "failure_recovery", "human_approval": "human_approval"},
    )
    graph.add_conditional_edges(
        "failure_recovery",
        route_after_failure_recovery,
        {"executor": "executor", "human_approval": "human_approval"},
    )
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"call_llm": "call_llm", "save_context": "save_context", "end": END},
    )
    graph.add_edge("call_llm", "human_approval")
    graph.add_edge("save_context", END)
    return graph.compile()
