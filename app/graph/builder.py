"""Builder utilities for the LangGraph assistant runtime."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import call_llm, human_approval, load_context, route_after_approval, save_context
from app.graph.state import AssistantState


def build_graph():
    """Build and compile the assistant graph.

    Graph shape:
        START -> load_context -> human_approval -> call_llm -> human_approval
        -> save_context -> END

    In human-in-the-loop mode, the approval node can safely route rejected LLM
    calls or memory writes to END without executing the proposed action.
    """

    graph = StateGraph(AssistantState)
    graph.add_node("load_context", load_context)
    graph.add_node("human_approval", human_approval)
    graph.add_node("call_llm", call_llm)
    graph.add_node("save_context", save_context)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "human_approval")
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"call_llm": "call_llm", "save_context": "save_context", "end": END},
    )
    graph.add_edge("call_llm", "human_approval")
    graph.add_edge("save_context", END)
    return graph.compile()
