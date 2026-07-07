"""Builder utilities for the LangGraph assistant runtime."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import call_llm, load_context, save_context
from app.graph.state import AssistantState


def build_graph():
    """Build and compile the assistant graph.

    Graph shape:
        START -> load_context -> call_llm -> save_context -> END
    """

    graph = StateGraph(AssistantState)
    graph.add_node("load_context", load_context)
    graph.add_node("call_llm", call_llm)
    graph.add_node("save_context", save_context)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "call_llm")
    graph.add_edge("call_llm", "save_context")
    graph.add_edge("save_context", END)
    return graph.compile()
