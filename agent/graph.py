# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import AgentState

from agent.analyzer import analyzer_node_factory
from agent.planner import planner_node_factory
from agent.summarizer import summarization_node_factory
from agent.qa import qa_node_factory
from llm.inference import LocalLLM


def route_by_plan(state: AgentState) -> str:
    return state["plan"]["chain"]


def build_graph(llm: LocalLLM):
    graph = StateGraph(AgentState)

    graph.add_node("analyzer", analyzer_node_factory(llm))
    graph.add_node("planner", planner_node_factory(llm))
    graph.add_node("summarization_chain", summarization_node_factory(llm))
    graph.add_node("qa_chain", qa_node_factory(llm))

    graph.set_entry_point("analyzer")
    graph.add_edge("analyzer", "planner")

    graph.add_conditional_edges(
        "planner",
        route_by_plan,
        {
            "summarization_chain": "summarization_chain",
            "qa_chain": "qa_chain",
        },
    )

    graph.add_edge("summarization_chain", END)
    graph.add_edge("qa_chain", END)

    return graph.compile()
