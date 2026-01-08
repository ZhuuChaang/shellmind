# agent/nodes/qa.py
from agent.state import AgentState
from llm.inference import LocalLLM


def qa_node_factory(llm: LocalLLM):
    def qa_node(state: AgentState) -> AgentState:
        raise NotImplementedError("QA chain not implemented yet")
    return qa_node
