# agent/nodes/planner.py
import json
from agent.state import AgentState, PlanState
from llm.inference import LocalLLM



def planner_node_factory(llm: LocalLLM):
    def planner_node(state: AgentState) -> AgentState:
        analysis = state["analysis"]

        prompt = f"""
You are a planner.

Based on the analysis below, output ONLY valid JSON.

Analysis:
{analysis}

Schema:
- chain: "summarization_chain" | "qa_chain"
- execution_mode: "single_step"
- policy:
    - compression_ratio: float (0-1)
    - output_style: "paragraph" | "bullet"
    - tone: "academic" | "neutral"
- use_retrieval: boolean
- use_citation: boolean
"""

        raw = llm.generate(prompt, do_sample=False)
        plan: PlanState = json.loads(raw)

        state["plan"] = plan
        return state

    return planner_node
