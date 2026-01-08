import json
from agent.state import AgentState, AnalysisState
from llm.inference import LocalLLM


def analyzer_node_factory(llm: LocalLLM):
    def analyzer_node(state: AgentState) -> AgentState:
        user_text = state["messages"][-1].content

        prompt = f"""
You are an analyzer.

Output ONLY valid JSON with fields:
- task_type: "summarization" or "qa"
- input_type: "raw_text" or "conversation"
- input_length: "short" | "medium" | "long"
- language: "zh" | "en"

JSON only.

User input:
{user_text}
"""

        raw = llm.generate(prompt, do_sample=False)
        analysis: AnalysisState = json.loads(raw)

        state["analysis"] = analysis
        return state

    return analyzer_node