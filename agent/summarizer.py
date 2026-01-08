# agent/nodes/summarizer.py
from agent.state import AgentState
from llm.inference import LocalLLM


def summarization_node_factory(llm: LocalLLM):
    def summarization_node(state: AgentState) -> AgentState:
        plan = state["plan"]
        policy = plan.get("policy", {})
        user_text = state["messages"][-1].content

        prompt = f"""
Summarize the following text.

Requirements:
- compression ratio: {policy.get("compression_ratio", 0.3)}
- style: {policy.get("output_style", "paragraph")}
- tone: {policy.get("tone", "neutral")}

Text:
{user_text}
"""

        state["execution"] = {
            "prompt_template": "summarization",
            "model_name": "local_llm",
            "status": "running",
        }

        output = llm.generate(prompt)

        state["execution"]["status"] = "done"
        state["result"] = {
            "raw_output": output,
            "final_output": output,
        }

        return state

    return summarization_node
