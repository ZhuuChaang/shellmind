# agent/nodes/planner.py
import json
from agent.state import AgentState, PlanState
from llm.inference import LocalLLM
from agent.checker import extract_json



def planner_node_factory(llm: LocalLLM):
    def planner_node(state: AgentState) -> AgentState:
        analysis = state["analysis"]

        prompt = f"""
You are a planner.

Based on the analysis below:

Analysis:
{analysis}

Output ONLY valid JSON. Example:

{{
  "chain": "summarization_chain",     #"summarization_chain" or "qa_chain"
  "execution_mode": "single_step",    #"single_step" or "multi_step"
  "policy": {{                          
      "compression_ratio": 0.2,       # float 0-1
      "output_style": "paragraph",    # "paragraph" or "bullet"
      "tone": "neutral"               # "academic" or "neutral"
  }},
  "use_retrieval": false,             #bool
  "use_citation": false               #bool
}}

Generate JSON following this example exactly. Do not output any extra text.
"""

        raw = llm.generate(prompt, do_sample=False, temperature=0.3, top_p=0.9)
        raw = extract_json(raw)
        plan: PlanState = json.loads(raw)

        state["plan"] = plan
        return state

    return planner_node
