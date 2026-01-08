# run.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import HumanMessage
import torch

from llm.inference import LocalLLM
from llm.loader import load_llm
from agent.graph import build_graph



llm = load_llm("/mnt/sdb1/zc/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct")

graph = build_graph(llm)


with open("./test/testinput.txt", "r", encoding="utf-8") as f:
    content = f.read()

state = {
    "messages": [
        HumanMessage(
            content=content
        )
    ]
}

final_state = graph.invoke(state)

print(final_state["result"]["final_output"])
