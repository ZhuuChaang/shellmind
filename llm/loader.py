from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llm.inference import LocalLLM

def load_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto"
    )
    return LocalLLM(tokenizer, model)
