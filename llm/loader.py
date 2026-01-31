from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llm.inference import LocalLLM, QWenAPILLM

def load_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto"
    )
    return LocalLLM(tokenizer, model)


def load_api_llm(model: str = "qwen-plus", api_key: str = None, base_url: str = None, timeout: float = 30.0):
    return QWenAPILLM(model=model, api_key= api_key, base_url=base_url, timeout=timeout)
