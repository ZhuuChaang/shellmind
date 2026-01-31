import torch
from transformers import GenerationConfig
from dataclasses import dataclass
from typing import Any, Optional
from openai import OpenAI

class LocalLLM:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        no_repeat_ngram_size = 3,
        repetition_penalty = 1.2 
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        # 基础生成参数
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        # 只在采样模式下传递 temperature/top_p
        if do_sample:
            gen_config.temperature = temperature
            gen_config.top_p = top_p
            gen_config.no_repeat_ngram_size = no_repeat_ngram_size
            gen_config.repetition_penalty = repetition_penalty

        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config
        )

        # 只 decode 新生成部分
        gen_tokens = outputs[0][input_len:]

        return self.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True
        )
    




@dataclass
class LLMResult:
    content: Optional[str]
    tool_calls: Optional[list[dict]]
    raw: Any


class QWenAPILLM:
    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model

    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        if response_format is not None:
            kwargs["response_format"] = response_format
            # e.g. {"type": "json_schema", "json_schema": {...}}

        resp = self.client.chat.completions.create(**kwargs)

        msg = resp.choices[0].message

        return LLMResult(
            content=msg.content,
            tool_calls=msg.tool_calls,
            raw=resp,
        )