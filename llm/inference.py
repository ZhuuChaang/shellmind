import torch
from transformers import GenerationConfig

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
