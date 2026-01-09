import torch

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

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
            })

        outputs = self.model.generate(
            **inputs,
            **gen_kwargs
        )

        # 只 decode 新生成部分
        gen_tokens = outputs[0][input_len:]

        return self.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True
        )
