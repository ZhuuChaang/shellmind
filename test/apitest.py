import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("HAPPY_NUMBER"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，你是谁？"},
    ]
)


print(completion.model_dump_json())