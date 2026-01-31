from llm.loader import load_api_llm
from llm.inference import QWenAPILLM, LLMResult
import os

api_key=os.getenv("HAPPY_NUMBER")
client: QWenAPILLM = load_api_llm(model = "qwen-plus", api_key = api_key, base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1")



response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "analysis_result",
        "schema": {
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "enum": ["summarization", "qa"]},
                "input_type": {"type": "string", "enum": ["raw_text", "conversation"]},
                "input_length": {"type": "string", "enum": ["short", "medium", "long"]},
                "language": {"type": "string", "enum": ["zh", "en"]}
            },
            "required": ["task_type", "input_type", "language"],
        },
    },
}


res: LLMResult = client.generate(
    messages=[
        {"role": "system", "content": "You are a task analyzer, please analyse the user input, and then fill a format output."},
        {"role": "user", "content": "What is HH-PIM?"},
    ],

    response_format=response_format
)

print(res)
