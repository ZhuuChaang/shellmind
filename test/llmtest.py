# /mnt/sdb1/zc/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct


from llm.loader import load_llm

llm = load_llm("/mnt/sdb1/zc/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct")

print(
    llm.generate("How to list files in Linux?")
)

