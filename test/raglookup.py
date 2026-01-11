import numpy as np
import json
from rag.vectorindex import FaissVectorIndex
from rag.ebdmanager import EmbeddingManager



# 假设已经加载：
# embedder: EmbeddingManager 实例
# index: FaissVectorIndex 实例
# chunk_id: {chunk_id: text}


with open("data/vector_index/chunk_ids.json", "r") as f:
    chunk_ids = json.load(f)

query = "What is HH-PIM? What is it's difference with powerlens?"
embeddings = np.load("data/embeddings/embeddings.npy") 
index = FaissVectorIndex(dim=embeddings.shape[1])
index.build(embeddings, chunk_ids)
emb_mgr = EmbeddingManager(model_name="/mnt/sdb1/zc/.cache/modelscope/hub/models/SJ007NB/text2vec-base-multilingual")

# 1. 先把 query 转成向量（注意 encode 输入是 list）
query_embedding = emb_mgr.encode([query])  # 输出 shape [1, D]

# 2. 在 Faiss 中搜索 top-k 相似向量
top_k = 5
results = index.search(query_embedding, top_k=top_k)  
# 这里 results 假设是 [(chunk_id, score), ...]

# 3. 映射回文本
with open("data/corpus/chunks.json", "r") as f:
    chunk_texts = json.load(f)

chunk_text_map = {cid: text for cid, text in zip(chunk_ids, chunk_texts)}  
retrieved_texts = [chunk_text_map[cid] for cid, score in results]

# 4. 打印结果
for i, text in enumerate(retrieved_texts, 1):
    print(f"[{i}] {text}\n")
