import numpy as np
import json
from rag.vectorindex import FaissVectorIndex
from rag.ebdmanager import EmbeddingManager
from typing import List, Optional

def retrieve_rag_chunks(
    query: str,
    top_k: int = 5,
    embeddings_path: str = "data/embeddings/embeddings.npy",
    chunk_ids_path: str = "data/vector_index/chunk_ids.json",
    chunk_texts_path: str = "data/corpus/chunks.json",
    embedding_model: Optional[str] = "/mnt/sdb1/zc/.cache/modelscope/hub/models/SJ007NB/text2vec-base-multilingual"
) -> List[str]:
    """
    使用 RAG 从向量索引中检索与 query 最相关的文本 chunk。

    参数:
        query: 用户查询字符串
        top_k: 返回最相似的文本数量
        embeddings_path: npy 文件路径，包含向量索引的 embeddings
        chunk_ids_path: json 文件路径，包含 embeddings 对应的 chunk_id
        chunk_texts_path: json 文件路径，包含 chunk_id 对应的文本
        embedding_model: EmbeddingManager 模型名称或路径

    返回:
        List[str]: 检索到的文本 chunk 列表，按相似度从高到低排序
    """
    # 1. 加载 chunk_ids
    with open(chunk_ids_path, "r") as f:
        chunk_ids = json.load(f)

    # 2. 加载 embeddings 并构建 Faiss 索引
    embeddings = np.load(embeddings_path)
    index = FaissVectorIndex(dim=embeddings.shape[1])
    index.build(embeddings, chunk_ids)

    # 3. 初始化 EmbeddingManager
    emb_mgr = EmbeddingManager(model_name=embedding_model)

    # 4. 将 query 转成向量
    query_embedding = emb_mgr.encode([query])  # shape [1, D]

    # 5. 搜索 top_k 相似向量
    results = index.search(query_embedding, top_k=top_k)  # [(chunk_id, score), ...]

    # 6. 映射回文本
    with open(chunk_texts_path, "r") as f:
        chunk_texts = json.load(f)

    chunk_text_map = {cid: text for cid, text in zip(chunk_ids, chunk_texts)}
    retrieved_texts = [chunk_text_map[cid] for cid, score in results]

    return retrieved_texts