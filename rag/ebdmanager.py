import os
import json
import numpy as np
from typing import List, Optional
from loguru import logger
import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingManager:
    """
    文本 embedding 管理器
    功能：
      - 文本 chunk → embedding
      - 支持保存和加载 embedding
      - 支持增量更新
    """

    def __init__(self, model_name: str = "shibing624/text2vec-base-multilingual", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.embeddings = None  # numpy array: [num_chunks, dim]
        self.chunk_ids = []  # 与 embeddings 对应的 chunk 索引

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        文本列表转 embedding
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            # 获取句向量: mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state  # [B, L, D]
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                emb = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1)
            elif hasattr(outputs, 'pooler_output'):
                emb = outputs.pooler_output
            else:
                raise ValueError("Cannot get sentence embedding from model output")
            all_embeddings.append(emb.cpu().numpy())
        return np.vstack(all_embeddings)

    def build_embeddings(self, chunks: List[str], chunk_ids: Optional[List[int]] = None, batch_size: int = 32):
        """
        生成 embedding 并保存到内存
        """
        embs = self.encode(chunks, batch_size=batch_size)
        if self.embeddings is None:
            self.embeddings = embs
            self.chunk_ids = chunk_ids or list(range(len(chunks)))
        else:
            # 增量添加
            self.embeddings = np.vstack([self.embeddings, embs])
            if chunk_ids:
                self.chunk_ids.extend(chunk_ids)
            else:
                start_idx = max(self.chunk_ids) + 1
                self.chunk_ids.extend(list(range(start_idx, start_idx + len(chunks))))
        logger.info(f"当前 embedding 数量: {len(self.embeddings)}")

    def save_embeddings(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(save_dir, "chunk_ids.json"), 'w', encoding='utf-8') as f:
            json.dump(self.chunk_ids, f, ensure_ascii=False, indent=2)
        logger.info(f"Embedding 已保存到 {save_dir}")

    def load_embeddings(self, save_dir: str):
        self.embeddings = np.load(os.path.join(save_dir, "embeddings.npy"))
        with open(os.path.join(save_dir, "chunk_ids.json"), 'r', encoding='utf-8') as f:
            self.chunk_ids = json.load(f)
        logger.info(f"Embedding 已加载，数量: {len(self.embeddings)}")

    def get_embeddings(self) -> np.ndarray:
        return self.embeddings

    def get_chunk_ids(self) -> List[int]:
        return self.chunk_ids
