import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np


class FaissVectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids: List[int] = []

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def build(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[int],
        normalize: bool = True,
    ):
        """
        embeddings: [N, D] float32
        chunk_ids:  length N
        """
        assert embeddings.shape[1] == self.dim
        assert len(chunk_ids) == embeddings.shape[0]

        if normalize:
            embeddings = self._normalize(embeddings)

        self.index.add(embeddings)
        self.chunk_ids = chunk_ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        normalize: bool = True,
    ) -> List[Tuple[int, float]]:
        """
        return: [(chunk_id, score), ...]
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[None, :]

        if normalize:
            query_embedding = self._normalize(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for emb_idx, score in zip(indices[0], scores[0]):
            if emb_idx == -1:
                continue
            chunk_id = self.chunk_ids[emb_idx]
            results.append((chunk_id, float(score)))
        return results

    def save(self, dir_path: str):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(dir_path / "faiss.index"))
        with open(dir_path / "chunk_ids.json", "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f)

        meta = {
            "dim": self.dim,
            "metric": "ip",
            "normalized": True,
            "num_vectors": len(self.chunk_ids),
        }
        with open(dir_path / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, dir_path: str):
        dir_path = Path(dir_path)

        index = faiss.read_index(str(dir_path / "faiss.index"))
        with open(dir_path / "chunk_ids.json", "r", encoding="utf-8") as f:
            chunk_ids = json.load(f)

        dim = index.d
        obj = cls(dim)
        obj.index = index
        obj.chunk_ids = chunk_ids
        return obj
