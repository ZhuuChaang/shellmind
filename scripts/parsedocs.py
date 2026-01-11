import os
import numpy as np
import json

from rag.docmanager import DocumentManager
from rag.ebdmanager import EmbeddingManager
from rag.vectorindex import FaissVectorIndex

folder_path = "./data/docs/"
docs = os.listdir(folder_path) 
docs = [os.path.join(folder_path, f) 
             for f in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, f))]



dm = DocumentManager(chunk_size=500, chunk_overlap=50)
dm.load_pdfs(docs)
dm.save_metadata("data/corpus")




# 假设已经有 DocumentManager
# dm = DocumentManager(chunk_size=500, chunk_overlap=50)
# dm.load_metadata("data/corpus")
chunks = dm.get_chunks()
chunk_ids = list(range(len(chunks)))

emb_mgr = EmbeddingManager(model_name="/mnt/sdb1/zc/.cache/modelscope/hub/models/SJ007NB/text2vec-base-multilingual")
emb_mgr.build_embeddings(chunks, chunk_ids)
emb_mgr.save_embeddings("data/embeddings")

# 加载
emb_mgr.load_embeddings("data/embeddings")
print(emb_mgr.get_embeddings().shape)





embeddings = np.load("data/embeddings/embeddings.npy")  # [N, D]
with open("data/embeddings/chunk_ids.json", "r") as f:
    chunk_ids = json.load(f)

index = FaissVectorIndex(dim=embeddings.shape[1])
index.build(embeddings, chunk_ids)
index.save("data/vector_index")

