from rag.docmanager import DocumentManager
from rag.ebdmanager import EmbeddingManager
import os

folder_path = "./data/docs/"
docs = os.listdir(folder_path) 
docs = [os.path.join(folder_path, f) 
             for f in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, f))]



dm = DocumentManager(chunk_size=500, chunk_overlap=50)
# dm.load_pdfs(docs)
# dm.save_metadata("data/corpus")




# 假设已经有 DocumentManager
dm = DocumentManager(chunk_size=500, chunk_overlap=50)
dm.load_metadata("data/corpus")
chunks = dm.get_chunks()
chunk_ids = list(range(len(chunks)))

emb_mgr = EmbeddingManager(model_name="shibing624/text2vec-base-multilingual")
emb_mgr.build_embeddings(chunks, chunk_ids)
emb_mgr.save_embeddings("data/embeddings")

# 加载
emb_mgr.load_embeddings("data/embeddings")
print(emb_mgr.get_embeddings().shape)
