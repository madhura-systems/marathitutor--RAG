import sys
from app.config.settings import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_service import ChromaService

s = get_settings()
emb = EmbeddingService(s)
emb.initialize()
cs = ChromaService(emb, s)
store = cs.create()

collection = store._collection
all_docs = collection.get(include=["documents", "metadatas"])

found = []
for i, doc in enumerate(all_docs["documents"]):
    meta = all_docs["metadatas"][i]
    if "धास्तावलेला" in doc or "धास्ताव" in doc or "घाबरलेला" in doc:
        found.append((meta, doc))

with open("scratch/found_docs.txt", "w", encoding="utf-8") as f:
    f.write(f"Total docs in collection: {len(all_docs['documents'])}\n")
    f.write(f"Found matches: {len(found)}\n\n")
    for meta, doc in found:
        f.write(f"--- Meta: {meta} ---\n{doc}\n\n")

print(f"Total docs: {len(all_docs['documents'])}, Matches: {len(found)}")
