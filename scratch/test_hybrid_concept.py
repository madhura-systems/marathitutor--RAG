import re
from app.config.settings import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_service import ChromaService
from app.preprocessing.transliteration_service import TransliterationService

s = get_settings()
emb = EmbeddingService(s)
emb.initialize()
cs = ChromaService(emb, s)
store = cs.create()

collection = store._collection
all_data = collection.get(include=["documents", "metadatas"])
docs = all_data["documents"]
metas = all_data["metadatas"]

def hybrid_find(word, std=8):
    matches = []
    for i, d in enumerate(docs):
        m = metas[i]
        if std and m.get("standard") != std:
            continue
        if word in d:
            matches.append((m.get("page_number"), m.get("chapter"), m.get("chunk_id")))
    return matches

with open("scratch/test_hybrid_concept.txt", "w", encoding="utf-8") as f:
    f.write(f"Total documents loaded: {len(docs)}\n")
    for test_word in ["धास्तावलेला", "सामसूम", "शहामत", "खजील", "भूमिगत", "रंगपट"]:
        found = hybrid_find(test_word, std=8)
        f.write(f"Word '{test_word}' found in {len(found)} chunks: {found}\n")
