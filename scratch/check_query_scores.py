import os
import sys
from app.config.settings import get_settings
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.chroma_service import ChromaService

s = get_settings()
emb = EmbeddingService(s)
emb.initialize()
cs = ChromaService(emb, s)
cs.create()

with open("scratch/scores_out.txt", "w", encoding="utf-8") as f:
    for query in ["धास्तावलेला", "धास्तावलेला अर्थ", "धास्तावलेला म्हणजे काय", "धास्तावलेला शब्दाचा अर्थ काय आहे?"]:
        res = cs.search(query=query, k=20, filter_dict={"standard": 8})
        f.write(f"\n=== QUERY: {query} ===\n")
        for i, r in enumerate(res):
            p = r.document.metadata.get("page_number")
            cid = r.document.metadata.get("chunk_id")
            has_word = "धास्तावलेला" in r.document.page_content
            f.write(f"#{i+1}: p{p} (score: {r.score:.4f}, has_word: {has_word}) | {r.document.page_content[:70]}...\n")
