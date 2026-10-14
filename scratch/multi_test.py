"""Test multiple queries to verify universal hybrid retrieval works."""
import urllib.request
import json

tests = [
    ("dhastaavlela", 8),           # Single word - vocab lookup
    ("saamsum hone mhanje", 8),     # Saamsum
    ("shabdache artha sanga", 8),   # General vocab question
    ("give me all chapter names", 8),  # TOC query
    ("Shankar kon hota", 8),         # Story character
    ("kavicha nav kaay", 7),         # Std 7 question
]

with open("scratch/multi_test_results.json", "w", encoding="utf-8") as out:
    results = []
    for question, standard in tests:
        try:
            data = json.dumps({"question": question, "standard": standard}).encode("utf-8")
            req = urllib.request.Request(
                "http://localhost:8000/api/ask",
                data,
                {"Content-Type": "application/json"}
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            results.append({
                "question": question,
                "standard": standard,
                "marathi_q": resp.get("marathi_question"),
                "answer": resp.get("answer", "")[:200],
                "pages": resp.get("page_numbers", []),
                "num_chunks": len(resp.get("retrieved_chunks", [])),
                "top_chunk_score": resp["retrieved_chunks"][0]["score"] if resp.get("retrieved_chunks") else None,
            })
        except Exception as e:
            results.append({"question": question, "error": str(e)})

    json.dump(results, out, indent=2, ensure_ascii=False)

print("Done")
