# # test_chroma_count.py
# import chromadb

# client = chromadb.PersistentClient(path="./chroma_store")
# collection = client.get_collection("utd_chunks_mistral")
# count = collection.count()
# print(f"📦 Total documents in ChromaDB: {count}")

# semantic_search_debug.py
# from sentence_transformers import SentenceTransformer
# import chromadb

# model = SentenceTransformer("intfloat/e5-small-v2")
# client = chromadb.PersistentClient(path="./chroma_store")
# collection = client.get_collection("utd_chunks_mistral")

# query = "what programs are offered at utd?"
# embedding = model.encode(f"query: {query.lower()}").tolist()

# results = collection.query(query_embeddings=[embedding], n_results=5)

# for i, doc in enumerate(results["documents"][0]):
#     print(f"\n[{i+1}] 📄 {doc[:300]}...\n")

import json
from pathlib import Path

# Paths to your summary files
BASE = Path("../../../../../UTD_chatbot/chatbot_mini/hybrid_chunks_mistral")
current_file = BASE / "summarized.json"
older_file   = BASE / "summarized1.json"
output_file  = BASE / "summarized_merged.json"

# Load both lists of documents
with open(current_file, "r", encoding="utf-8") as f:
    current = json.load(f)
with open(older_file, "r", encoding="utf-8") as f:
    older = json.load(f)

# Build a set of the full‐content signatures from 'current'
seen = set()
for doc in current:
    # Create a canonical string of the entire doc for comparison
    seen.add(json.dumps(doc, sort_keys=True))

# Start merged list with all current docs
merged = list(current)

# Append only those older docs whose full content isn't already seen
for doc in older:
    signature = json.dumps(doc, sort_keys=True)
    if signature not in seen:
        seen.add(signature)
        merged.append(doc)

# Save the merged, deduplicated list
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"✅ Merged {len(current)} + {len(older)} ➝ {len(merged)} unique documents saved to {output_file}")

