# summarize.py (cleaned & optimized for performance + future fact use)

import json
import requests
from pymongo import MongoClient
from tqdm import tqdm
import os

# 🔹 Mistral wrapper via Ollama
def summarize_with_mistral(text):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": f"Summarize this for a UTD academic chatbot:\n\n{text}\n\nSummary:",
                "stream": False
            }
        )
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return None

# 🔹 Summarize and save
def main():
    os.makedirs("../../../../../UTD_chatbot/chatbot_mini/hybrid_chunks_mistral", exist_ok=True)

    # Connect to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["utdallas_db_mini"]
    collection = db["scraped_data"]

    docs = list(collection.find({}))
    print(f"🔁 Loaded {len(docs)} documents from MongoDB")

    summarized_data = []
    skipped_urls = []

    for doc in tqdm(docs, desc="🔍 Summarizing"):
        try:
            full_text = "\n".join([
                doc.get("title", ""),
                *doc.get("headings", []),
                *doc.get("paragraphs", []),
                *doc.get("lists", []),
                *doc.get("divs", []),
                *doc.get("spans", [])
            ]).strip()

            if not full_text or len(full_text.split()) < 20:
                skipped_urls.append(doc.get("url", "unknown_url"))
                continue

            # Truncate to avoid overloading LLM
            prompt_text = full_text[:5000]

            summary = summarize_with_mistral(prompt_text)

            if not summary:
                skipped_urls.append(doc.get("url", "unknown_url"))
                continue

            summarized_data.append({
                "url": doc.get("url", ""),
                "title": doc.get("title", ""),
                "summary": summary,
                "full_text": full_text
            })

        except Exception as e:
            print(f"❌ Error on {doc.get('url')}: {e}")
            skipped_urls.append(doc.get("url", "unknown_url"))

    # ✅ Save summarized output
    with open("../../../../../UTD_chatbot/chatbot_mini/hybrid_chunks_mistral/summarized1.json", "w", encoding="utf-8") as f:
        json.dump(summarized_data, f, indent=2)

    with open("../../../../../UTD_chatbot/chatbot_mini/hybrid_chunks_mistral/skipped_urls.txt", "w", encoding="utf-8") as f:
        for url in skipped_urls:
            f.write(url + "\n")

    print(f"\n✅ Done! {len(summarized_data)} summarized, {len(skipped_urls)} skipped.")

# 🔹 Only run if this file is directly executed
if __name__ == "__main__":
    main()
