﻿# UTDIGI Chatbot

**UTDIGI** is a retrieval-based chatbot (“Your Smart Guide to UT Dallas”) that answers academic questions by:

1. **Scraping** UT Dallas academic pages  
2. **Storing** content in MongoDB  
3. **Preprocessing** with spaCy NER & LLM summarization  
4. **Embedding** & **semantic search** for retrieval  
5. Exposing a **Streamlit**/CLI interface for Q&A with source links  

---

## 📁 Repository Structure

```
.
├── utdigi_chatbot/          
│   ├── clear_mongodb.py     # utility to clear your MongoDB collections
│   ├── jindal_scrapy.py     # Scrapy spider for Jindal school pages
│   ├── scrapy_academics.py  # main Scrapy crawler for /academics
│   ├── ner_with_spacy_transformer.py  # spaCy transformer NER pipeline  
│   ├── summarize.py         # LLM-based summarization & chunking  
│   ├── utdigi_chatbot.py    # entry point: semantic search + QA loop  
│   └── test.py              # (optional) QA rerank evaluation  
├── requirements.txt         # pinned Python dependencies  
└── README.md                
```

---

## ⚙️ Setup

1. **Clone** the repo  
   ```bash
   git clone https://github.com/prudhvi-marpina/utdigi_chatbot.git
   cd utdigi_chatbot
   ```
2. **Create & activate** a virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. **Install** dependencies  
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure** your MongoDB URI  
   - Export `MONGO_URI` as an environment variable or update it in each script.

---

## ▶️ Usage

### 1. Scrape & ingest
```bash
# Scrape UT Dallas Academics
python utdigi_chatbot/scrapy_academics.py

# (Optional) Scrape Jindal site
python utdigi_chatbot/jindal_scrapy.py
```

### 2. Preprocess
```bash
# Clear old data (if needed)
python utdigi_chatbot/clear_mongodb.py

# Run NER & summarization pipelines
python utdigi_chatbot/ner_with_spacy_transformer.py
python utdigi_chatbot/summarize.py
```

### 3. Launch chatbot
```bash
python utdigi_chatbot/utdigi_chatbot.py
```
- Follow the prompts in your terminal (or Streamlit UI if you integrate it).

---

## 🔍 How it works

- **Web Scraping:** uses Scrapy (and Playwright when needed) to harvest HTML.  
- **NER:** spaCy transformer models tag entities for smarter chunking.  
- **Summarization:** LLM calls condense paragraphs into bite-size facts.  
- **Embedding & Search:** sentence-transformers + FAISS/ChromaDB for vector search.  
- **Reranking:** Cross-Encoder refines top hits by relevance score.  

---

## 🤝 Contributing

1. Fork & clone  
2. Create a feature branch  
3. Submit a PR with descriptions & tests  

---

## 📄 License

MIT © [prudhvi-marpina]
