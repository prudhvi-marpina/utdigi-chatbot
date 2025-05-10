import os, json
from pathlib import Path
from datetime import datetime
import time

import streamlit as st
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ────────────────────────────────────────────────────────────────────────────────
# 0) Streamlit config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 UTDIGI ChatBot", layout="wide")

# ────────────────────────────────────────────────────────────────────────────────
# 1) Load & cache embedding model
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# ────────────────────────────────────────────────────────────────────────────────
# 2) Load & precompute static chunks and embeddings
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_static_data():
    data_path = Path(__file__).parent / "hybrid_chunks_mistral" / "summarized_merged.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    documents = []
    for entry in data:
        full = entry["summary"] + "\n\n" + entry["full_text"]
        documents.append(Document(page_content=full, metadata=entry))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    static_chunks = splitter.split_documents(documents)
    model = load_model()
    static_embs = model.encode([c.page_content for c in static_chunks])
    return static_chunks, static_embs

# ────────────────────────────────────────────────────────────────────────────────
# 3) Retrieval helpers
# ────────────────────────────────────────────────────────────────────────────────
def get_relevant(query, model, chunks, embs, threshold=0.5, min_chunks=4, max_chunks=8):
    q_emb = model.encode([query])
    sims = cosine_similarity(q_emb, embs)[0]
    idxs = np.argsort(sims)[::-1]
    selected = []
    for i in idxs:
        if sims[i] >= threshold:
            selected.append(chunks[i])
        if len(selected) >= max_chunks:
            break
    if len(selected) < min_chunks:
        for i in idxs:
            if chunks[i] not in selected:
                selected.append(chunks[i])
            if len(selected) >= min_chunks:
                break
    return selected

def dynamic_split(doc, query, min_size=300, max_size=600):
    size = max(min(len(query) * 10, max_size), min_size)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=int(size * 0.1))
    return splitter.split_documents([doc])

# ────────────────────────────────────────────────────────────────────────────────
# 4) Initialize data and session state
# ────────────────────────────────────────────────────────────────────────────────
model = load_model()
static_chunks, static_embs = load_static_data()
if 'history' not in st.session_state:
    st.session_state.history = []  # list of {role, content, time, url, score}
if 'feedback' not in st.session_state:
    st.session_state.feedback = {}  # assistant turn index -> bool

# ────────────────────────────────────────────────────────────────────────────────
# 5) Render header
# ────────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#004B2D; background-color:#FF8200; padding:8px; border-radius:6px;'>"
    "📚 UTDIGI ChatBot</h1>",
    unsafe_allow_html=True
)

# ────────────────────────────────────────────────────────────────────────────────
# 6) User input and response generation
# ────────────────────────────────────────────────────────────────────────────────
user_query = st.chat_input("Enter your question about UT Dallas…")
if user_query:
    # record user
    st.session_state.history.append({
        'role': 'user',
        'content': user_query,
        'time': datetime.now(),
        'url': ''
    })

    # Two-stage retrieval
    coarse = get_relevant(user_query, model, static_chunks, static_embs,
                           threshold=0.6, min_chunks=3, max_chunks=5)
    fine = []
    for doc in coarse:
        fine.extend(dynamic_split(doc, user_query))
    fine_embs = model.encode([d.page_content for d in fine])
    top_chunks = get_relevant(user_query, model, fine, fine_embs,
                               threshold=0.5, min_chunks=4, max_chunks=4)

    # Compute relevance score
    q_emb = model.encode([user_query])
    sims = cosine_similarity(q_emb, fine_embs)[0]
    best_score = 0.0
    for c in top_chunks:
        for idx, doc in enumerate(fine):
            if doc.page_content == c.page_content:
                best_score = max(best_score, sims[idx])
                break

    # Build prompt
    context = "\n\n".join([c.page_content for c in top_chunks])
    urls = [c.metadata['url'] for c in top_chunks]
    first_url = list(dict.fromkeys(urls))[0] if urls else ''
    prompt = f"""
Context:
{context}

Query:
{user_query}

Instruction: Answer the above query based only on the context provided which will have a 
summary and full text for reference. Try answering first from the most relevant chunk and then from the remaining chunk without explicitly stating your approach.
If you can't find relevant info, say you are unable to find it. The answer must be around 3 to 4 lines.
End with:
"For more information, you can refer to the following link:"
{first_url}
"""

    # Call the LLM API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3:latest", "prompt": prompt, "stream": False}
    )
    if response.status_code == 200:
        answer = response.json().get('response', 'No response')
    else:
        answer = f"Error {response.status_code}: {response.text}"

    # record assistant with relevance score
    st.session_state.history.append({
        'role': 'assistant',
        'content': answer,
        'time': datetime.now(),
        'url': first_url,
        'score': best_score
    })

# ────────────────────────────────────────────────────────────────────────────────
# 7) Display chat history, relevance & inline feedback
# ────────────────────────────────────────────────────────────────────────────────
for idx, entry in enumerate(st.session_state.history):
    st.chat_message(entry['role']).write(entry['content'])
    if entry['role'] == 'assistant':
        # relevance
        if 'score' in entry:
            st.caption(f"Relevance: {entry['score']:.2f}")
        # thumbs up/down
        if idx not in st.session_state.feedback:
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button('👍', key=f'up_{idx}'):
                    st.session_state.feedback[idx] = True
            with col2:
                if st.button('👎', key=f'down_{idx}'):
                    st.session_state.feedback[idx] = False
        else:
            sel = st.session_state.feedback[idx]
            st.write('Feedback:', '👍' if sel else '👎')
