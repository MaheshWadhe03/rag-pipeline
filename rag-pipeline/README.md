# 🔍 RAG Pipeline

> **Retrieval-Augmented Generation** from scratch — FAISS · LangChain · Sentence Transformers · Groq (Llama-3)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Meta%20AI-orange)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 What is RAG?

Standard LLMs hallucinate because they answer from fixed training weights. **RAG fixes this** by retrieving relevant documents at inference time and conditioning the answer on them — like letting the model "look things up" before speaking.

```
Query → Embed → FAISS Search → Top-K Chunks → LLM (Groq) → Grounded Answer
```

---

## 🏗️ Architecture

```
rag-pipeline/
├── src/
│   └── rag_pipeline.py     # Core: ingest · retrieve · generate
├── main.py                 # CLI: ingest / query / chat
├── app.py                  # Streamlit web UI
├── notebooks/
│   └── quickstart.ipynb    # Step-by-step notebook demo
├── tests/
│   └── test_pipeline.py    # Pytest suite
├── data/
│   └── sample.txt          # Example document
└── requirements.txt
```

### Tech Stack

| Layer | Library | Why |
|---|---|---|
| **Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | Fast, free, 384-dim, strong semantic quality |
| **Vector Store** | `faiss-cpu` (IndexFlatIP) | Sub-millisecond search; cosine similarity on normalised vecs |
| **Doc Loading** | `langchain-community` | PDF, TXT, MD — one unified API |
| **Chunking** | `RecursiveCharacterTextSplitter` | Smart overlap to preserve context |
| **LLM** | `groq` SDK — Llama-3 8B | Free tier, ~300 tok/s, zero cold-starts |
| **Web UI** | `streamlit` | Upload docs & query in-browser |

---

## ⚡ Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/rag-pipeline.git
cd rag-pipeline
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your free Groq key from https://console.groq.com
```

### 3. Ingest documents

```bash
python main.py ingest --path data/sample.txt
# Or ingest an entire folder:
python main.py ingest --path data/
```

### 4. Query

```bash
# One-shot query
python main.py query --question "What is FAISS?" --verbose

# Interactive chat
python main.py chat
```

### 5. Web UI (optional)

```bash
streamlit run app.py
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

All tests mock the LLM — **no API key needed to run the test suite.**

---

## 📖 How It Works (Step by Step)

### Step 1 — Ingest

```python
rag = RAGPipeline()
rag.ingest("data/")           # load → chunk → embed → FAISS index
rag.save_index("index/")      # persist to disk
```

Documents → `PyPDFLoader` / `TextLoader` → `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) → `SentenceTransformer.encode()` → `faiss.IndexFlatIP.add()`

### Step 2 — Retrieve

```python
results = rag.retrieve("What is RAG?", top_k=4)
# [{"text": "...", "source": "data/sample.txt", "score": 0.91}, ...]
```

Query string → embed → `faiss.index.search()` → top-k chunks ranked by cosine similarity.

### Step 3 — Generate

```python
result = rag.query("Explain RAG in simple terms.")
print(result["answer"])   # Grounded answer, no hallucination
print(result["sources"])  # Which documents were used
```

Retrieved chunks are injected into a strict prompt:
> *"Answer using ONLY the context below. If the answer isn't there, say so."*

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `LLM_MODEL` | `llama3-8b-8192` | Groq model ID |

Edit `src/rag_pipeline.py` to change these.

---

## 🚀 Extending This Project

| Idea | How |
|---|---|
| Swap to OpenAI | Replace `Groq` client with `openai.OpenAI`; update `LLM_MODEL` |
| Use ChromaDB | Replace FAISS with `chromadb`; keep embedder & splitter |
| Add re-ranking | Add `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker after retrieval |
| Multi-turn chat | Maintain conversation history, append `assistant` turns to messages |
| Eval with RAGAS | Use [ragas](https://github.com/explodinggradients/ragas) for faithfulness & relevance scoring |

---

## 📚 References

- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (2020)
- Johnson et al., [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734) (2017)
- Reimers & Gurevych, [Sentence-BERT](https://arxiv.org/abs/1908.10084) (2019)

---

## 👤 Author

**Mahesh (Max)** — BTech CSE (AI & ML)  
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)

---

## 📄 License

MIT — free to use, modify, and distribute.
