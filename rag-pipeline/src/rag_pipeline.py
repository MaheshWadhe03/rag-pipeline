"""
RAG Pipeline — Core Module
Retrieval-Augmented Generation using FAISS + HuggingFace + Groq/OpenAI
Author: Mahesh (Max)
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

# Vector store + embeddings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Document loading & splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)

# LLM (supports Groq or OpenAI)
from groq import Groq

# ── Config ─────────────────────────────────────────────────────────────────────

EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, free, 384-dim
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 4                     # retrieved chunks per query
LLM_MODEL     = "llama3-8b-8192"     # Groq model (free tier)

# ── RAG Pipeline ───────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Load documents  (PDF / TXT / directory)
      2. Chunk & embed   (SentenceTransformers → FAISS)
      3. Retrieve        (top-k cosine similarity)
      4. Generate        (Groq LLM with retrieved context)
    """

    def __init__(self, groq_api_key: Optional[str] = None):
        print("⚙  Loading embedding model …")
        self.embedder   = SentenceTransformer(EMBED_MODEL)
        self.splitter   = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.index      = None   # FAISS index
        self.chunks     = []     # raw text of every stored chunk
        self.metadata   = []     # source file per chunk

        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GROQ_API_KEY env var or pass groq_api_key= to RAGPipeline()"
            )
        self.llm = Groq(api_key=api_key)
        print("✅ Pipeline ready.\n")

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def load_documents(self, path: str) -> list:
        """Load from a file (PDF/TXT) or a directory."""
        p = Path(path)
        docs = []
        if p.is_dir():
            for fp in p.rglob("*"):
                docs.extend(self._load_file(str(fp)))
        else:
            docs.extend(self._load_file(str(p)))
        print(f"📄 Loaded {len(docs)} document segment(s) from '{path}'")
        return docs

    def _load_file(self, path: str) -> list:
        if path.endswith(".pdf"):
            return PyPDFLoader(path).load()
        elif path.endswith(".txt") or path.endswith(".md"):
            return TextLoader(path, encoding="utf-8").load()
        return []

    def ingest(self, path: str):
        """Load → chunk → embed → index."""
        docs   = self.load_documents(path)
        chunks = self.splitter.split_documents(docs)
        texts  = [c.page_content for c in chunks]
        metas  = [c.metadata.get("source", path) for c in chunks]

        print(f"✂  Split into {len(texts)} chunk(s). Embedding …")
        t0       = time.time()
        vectors  = self.embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        vectors  = np.array(vectors, dtype="float32")
        print(f"   Done in {time.time()-t0:.1f}s")

        dim = vectors.shape[1]
        if self.index is None:
            # Inner-product on normalized vecs == cosine similarity
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(vectors)
        self.chunks.extend(texts)
        self.metadata.extend(metas)
        print(f"🗄  FAISS index now has {self.index.ntotal} vector(s).\n")

    def save_index(self, dir_path: str = "index"):
        """Persist FAISS index + chunk store to disk."""
        Path(dir_path).mkdir(exist_ok=True)
        faiss.write_index(self.index, f"{dir_path}/faiss.index")
        with open(f"{dir_path}/chunks.json", "w") as f:
            json.dump({"chunks": self.chunks, "metadata": self.metadata}, f)
        print(f"💾 Index saved to '{dir_path}/'")

    def load_index(self, dir_path: str = "index"):
        """Reload a previously saved index."""
        self.index = faiss.read_index(f"{dir_path}/faiss.index")
        with open(f"{dir_path}/chunks.json") as f:
            data = json.load(f)
        self.chunks   = data["chunks"]
        self.metadata = data["metadata"]
        print(f"📂 Loaded index with {self.index.ntotal} vector(s) from '{dir_path}/'")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Return top-k chunks most relevant to query."""
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("Index is empty — call ingest() first.")
        qvec = self.embedder.encode([query], normalize_embeddings=True)
        qvec = np.array(qvec, dtype="float32")
        scores, indices = self.index.search(qvec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text":   self.chunks[idx],
                "source": self.metadata[idx],
                "score":  float(score),
            })
        return results

    # ── Generation ─────────────────────────────────────────────────────────────

    def _build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
        )
        return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    def query(self, question: str, top_k: int = TOP_K, verbose: bool = False) -> dict:
        """Full RAG query: retrieve → generate → return answer + sources."""
        retrieved = self.retrieve(question, top_k=top_k)
        prompt    = self._build_prompt(question, retrieved)

        if verbose:
            print(f"\n🔍 Retrieved {len(retrieved)} chunk(s):")
            for i, r in enumerate(retrieved, 1):
                print(f"  [{i}] score={r['score']:.3f} | {r['source']}")
                print(f"      {r['text'][:120]}…\n")

        response = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()
        sources = list({r["source"] for r in retrieved})

        return {
            "question":  question,
            "answer":    answer,
            "sources":   sources,
            "retrieved": retrieved,
        }
