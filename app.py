"""
RAG Pipeline — Streamlit Web Demo
Run: streamlit run app.py
"""

import os
import streamlit as st
from src.rag_pipeline import RAGPipeline

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline Demo",
    page_icon="🔍",
    layout="wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Groq API Key", type="password",
                            value=os.getenv("GROQ_API_KEY", ""))
    top_k   = st.slider("Top-K chunks retrieved", 1, 8, 4)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)
    st.divider()
    st.markdown("""
**Model stack**
- 🧠 Embeddings: `all-MiniLM-L6-v2`
- 🗄 Vector store: FAISS (IndexFlatIP)
- 🤖 LLM: Llama-3 8B via Groq

**GitHub**
[mahesh-max/rag-pipeline](https://github.com)
""")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("RAG Pipeline Demo")
st.caption("Upload documents → ask questions → get grounded answers with sources.")

tab1, tab2 = st.tabs(["📄 Ingest Documents", "💬 Ask a Question"])

# ── Tab 1: Ingest ──────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload & Index Documents")
    uploaded = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Ingest", disabled=not uploaded or not api_key):
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            import tempfile, pathlib
            with st.spinner("Chunking & embedding …"):
                rag = RAGPipeline(groq_api_key=api_key)
                tmp = pathlib.Path(tempfile.mkdtemp())
                for uf in uploaded:
                    (tmp / uf.name).write_bytes(uf.read())
                rag.ingest(str(tmp))
                rag.save_index("index")
                st.session_state["rag_ready"] = True
            st.success(f"Indexed {len(uploaded)} file(s)!")

# ── Tab 2: Query ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Ask Your Documents")

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar first.")
    else:
        question = st.text_input("Your question", placeholder="What is the main topic of the document?")
        if st.button("Search & Answer", disabled=not question):
            try:
                rag = RAGPipeline(groq_api_key=api_key)
                rag.load_index("index")
                with st.spinner("Retrieving & generating …"):
                    result = rag.query(question, top_k=top_k)

                st.markdown("### 💬 Answer")
                st.success(result["answer"])

                st.markdown(f"**📎 Sources:** `{'` · `'.join(result['sources'])}`")

                if show_chunks:
                    with st.expander("Retrieved Chunks"):
                        for i, chunk in enumerate(result["retrieved"], 1):
                            st.markdown(f"**Chunk {i}** — score `{chunk['score']:.3f}` | `{chunk['source']}`")
                            st.code(chunk["text"], language=None)

            except FileNotFoundError:
                st.error("No index found — ingest some documents first (Tab 1).")
            except Exception as e:
                st.error(f"Error: {e}")
