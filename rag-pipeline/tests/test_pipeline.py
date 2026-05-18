"""
Tests for RAG Pipeline
Run: pytest tests/ -v
"""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_txt(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    f = d / "test.txt"
    f.write_text(
        "RAG stands for Retrieval-Augmented Generation. "
        "FAISS is a library for efficient similarity search. "
        "Sentence Transformers produce dense embeddings for semantic search. "
        "LangChain is a framework for building LLM applications. "
        "Groq provides ultra-fast LLM inference using LPU hardware.",
        encoding="utf-8",
    )
    return str(f)


@pytest.fixture(scope="module")
def pipeline(sample_txt):
    """Build a pipeline with a mock LLM to avoid needing a real API key."""
    from unittest.mock import MagicMock, patch

    fake_key = "gsk_test_fake_key_12345"
    with patch("src.rag_pipeline.Groq") as MockGroq:
        # Mock the LLM response
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "RAG stands for Retrieval-Augmented Generation."
        MockGroq.return_value.chat.completions.create.return_value = mock_resp

        from src.rag_pipeline import RAGPipeline
        rag = RAGPipeline(groq_api_key=fake_key)
        rag.ingest(sample_txt)
        return rag


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_index_not_empty(self, pipeline):
        assert pipeline.index is not None
        assert pipeline.index.ntotal > 0

    def test_chunks_populated(self, pipeline):
        assert len(pipeline.chunks) > 0

    def test_metadata_matches_chunks(self, pipeline):
        assert len(pipeline.metadata) == len(pipeline.chunks)


class TestRetrieval:
    def test_retrieve_returns_results(self, pipeline):
        results = pipeline.retrieve("What is FAISS?", top_k=2)
        assert len(results) > 0

    def test_retrieve_fields(self, pipeline):
        results = pipeline.retrieve("LLM framework", top_k=1)
        assert "text" in results[0]
        assert "score" in results[0]
        assert "source" in results[0]

    def test_scores_between_0_and_1(self, pipeline):
        results = pipeline.retrieve("semantic search", top_k=3)
        for r in results:
            assert -0.1 <= r["score"] <= 1.1  # cosine can be slightly >1 due to float

    def test_top_k_respected(self, pipeline):
        results = pipeline.retrieve("anything", top_k=2)
        assert len(results) <= 2


class TestQuery:
    def test_query_returns_answer(self, pipeline):
        result = pipeline.query("What is RAG?")
        assert "answer" in result
        assert len(result["answer"]) > 0

    def test_query_returns_sources(self, pipeline):
        result = pipeline.query("What is RAG?")
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_query_returns_retrieved(self, pipeline):
        result = pipeline.query("LLM")
        assert "retrieved" in result


class TestPersistence:
    def test_save_and_load(self, pipeline, tmp_path):
        idx_dir = str(tmp_path / "idx")
        pipeline.save_index(idx_dir)

        from unittest.mock import MagicMock, patch
        with patch("src.rag_pipeline.Groq"):
            from src.rag_pipeline import RAGPipeline
            rag2 = RAGPipeline(groq_api_key="fake")
            rag2.load_index(idx_dir)

        assert rag2.index.ntotal == pipeline.index.ntotal
        assert rag2.chunks == pipeline.chunks
