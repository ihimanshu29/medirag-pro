"""
Phase 2 tests: ingestion pipeline components.
All tests run without external services (Qdrant/Postgres mocked where needed).
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")

# ── Model tests ───────────────────────────────────────────────────────────────

def test_raw_page_creation():
    from src.models import RawPage
    page = RawPage(page_num=1, text="Blood pressure is 120/80 mmHg.", section="Vitals")
    assert page.page_num == 1
    assert "Blood pressure" in page.text
    assert page.tables == []


def test_child_chunk_defaults():
    from src.models import ChildChunk, ChunkType
    chunk = ChildChunk(
        chunk_id="abc123",
        parent_id="parent1",
        text="Hypertension is defined as systolic BP > 140 mmHg.",
        source_file="test.pdf",
        page=1,
    )
    assert chunk.chunk_type == ChunkType.TEXT
    assert chunk.embedding == []


# ── Chunker tests ─────────────────────────────────────────────────────────────

def make_raw_pages(n: int = 3):
    from src.models import RawPage
    return [
        RawPage(
            page_num=i + 1,
            text=f"# Section {i+1}\n\n"
                 + "The patient presented with symptoms of hypertension and tachycardia. "
                 * 20,
            section=f"Section {i+1}",
        )
        for i in range(n)
    ]


def test_chunker_produces_output():
    from src.ingestion.chunker import chunk_pages
    pages = make_raw_pages(3)
    parents, children = chunk_pages(pages, source_file="test.pdf")
    assert len(parents) > 0
    assert len(children) > 0


def test_chunker_child_has_parent_ref():
    from src.ingestion.chunker import chunk_pages
    pages = make_raw_pages(2)
    parents, children = chunk_pages(pages, source_file="test.pdf")

    parent_ids = {p.chunk_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids, (
            f"Child {child.chunk_id} has orphaned parent_id {child.parent_id}"
        )


def test_chunker_child_text_not_empty():
    from src.ingestion.chunker import chunk_pages
    pages = make_raw_pages(2)
    _, children = chunk_pages(pages, source_file="test.pdf")
    for child in children:
        assert len(child.text.strip()) >= 50, f"Child chunk too short: '{child.text[:30]}'"


def test_chunker_source_file_propagated():
    from src.ingestion.chunker import chunk_pages
    pages = make_raw_pages(1)
    parents, children = chunk_pages(pages, source_file="my_doc.pdf")
    assert all(p.source_file == "my_doc.pdf" for p in parents)
    assert all(c.source_file == "my_doc.pdf" for c in children)


def test_chunker_deterministic_ids():
    """Same content must produce same chunk IDs (idempotent re-ingest)."""
    from src.ingestion.chunker import chunk_pages
    pages = make_raw_pages(2)
    _, children1 = chunk_pages(pages, source_file="test.pdf")
    _, children2 = chunk_pages(pages, source_file="test.pdf")
    ids1 = {c.chunk_id for c in children1}
    ids2 = {c.chunk_id for c in children2}
    assert ids1 == ids2


def test_chunker_with_table():
    from src.models import RawPage
    from src.ingestion.chunker import chunk_pages
    page = RawPage(
        page_num=1,
        text="Drug dosage information follows.",
        tables=["| Drug | Dose |\n| --- | --- |\n| Aspirin | 100mg |"],
        section="Dosage",
    )
    parents, children = chunk_pages([page], source_file="test.pdf")
    # Table text should appear somewhere in the chunks
    all_text = " ".join(c.text for c in children) + " ".join(p.text for p in parents)
    assert "Aspirin" in all_text or "Drug" in all_text


# ── BM25 tests ───────────────────────────────────────────────────────────────

def make_child_chunks(n: int = 5):
    from src.models import ChildChunk, ChunkType
    terms = ["hypertension", "diabetes", "tachycardia", "pneumonia", "arthritis"]
    return [
        ChildChunk(
            chunk_id=f"chunk_{i}",
            parent_id=f"parent_{i}",
            text=f"Patient has {terms[i % len(terms)]}. Treatment includes medication.",
            source_file="test.pdf",
            page=i + 1,
            chunk_type=ChunkType.TEXT,
        )
        for i in range(n)
    ]


def test_bm25_add_and_search(tmp_path):
    from src.retrieval.bm25_store import BM25Store
    store = BM25Store(index_path=str(tmp_path / "bm25.pkl"))
    chunks = make_child_chunks(5)
    store.add_chunks(chunks)

    results = store.search("hypertension treatment", top_k=3)
    assert len(results) > 0
    assert results[0].score >= 0.0


def test_bm25_search_empty_index(tmp_path):
    from src.retrieval.bm25_store import BM25Store
    store = BM25Store(index_path=str(tmp_path / "bm25_empty.pkl"))
    results = store.search("anything", top_k=5)
    assert results == []


def test_bm25_remove_by_source(tmp_path):
    from src.retrieval.bm25_store import BM25Store
    store = BM25Store(index_path=str(tmp_path / "bm25_rm.pkl"))
    chunks = make_child_chunks(5)
    store.add_chunks(chunks)
    assert store.document_count == 5

    store.remove_by_source("test.pdf")
    assert store.document_count == 0


def test_bm25_source_filter(tmp_path):
    from src.models import ChildChunk, ChunkType
    from src.retrieval.bm25_store import BM25Store

    store = BM25Store(index_path=str(tmp_path / "bm25_filter.pkl"))
    chunks_a = [ChildChunk(
        chunk_id="a1", parent_id="pa", text="hypertension treatment aspirin",
        source_file="doc_a.pdf", page=1, chunk_type=ChunkType.TEXT,
    )]
    chunks_b = [ChildChunk(
        chunk_id="b1", parent_id="pb", text="hypertension medication",
        source_file="doc_b.pdf", page=1, chunk_type=ChunkType.TEXT,
    )]
    store.add_chunks(chunks_a + chunks_b)

    results = store.search("hypertension", top_k=5, source_filter="doc_a.pdf")
    assert all(r.source_file == "doc_a.pdf" for r in results)


# ── Parent store tests ────────────────────────────────────────────────────────

def test_parent_store_add_and_get(tmp_path, monkeypatch):
    monkeypatch.setenv("PARENT_CHUNKS_PATH", str(tmp_path / "parents.pkl"))
    from src.config import get_settings
    get_settings.cache_clear()

    from src.models import ParentChunk
    from src.retrieval.parent_store import ParentStore

    store = ParentStore()
    parent = ParentChunk(
        chunk_id="p1",
        text="Full context about hypertension management...",
        source_file="test.pdf",
        page_start=1,
        page_end=2,
        section="Treatment",
    )
    store.add([parent])
    result = store.get("p1")
    assert result is not None
    assert result.text == parent.text


def test_parent_store_missing_key(tmp_path, monkeypatch):
    monkeypatch.setenv("PARENT_CHUNKS_PATH", str(tmp_path / "parents2.pkl"))
    from src.config import get_settings
    get_settings.cache_clear()

    from src.retrieval.parent_store import ParentStore
    store = ParentStore()
    assert store.get("nonexistent") is None


# ── Embedder unit test (no model download in CI) ──────────────────────────────

def test_embedder_cosine_similarity():
    """Test cosine similarity math without loading the model."""
    import numpy as np
    from src.retrieval.embedder import Embedder

    emb = Embedder()
    # Two identical vectors should have similarity ~1.0
    v = [1.0, 0.0, 0.0]
    assert abs(emb.cosine_similarity(v, v) - 1.0) < 1e-6

    # Orthogonal vectors should have similarity ~0.0
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert abs(emb.cosine_similarity(v1, v2)) < 1e-6


# ── Loader unit tests (no real PDF needed) ────────────────────────────────────

def test_loader_rejects_unsupported_type():
    from src.ingestion.loader import load_document
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(Path("some_file.csv"))


def test_serialize_table():
    from src.ingestion.loader import _serialize_table
    table = [["Drug", "Dose", "Frequency"], ["Aspirin", "100mg", "Daily"]]
    result = _serialize_table(table)
    assert "Drug" in result
    assert "Aspirin" in result
    assert "|" in result  # Markdown table format


def test_serialize_table_handles_none():
    from src.ingestion.loader import _serialize_table
    table = [["Drug", None], [None, "100mg"]]
    result = _serialize_table(table)
    assert "Drug" in result
    assert "100mg" in result
