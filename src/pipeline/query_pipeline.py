"""
Full RAG query pipeline — Phase 3 complete implementation.

Execution order:
  1. Emergency guard          → bypass entire pipeline if crisis detected
  2. Input sanitization       → block prompt injection
  3. Semantic cache lookup    → return cached answer if similar query exists
  4. Query embedding          → BGE embed with query prefix
  5. Parallel retrieval       → dense (Qdrant) + sparse (BM25) simultaneously
  6. RRF fusion               → merge ranked lists into single ranking
  7. Parent expansion         → swap child chunk text for full parent context
  8. Cross-encoder reranking  → BGE reranker scores (query, passage) pairs
  9. Session memory load      → fetch conversation history from PostgreSQL
 10. LLM generation           → Groq with structured prompt + citations
 11. Cache store              → persist answer for future similar queries
 12. Session memory save      → persist this turn to PostgreSQL

Every step is logged. Latency tracked end-to-end and per-step.
"""
import asyncio
import time
from typing import Any

from src.api.schemas import SourceDocument
from src.cache.semantic_cache import SemanticCache
from src.config import settings
from src.generation.llm import generate_answer
from src.guardrails.emergency import check_emergency
from src.guardrails.safety import sanitize_query
from src.logging_config import get_logger
from src.memory.session import SessionMemory
from src.models import RetrievedChunk
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.parent_store import ParentStore
from src.retrieval.qdrant_store import QdrantStore
from src.retrieval.reranker import Reranker

logger = get_logger(__name__)

# Singletons instantiated once per process
_embedder: Embedder | None = None
_qdrant: QdrantStore | None = None
_bm25: BM25Store | None = None
_parent_store: ParentStore | None = None
_reranker: Reranker | None = None
_cache: SemanticCache | None = None


def _get_components() -> tuple[Embedder, QdrantStore, BM25Store, ParentStore, Reranker, SemanticCache]:
    """Lazy-init all pipeline components as process-level singletons."""
    global _embedder, _qdrant, _bm25, _parent_store, _reranker, _cache
    if _embedder is None:
        _embedder = Embedder()
        _qdrant = QdrantStore()
        _bm25 = BM25Store()
        _parent_store = ParentStore()
        _reranker = Reranker()
        _cache = SemanticCache()
    # 💡 Added: Explicit type narrowing assertions for mypy compliance
    assert _embedder is not None
    assert _qdrant is not None
    assert _bm25 is not None
    assert _parent_store is not None
    assert _reranker is not None
    assert _cache is not None    
    return _embedder, _qdrant, _bm25, _parent_store, _reranker, _cache


def _expand_to_parents(
    chunks: list[RetrievedChunk],
    parent_store: ParentStore,
) -> list[RetrievedChunk]:
    """
    Replace child chunk text with parent chunk text for richer LLM context.
    If a parent is not found (e.g. index rebuilt), keeps child text.
    Deduplicates by parent_id — same parent from multiple children → one entry.
    """
    seen_parents: set[str] = set()
    expanded: list[RetrievedChunk] = []

    for chunk in chunks:
        if chunk.parent_id in seen_parents:
            continue
        parent = parent_store.get(chunk.parent_id)
        if parent:
            expanded.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                text=parent.text,           # full parent context
                source_file=chunk.source_file,
                page=chunk.page,
                section=parent.section or chunk.section,
                score=chunk.score,
                retrieval_method=chunk.retrieval_method,
            ))
        else:
            expanded.append(chunk)          # fallback: keep child text
        seen_parents.add(chunk.parent_id)

    return expanded


def _chunks_to_sources(chunks: list[RetrievedChunk]) -> list[SourceDocument]:
    """Convert internal RetrievedChunk objects to API SourceDocument schema."""
    return [
        SourceDocument(
            doc_id=chunk.chunk_id,
            content=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
            source_file=chunk.source_file,
            page=chunk.page,
            section=chunk.section or None,
            score=round(chunk.score, 4),
        )
        for chunk in chunks
    ]


class QueryPipeline:
    """
    Stateless orchestrator — components are process-level singletons injected via _get_components().
    A new QueryPipeline() per request is cheap (no model loading).
    """

    async def run(
        self,
        query: str,
        session_id: str,
        source_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full RAG query pipeline.

        Returns dict with:
            answer, sources, confidence, cache_hit, is_emergency
        """
        t_total = time.perf_counter()

        # ── Step 1: Emergency guard ───────────────────────────────────────────
        is_emergency, emergency_response = check_emergency(query)
        if is_emergency:
            return {
                "answer": emergency_response,
                "sources": [],
                "confidence": 1.0,   # maximum — this is a definitive response
                "cache_hit": False,
                "is_emergency": True,
            }

        # ── Step 2: Input sanitization ────────────────────────────────────────
        is_safe, cleaned_query = sanitize_query(query)
        if not is_safe:
            return {
                "answer": cleaned_query,   # contains the rejection reason
                "sources": [],
                "confidence": 0.0,
                "cache_hit": False,
                "is_emergency": False,
            }

        embedder, qdrant, bm25, parent_store, reranker, cache = _get_components()

        # ── Step 3: Semantic cache lookup ─────────────────────────────────────
        cached = cache.get(cleaned_query)
        if cached:
            logger.info("query_cache_hit", session=session_id)
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "confidence": cached["confidence"],
                "cache_hit": True,
                "is_emergency": False,
            }

        # ── Step 4: Embed query ───────────────────────────────────────────────
        t_embed = time.perf_counter()
        query_vector = embedder.embed_query(cleaned_query)
        logger.debug("query_embedded", ms=round((time.perf_counter() - t_embed) * 1000, 1))

        # ── Step 5: Parallel retrieval (dense + BM25) ─────────────────────────
        t_retrieve = time.perf_counter()
        top_k = settings.retrieval_top_k

        # Run both retrievals — asyncio allows overlap even though both are CPU-bound
        # In production with async Qdrant client these would be truly parallel
        dense_results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: qdrant.search(query_vector, top_k, source_filter)
        )
        bm25_results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: bm25.search(cleaned_query, top_k, source_filter)
        )

        logger.info(
            "retrieval_complete",
            dense=len(dense_results),
            bm25=len(bm25_results),
            ms=round((time.perf_counter() - t_retrieve) * 1000, 1),
        )

        # ── Step 6: RRF fusion ────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)

        if not fused:
            no_context_answer = (
                "I don't have enough information in the available documents to answer this question.\n\n"
                "---\n⚠️ *Please ensure relevant medical documents have been ingested first.*"
            )
            return {
                "answer": no_context_answer,
                "sources": [],
                "confidence": 0.0,
                "cache_hit": False,
                "is_emergency": False,
            }

        # ── Step 7: Parent expansion ──────────────────────────────────────────
        expanded = _expand_to_parents(fused, parent_store)

        # ── Step 8: Cross-encoder reranking ──────────────────────────────────
        t_rerank = time.perf_counter()
        reranked = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: reranker.rerank(cleaned_query, expanded, top_n=settings.rerank_top_n),
        )
        logger.info(
            "reranking_complete",
            candidates=len(expanded),
            selected=len(reranked),
            ms=round((time.perf_counter() - t_rerank) * 1000, 1),
        )

        # ── Step 9: Session memory load ───────────────────────────────────────
        memory = SessionMemory(session_id)
        chat_history = memory.format_for_prompt()

        # ── Step 10: LLM generation ───────────────────────────────────────────
        t_llm = time.perf_counter()
        answer, confidence = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_answer(cleaned_query, reranked, chat_history),
        )
        logger.info(
            "generation_complete",
            ms=round((time.perf_counter() - t_llm) * 1000, 1),
            confidence=round(confidence, 3),
        )

        sources = _chunks_to_sources(reranked)

        # ── Step 11: Cache store ──────────────────────────────────────────────
        cache.set(cleaned_query, answer, [s.model_dump() for s in sources], confidence)

        # ── Step 12: Session memory save ──────────────────────────────────────
        memory.add_turn(user_message=cleaned_query, assistant_message=answer)

        total_ms = round((time.perf_counter() - t_total) * 1000, 1)
        logger.info(
            "query_pipeline_complete",
            session=session_id,
            total_ms=total_ms,
            sources=len(sources),
            confidence=round(confidence, 3),
        )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "cache_hit": False,
            "is_emergency": False,
        }
