"""
Semantic cache using diskcache + BGE embeddings.

Why semantic cache over exact-match cache?
  "What is hypertension?" and "Define hypertension" should return the same answer.
  Exact-match (Redis key-value) would miss this. Semantic cache stores
  the query embedding and does cosine similarity at lookup time.

Implementation:
  - diskcache: persists to disk, zero infrastructure (no Redis required)
  - BGE embeddings: consistent with our retrieval embedder
  - Threshold 0.92: high enough to avoid false positives (clinical queries
    can be superficially similar but clinically distinct)

Cache structure per entry:
  key:   f"sem_cache:{cache_index}"   (sequential int)
  value: {embedding, answer, sources, confidence}

Lookup: embed query → scan cache → if max_cosine >= threshold → return cached

At scale: replace diskcache with Redis + FAISS index over cached embeddings.
For portfolio scale (< 10K cached queries), this approach is fast enough.

Cache hit metric is tracked and exposed via Prometheus.
"""
import time
from pathlib import Path

import diskcache
import numpy as np

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class SemanticCache:
    """Disk-backed semantic cache for RAG query-answer pairs."""

    def __init__(self) -> None:
        cache_path = Path(settings.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        size_bytes = int(settings.cache_max_size_gb * 1024 ** 3)
        self._cache = diskcache.Cache(str(cache_path), size_limit=size_bytes)
        self._threshold = settings.cache_similarity_threshold
        logger.info(
            "semantic_cache_init",
            path=str(cache_path),
            threshold=self._threshold,
            entries=len(self._cache),
        )

    def _embed(self, query: str) -> list[float]:
        """Embed using the same BGE model as retrieval (lazy import — singleton)."""
        from src.retrieval.embedder import Embedder
        return Embedder().embed_query(query)

    def _cosine(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        return float(np.dot(va, vb))   # both are L2-normalized by BGE

    def get(self, query: str) -> dict | None:
        """
        Look up a semantically similar cached response.
        Returns the cached payload or None if no match above threshold.
        """
        if len(self._cache) == 0:
            return None

        t0 = time.perf_counter()
        query_emb = self._embed(query)

        best_score = -1.0
        best_payload = None

        for key in self._cache:
            entry = self._cache.get(key)
            if entry is None:
                continue
            score = self._cosine(query_emb, entry["embedding"])
            if score > best_score:
                best_score = score
                best_payload = entry

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if best_score >= self._threshold and best_payload is not None:
            logger.info(
                "cache_hit",
                score=round(best_score, 4),
                elapsed_ms=round(elapsed_ms, 1),
            )
            self._increment_hit_counter()
            return best_payload

        logger.debug(
            "cache_miss",
            best_score=round(best_score, 4),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return None

    def set(self, query: str, answer: str, sources: list, confidence: float) -> None:
        """Store a new query-answer pair in the cache."""
        embedding = self._embed(query)
        key = f"sem:{hash(query) & 0xFFFFFFFF}"
        self._cache.set(key, {
            "embedding": embedding,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        })
        logger.debug("cache_set", key=key, entries=len(self._cache))

    def _increment_hit_counter(self) -> None:
        try:
            from src.api.main import CACHE_HITS
            CACHE_HITS.inc()
        except Exception:
            pass

    @property
    def size(self) -> int:
        return len(self._cache)
