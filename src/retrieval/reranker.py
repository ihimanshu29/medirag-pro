"""
Cross-encoder reranker using BAAI/bge-reranker-base.

Why reranking matters (the most impactful single retrieval improvement):
- Bi-encoder (BGE embeddings): encodes query and document SEPARATELY.
  Fast, but loses fine-grained query-document interaction.
  Retrieves 20 candidates in ~50ms.

- Cross-encoder (BGE reranker): takes (query, document) as a PAIR.
  Full attention between query and document tokens → much higher precision.
  Slow per-pair (~5-10ms each), so only runs on the top-20 candidates.

Two-stage pipeline:
  Stage 1 (bi-encoder):    retrieve 20 candidates   — fast, high recall
  Stage 2 (cross-encoder): rerank to top 5          — slow, high precision

This is the standard production retrieval pattern at every serious ML company.
Without reranking, noisy chunks make it into the LLM context → hallucinations.

Model: BAAI/bge-reranker-base (278M params, runs comfortably on CPU)
  - Scores range: logits (unbounded), higher = more relevant
  - We normalise to [0, 1] via sigmoid for confidence reporting
"""
import threading
from typing import ClassVar

from src.config import settings
from src.logging_config import get_logger
from src.models import RetrievedChunk

logger = get_logger(__name__)


class Reranker:
    """
    Thread-safe singleton cross-encoder reranker.
    Loads model once, reuses across requests.
    """

    _instance: ClassVar["Reranker | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "Reranker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[has-type]
            return
        self._model_name = settings.reranker_model
        self._model = None
        self._initialized = True
        logger.info("reranker_singleton_created", model=self._model_name)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("reranker_loading", model=self._model_name)
            self._model = CrossEncoder(
                self._model_name,
                device="cpu",
                max_length=512,
            )
            logger.info("reranker_ready", model=self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """
        Rerank candidate chunks by relevance to query.

        Args:
            query:      The user's original query.
            candidates: Top-K chunks from hybrid retrieval (e.g. 20).
            top_n:      How many to keep for the LLM (e.g. 5).

        Returns:
            Top-N chunks sorted by cross-encoder score (best first).
            Scores normalised to [0, 1] via sigmoid.
        """
        if not candidates:
            return []

        # Clamp top_n to available candidates
        top_n = min(top_n, len(candidates))

        model = self._get_model()

        # Build (query, passage) pairs — cross-encoder input format
        pairs = [(query, chunk.text) for chunk in candidates]

        import numpy as np

        raw_scores: list[float] = model.predict(pairs, show_progress_bar=False).tolist()

        # Sigmoid normalisation: maps logits to [0, 1]
        def sigmoid(x: float) -> float:
            return float(1.0 / (1.0 + np.exp(-x)))

        scored = sorted(
            zip(raw_scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked: list[RetrievedChunk] = []
        for raw_score, chunk in scored[:top_n]:
            reranked.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                text=chunk.text,
                source_file=chunk.source_file,
                page=chunk.page,
                section=chunk.section,
                score=sigmoid(raw_score),
                retrieval_method="reranked",
            ))

        logger.debug(
            "reranker_complete",
            candidates_in=len(candidates),
            top_n_out=len(reranked),
            top_score=round(reranked[0].score, 3) if reranked else 0,
            bottom_score=round(reranked[-1].score, 3) if reranked else 0,
        )
        return reranked
