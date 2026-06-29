"""
BGE Embedding wrapper — full Phase 2 implementation.

Key decisions:
- BAAI/bge-small-en-v1.5: best quality/speed ratio for retrieval tasks (MTEB leaderboard).
  384-dim, much faster than large models, outperforms all-MiniLM on retrieval benchmarks.
- BGE requires query prefix "Represent this sentence for searching relevant passages: "
  for asymmetric retrieval (query vs document). Documents are embedded without prefix.
- normalize_embeddings=True: enables cosine similarity via dot product (faster at search time).
- Singleton pattern: model loads once per process, not per request.
"""
import threading
from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# BGE query prefix — required for correct asymmetric retrieval
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """
    Thread-safe singleton BGE embedding model.
    One instance per process, shared across requests.
    """

    _instance: ClassVar["Embedder | None"] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:  # type: ignore[has-type]
            return
        self._model_name = settings.embedding_model
        self._model: SentenceTransformer | None = None
        self._initialized = True
        logger.info("embedder_singleton_created", model=self._model_name)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("embedder_loading", model=self._model_name)
            self._model = SentenceTransformer(
                self._model_name,
                device="cpu",
            )
            logger.info(
                "embedder_ready",
                model=self._model_name,
                dim=self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Embed document chunks (no query prefix).
        Used during ingestion.
        """
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )
        return embeddings.tolist()  # type: ignore[union-attr]

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query with BGE prefix.
        Used at retrieval time.
        """
        model = self._get_model()
        prefixed = BGE_QUERY_PREFIX + query
        embedding = model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()  # type: ignore[union-attr]

    def cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Cosine similarity between two normalized vectors (dot product suffices)."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        return float(np.dot(a, b))
