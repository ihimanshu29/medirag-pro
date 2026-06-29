"""
Parent chunk store — persists the large parent chunks to disk.

At retrieval time:
  1. Search returns child chunk IDs (small, precise)
  2. We expand to the parent chunk (large, full context)
  3. Parent text goes to the LLM — richer context, less hallucination

Why not store parents in Qdrant?
  Parents are NOT embedded — they're only fetched by ID.
  A simple dict on disk (pickle) is faster and simpler than a DB query.
  At scale (>1M parents), move to Redis or PostgreSQL.
"""
import pickle
from pathlib import Path

from src.config import settings
from src.logging_config import get_logger
from src.models import ParentChunk

logger = get_logger(__name__)


class ParentStore:
    """Key-value store: parent_id → ParentChunk. Persisted as a pickle file."""

    def __init__(self) -> None:
        self._path = Path(settings.parent_chunks_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, ParentChunk] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "rb") as f:
                    self._store = pickle.load(f)
                logger.info("parent_store_loaded", count=len(self._store))
            except Exception as e:
                logger.warning("parent_store_load_failed", error=str(e))
                self._store = {}
        else:
            logger.info("parent_store_new")

    def _save(self) -> None:
        with open(self._path, "wb") as f:
            pickle.dump(self._store, f, protocol=pickle.HIGHEST_PROTOCOL)

    def add(self, parents: list[ParentChunk]) -> None:
        """Add parent chunks. Overwrites existing entries with same ID (idempotent)."""
        for p in parents:
            self._store[p.chunk_id] = p
        self._save()
        logger.info("parent_store_added", new=len(parents), total=len(self._store))

    def get(self, parent_id: str) -> ParentChunk | None:
        return self._store.get(parent_id)

    def remove_by_source(self, source_file: str) -> int:
        before = len(self._store)
        self._store = {
            pid: chunk for pid, chunk in self._store.items()
            if chunk.source_file != source_file
        }
        removed = before - len(self._store)
        if removed:
            self._save()
        return removed

    @property
    def count(self) -> int:
        return len(self._store)
