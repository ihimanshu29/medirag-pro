"""
BM25 sparse retrieval index.

Why BM25 alongside dense retrieval?
- Dense embeddings excel at semantic similarity ("heart attack" ↔ "myocardial infarction")
- BM25 excels at exact keyword matches (drug names, ICD codes, specific lab values)
- Medical queries frequently contain both — hybrid always outperforms either alone
- rank_bm25: pure Python, zero infrastructure, persists as a pickle file

Index is rebuilt incrementally on each ingest by loading existing index,
merging new documents, and saving back to disk.
"""
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import settings
from src.logging_config import get_logger
from src.models import ChildChunk, RetrievedChunk

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer. Good enough for BM25."""
    return text.lower().split()


class BM25Store:
    """
    Persistent BM25 index over child chunk texts.
    Saved to disk as a pickle (corpus + metadata), rebuilt into BM25Okapi on load.
    """

    def __init__(self, index_path: str | None = None) -> None:
        self._index_path = Path(index_path or settings.bm25_index_path)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory state
        self._corpus_tokens: list[list[str]] = []
        self._metadata: list[dict] = []   # parallel list to corpus
        self._bm25: BM25Okapi | None = None

        self._load_from_disk()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_from_disk(self) -> None:
        """Load existing index from disk if available."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "rb") as f:
                    saved = pickle.load(f)
                self._corpus_tokens = saved["corpus_tokens"]
                self._metadata = saved["metadata"]
                self._bm25 = BM25Okapi(self._corpus_tokens)
                logger.info(
                    "bm25_loaded",
                    path=str(self._index_path),
                    documents=len(self._corpus_tokens),
                )
            except Exception as e:
                logger.warning("bm25_load_failed", error=str(e))
                self._reset()
        else:
            logger.info("bm25_new_index", path=str(self._index_path))

    def _save_to_disk(self) -> None:
        """Persist the current index to disk."""
        with open(self._index_path, "wb") as f:
            pickle.dump(
                {"corpus_tokens": self._corpus_tokens, "metadata": self._metadata},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("bm25_saved", documents=len(self._corpus_tokens))

    def _reset(self) -> None:
        self._corpus_tokens = []
        self._metadata = []
        self._bm25 = None

    # ── Indexing ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[ChildChunk]) -> None:
        """
        Add new chunks to the BM25 index.
        Deduplicates by chunk_id — safe to call on re-ingest.
        """
        if not chunks:
            return

        existing_ids = {m["chunk_id"] for m in self._metadata}
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("bm25_no_new_chunks", skipped=len(chunks))
            return

        for chunk in new_chunks:
            tokens = _tokenize(chunk.text)
            self._corpus_tokens.append(tokens)
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_type": chunk.chunk_type.value,
            })

        # Rebuild BM25 index with updated corpus
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._save_to_disk()

        logger.info("bm25_indexed", new=len(new_chunks), total=len(self._corpus_tokens))

    def remove_by_source(self, source_file: str) -> None:
        """Remove all chunks for a given source file (for re-ingestion support)."""
        before = len(self._corpus_tokens)
        combined = [
            (tokens, meta)
            for tokens, meta in zip(self._corpus_tokens, self._metadata, strict=False)
            if meta["source_file"] != source_file
        ]
        if not combined:
            self._reset()
        else:
            self._corpus_tokens, self._metadata = map(list, zip(*combined, strict=False))
            self._bm25 = BM25Okapi(self._corpus_tokens)

        self._save_to_disk()
        logger.info("bm25_removed_source", source=source_file, removed=before - len(self._corpus_tokens))

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        BM25 keyword search. Returns top_k results scored by BM25 relevance.
        Scores are normalised to [0, 1] for fusion with dense scores.
        """
        if self._bm25 is None or not self._corpus_tokens:
            logger.warning("bm25_search_on_empty_index")
            return []

        query_tokens = _tokenize(query)
        raw_scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

        # Pair scores with metadata
        scored = sorted(
            zip(raw_scores, self._metadata, strict=False),
            key=lambda x: x[0],
            reverse=True,
        )

        # Apply source filter
        if source_filter:
            scored = [(s, m) for s, m in scored if m["source_file"] == source_filter]

        # Take top_k
        scored = scored[:top_k]

        if not scored:
            return []

        # Normalise scores to [0, 1] using min-max.
        # BM25Okapi can return negative scores on small corpora — min-max handles this.
        max_score = scored[0][0]
        min_score = scored[-1][0]
        score_range = max_score - min_score

        if score_range == 0:
            # All scores identical — assign uniform small score
            return [
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    parent_id=meta["parent_id"],
                    text=meta["text"],
                    source_file=meta["source_file"],
                    page=meta["page"],
                    section=meta.get("section", ""),
                    score=0.1,
                    retrieval_method="bm25",
                )
                for _, meta in scored
            ]

        return [
            RetrievedChunk(
                chunk_id=meta["chunk_id"],
                parent_id=meta["parent_id"],
                text=meta["text"],
                source_file=meta["source_file"],
                page=meta["page"],
                section=meta.get("section", ""),
                score=(score - min_score) / score_range,
                retrieval_method="bm25",
            )
            for score, meta in scored
        ]

    @property
    def document_count(self) -> int:
        return len(self._corpus_tokens)
