"""
Full document ingestion pipeline.

Flow:
  PDF/DOCX file
    -> load_document()          [loader.py]   -> list[RawPage]
    -> chunk_pages()            [chunker.py]  -> (list[ParentChunk], list[ChildChunk])
    -> embedder.embed_documents()             -> embeddings on ChildChunks
    -> qdrant_store.upsert_chunks()           -> dense vectors stored
    -> bm25_store.add_chunks()               -> sparse index updated
    -> parent_store.add()                    -> parent context stored

Idempotent: re-ingesting the same file deletes old data first, then re-indexes.
"""
from pathlib import Path
from typing import Any

from src.ingestion.chunker import chunk_pages
from src.ingestion.loader import load_document
from src.logging_config import get_logger
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import Embedder
from src.retrieval.parent_store import ParentStore
from src.retrieval.qdrant_store import QdrantStore

logger = get_logger(__name__)


class IngestPipeline:
    """
    Orchestrates the full document ingestion pipeline.
    Components are instantiated once per pipeline run.
    """

    def __init__(self) -> None:
        self._embedder = Embedder()       # singleton — loads model once
        self._qdrant = QdrantStore()
        self._bm25 = BM25Store()
        self._parents = ParentStore()

    async def run(self, file_path: Path, filename: str) -> dict[str, Any]:
        """
        Run the full ingestion pipeline on a single document.

        Args:
            file_path: Path to the temp file on disk.
            filename:  Original filename (used as source identifier in metadata).

        Returns:
            dict with chunks_created, tables_extracted
        """
        logger.info("ingest_pipeline_start", filename=filename)

        # Step 1: Remove existing data for this source (idempotent re-ingest)
        self._qdrant.delete_by_source(filename)
        self._bm25.remove_by_source(filename)
        self._parents.remove_by_source(filename)

        # Step 2: Load document -> RawPages
        pages = load_document(file_path)
        if not pages:
            raise ValueError(f"No content extracted from {filename}")

        tables_extracted = sum(len(p.tables) for p in pages)
        logger.info(
            "ingest_loaded",
            filename=filename,
            pages=len(pages),
            tables=tables_extracted,
        )

        # Step 3: Chunk -> ParentChunks + ChildChunks
        parent_chunks, child_chunks = chunk_pages(pages, source_file=filename)

        if not child_chunks:
            raise ValueError(
                f"No chunks produced from {filename} — document may be empty or image-only."
            )

        logger.info(
            "ingest_chunked",
            filename=filename,
            parents=len(parent_chunks),
            children=len(child_chunks),
        )

        # Step 4: Embed child chunks (batch)
        texts = [c.text for c in child_chunks]
        embeddings = self._embedder.embed_documents(texts)

        for chunk, emb in zip(child_chunks, embeddings, strict=False):
            chunk.embedding = emb

        logger.info("ingest_embedded", filename=filename, vectors=len(embeddings))

        # Step 5: Store in Qdrant (dense)
        upserted = self._qdrant.upsert_chunks(child_chunks)

        # Step 6: Update BM25 index (sparse)
        self._bm25.add_chunks(child_chunks)

        # Step 7: Store parent chunks for context expansion
        self._parents.add(parent_chunks)

        logger.info(
            "ingest_pipeline_complete",
            filename=filename,
            chunks_created=upserted,
            tables_extracted=tables_extracted,
            parents_stored=len(parent_chunks),
            bm25_total=self._bm25.document_count,
            qdrant_total=self._qdrant.count(),
        )

        return {
            "chunks_created": upserted,
            "tables_extracted": tables_extracted,
        }
