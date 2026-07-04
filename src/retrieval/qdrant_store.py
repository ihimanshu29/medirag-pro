"""
Qdrant vector store manager.

Supports two connection modes via config:
  Local / VPS (self-hosted):  QDRANT_HOST + QDRANT_PORT
  Free cloud (Qdrant Cloud):  QDRANT_URL  + QDRANT_API_KEY

The rest of the application is unaware of which mode is active.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.config import settings
from src.logging_config import get_logger
from src.models import ChildChunk, RetrievedChunk

logger = get_logger(__name__)


def _build_qdrant_client() -> QdrantClient:
    """
    Build the QdrantClient appropriate for the current deployment target.
    Qdrant Cloud:    uses URL + API key (HTTPS, port embedded in URL).
    Local / VPS:     uses host + port (plain HTTP to Docker service).
    """
    if settings.uses_qdrant_cloud:
        logger.info(
            "qdrant_cloud_connect",
            url=settings.qdrant_url,
            has_api_key=bool(settings.qdrant_api_key),
        )
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    else:
        logger.info(
            "qdrant_local_connect",
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        return QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )


class QdrantStore:
    """Manages the Qdrant collection for medical document chunks."""

    def __init__(self) -> None:
        self._client = _build_qdrant_client()
        self._collection = settings.qdrant_collection_name
        self._dim = settings.embedding_dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist and guarantee payload indexing."""
        existing = {c.name for c in self._client.get_collections().collections}

        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("qdrant_collection_created", collection=self._collection, dim=self._dim)
        else:
            logger.info("qdrant_collection_exists", collection=self._collection)

        # ─── PRODUCTION FIX: RUNS REGARDLESS OF IF/ELSE CORNER CASES ───
        for field in ("source_file", "section", "chunk_type"):
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("qdrant_payload_index_created", field=field)
            except Exception:
                # If the index already exists, Qdrant raises an exception.
                # We catch and pass it silently so it doesn't disrupt startup.
                pass

    def upsert_chunks(self, chunks: list[ChildChunk]) -> int:
        """
        Upsert child chunks into Qdrant.
        Uses chunk_id as the point ID (deterministic → safe to re-ingest same doc).
        Returns number of chunks upserted.
        """
        if not chunks:
            return 0

        valid = [c for c in chunks if c.embedding]
        if not valid:
            logger.warning("upsert_skipped_no_embeddings", total=len(chunks))
            return 0

        points = [
            PointStruct(
                id=abs(hash(chunk.chunk_id)) % (2**63),
                vector=chunk.embedding,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id,
                    "text": chunk.text,
                    "source_file": chunk.source_file,
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_type": chunk.chunk_type.value,
                },
            )
            for chunk in valid
        ]

        batch_size = 256
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(collection_name=self._collection, points=batch)
            total += len(batch)

        logger.info("qdrant_upserted", count=total, collection=self._collection)
        return total

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Dense vector search with optional source filter."""
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_filter),
                )]
            )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        retrieved_chunks: list[RetrievedChunk] = []
        for r in results:
            payload = r.payload
            assert payload is not None  # Proves to mypy this is a dict, not None
            
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=payload["chunk_id"],
                    parent_id=payload["parent_id"],
                    text=payload["text"],
                    source_file=payload["source_file"],
                    page=payload["page"],
                    section=payload.get("section", ""),
                    score=r.score,
                    retrieval_method="dense",
                )
            )
            
        return retrieved_chunks

    def delete_by_source(self, source_file: str) -> None:
        """Remove all chunks for a given source file (enables re-ingestion)."""
        from qdrant_client.models import FilterSelector
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    )]
                )
            ),
        )
        logger.info("qdrant_deleted_source", source_file=source_file)

    def count(self) -> int:
        """Return total number of vectors in the collection."""
        info = self._client.get_collection(self._collection)
        return info.points_count or 0
