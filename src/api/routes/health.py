"""
Health check endpoint.
Verifies Qdrant, PostgreSQL, embedding model, cache, and reports deployment info.
Returns HTTP 200 even when degraded — load balancers need to distinguish
'app is down' (connection refused) from 'app is up but dependency is struggling'.
"""
import time

from fastapi import APIRouter
from qdrant_client import QdrantClient
from sqlalchemy import text
from sqlalchemy.engine import create_engine

from src.api.schemas import ComponentStatus, HealthResponse
from src.config import settings
from src.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _check_qdrant() -> dict:
    try:
        if settings.uses_qdrant_cloud:
            client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=3,
            )
        else:
            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=3,
            )
        collections = client.get_collections()
        return {
            "status": ComponentStatus.HEALTHY,
            "collections": len(collections.collections),
            "mode": "cloud" if settings.uses_qdrant_cloud else "local",
        }
    except Exception as e:
        return {"status": ComponentStatus.DOWN, "error": str(e)}


def _check_postgres() -> dict:
    try:
        # Neon requires SSL — SQLAlchemy handles this automatically
        # when the DATABASE_URL contains sslmode=require
        engine = create_engine(
            settings.database_url,
            connect_args={"connect_timeout": 3},
            pool_pre_ping=True,
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {
            "status": ComponentStatus.HEALTHY,
            "mode": "managed" if settings.database_url_override else "local",
        }
    except Exception as e:
        return {"status": ComponentStatus.DOWN, "error": str(e)}


def _check_embedder() -> dict:
    try:
        from src.retrieval.embedder import Embedder
        emb = Embedder()
        # Only test if model is already loaded — don't trigger download at health check time
        if emb._model is not None:
            return {"status": ComponentStatus.HEALTHY, "model": settings.embedding_model}
        return {"status": ComponentStatus.DEGRADED, "note": "model not yet loaded (loads on first request)"}
    except Exception as e:
        return {"status": ComponentStatus.DOWN, "error": str(e)}


def _check_cache() -> dict:
    try:
        from src.cache.semantic_cache import SemanticCache
        cache = SemanticCache()
        return {"status": ComponentStatus.HEALTHY, "entries": cache.size}
    except Exception as e:
        return {"status": ComponentStatus.DOWN, "error": str(e)}


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """
    System health check. Returns status of all dependent services.
    Always returns HTTP 200 — status field indicates true health.
    """
    t0 = time.perf_counter()

    qdrant_status = _check_qdrant()
    postgres_status = _check_postgres()
    embedder_status = _check_embedder()
    cache_status = _check_cache()

    critical = [qdrant_status, postgres_status]
    all_critical_healthy = all(
        c["status"] == ComponentStatus.HEALTHY for c in critical
    )
    overall = ComponentStatus.HEALTHY if all_critical_healthy else ComponentStatus.DEGRADED

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info("health_check", overall=overall, latency_ms=round(latency_ms, 2))

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        components={
            "qdrant": qdrant_status,
            "postgres": postgres_status,
            "embedder": embedder_status,
            "cache": cache_status,
            "environment": settings.app_env,
            "latency_ms": round(latency_ms, 2),
        },
    )
