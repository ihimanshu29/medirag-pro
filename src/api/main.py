"""
FastAPI application entry point.
- Lifespan context manager for startup/shutdown + startup validation
- Structured logging middleware
- Prometheus metrics middleware
- Config-driven CORS (supports all deployment targets)
- All routers registered here
"""
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.limiter import limiter
from src.api.routes import chat, feedback, health, ingest
from src.config import settings
from src.logging_config import get_logger, setup_logging

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "medirag_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "medirag_request_latency_seconds",
    "HTTP request latency",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)
CACHE_HITS = Counter("medirag_cache_hits_total", "Semantic cache hits")
EMERGENCY_QUERIES = Counter("medirag_emergency_queries_total", "Emergency queries detected")

logger = get_logger(__name__)


def _validate_startup() -> None:
    """
    Validate critical configuration at startup.
    Fails immediately with a clear message rather than mid-request.
    """
    errors: list[str] = []

    if not settings.groq_api_key:
        errors.append("GROQ_API_KEY is not set.")

    if settings.uses_qdrant_cloud and not settings.qdrant_url:
        errors.append("QDRANT_URL must be set when using Qdrant Cloud.")

    if errors:
        for err in errors:
            logger.error("startup_validation_failed", reason=err)
        raise RuntimeError(
            "Startup validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(
        "startup_config_ok",
        env=settings.app_env,
        qdrant_mode="cloud" if settings.uses_qdrant_cloud else "local",
        db_mode="managed" if settings.database_url_override else "local",
        version=settings.app_version,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup and shutdown lifecycle."""
    setup_logging()

    # Validate config before anything else
    _validate_startup()

    logger.info("medirag_starting", env=settings.app_env, port=settings.api_port)

    # Warm up embedding model (avoids cold start on first request)
    try:
        from src.retrieval.embedder import Embedder
        Embedder()
        logger.info("embedder_warm", model=settings.embedding_model)
    except Exception as e:
        logger.warning("embedder_warm_failed", error=str(e))

    yield

    logger.info("medirag_shutdown")


# ── App Factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="MediRAG Pro",
        description="Production-grade Medical RAG API with hybrid retrieval, reranking, and evaluation.",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # type: ignore[arg-type]

    # CORS — driven entirely by CORS_ORIGINS env var
    # Local/Free: "*"   VPS: "https://yourdomain.com"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(feedback.router, prefix="/api/v1")

    # Prometheus scrape endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Request logging + metrics middleware
    @app.middleware("http")
    async def observability_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        t0 = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - t0

        endpoint = request.url.path
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        logger.info(
            "http_request",
            method=request.method,
            path=endpoint,
            status=response.status_code,
            latency_ms=round(latency * 1000, 2),
        )
        return response

    return app


app = create_app()
