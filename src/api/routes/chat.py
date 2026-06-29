"""
/chat endpoint.
Accepts a user query, runs the full RAG pipeline, returns a cited answer.
"""
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.limiter import limiter
from src.api.schemas import ChatRequest, ChatResponse
from src.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/chat", response_model=ChatResponse, tags=["RAG"])
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Main RAG chat endpoint.

    Pipeline:
    1. Emergency guard (pre-retrieval)
    2. Semantic cache lookup
    3. Hybrid retrieval (dense + BM25) with RRF
    4. Cross-encoder reranking
    5. LLM generation with citations
    6. Session memory update
    """
    from src.pipeline.query_pipeline import QueryPipeline  # lazy import avoids startup cost

    t0 = time.perf_counter()

    try:
        pipeline = QueryPipeline()
        result = await pipeline.run(
            query=body.query,
            session_id=str(body.session_id),
            source_filter=body.source_filter,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("chat_pipeline_error", error=str(e), query=body.query[:100])
        raise HTTPException(status_code=500, detail="Internal pipeline error") from e

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "chat_complete",
        session_id=str(body.session_id),
        latency_ms=round(latency_ms, 2),
        cache_hit=result.get("cache_hit", False),
        sources_count=len(result.get("sources", [])),
        is_emergency=result.get("is_emergency", False),
    )

    return ChatResponse(
        session_id=body.session_id,
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        is_emergency=result.get("is_emergency", False),
        cache_hit=result.get("cache_hit", False),
        latency_ms=round(latency_ms, 2),
    )
