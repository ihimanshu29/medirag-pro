"""
/feedback endpoint.
Stores thumbs-up / thumbs-down per response to PostgreSQL.
Feeds the online evaluation loop — daily aggregation drives quality monitoring.
"""
import psycopg2

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    rating: int = Field(..., ge=-1, le=1, description="-1 = thumbs down, 1 = thumbs up")
    comment: str = ""


@router.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest) -> dict:
    """Store user feedback. Non-fatal if DB is unavailable."""
    try:
        from src.config import settings

        with psycopg2.connect(settings.database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO feedback (session_id, query, answer, rating, comment)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        request.session_id,
                        request.query[:2000],
                        request.answer[:5000],
                        request.rating,
                        request.comment[:500],
                    ),
                )
            conn.commit()

        logger.info(
            "feedback_stored",
            session=request.session_id[:8],
            rating=request.rating,
        )
        return {"status": "ok"}

    except Exception as e:
        logger.warning("feedback_store_failed", error=str(e))
        return {"status": "degraded", "error": str(e)}
