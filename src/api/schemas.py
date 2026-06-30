"""
Pydantic schemas for all API request and response bodies.
Single source of truth for data contracts.
"""
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# ── Shared ───────────────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """A retrieved source chunk that grounded the answer."""
    doc_id: str
    content: str
    source_file: str
    page: int | None = None
    section: str | None = None
    score: float = Field(ge=0.0, le=1.0)


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: UUID = Field(default_factory=uuid4)
    source_filter: str | None = Field(
        default=None,
        description="Filter retrieval to a specific source document filename",
    )


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    sources: list[SourceDocument]
    confidence: float = Field(ge=0.0, le=1.0, description="Retrieval confidence score")
    is_emergency: bool = Field(default=False)
    cache_hit: bool = Field(default=False)
    latency_ms: float


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class IngestResponse(BaseModel):
    filename: str
    status: IngestStatus
    chunks_created: int
    tables_extracted: int
    message: str


# ── Health ───────────────────────────────────────────────────────────────────

class ComponentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class HealthResponse(BaseModel):
    status: ComponentStatus
    version: str = "0.1.0"
    components: dict[str, Any]


# ── Evaluation (internal) ─────────────────────────────────────────────────────

class EvalSample(BaseModel):
    question: str
    ground_truth: str
    contexts: list[str] = Field(default_factory=list)
    answer: str = ""
