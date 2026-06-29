"""
Phase 1 tests: API structure, health endpoint, schema validation.
These tests run without Qdrant/Postgres (mocked) to verify structure.
"""
import os

import pytest
from fastapi.testclient import TestClient

# Set minimum required env vars before importing app
os.environ.setdefault("GROQ_API_KEY", "test-key-for-ci")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("POSTGRES_HOST", "localhost")


@pytest.fixture
def client():
    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def test_docs_available(client):
    """OpenAPI docs should be accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_health_endpoint_exists(client):
    """Health endpoint must exist and return JSON."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data


def test_chat_requires_query(client):
    """Chat endpoint must reject empty body."""
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422  # Validation error


def test_chat_query_too_long(client):
    """Chat endpoint must reject queries over 2000 chars."""
    response = client.post("/api/v1/chat", json={"query": "x" * 2001})
    assert response.status_code == 422


def test_chat_valid_request(client):
    """Valid chat request should return 200 with expected schema (pipeline mocked)."""
    from unittest.mock import AsyncMock, patch
    mock_result = {
        "answer": "Hypertension is high blood pressure.",
        "sources": [],
        "confidence": 0.85,
        "cache_hit": False,
        "is_emergency": False,
    }
    with patch("src.pipeline.query_pipeline.QueryPipeline") as MockPipeline:
        MockPipeline.return_value.run = AsyncMock(return_value=mock_result)
        response = client.post("/api/v1/chat", json={"query": "What is hypertension?"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "latency_ms" in data
    assert isinstance(data["sources"], list)


def test_ingest_rejects_non_pdf(client):
    """Ingest endpoint must reject non-PDF files."""
    response = client.post(
        "/api/v1/ingest",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_config_loads_without_error():
    """Config must load from env vars without raising."""
    from src.config import settings
    assert settings.groq_model_name is not None
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.retrieval_top_k == 20
    assert settings.rerank_top_n == 5


def test_database_url_computed():
    """database_url must be correctly assembled from parts."""
    from src.config import settings
    assert "postgresql://" in settings.database_url
    assert settings.postgres_db in settings.database_url
