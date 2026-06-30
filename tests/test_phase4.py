"""
Phase 4 tests: evaluation framework, feedback endpoint, metric computation.
All LLM calls mocked — tests verify parsing logic and API contracts.
"""
import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")


# ── Score extraction tests ────────────────────────────────────────────────────

def test_extract_score_from_label():
    from src.evaluation.ragas_eval import _extract_score
    assert _extract_score("Score: 0.85") == pytest.approx(0.85)


def test_extract_score_from_fraction():
    from src.evaluation.ragas_eval import _extract_score
    assert _extract_score("0.9 / 1") == pytest.approx(0.9)


def test_extract_score_bare_float():
    from src.evaluation.ragas_eval import _extract_score
    assert _extract_score("0.72") == pytest.approx(0.72)


def test_extract_score_clamps_to_one():
    from src.evaluation.ragas_eval import _extract_score
    assert _extract_score("Score: 1.5") == pytest.approx(1.0)


def test_extract_score_clamps_to_zero():
    from src.evaluation.ragas_eval import _extract_score
    # The regex won't match "Score: -0.3" because our patterns don't capture negatives.
    # Result should be 0.0 (fallback). This is correct behaviour — negative scores are invalid.
    result = _extract_score("Score: -0.3")
    assert result == pytest.approx(0.0)


def test_extract_score_fallback_no_score():
    from src.evaluation.ragas_eval import _extract_score
    # Unparseable → returns 0.0
    result = _extract_score("I cannot determine a score from this.")
    assert result == pytest.approx(0.0)


def test_extract_score_case_insensitive():
    from src.evaluation.ragas_eval import _extract_score
    assert _extract_score("score: 0.77") == pytest.approx(0.77)


# ── Faithfulness metric tests ─────────────────────────────────────────────────

def test_faithfulness_empty_context():
    from src.evaluation.ragas_eval import compute_faithfulness
    score = compute_faithfulness("Some answer", contexts=[])
    assert score == 0.0


def test_faithfulness_empty_answer():
    from src.evaluation.ragas_eval import compute_faithfulness
    score = compute_faithfulness("", contexts=["some context"])
    assert score == 0.0


def test_faithfulness_calls_llm_judge():
    from src.evaluation.ragas_eval import compute_faithfulness
    with patch("src.evaluation.ragas_eval._call_judge", return_value="Score: 0.9") as mock_judge:
        score = compute_faithfulness("Aspirin inhibits COX enzymes.", ["Aspirin blocks COX-1 and COX-2."])
    assert score == pytest.approx(0.9)
    mock_judge.assert_called_once()


def test_faithfulness_strips_disclaimer():
    """Disclaimer text must not confuse the judge — strip it before sending."""
    from src.evaluation.ragas_eval import compute_faithfulness
    answer_with_disclaimer = "Aspirin is an NSAID.\n\n---\n⚠️ *This is not medical advice.*"
    with patch("src.evaluation.ragas_eval._call_judge", return_value="Score: 0.95") as mock_judge:
        score = compute_faithfulness(answer_with_disclaimer, ["Aspirin is a non-steroidal anti-inflammatory drug."])
    call_args = mock_judge.call_args[0][0]
    assert "⚠️" not in call_args   # disclaimer stripped from judge prompt
    assert score == pytest.approx(0.95)


# ── Answer relevance tests ────────────────────────────────────────────────────

def test_answer_relevance_empty_answer():
    from src.evaluation.ragas_eval import compute_answer_relevance
    score = compute_answer_relevance("What is aspirin?", "")
    assert score == 0.0


def test_answer_relevance_calls_judge():
    from src.evaluation.ragas_eval import compute_answer_relevance
    with patch("src.evaluation.ragas_eval._call_judge", return_value="Score: 0.82"):
        score = compute_answer_relevance("What is hypertension?", "Hypertension is high blood pressure.")
    assert score == pytest.approx(0.82)


# ── Context recall tests ──────────────────────────────────────────────────────

def test_context_recall_empty_contexts():
    from src.evaluation.ragas_eval import compute_context_recall
    score = compute_context_recall("ground truth answer", contexts=[])
    assert score == 0.0


def test_context_recall_calls_judge():
    from src.evaluation.ragas_eval import compute_context_recall
    with patch("src.evaluation.ragas_eval._call_judge", return_value="Score: 0.75"):
        score = compute_context_recall("Hypertension is >140/90.", ["Blood pressure above 140/90 is hypertension."])
    assert score == pytest.approx(0.75)


# ── Context precision tests ───────────────────────────────────────────────────

def test_context_precision_empty_contexts():
    from src.evaluation.ragas_eval import compute_context_precision
    score = compute_context_precision("What is diabetes?", contexts=[])
    assert score == 0.0


def test_context_precision_calls_judge():
    from src.evaluation.ragas_eval import compute_context_precision
    with patch("src.evaluation.ragas_eval._call_judge", return_value="Score: 0.88"):
        score = compute_context_precision("What is diabetes?", ["Diabetes is a metabolic disorder."])
    assert score == pytest.approx(0.88)


# ── Golden test set tests ─────────────────────────────────────────────────────

def test_golden_test_set_loads():
    """Test set must be valid JSON with required fields."""
    with open("evaluation/golden_test_set.json") as f:
        data = json.load(f)
    assert len(data) >= 50, f"Expected 50+ samples, got {len(data)}"
    for sample in data:
        assert "question" in sample
        assert "ground_truth" in sample
        assert "category" in sample
        assert len(sample["question"]) > 10
        assert len(sample["ground_truth"]) > 10


def test_golden_test_set_categories():
    """Test set must cover diverse medical categories."""
    with open("evaluation/golden_test_set.json") as f:
        data = json.load(f)
    categories = {s["category"] for s in data}
    required = {"cardiovascular", "pharmacology", "neurology", "endocrinology"}
    assert required.issubset(categories), f"Missing categories: {required - categories}"


# ── Feedback API tests ────────────────────────────────────────────────────────

@pytest.fixture
def client():
    os.environ.setdefault("GROQ_API_KEY", "test-key")
    from fastapi.testclient import TestClient

    from src.api.main import app
    return TestClient(app, raise_server_exceptions=False)


def test_feedback_endpoint_exists(client):
    """Feedback endpoint must accept valid feedback and return 200."""
    with patch("src.api.routes.feedback.psycopg2") as mock_psycopg2:
        mock_conn = MagicMock()
        mock_psycopg2.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_psycopg2.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        response = client.post("/api/v1/feedback", json={
            "session_id": "test-123",
            "query": "What is hypertension?",
            "answer": "Hypertension is high blood pressure.",
            "rating": 1,
        })
    assert response.status_code == 200


def test_feedback_rejects_invalid_rating(client):
    """Rating must be -1 or 1."""
    response = client.post("/api/v1/feedback", json={
        "session_id": "test-123",
        "query": "test",
        "answer": "test",
        "rating": 5,   # invalid
    })
    assert response.status_code == 422


def test_feedback_db_failure_is_non_fatal(client):
    """DB write failure must return degraded status, not 500."""
    with patch("src.api.routes.feedback.psycopg2") as mock_psycopg2:
        mock_psycopg2.connect.side_effect = Exception("DB down")
        response = client.post("/api/v1/feedback", json={
            "session_id": "test-456",
            "query": "What is aspirin?",
            "answer": "Aspirin is an NSAID.",
            "rating": -1,
        })
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


# ── Full suite smoke test ─────────────────────────────────────────────────────

def test_api_has_all_expected_routes(client):
    """All Phase 1–4 routes must be registered."""
    response = client.get("/docs")
    assert response.status_code == 200
    # OpenAPI JSON lists all paths
    openapi = client.get("/openapi.json").json()
    paths = set(openapi["paths"].keys())
    assert "/health" in paths
    assert "/api/v1/chat" in paths
    assert "/api/v1/ingest" in paths
    assert "/api/v1/feedback" in paths
    assert "/metrics" not in paths   # excluded from schema
