.PHONY: install test lint format typecheck check \
        docker-up docker-down docker-build \
        run-api run-ui ingest eval benchmark clean help

# ── Setup ─────────────────────────────────────────────────────────────────────

install:  ## Install all dependencies and pre-commit hooks
	pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
	pip install pre-commit
	pre-commit install
	@echo "✅ Installation complete. Copy .env.example to .env and add your GROQ_API_KEY."

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:  ## Run ruff linter
	ruff check src/ tests/

format:  ## Auto-format with ruff
	ruff format src/ tests/

format-check:  ## Check formatting without modifying files (for CI)
	ruff format --check src/ tests/

typecheck:  ## Run mypy type checker
	mypy src/ --ignore-missing-imports

check: lint format-check typecheck  ## Run all quality checks (no modification)
	@echo "✅ All quality checks passed."

# ── Testing ───────────────────────────────────────────────────────────────────

test:  ## Run all tests with coverage
	GROQ_API_KEY=test-key pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-fail-under=70 \
		-v

test-fast:  ## Run tests without coverage (faster)
	GROQ_API_KEY=test-key pytest tests/ -q

test-phase1:  ## Run Phase 1 tests only
	GROQ_API_KEY=test-key pytest tests/test_phase1.py -v

test-phase2:  ## Run Phase 2 tests only
	GROQ_API_KEY=test-key pytest tests/test_phase2.py -v

test-phase3:  ## Run Phase 3 tests only
	GROQ_API_KEY=test-key pytest tests/test_phase3.py -v

test-phase4:  ## Run Phase 4 tests only
	GROQ_API_KEY=test-key pytest tests/test_phase4.py -v

# ── Infrastructure ────────────────────────────────────────────────────────────

docker-up:  ## Start Qdrant and PostgreSQL
	docker compose up qdrant postgres -d
	@echo "⏳ Waiting for services to be healthy..."
	@sleep 5
	@docker compose ps

docker-down:  ## Stop all services
	docker compose down

docker-up-full:  ## Start entire stack (Qdrant + Postgres + API)
	docker compose up -d

docker-build:  ## Build the API Docker image
	docker build -t medirag-pro:latest .

docker-logs:  ## Tail API logs
	docker compose logs api -f

# ── Running ───────────────────────────────────────────────────────────────────

run-api:  ## Start FastAPI development server
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:  ## Start Streamlit frontend
	streamlit run frontend/app.py

# ── Data Operations ───────────────────────────────────────────────────────────

ingest:  ## Ingest a PDF — usage: make ingest FILE=data/your_doc.pdf
	@test -n "$(FILE)" || (echo "Usage: make ingest FILE=path/to/doc.pdf" && exit 1)
	curl -X POST http://localhost:8000/api/v1/ingest \
		-F "file=@$(FILE)" | python -m json.tool

# ── Evaluation ────────────────────────────────────────────────────────────────

eval:  ## Run RAG evaluation (20 samples)
	python -m src.evaluation.ragas_eval \
		--test-set evaluation/golden_test_set.json \
		--output evaluation/results.json \
		--max-samples 20

eval-full:  ## Run full 50-sample evaluation
	python -m src.evaluation.ragas_eval \
		--test-set evaluation/golden_test_set.json \
		--output evaluation/results.json \
		--max-samples 50

benchmark:  ## Run latency benchmark (requires running API)
	python scripts/benchmark.py --url http://localhost:8000 --users 10 --requests 50

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:  ## Remove all generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml
	@echo "✅ Clean complete."

clean-data:  ## Remove ingested data (BM25 index, parent chunks, cache)
	rm -f data/bm25_index.pkl data/parent_chunks.pkl
	rm -rf data/semantic_cache/
	@echo "⚠️  Data cleaned. You will need to re-ingest documents."

# ── Help ─────────────────────────────────────────────────────────────────────

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
