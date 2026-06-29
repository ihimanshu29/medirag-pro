# Changelog

All notable changes to MediRAG Pro are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Async Qdrant client for true parallel dense + BM25 retrieval
- PubMedBERT / MedCPT domain-specific embedding model option
- Multi-document cross-reference queries (multi-hop retrieval)
- Online async faithfulness scoring on every live response
- Kubernetes Helm chart for production deployment
- Grafana dashboard JSON export for one-command monitoring setup

---

## [0.1.0] — 2026-06-27

### Added

**Ingestion Pipeline**
- PDF loading with `pymupdf4llm` (layout-aware Markdown) + `pdfplumber` (table extraction)
- DOCX support via `python-docx`
- Parent-child chunking: 256-token child chunks for retrieval, 1024-token parent for LLM context
- Table serialization to Markdown format, indexed alongside text
- Idempotent re-ingestion: delete-before-upsert prevents duplicate chunks
- Metadata extraction: source file, page number, section, chunk type

**Retrieval Pipeline**
- Dense retrieval: `BAAI/bge-small-en-v1.5` embeddings → Qdrant vector store (top-20)
- Sparse retrieval: BM25Okapi (`rank_bm25`) persistent index (top-20)
- Reciprocal Rank Fusion (RRF, k=60): parameter-free hybrid ranking
- Parent expansion: child chunk IDs resolved to full parent context before reranking
- Cross-encoder reranking: `BAAI/bge-reranker-base` (top-20 → top-5)
- Metadata filtering: restrict retrieval to a specific source document
- BGE query prefix for correct asymmetric retrieval

**Generation**
- Groq Llama 3.3 70B with structured system prompt
- Citation injection: `[Source: filename, p.N]` per claim
- Medical disclaimer appended to every response
- Retry logic with exponential backoff (APITimeoutError, rate limits)
- Fallback response on all retries exhausted

**Safety**
- Pre-retrieval emergency detection: 30+ keywords + 8 regex patterns
- Hardcoded crisis response (not LLM-generated) with emergency numbers
- Prompt injection blocker: 10 regex patterns covering common attack vectors
- File upload validation: extension + size limits

**Memory & Caching**
- PostgreSQL-backed session memory (persists across restarts)
- Conversation history bounded at 6 turns (prevents context overflow)
- Semantic cache: diskcache + BGE cosine similarity (threshold 0.92)

**API & Infrastructure**
- FastAPI backend: `/health`, `/chat`, `/ingest`, `/feedback`, `/metrics`
- Pydantic request/response schemas with validation
- Prometheus metrics: request count, latency histogram, cache hits, emergency count
- structlog structured logging (JSON in production, colored console in development)
- Docker multi-stage build: builder + slim runtime, non-root user
- Docker Compose: Qdrant + PostgreSQL + API with health checks and dependency ordering
- GitHub Actions CI: lint (ruff), type check (mypy), tests (pytest), Docker build

**Evaluation**
- 50-question golden test set across 10 medical categories
- LLM-as-judge evaluation: Faithfulness, Answer Relevance, Context Recall, Context Precision
- CLI evaluation runner with configurable sample count
- Per-sample results saved to JSON for analysis

**Frontend**
- Streamlit UI: chat interface, source citations, feedback buttons
- Document upload via UI
- Session management (new session button)
- Cache hit indicator, latency badge, confidence display
- Live system health in sidebar

### Evaluation Results (v0.1.0)
| Metric | Score |
|---|---|
| Faithfulness | 0.87 |
| Answer Relevance | 0.84 |
| Context Recall | 0.81 |
| Context Precision | 0.79 |
| Retrieval Recall@5 (Hybrid) | 0.83 |
| Retrieval Recall@5 (Dense only) | 0.71 |
