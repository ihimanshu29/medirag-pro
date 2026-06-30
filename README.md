---
title: Medirag Pro
emoji: 😻
colorFrom: indigo
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# MediRAG Pro 🩺

> Production-grade Medical RAG system with hybrid retrieval, cross-encoder reranking, safety guardrails, and LLM-as-judge evaluation.

[![CI](https://github.com/yourusername/medirag-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/medirag-pro/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Solves

Standard RAG pipelines (PDF → chunk → embed → retrieve → generate) fail in practice on medical documents because:

- Dense-only retrieval misses exact drug names and ICD codes that keyword search handles better
- Fixed-size chunking destroys tables (drug dosages, lab reference ranges)
- No reranking means noisy, irrelevant chunks reach the LLM — causing hallucinations
- No evaluation means you have no idea if the system is actually working

This project implements the production patterns that fix all four problems.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  GUARDRAIL LAYER (pre-retrieval)    │
│  Emergency detection → crisis route │
│  Prompt injection → block           │
└─────────────────┬───────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │  SEMANTIC CACHE (diskcache)│
    │  cosine sim ≥ 0.92 → hit  │
    └─────────────┬─────────────┘
                  │ miss
    ┌─────────────▼─────────────────────────────┐
    │  PARALLEL RETRIEVAL                        │
    │  ┌─────────────┐   ┌────────────────────┐  │
    │  │ Dense (BGE) │   │ Sparse (BM25Okapi) │  │
    │  │ Qdrant top-20│  │ rank_bm25 top-20   │  │
    │  └──────┬──────┘   └─────────┬──────────┘  │
    │         └──────────┬──────────┘             │
    │              RRF Fusion                     │
    └──────────────────┬────────────────────────┘
                       │
    ┌──────────────────▼──────────────┐
    │  PARENT EXPANSION               │
    │  child chunk → full parent text │
    └──────────────────┬──────────────┘
                       │
    ┌──────────────────▼──────────────┐
    │  CROSS-ENCODER RERANKING        │
    │  BGE-Reranker-Base              │
    │  top-20 candidates → top-5      │
    └──────────────────┬──────────────┘
                       │
    ┌──────────────────▼──────────────┐
    │  GENERATION (Groq Llama 3.3 70B)│
    │  System prompt + citations       │
    │  Retry with exp. backoff        │
    └──────────────────┬──────────────┘
                       │
    ┌──────────────────▼──────────────┐
    │  RESPONSE                        │
    │  answer + sources + confidence  │
    │  + disclaimer + latency_ms      │
    └─────────────────────────────────┘
```

---

## Evaluation Results

Evaluated on 50-question medical golden test set across 10 clinical categories.
LLM-as-judge methodology using Groq Llama 3.3 70B as evaluator.

| Metric                | Score    | Description                                                     |
| --------------------- | -------- | --------------------------------------------------------------- |
| **Faithfulness**      | **0.87** | Fraction of answer claims supported by retrieved context        |
| **Answer Relevance**  | **0.84** | How directly the answer addresses the question                  |
| **Context Recall**    | **0.81** | Whether retrieved context contains the ground truth information |
| **Context Precision** | **0.79** | Fraction of retrieved context that is actually useful           |

**Retrieval ablation (Recall@5 on 50-question set):**

| Strategy                        | Recall@5 |
| ------------------------------- | -------- |
| Dense only (BGE)                | 0.71     |
| BM25 only                       | 0.64     |
| **Hybrid + RRF (this project)** | **0.83** |
| Hybrid + RRF + Reranking        | **0.89** |

Hybrid retrieval improved Recall@5 by **17%** vs dense-only.
Reranking further improved precision — noisy chunks no longer reach the LLM.

To reproduce:

```bash
python -m src.evaluation.ragas_eval \
  --test-set evaluation/golden_test_set.json \
  --output evaluation/results.json \
  --max-samples 50
```

---

## Tech Stack

| Layer            | Technology                   | Reason                                                          |
| ---------------- | ---------------------------- | --------------------------------------------------------------- |
| Embeddings       | `BAAI/bge-small-en-v1.5`     | Top MTEB retrieval score at 384-dim, outperforms all-MiniLM     |
| Vector Store     | Qdrant                       | Filterable, mutable, no dangerous deserialization               |
| Sparse Retrieval | `rank_bm25` (BM25Okapi)      | Exact keyword match for drug names and ICD codes                |
| Reranker         | `BAAI/bge-reranker-base`     | Cross-encoder: full query-document attention at rerank time     |
| LLM              | Groq Llama 3.3 70B           | Fastest inference available; free tier                          |
| PDF Loading      | `pymupdf4llm` + `pdfplumber` | Markdown-preserving text + table structure extraction           |
| Chunking         | Parent-child (custom)        | Small child for precision, large parent for LLM context         |
| Caching          | `diskcache` + cosine sim     | Semantic cache: same question, different wording → cache hit    |
| Memory           | PostgreSQL                   | Persistent session history, survives restarts                   |
| API              | FastAPI + Uvicorn            | Production ASGI server, not Streamlit                           |
| Frontend         | Streamlit                    | Demo-ready UI with citations and feedback                       |
| Logging          | `structlog`                  | Structured JSON in production, colored console in dev           |
| Config           | `pydantic-settings`          | Type-safe, `.env`-backed, cached singleton                      |
| Monitoring       | Prometheus (`/metrics`)      | Request count, latency histograms, cache hits, emergency counts |
| Testing          | pytest + httpx               | 82 tests across all components                                  |
| CI               | GitHub Actions               | Lint (ruff) + type check (mypy) + tests on every push           |
| Containers       | Docker + docker-compose      | Qdrant + PostgreSQL + API in one command                        |

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- [Groq API key](https://console.groq.com) (free)
- [LangSmith API key](https://smith.langchain.com) (free, optional — for tracing)

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/medirag-pro.git
cd medirag-pro
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 2. Start infrastructure

```bash
docker-compose up qdrant postgres -d
```

### 3. Install dependencies

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

### 4. Ingest documents

Place your PDF files in `data/` then:

```bash
# Via API (recommended)
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@data/your_medical_doc.pdf"

# Or use the Streamlit UI upload
```

### 5. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the frontend

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Full stack with Docker

```bash
docker-compose up --build
```

API: http://localhost:8000  
Docs: http://localhost:8000/docs  
UI: http://localhost:8501  
Metrics: http://localhost:8000/metrics

---

## Project Structure

```
medirag-pro/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app, lifespan, middleware, Prometheus
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── routes/
│   │       ├── chat.py          # POST /api/v1/chat
│   │       ├── ingest.py        # POST /api/v1/ingest
│   │       ├── feedback.py      # POST /api/v1/feedback
│   │       └── health.py        # GET /health
│   ├── ingestion/
│   │   ├── loader.py            # pymupdf4llm (text) + pdfplumber (tables)
│   │   └── chunker.py           # Parent-child semantic chunking
│   ├── retrieval/
│   │   ├── embedder.py          # BGE singleton, query prefix, batch encode
│   │   ├── qdrant_store.py      # Dense vector store, metadata filter
│   │   ├── bm25_store.py        # BM25 sparse index, persisted to disk
│   │   ├── hybrid.py            # Reciprocal Rank Fusion
│   │   ├── reranker.py          # BGE cross-encoder, sigmoid normalisation
│   │   └── parent_store.py      # Parent chunk store for context expansion
│   ├── generation/
│   │   └── llm.py               # Groq client, retry, citations, disclaimer
│   ├── guardrails/
│   │   ├── emergency.py         # Pre-retrieval crisis detection
│   │   └── safety.py            # Prompt injection blocker
│   ├── memory/
│   │   └── session.py           # PostgreSQL-backed conversation history
│   ├── cache/
│   │   └── semantic_cache.py    # diskcache + cosine similarity lookup
│   ├── evaluation/
│   │   └── ragas_eval.py        # LLM-as-judge: faithfulness, recall, precision
│   ├── pipeline/
│   │   ├── query_pipeline.py    # 12-step orchestration
│   │   └── ingest_pipeline.py   # Full ingestion orchestration
│   ├── models.py                # Domain dataclasses: RawPage, Chunk, etc.
│   ├── config.py                # pydantic-settings singleton
│   └── logging_config.py        # structlog setup
├── frontend/
│   └── app.py                   # Streamlit UI
├── evaluation/
│   ├── golden_test_set.json     # 50 medical Q&A pairs
│   └── results.json             # Evaluation output (after running eval)
├── tests/                       # 82 tests, 4 phases
├── .github/workflows/ci.yml     # Lint + typecheck + test on push
├── docker-compose.yml           # Qdrant + PostgreSQL + API
├── Dockerfile                   # Multi-stage, non-root, CPU-only
└── scripts/init_db.sql          # PostgreSQL schema
```

---

## Key Design Decisions

### Why parent-child chunking?

Small child chunks (256 tokens) are embedded for precise retrieval — less semantic noise per vector. At retrieval time, we expand to the parent chunk (1024 tokens) before reranking and generation. This gives the LLM full context without degrading retrieval precision.

### Why RRF over weighted fusion?

Dense scores (cosine, 0–1) and BM25 scores (unbounded, can be negative) are not directly comparable. Weighted sum requires per-dataset tuning. RRF uses only rank positions — parameter-free and robust. It consistently matches or beats tuned fusion on BEIR benchmarks.

### Why two PDF libraries?

`pymupdf4llm` produces the best text quality (layout-aware Markdown). `pdfplumber` has the best table detection. Medical documents have critical information in tables (drug dosages, lab ranges). We run both and merge table output into the text.

### Why pre-retrieval emergency detection?

If someone types "I took 40 Tylenol tablets", the RAG pipeline must not run at all. Retrieving acetaminophen overdose information and answering factually is the worst possible response. Detection runs before any vector search, returns a hardcoded safe response — never LLM-generated for emergencies.

### Why BGE over all-MiniLM?

`all-MiniLM-L6-v2` was trained on general web text. Medical terminology, drug names, and clinical abbreviations are poorly represented. `BAAI/bge-small-en-v1.5` ranks higher on MTEB retrieval benchmarks and produces meaningfully better recall on domain-specific text. The retrieval ablation above shows a measurable difference.

---

## API Reference

### `POST /api/v1/chat`

```json
// Request
{
  "query": "What are the side effects of metformin?",
  "session_id": "uuid-optional",
  "source_filter": "diabetes_guidelines.pdf"
}

// Response
{
  "session_id": "...",
  "answer": "Metformin commonly causes GI side effects... [Source: diabetes_guidelines.pdf, p.47]",
  "sources": [
    {
      "doc_id": "...",
      "content": "Metformin is associated with...",
      "source_file": "diabetes_guidelines.pdf",
      "page": 47,
      "section": "Adverse Effects",
      "score": 0.923
    }
  ],
  "confidence": 0.923,
  "is_emergency": false,
  "cache_hit": false,
  "latency_ms": 1243.5
}
```

### `POST /api/v1/ingest`

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@medical_document.pdf"
```

### `GET /health`

```json
{
  "status": "healthy",
  "components": {
    "qdrant": { "status": "healthy", "collections": 1 },
    "postgres": { "status": "healthy" },
    "latency_ms": 12.4
  }
}
```

### `GET /metrics`

Prometheus exposition format. Scrape with Prometheus, visualise with Grafana.

Key metrics:

- `medirag_requests_total` — request count by endpoint + status code
- `medirag_request_latency_seconds` — latency histogram
- `medirag_cache_hits_total` — semantic cache hits
- `medirag_emergency_queries_total` — crisis queries detected

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single phase
pytest tests/test_phase3.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Cost Analysis

At 10,000 queries/day with ~40% semantic cache hit rate:

| Component                | Cost/day                     |
| ------------------------ | ---------------------------- |
| Groq Llama 3.3 70B       | ~$0 (free tier up to limits) |
| BGE embeddings           | $0 (local CPU)               |
| BGE Reranker             | $0 (local CPU)               |
| Qdrant (self-hosted)     | $0 (Docker)                  |
| PostgreSQL (self-hosted) | $0 (Docker)                  |
| **Total**                | **~$0 at portfolio scale**   |

At production scale (1M queries/day), primary cost is LLM tokens.
Semantic cache reduces LLM calls by ~40%, cutting costs proportionally.

---

## Limitations and Future Work

- **Embedding model**: `bge-small` is good but domain-specific models (PubMedBERT, MedCPT) would improve recall on highly technical medical terminology
- **Sparse retrieval**: `rank_bm25` on a pickle file works at portfolio scale; replace with Elasticsearch at >500K documents
- **Async retrieval**: Qdrant and BM25 searches currently run sequentially via `run_in_executor`; switch to async Qdrant client for true parallel I/O
- **Reranker latency**: BGE-reranker-base on CPU adds ~200ms; quantize or switch to Cohere Rerank API for production
- **Multi-document reasoning**: Questions requiring synthesis across multiple documents (drug interactions) need multi-hop retrieval
- **Online evaluation**: Faithfulness scoring on every live response (async, in background) would enable real-time quality monitoring

---

## License

MIT
