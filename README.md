# MediRAG Pro 🩺

> Production-grade Medical RAG system with hybrid retrieval, cross-encoder reranking, safety guardrails, and LLM-as-judge evaluation.

[![CI](https://img.shields.io/badge/CI-Passing-brightgreen?style=flat&logo=github)](https://github.com/ihimanshu29/medirag-pro/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![🤗 Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/nightKing29/medirag-pro)
[![Deploy](https://img.shields.io/badge/deploy-Local%20%7C%20Free%20Cloud%20%7C%20VPS-blue)](docs/deployment.md)

---

## Live Demo

🔗 **https://huggingface.co/spaces/nightKing29/medirag-pro** — deployed free on HuggingFace Spaces, backed by Qdrant Cloud + Neon PostgreSQL + Groq.

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
## ⬇️ *CLICK THE IMAGE BELOW* ⬇️
## 🏗️ LIVE INTERACTIVE ARCHITECTURE EXPLORER
[![MediRAG Architecture](https://github.com/user-attachments/assets/25abff66-9282-49cc-8407-1d905aa283be)](https://medirag-architecture.vercel.app)

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

## Deployment — One Codebase, Three Targets

MediRAG Pro deploys identically to local, free cloud, or a paid VPS. **Zero application code changes between targets** — only environment variables differ. This is enforced by a single config abstraction (`src/config.py`) that switches Qdrant between self-hosted (host+port) and Qdrant Cloud (URL+API key) mode, and PostgreSQL between local and a managed `DATABASE_URL` (Neon/Supabase), based on which env vars are present.

|                 | Local               | Free Cloud                                         | VPS                 |
| --------------- | ------------------- | -------------------------------------------------- | ------------------- |
| **Cost**        | $0                  | $0 forever                                         | ~$5–6/mo            |
| **Compute**     | Your machine        | HF Spaces (16GB RAM)                               | Any VPS             |
| **Vector DB**   | Qdrant (Docker)     | Qdrant Cloud (1GB free)                            | Qdrant (Docker)     |
| **Database**    | PostgreSQL (Docker) | Neon (512MB free)                                  | PostgreSQL (Docker) |
| **Persistence** | Full                | Vectors persist; BM25/cache reset on Space restart | Full                |

### Target 1 — Local (Instructions)

```bash
git clone https://github.com/yourusername/medirag-pro.git && cd medirag-pro
cp .env.example .env        # add GROQ_API_KEY
docker compose up qdrant postgres -d
uvicorn src.api.main:app --reload      # API → :8000
streamlit run frontend/app.py          # UI  → :8501
```

Or full Docker stack: `docker compose up -d`

### Target 2 — Free Cloud (HuggingFace Spaces + Qdrant Cloud + Neon)

1. **Qdrant Cloud** (cloud.qdrant.io) → free cluster → copy Cluster URL + API key
2. **Neon** (console.neon.tech) → new project → copy connection string (`?sslmode=require` required) → run `scripts/init_db.sql` in SQL Editor
3. **Groq** (console.groq.com) → API key
4. **HF Space** (huggingface.co/new-space) → SDK: Docker → Hardware: CPU Basic (free) → add all secrets from `.env.free` in Settings → Repository secrets (critically: `API_BACKEND_URL=http://localhost:8000`, not the public URL)
5. Push directly to the Space (HF Spaces _is_ a git repo — no GitHub linking needed):
   ```bash
   git checkout -b hf-deploy
   cp Dockerfile.spaces Dockerfile
   git add Dockerfile && git commit -m "chore: HF Spaces Dockerfile"
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/medirag-pro
   git push hf hf-deploy:main
   ```
6. First build: 10–20 min. Verify: `curl https://YOUR_USERNAME-medirag-pro.hf.space/health`

Full walkthrough with troubleshooting: [`docs/deployment.md`](docs/deployment.md)

### Target 3 — VPS (Nginx + Docker Compose, self-hosted)

```bash
ssh ubuntu@YOUR_IP
git clone https://github.com/yourusername/medirag-pro.git && cd medirag-pro
cp .env.vps .env            # set GROQ_API_KEY, POSTGRES_PASSWORD, API_BACKEND_URL
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d
```

Stack: Nginx (80/443, WebSocket-aware) → FastAPI (8000) → Streamlit (8501) → Qdrant + PostgreSQL, all in Docker. Add HTTPS via Certbot — see `docs/deployment.md`.

### Migrating Free → VPS

```bash
cp .env.vps .env   # edit POSTGRES_PASSWORD + API_BACKEND_URL
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d
```

**Under 10 minutes. Zero code changes** — this is the entire point of the config abstraction.

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
│   ├── config.py                # pydantic-settings — local/cloud/VPS via env vars
│   └── logging_config.py        # structlog setup
├── frontend/
│   └── app.py                   # Streamlit UI (backend URL via API_BACKEND_URL)
├── evaluation/
│   ├── golden_test_set.json     # 50 medical Q&A pairs
│   └── results.json             # Evaluation output (after running eval)
├── tests/                       # 82 tests, 4 phases
├── docs/
│   └── deployment.md            # Full deployment walkthrough, all 3 targets
├── nginx/
│   └── nginx.conf               # VPS reverse proxy (WebSocket-aware)
├── .github/workflows/ci.yml     # Lint + typecheck + test + security scan
├── .env.example                 # Local dev template (all 3 modes documented)
├── .env.free                    # Free cloud template (HF + Qdrant Cloud + Neon)
├── .env.vps                     # VPS template (Docker service names)
├── docker-compose.yml           # Base: Qdrant + PostgreSQL + API
├── docker-compose.vps.yml       # VPS override: + Nginx + Streamlit container
├── docker-compose.monitoring.yml # Prometheus + Grafana (optional, any target)
├── Dockerfile                   # Multi-stage, non-root — local/VPS
├── Dockerfile.spaces            # Combined API+UI single container — HF Spaces
├── Dockerfile.frontend          # Streamlit-only, lightweight — VPS frontend
├── Makefile                     # make test / lint / docker-up / eval / deploy-free
└── scripts/init_db.sql          # PostgreSQL schema (run manually on Neon)
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

### Why config-driven deployment instead of separate branches/forks?

Maintaining separate codebases per environment guarantees drift — a fix applied to "the free version" silently doesn't reach "the VPS version." Instead, `src/config.py` exposes a single `uses_qdrant_cloud` computed property: when `QDRANT_URL` is set, `qdrant_store.py` connects via Qdrant Cloud's URL+API-key mode; otherwise it falls back to self-hosted host+port. Same pattern for PostgreSQL via `DATABASE_URL` (Neon/Supabase) vs assembled `POSTGRES_*` vars. The retrieval, generation, guardrail, and evaluation code never branches on environment — it only ever calls `settings.database_url` or asks the store for a client. This is what makes the Free → VPS migration a 10-minute `.env` swap with zero code diff.

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
