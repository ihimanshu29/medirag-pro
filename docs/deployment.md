# MediRAG Pro — Deployment Guide

Three deployment targets. One codebase. Only environment variables change.

---

## Quick Reference

| Target | Command | Public URL |
|---|---|---|
| Local | `docker compose up -d` | http://localhost:8501 |
| Free Cloud | Push Dockerfile.spaces to HF Space | https://your-space.hf.space |
| VPS | `docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d` | http://YOUR_IP:80 |

---

## Target 1 — Local Development

```bash
cp .env.example .env          # add GROQ_API_KEY
docker compose up qdrant postgres -d
uvicorn src.api.main:app --reload   # API on :8000
streamlit run frontend/app.py       # UI on :8501
```

Or full Docker stack: `docker compose up -d`

---

## Target 2 — Free Cloud (HF Spaces + Qdrant Cloud + Neon)

### Step 1: Qdrant Cloud (5 min)
1. https://cloud.qdrant.io → free cluster → copy **Cluster URL** + **API Key**

### Step 2: Neon PostgreSQL (5 min)
1. https://console.neon.tech → new project → copy connection string
2. SQL Editor → run `scripts/init_db.sql`
3. URL format: `postgresql://user:pass@ep-xxx.neon.tech/medirag?sslmode=require`

### Step 3: HuggingFace Space (10 min)
1. https://huggingface.co/new-space → SDK: Docker → Hardware: CPU Basic (free)
2. In Space Settings → Repository secrets, add:

```
GROQ_API_KEY               = your_key
QDRANT_URL                 = https://YOUR_CLUSTER.qdrant.io
QDRANT_API_KEY             = your_qdrant_key
QDRANT_COLLECTION_NAME     = medical_docs
DATABASE_URL               = postgresql://...neon.tech/medirag?sslmode=require
API_BACKEND_URL            = http://localhost:8000
APP_ENV                    = production
CORS_ORIGINS               = *
LOG_LEVEL                  = INFO
GROQ_MODEL_NAME            = llama-3.3-70b-versatile
EMBEDDING_MODEL            = BAAI/bge-small-en-v1.5
RERANKER_MODEL             = BAAI/bge-reranker-base
RETRIEVAL_TOP_K            = 20
RERANK_TOP_N               = 5
BM25_INDEX_PATH            = ./data/bm25_index.pkl
PARENT_CHUNKS_PATH         = ./data/parent_chunks.pkl
CACHE_DIR                  = ./data/semantic_cache
CACHE_SIMILARITY_THRESHOLD = 0.92
CACHE_MAX_SIZE_GB          = 1
```

### Step 4: Push to HF Space

```bash
# Create deployment branch with HF Dockerfile
git checkout -b hf-deploy
cp Dockerfile.spaces Dockerfile
git add Dockerfile && git commit -m "chore: HF Spaces Dockerfile"
git remote add spaces https://huggingface.co/spaces/YOUR_USERNAME/medirag-pro
git push spaces hf-deploy:main
git checkout main
```

First build: 8–15 minutes. Verify: `curl https://YOUR_USERNAME-medirag-pro.hf.space/health`

---

## Target 3 — VPS

```bash
# On VPS
git clone https://github.com/YOURUSERNAME/medirag-pro.git && cd medirag-pro
cp .env.vps .env
nano .env   # set GROQ_API_KEY, POSTGRES_PASSWORD, API_BACKEND_URL
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d
```

Stack: Nginx (80/443) → FastAPI (8000) → Streamlit (8501) → Qdrant + PostgreSQL

### Fly.io (free, no sleep)

```bash
fly launch && fly scale memory 1024
fly secrets set GROQ_API_KEY=... QDRANT_URL=... DATABASE_URL=... APP_ENV=production
fly deploy
```

---

## Migration: Free Cloud → VPS

```bash
cp .env.vps .env
# Edit: POSTGRES_PASSWORD, API_BACKEND_URL
docker compose -f docker-compose.yml -f docker-compose.vps.yml up -d
```

**Total time: under 10 minutes. Zero code changes.**

---

## Environment Variable Reference

| Variable | Local | Free Cloud | VPS | Notes |
|---|---|---|---|---|
| `GROQ_API_KEY` | required | required | required | |
| `QDRANT_URL` | — | required | — | Qdrant Cloud URL |
| `QDRANT_API_KEY` | — | required | — | Qdrant Cloud key |
| `QDRANT_HOST` | localhost | — | qdrant | Docker service name |
| `DATABASE_URL` | — | required | — | Full Neon URL with ?sslmode=require |
| `POSTGRES_HOST` | localhost | — | postgres | Docker service name |
| `API_BACKEND_URL` | http://localhost:8000 | http://localhost:8000 | http://IP:8000 | Frontend discovers API here |
| `CORS_ORIGINS` | * | * | https://domain.com | |
| `APP_ENV` | development | production | production | |

---

## Troubleshooting

**Qdrant Cloud: connection refused** — ensure `QDRANT_URL` includes `https://` and `QDRANT_API_KEY` is set.

**Neon: SSL error** — ensure `DATABASE_URL` ends with `?sslmode=require`.

**HF Space: "Cannot connect to API"** — set `API_BACKEND_URL=http://localhost:8000` (API and UI share the same container).

**VPS: Streamlit blank page** — nginx WebSocket headers are in `nginx/nginx.conf` — ensure nginx container is running.

**Health shows embedder degraded** — normal on first startup. Model loads on first `/chat` request, then shows healthy.
