# ════════════════════════════════════════════════════════════════════
# MediRAG Pro — HuggingFace Spaces Dockerfile
#
# HF Spaces constraints:
#   - Exposes exactly ONE port (7860)
#   - No docker-compose (single container only)
#   - External services required: Qdrant Cloud + Neon PostgreSQL
#   - Secrets injected via Space Settings → Repository Secrets
#   - Free tier: 16GB RAM CPU, ~2 vCPU
#
# Architecture inside this container:
#   Port 7860 → Streamlit (public-facing UI)
#   Port 8000 → FastAPI (internal, Streamlit talks to it via localhost)
#   API_BACKEND_URL=http://localhost:8000 (set in Space secrets)
#
# Usage:
#   1. Create HF Space: huggingface.co/new-space → SDK: Docker
#   2. Set all secrets in Space Settings → Repository secrets
#   3. Push this file as Dockerfile to the Space repo
# ════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# CPU-only torch keeps the image lean for HF free tier
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Create data directory (for BM25 index, parent chunks, semantic cache)
RUN mkdir -p /app/data

# HF Spaces runs as root by default — acceptable for free tier
# (non-root user is in the main Dockerfile for VPS/self-hosted)

# HF Spaces public port
EXPOSE 7860

# Startup script: run FastAPI on 8000 (background) + Streamlit on 7860 (foreground)
# API_BACKEND_URL must be set to http://localhost:8000 in Space secrets
CMD ["sh", "-c", \
  "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 & \
   echo 'Waiting for API to be ready...' && \
   sleep 12 && \
   streamlit run frontend/app.py \
     --server.port=7860 \
     --server.address=0.0.0.0 \
     --server.headless=true \
     --server.enableCORS=false \
     --server.enableXsrfProtection=false"]
