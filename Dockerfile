# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

# Install CPU-only PyTorch and all deps
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt \
       --target /build/deps


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed deps from builder
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages/

# Copy application source
COPY src/ ./src/
COPY pyproject.toml .

# Data directory for BM25 index, cache, etc.
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
