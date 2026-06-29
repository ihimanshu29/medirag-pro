"""
Rate limiting for MediRAG Pro API.

Uses slowapi (FastAPI-native, wraps limits library).
Limits are per-IP to prevent abuse and Groq token overruns.

Limits:
  /chat    — 20 requests/minute  (LLM calls are expensive)
  /ingest  — 5 requests/minute   (embedding is CPU-intensive)
  /feedback — 60 requests/minute (cheap DB write, more lenient)

In production with multiple API instances, use Redis as the storage
backend instead of in-memory:
  from limits.storage import RedisStorage
  limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

# Module-level limiter instance — imported by route handlers
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],   # global fallback
)
