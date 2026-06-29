"""
Input sanitization — runs after emergency check, before retrieval.

Catches:
  1. Prompt injection: attempts to override system instructions
  2. Jailbreak patterns: "ignore previous", "pretend you are", "DAN mode"
  3. Extreme query length after stripping (catches token-stuffing)

This is intentionally lightweight — a regex pre-filter.
A production system would add a fine-tuned classifier (e.g. Llama Guard).
For a portfolio project, regex is the correct tradeoff: demonstrable,
explainable, zero latency, no external dependency.
"""
import re

from src.logging_config import get_logger

logger = get_logger(__name__)

INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+", re.I),
    re.compile(r"you\s+are\s+now\s+(a\s+)?(DAN|jailbreak|unrestricted)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?!a\s+doctor|a\s+nurse)", re.I),
    re.compile(r"system\s*:\s*you\s+are", re.I),
    re.compile(r"<\s*system\s*>", re.I),
    re.compile(r"\[INST\]|\[\/INST\]|<<SYS>>|<</SYS>>", re.I),
]

MAX_EFFECTIVE_QUERY_LENGTH = 1500  # chars after strip


def sanitize_query(query: str) -> tuple[bool, str]:
    """
    Check query for injection attempts.

    Returns:
        (is_safe: bool, cleaned_query_or_reason: str)
        If safe: (True, stripped_query)
        If unsafe: (False, reason_string)
    """
    stripped = query.strip()

    # Length guard (API schema allows 2000 chars; effective limit is lower)
    if len(stripped) > MAX_EFFECTIVE_QUERY_LENGTH:
        stripped = stripped[:MAX_EFFECTIVE_QUERY_LENGTH]
        logger.warning("query_truncated", original_len=len(query))

    # Injection pattern scan
    for pattern in INJECTION_PATTERNS:
        if pattern.search(stripped):
            logger.warning(
                "injection_attempt_blocked",
                pattern=pattern.pattern,
                query_preview=stripped[:80],
            )
            return False, "Query contains content that cannot be processed."

    return True, stripped
