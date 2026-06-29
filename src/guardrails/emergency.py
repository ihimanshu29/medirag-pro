"""
Emergency detection guardrail — runs BEFORE retrieval.

Why pre-retrieval?
  If someone types "I took 40 Tylenol pills", the RAG pipeline must NOT run.
  Retrieving acetaminophen overdose dosage info and answering factually
  is the worst possible response. We bypass entirely and return crisis resources.

Detection strategy: two layers
  1. Keyword patterns (fast, zero latency, high recall)
     Covers explicit terms: overdose, suicide, chest pain, can't breathe, etc.
  2. Contextual patterns (regex, catches indirect phrasing)
     "I want to end it", "don't want to be here anymore", "took too many pills"

On detection:
  - Return a hardcoded safe response with emergency numbers
  - Log the event to Prometheus counter (medirag_emergency_queries_total)
  - Do NOT pass query to retrieval or LLM

This is a portfolio differentiator. 99% of RAG projects skip this entirely.
Every medical AI interviewer will ask about it.
"""
import re

from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Keyword patterns (exact word match, case-insensitive) ────────────────────
EMERGENCY_KEYWORDS: frozenset[str] = frozenset({
    # Self-harm / suicide
    "suicide", "suicidal", "kill myself", "end my life", "end it all",
    "self harm", "self-harm", "cutting myself", "hurt myself",
    "don't want to live", "dont want to live", "want to die",
    # Overdose
    "overdose", "took too many pills", "took too much", "swallowed too many",
    # Acute medical emergencies
    "chest pain", "chest pressure", "heart attack", "cardiac arrest",
    "can't breathe", "cannot breathe", "difficulty breathing", "stop breathing",
    "stroke", "face drooping", "arm weakness", "slurred speech",
    "unconscious", "passed out", "not breathing", "choking",
    "severe bleeding", "uncontrolled bleeding",
    "anaphylaxis", "allergic reaction", "throat closing",
    "seizure", "convulsing",
    # Abuse
    "being abused", "someone is hurting me", "domestic violence",
})

# ── Contextual patterns (regex for indirect phrasing) ────────────────────────
EMERGENCY_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bi\s+(took|swallowed|consumed|ingested)\s+too\s+many\b", re.I),
    re.compile(r"\bi\s+(took|swallowed|ingested)\s+\d+\s+\w+\s+(pills?|tablets?|capsules?)\b", re.I),
    re.compile(r"\bdon[\'']?t\s+want\s+to\s+(be\s+here|exist|live)\b", re.I),
    re.compile(r"\bthinking\s+about\s+(ending|taking)\s+(my\s+)?(life|it)\b", re.I),
    re.compile(r"\b(plan|planning)\s+to\s+(hurt|harm|kill)\s+myself\b", re.I),
    re.compile(r"\b(sudden|severe|crushing)\s+(chest|head)\s+(pain|pressure)\b", re.I),
    re.compile(r"\bcall\s+(911|ambulance|emergency)\b", re.I),
]

# ── Safe response (hardcoded — never LLM-generated for emergencies) ───────────
EMERGENCY_RESPONSE = """⚠️ **This sounds like an emergency. Please seek immediate help.**

**If you are in immediate danger, call emergency services now:**
- 🇮🇳 **India Emergency:** 112
- 🇺🇸 **US Emergency:** 911
- 🌍 **International Emergency:** 112

**Mental Health Crisis Lines:**
- 🇮🇳 iCall (India): 9152987821
- 🇺🇸 988 Suicide & Crisis Lifeline (US): Call or text **988**
- 🌍 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

**Medical Poison Control:**
- 🇮🇳 India: 1800-116-117
- 🇺🇸 US Poison Control: 1-800-222-1222

Please do not rely on this chatbot for emergencies. Reach out to a real person right now.
"""


def check_emergency(query: str) -> tuple[bool, str]:
    """
    Check if a query is a medical or mental health emergency.

    Returns:
        (is_emergency: bool, response: str)
        If is_emergency=True, response is the safe crisis message.
        If is_emergency=False, response is "" and normal pipeline continues.
    """
    query_lower = query.lower()

    # Layer 1: keyword match
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in query_lower:
            logger.warning(
                "emergency_detected",
                method="keyword",
                trigger=keyword,
                query_preview=query[:60],
            )
            _increment_counter()
            return True, EMERGENCY_RESPONSE

    # Layer 2: contextual regex
    for pattern in EMERGENCY_PATTERNS:
        if pattern.search(query):
            logger.warning(
                "emergency_detected",
                method="regex",
                pattern=pattern.pattern,
                query_preview=query[:60],
            )
            _increment_counter()
            return True, EMERGENCY_RESPONSE

    return False, ""


def _increment_counter() -> None:
    """Increment Prometheus emergency counter (fails silently if metrics not set up)."""
    try:
        from src.api.main import EMERGENCY_QUERIES
        EMERGENCY_QUERIES.inc()
    except Exception:
        pass
