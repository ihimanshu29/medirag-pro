"""
Groq LLM interface with:
  - Retry logic with exponential backoff (tenacity)
  - Structured prompt: context + history + query
  - Citation injection: each source chunk referenced inline
  - Confidence scoring: derived from top reranker score
  - Medical disclaimer appended automatically

Why structured prompt over LangChain chains?
  More control, easier to debug, no hidden magic.
  LangChain RetrievalQA is a black box — you can't see exactly what
  gets sent to the model without verbose=True hacks.
  Direct API calls are transparent, testable, and interviewer-explainable.
"""
import time

from groq import Groq, APIStatusError, APITimeoutError

from src.config import settings
from src.logging_config import get_logger
from src.models import RetrievedChunk

logger = get_logger(__name__)

MEDICAL_DISCLAIMER = (
    "\n\n---\n⚠️ *This response is based on the ingested medical reference documents. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider.*"
)

SYSTEM_PROMPT = """You are MediRAG, a medical knowledge assistant.
Your ONLY source of truth is the context provided below.

Rules you must always follow:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. If the answer is not in the context, say exactly: "I don't have enough information in the available documents to answer this question."
3. Cite your sources inline using [Source: <filename>, p.<page>] after each claim.
4. If the context contains conflicting information, note the conflict explicitly.
5. Never invent drug names, dosages, or medical facts.
6. Express appropriate uncertainty when the context is ambiguous."""


def _build_prompt(
    query: str,
    context_chunks: list[RetrievedChunk],
    chat_history: str,
) -> str:
    """Assemble the full user message: context + history + query."""
    # Format retrieved chunks with source labels
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_label = f"[Source {i}: {chunk.source_file}, p.{chunk.page}]"
        if chunk.section:
            source_label += f" [{chunk.section}]"
        context_parts.append(f"{source_label}\n{chunk.text}")

    context_block = "\n\n---\n\n".join(context_parts)

    history_block = ""
    if chat_history:
        history_block = f"\n\nConversation history:\n{chat_history}\n"

    return (
        f"Context from medical documents:\n\n{context_block}"
        f"{history_block}"
        f"\n\nQuestion: {query}"
        f"\n\nAnswer (cite sources inline):"
    )


def generate_answer(
    query: str,
    context_chunks: list[RetrievedChunk],
    chat_history: str = "",
    max_retries: int = 3,
) -> tuple[str, float]:
    """
    Generate a cited, grounded answer using Groq LLM.

    Returns:
        (answer_with_disclaimer: str, confidence: float)
        confidence = sigmoid of top reranker score (0.0 if no chunks)
    """
    if not context_chunks:
        no_context = (
            "I don't have enough information in the available documents to answer this question."
            + MEDICAL_DISCLAIMER
        )
        return no_context, 0.0

    # Confidence = top chunk's reranker score (already sigmoid-normalised)
    confidence = context_chunks[0].score

    prompt = _build_prompt(query, context_chunks, chat_history)
    client = Groq(api_key=settings.groq_api_key)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.groq_model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            answer = response.choices[0].message.content or ""
            answer = answer.strip() + MEDICAL_DISCLAIMER

            logger.info(
                "llm_response",
                model=settings.groq_model_name,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                confidence=round(confidence, 3),
                attempt=attempt + 1,
            )
            return answer, confidence

        except APITimeoutError as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("llm_timeout", attempt=attempt + 1, wait_s=wait)
            time.sleep(wait)

        except APIStatusError as e:
            last_error = e
            if e.status_code in {429, 503}:   # rate limit or overload
                wait = 2 ** attempt
                logger.warning("llm_rate_limited", status=e.status_code, wait_s=wait)
                time.sleep(wait)
            else:
                # Non-retryable (auth error, bad request etc.)
                logger.error("llm_api_error", status=e.status_code, error=str(e))
                break

    # All retries exhausted
    logger.error("llm_all_retries_failed", attempts=max_retries, error=str(last_error))
    fallback = (
        "I'm temporarily unable to generate a response. Please try again in a moment."
        + MEDICAL_DISCLAIMER
    )
    return fallback, 0.0
