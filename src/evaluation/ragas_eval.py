"""
RAG Evaluation Framework — custom implementation.

Why not RAGAS directly?
  RAGAS has deep LangChain version constraints that conflict with our pinned stack.
  More importantly: implementing metrics yourself means you can explain every
  calculation in an interview. "I used RAGAS" is a black box.
  "I implemented faithfulness as NLI entailment between answer and context" is engineering.

Metrics implemented:
  1. Faithfulness      — what fraction of answer claims are supported by retrieved context?
  2. Answer Relevance  — does the answer address what was actually asked?
  3. Context Recall    — does retrieved context contain the information in the ground truth?
  4. Context Precision — what fraction of retrieved context is actually useful?

All metrics use the same Groq LLM as the pipeline (no external dependency).
Each metric prompts the LLM as a judge and parses a score from its output.

Usage:
  python -m src.evaluation.ragas_eval --help
  python -m src.evaluation.ragas_eval --test-set evaluation/golden_test_set.json --output evaluation/results.json
"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from src.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ── LLM Judge helpers ─────────────────────────────────────────────────────────

def _call_judge(prompt: str, max_retries: int = 2) -> str:
    """Call Groq LLM as an evaluation judge. Returns the raw response text."""
    from groq import Groq, APITimeoutError
    from src.config import settings

    client = Groq(api_key=settings.groq_api_key)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=settings.groq_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            return response.choices[0].message.content or ""
        except APITimeoutError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return ""


def _extract_score(text: str) -> float:
    """Extract a float score from LLM output like 'Score: 0.8' or '0.75'."""
    patterns = [
        r"[Ss]core[:\s]+([0-9]+\.?[0-9]*)",
        r"([0-9]+\.?[0-9]*)\s*/\s*1",
        r"^([0-9]+\.?[0-9]*)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text.strip(), re.MULTILINE)
        if m:
            # Reject if the match is preceded by a minus sign (negative number)
            start = m.start(1)
            if start > 0 and text[start - 1] == "-":
                continue
            try:
                val = float(m.group(1))
                return min(max(val, 0.0), 1.0)
            except ValueError:
                continue
    # Fallback: look for any non-negative float in range [0, 1]
    floats = re.findall(r"(?<![-\d])(?:0\.\d+|1\.0)(?!\d)", text)
    if floats:
        return float(floats[0])
    return 0.0


# ── Individual Metrics ────────────────────────────────────────────────────────

def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: what fraction of the answer is supported by the retrieved context?

    Method: LLM-as-judge. Ask the model to score whether each answer claim
    is entailed by (supported by) the provided context.
    Score range: 0.0 (completely hallucinated) to 1.0 (fully grounded).
    """
    if not contexts or not answer:
        return 0.0

    context_text = "\n\n".join(contexts[:3])  # top-3 contexts max for token budget
    # Strip disclaimer before evaluating
    clean_answer = answer.split("---")[0].strip()

    prompt = f"""You are an evaluation judge. Rate the faithfulness of the answer below.

Faithfulness measures whether ALL claims in the answer are supported by the provided context.
A faithful answer contains no information not present in the context.

Context:
{context_text}

Answer:
{clean_answer}

Rate faithfulness on a scale from 0.0 to 1.0:
- 1.0: Every claim in the answer is directly supported by the context
- 0.5: Some claims are supported, others are not in the context
- 0.0: The answer contains claims not present in or contradicting the context

Respond with ONLY: Score: <number>"""

    response = _call_judge(prompt)
    score = _extract_score(response)
    logger.debug("faithfulness_scored", score=score)
    return score


def compute_answer_relevance(question: str, answer: str) -> float:
    """
    Answer Relevance: does the answer address the question that was asked?

    Method: LLM-as-judge checks whether the answer is on-topic and complete.
    Score range: 0.0 (completely off-topic) to 1.0 (directly and fully answers).
    """
    if not answer:
        return 0.0

    clean_answer = answer.split("---")[0].strip()

    prompt = f"""You are an evaluation judge. Rate how relevant this answer is to the question.

Question: {question}

Answer: {clean_answer}

Rate answer relevance from 0.0 to 1.0:
- 1.0: Answer directly and completely addresses the question
- 0.5: Answer is partially relevant but incomplete or partially off-topic
- 0.0: Answer does not address the question at all

Respond with ONLY: Score: <number>"""

    response = _call_judge(prompt)
    score = _extract_score(response)
    logger.debug("answer_relevance_scored", score=score)
    return score


def compute_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """
    Context Recall: does the retrieved context contain the information
    present in the ground truth answer?

    Method: LLM-as-judge checks whether the ground truth is supported by context.
    Score range: 0.0 (context is useless) to 1.0 (context contains all needed info).
    """
    if not contexts or not ground_truth:
        return 0.0

    context_text = "\n\n".join(contexts[:3])

    prompt = f"""You are an evaluation judge. Assess context recall.

Context Recall measures whether the retrieved context contains the information
needed to produce the ground truth answer.

Ground Truth Answer: {ground_truth}

Retrieved Context:
{context_text}

Rate context recall from 0.0 to 1.0:
- 1.0: The context contains all information present in the ground truth
- 0.5: The context contains some but not all information in the ground truth
- 0.0: The context does not contain information relevant to the ground truth

Respond with ONLY: Score: <number>"""

    response = _call_judge(prompt)
    score = _extract_score(response)
    logger.debug("context_recall_scored", score=score)
    return score


def compute_context_precision(question: str, contexts: list[str]) -> float:
    """
    Context Precision: what fraction of the retrieved context is actually
    relevant to answering the question?

    Method: LLM-as-judge rates the signal-to-noise ratio of retrieved context.
    Score range: 0.0 (all noise) to 1.0 (all relevant).
    """
    if not contexts:
        return 0.0

    context_text = "\n\n".join(contexts[:3])

    prompt = f"""You are an evaluation judge. Assess context precision.

Context Precision measures what fraction of the retrieved context is useful
for answering the question (vs. irrelevant noise).

Question: {question}

Retrieved Context:
{context_text}

Rate context precision from 0.0 to 1.0:
- 1.0: All retrieved context is relevant and useful for answering the question
- 0.5: About half the context is relevant, half is noise
- 0.0: None of the retrieved context is relevant to the question

Respond with ONLY: Score: <number>"""

    response = _call_judge(prompt)
    score = _extract_score(response)
    logger.debug("context_precision_scored", score=score)
    return score


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_evaluation(
    test_set_path: str,
    output_path: str,
    max_samples: int = 20,
) -> dict[str, Any]:
    """
    Run full offline evaluation on a golden test set.

    For each sample:
      1. Run the query pipeline (retrieval + generation)
      2. Compute all 4 metrics using LLM-as-judge
      3. Aggregate and save results

    Args:
        test_set_path: Path to golden_test_set.json
        output_path:   Path to write results JSON
        max_samples:   Limit samples (for cost control during dev)

    Returns:
        dict with aggregate metrics and per-sample results
    """
    import asyncio
    from src.pipeline.query_pipeline import QueryPipeline

    setup_logging()
    logger.info("evaluation_start", test_set=test_set_path, max_samples=max_samples)

    with open(test_set_path) as f:
        test_set = json.load(f)[:max_samples]

    pipeline = QueryPipeline()
    per_sample: list[dict] = []
    metric_sums = {
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
    }

    for i, sample in enumerate(test_set):
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        category = sample.get("category", "general")

        logger.info(f"evaluating_sample {i+1}/{len(test_set)}", question=question[:60])

        # Run pipeline
        try:
            result = asyncio.run(pipeline.run(
                query=question,
                session_id=f"eval-{i}",
            ))
            answer = result["answer"]
            contexts = [s["content"] if isinstance(s, dict) else s.content
                       for s in result.get("sources", [])]
        except Exception as e:
            logger.error("pipeline_error", sample=i, error=str(e))
            answer = ""
            contexts = []

        # Compute metrics
        faithfulness = compute_faithfulness(answer, contexts)
        answer_relevance = compute_answer_relevance(question, answer)
        context_recall = compute_context_recall(ground_truth, contexts)
        context_precision = compute_context_precision(question, contexts)

        sample_result = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "category": category,
            "metrics": {
                "faithfulness": faithfulness,
                "answer_relevance": answer_relevance,
                "context_recall": context_recall,
                "context_precision": context_precision,
            },
            "contexts_retrieved": len(contexts),
        }
        per_sample.append(sample_result)

        for k in metric_sums:
            metric_sums[k] += sample_result["metrics"][k]

        logger.info(
            "sample_evaluated",
            idx=i + 1,
            faithfulness=round(faithfulness, 3),
            answer_relevance=round(answer_relevance, 3),
        )

        # Rate limit: 1 second between samples
        time.sleep(1.0)

    n = len(per_sample)
    aggregate = {k: round(v / n, 4) for k, v in metric_sums.items()} if n > 0 else metric_sums

    results = {
        "summary": {
            "samples_evaluated": n,
            "aggregate_metrics": aggregate,
        },
        "per_sample": per_sample,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("evaluation_complete", aggregate=aggregate, output=output_path)
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG offline evaluation")
    parser.add_argument(
        "--test-set",
        default="evaluation/golden_test_set.json",
        help="Path to golden test set JSON",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results.json",
        help="Path to write evaluation results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum samples to evaluate (default 20 for cost control)",
    )
    args = parser.parse_args()

    results = run_evaluation(
        test_set_path=args.test_set,
        output_path=args.output,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in results["summary"]["aggregate_metrics"].items():
        print(f"  {metric:<25} {score:.4f}")
    print("=" * 50)
