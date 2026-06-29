"""
Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Why RRF over weighted sum?
- Weighted sum requires tuning alpha (dense weight) vs beta (sparse weight).
  Wrong weights hurt quality. Needs per-dataset calibration.
- RRF is parameter-free: score = sum(1 / (k + rank_i)) across retrieval methods.
  k=60 is empirically robust across datasets (from the original Cormack 2009 paper).
- RRF handles score scale mismatch naturally — it only uses rank positions,
  not raw scores. Dense scores (cosine, 0-1) and BM25 scores (unbounded) are
  incomparable. RRF sidesteps this entirely.
- Consistently matches or beats tuned weighted fusion on BEIR benchmarks.

Flow:
  dense_results (top-20)  ─┐
                             ├─ RRF ─→ unified_ranked_list (top-20 deduplicated)
  bm25_results  (top-20)  ─┘

Deduplication: same chunk_id from both sources → scores are summed (union, not
intersection). A chunk appearing #1 in dense AND #1 in BM25 gets maximum RRF score.
"""
from src.logging_config import get_logger
from src.models import RetrievedChunk

logger = get_logger(__name__)

RRF_K = 60  # Standard constant from Cormack et al. 2009


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Merge dense and BM25 results using Reciprocal Rank Fusion.

    Args:
        dense_results: Ranked list from vector search (best first).
        bm25_results:  Ranked list from BM25 keyword search (best first).
        top_k:         Number of results to return after fusion.

    Returns:
        Deduplicated, RRF-scored list of RetrievedChunks (best first).
    """
    # chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # chunk_id → best RetrievedChunk object (carry metadata)
    chunk_map: dict[str, RetrievedChunk] = {}

    def _accumulate(results: list[RetrievedChunk], method_label: str) -> None:
        for rank, chunk in enumerate(results):
            rrf_score = 1.0 / (RRF_K + rank + 1)   # rank is 0-indexed
            cid = chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
            # Keep the chunk object; prefer dense if duplicate (higher text quality)
            if cid not in chunk_map or method_label == "dense":
                chunk_map[cid] = chunk

    _accumulate(dense_results, "dense")
    _accumulate(bm25_results, "bm25")

    # Sort by descending RRF score
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    # Build output, normalise RRF scores to [0, 1]
    max_rrf = rrf_scores[sorted_ids[0]] if sorted_ids else 1.0

    results: list[RetrievedChunk] = []
    for cid in sorted_ids[:top_k]:
        chunk = chunk_map[cid]
        results.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            parent_id=chunk.parent_id,
            text=chunk.text,
            source_file=chunk.source_file,
            page=chunk.page,
            section=chunk.section,
            score=rrf_scores[cid] / max_rrf,   # normalised
            retrieval_method="hybrid",
        ))

    logger.debug(
        "rrf_fusion",
        dense_count=len(dense_results),
        bm25_count=len(bm25_results),
        fused_count=len(results),
        unique_sources={c.source_file for c in results},
    )
    return results
