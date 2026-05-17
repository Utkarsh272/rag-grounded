"""
Hybrid retrieval: vector search + BM25 fused via Reciprocal Rank Fusion (RRF),
then reranked with a cross-encoder.

Why RRF?
  - Simple: no tuning of per-source weights. The 60-constant in the denominator
    is the standard from the original Cormack et al. 2009 paper — it dampens the
    impact of very high ranks, which prevents one strong result from dominating.
  - Robust: a chunk that appears in both vector and BM25 results gets a combined
    score even if it isn't #1 in either list individually.
  - Calibration-free: unlike weighted score combination (0.7 * vec + 0.3 * bm25),
    RRF doesn't require normalising scores across retrieval methods.

Pipeline:
  vector search (top 20) ─┐
                           ├─ RRF fusion ─ top 10 candidates ─ cross-encoder ─ top 5
  BM25 search   (top 20) ─┘
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from app.retrieval.bm25 import BM25Result, bm25_search
from app.retrieval.reranker import RankedChunk, rerank
from app.retrieval.vector import SearchResult, vector_search


# RRF constant. Cormack et al. recommend k=60 for most IR tasks.
RRF_K = 60


@dataclass
class HybridResult:
    chunk_id: str
    document_id: str
    content: str
    section_title: str | None
    start_char: int
    end_char: int
    rrf_score: float       # after RRF fusion
    rerank_score: float    # after cross-encoder (0.0 if reranking was skipped)
    # For A/B comparison in eval harness:
    vector_rank: int | None   # rank in vector results (1-based), None if absent
    bm25_rank: int | None     # rank in BM25 results (1-based), None if absent


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    vector_candidates: int = 20,
    bm25_candidates: int = 20,
    rerank_candidates: int = 10,
) -> list[HybridResult]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query: User question.
        top_k: Final number of chunks to return (after reranking).
        document_id: If set, scopes retrieval to one document.
        vector_candidates: How many vector results to fetch before RRF.
        bm25_candidates: How many BM25 results to fetch before RRF.
        rerank_candidates: How many RRF candidates to send to cross-encoder.

    Returns:
        List of HybridResult, sorted by rerank_score descending, length = top_k.
    """

    # ── 1. Fetch candidates from both retrieval methods in parallel ──────────
    # asyncio.gather would be cleaner, but vector_search and bm25_search both
    # use the Supabase client which is synchronous under the hood (supabase-py
    # wraps httpx but doesn't expose async RPC properly in v1). Running them
    # sequentially keeps the code simple; typical latency is <100ms each so the
    # total is still acceptable for a portfolio demo.
    vector_results: list[SearchResult] = vector_search(
        query=query,
        top_k=vector_candidates,
        document_id=document_id,
    )
    bm25_results: list[BM25Result] = bm25_search(
        query=query,
        top_k=bm25_candidates,
        document_id=document_id,
    )

    # ── 2. Reciprocal Rank Fusion ────────────────────────────────────────────
    rrf_scores: dict[str, float] = {}
    vector_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, result in enumerate(vector_results, start=1):
        cid = result.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        vector_ranks[cid] = rank

    for rank, result in enumerate(bm25_results, start=1):
        cid = result.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        bm25_ranks[cid] = rank

    # Collect unique chunks from both result sets into one lookup dict
    all_chunks: dict[str, dict] = {}

    for r in vector_results:
        all_chunks[r.chunk_id] = {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "content": r.content,
            "section_title": r.section_title,
            "start_char": r.start_char,
            "end_char": r.end_char,
        }

    for r in bm25_results:
        if r.chunk_id not in all_chunks:
            all_chunks[r.chunk_id] = {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "content": r.content,
                "section_title": r.section_title,
                "start_char": r.start_char,
                "end_char": r.end_char,
            }

    # Sort by RRF score descending, take top rerank_candidates for the cross-encoder
    sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates_ids = [cid for cid, _ in sorted_by_rrf[:rerank_candidates]]
    top_candidates = [all_chunks[cid] for cid in top_candidates_ids if cid in all_chunks]

    # ── 3. Cross-encoder reranking ───────────────────────────────────────────
    if not top_candidates:
        return []

    reranked: list[RankedChunk] = rerank(
        query=query,
        candidates=top_candidates,
        top_k=top_k,
    )

    # ── 4. Assemble final results ─────────────────────────────────────────────
    results: list[HybridResult] = []
    for rc in reranked:
        results.append(
            HybridResult(
                chunk_id=rc.chunk_id,
                document_id=rc.document_id,
                content=rc.content,
                section_title=rc.section_title,
                start_char=rc.start_char,
                end_char=rc.end_char,
                rrf_score=rrf_scores.get(rc.chunk_id, 0.0),
                rerank_score=rc.rerank_score,
                vector_rank=vector_ranks.get(rc.chunk_id),
                bm25_rank=bm25_ranks.get(rc.chunk_id),
            )
        )

    return results


# ---------------------------------------------------------------------------
# A/B comparison helper — used by the /v1/search?mode= endpoint
# ---------------------------------------------------------------------------

def retrieve_all_modes(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> dict:
    """
    Run all three retrieval modes for direct A/B comparison.
    Returns a dict with keys: vector_only, hybrid_rrf, hybrid_reranked.
    Used by the eval harness and the /v1/search?mode=compare endpoint.
    """

    # Vector-only
    vector_results = vector_search(query=query, top_k=top_k, document_id=document_id)
    vector_only = [
        {
            "rank": i + 1,
            "chunk_id": r.chunk_id,
            "section": r.section_title,
            "similarity": round(r.similarity, 4),
            "content": r.content[:200],
        }
        for i, r in enumerate(vector_results[:top_k])
    ]

    # Full hybrid (includes RRF intermediate + final reranked)
    hybrid = hybrid_retrieve(
        query=query,
        top_k=top_k,
        document_id=document_id,
    )
    hybrid_reranked = [
        {
            "rank": i + 1,
            "chunk_id": r.chunk_id,
            "section": r.section_title,
            "rerank_score": round(r.rerank_score, 4),
            "rrf_score": round(r.rrf_score, 6),
            "vector_rank": r.vector_rank,
            "bm25_rank": r.bm25_rank,
            "content": r.content[:200],
        }
        for i, r in enumerate(hybrid)
    ]

    return {
        "query": query,
        "vector_only": vector_only,
        "hybrid_reranked": hybrid_reranked,
    }