"""
Search route — Day 3 update.

GET /v1/search
  ?q=<query>              required
  &document_id=<uuid>     optional — scope to one document
  &top_k=5                optional (default 5, max 20)
  &mode=hybrid            optional — 'vector' | 'hybrid' | 'compare'
                          default: 'hybrid'

mode=vector    → vector-only cosine similarity (Day 2 behaviour)
mode=hybrid    → vector + BM25 fused via RRF, then cross-encoder reranked
mode=compare   → runs all modes, returns side-by-side for the eval harness
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.retrieval.hybrid import hybrid_retrieve, retrieve_all_modes
from app.retrieval.vector import vector_search

router = APIRouter()


@router.get("/v1/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    document_id: str | None = Query(default=None, description="Filter to one document UUID"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return"),
    mode: str = Query(default="hybrid", description="'vector' | 'hybrid' | 'compare'"),
):
    """
    Semantic search over ingested document chunks.

    mode=vector   — fast, simple cosine similarity.
    mode=hybrid   — production path: vector + BM25 + RRF + cross-encoder.
    mode=compare  — A/B: returns both modes in one response (for eval harness).
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' must not be blank.")

    if mode not in ("vector", "hybrid", "compare"):
        raise HTTPException(
            status_code=400,
            detail="mode must be 'vector', 'hybrid', or 'compare'.",
        )

    if mode == "compare":
        return retrieve_all_modes(
            query=q,
            top_k=top_k,
            document_id=document_id,
        )

    if mode == "vector":
        results = vector_search(query=q, top_k=top_k, document_id=document_id)
        return [
            {
                "rank": i + 1,
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "section": r.section_title,
                "similarity": round(r.similarity, 4),
                "start_char": r.start_char,
                "end_char": r.end_char,
                "content": r.content,
            }
            for i, r in enumerate(results)
        ]

    # mode == "hybrid" (default)
    results = hybrid_retrieve(
        query=q,
        top_k=top_k,
        document_id=document_id,
    )
    return [
        {
            "rank": i + 1,
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "section": r.section_title,
            "rerank_score": round(r.rerank_score, 4),
            "rrf_score": round(r.rrf_score, 6),
            "vector_rank": r.vector_rank,
            "bm25_rank": r.bm25_rank,
            "start_char": r.start_char,
            "end_char": r.end_char,
            "content": r.content,
        }
        for i, r in enumerate(results)
    ]