"""
Hybrid retrieval: vector search (Jina embeddings) + BM25 fused via RRF,
then passthrough reranker (no local model needed).

Pipeline:
  vector search (top 20) ─┐
                           ├─ RRF fusion ─ top 5
  BM25 search   (top 20) ─┘
"""
from __future__ import annotations

from dataclasses import dataclass
from app.retrieval.bm25 import BM25Result, bm25_search
from app.retrieval.reranker import RankedChunk, rerank
from app.retrieval.vector import SearchResult, vector_search

RRF_K = 60


@dataclass
class HybridResult:
    chunk_id: str
    document_id: str
    content: str
    section_title: str | None
    start_char: int
    end_char: int
    rrf_score: float
    rerank_score: float
    vector_rank: int | None
    bm25_rank: int | None


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
    vector_candidates: int = 20,
    bm25_candidates: int = 20,
    rerank_candidates: int = 10,
) -> list[HybridResult]:

    vector_results: list[SearchResult] = vector_search(
        query=query, top_k=vector_candidates, document_id=document_id,
    )
    bm25_results: list[BM25Result] = bm25_search(
        query=query, top_k=bm25_candidates, document_id=document_id,
    )

    rrf_scores: dict[str, float] = {}
    vector_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, r in enumerate(vector_results, start=1):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        vector_ranks[r.chunk_id] = rank

    for rank, r in enumerate(bm25_results, start=1):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        bm25_ranks[r.chunk_id] = rank

    all_chunks: dict[str, dict] = {}
    for r in vector_results:
        all_chunks[r.chunk_id] = {
            "chunk_id": r.chunk_id, "document_id": r.document_id,
            "content": r.content, "section_title": r.section_title,
            "start_char": r.start_char, "end_char": r.end_char,
        }
    for r in bm25_results:
        if r.chunk_id not in all_chunks:
            all_chunks[r.chunk_id] = {
                "chunk_id": r.chunk_id, "document_id": r.document_id,
                "content": r.content, "section_title": r.section_title,
                "start_char": r.start_char, "end_char": r.end_char,
            }

    sorted_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [
        all_chunks[cid]
        for cid, _ in sorted_by_rrf[:rerank_candidates]
        if cid in all_chunks
    ]

    if not top_candidates:
        return []

    reranked: list[RankedChunk] = rerank(
        query=query, candidates=top_candidates, top_k=top_k,
    )

    return [
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
        for rc in reranked
    ]


def retrieve_all_modes(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> dict:
    hybrid = hybrid_retrieve(query=query, top_k=top_k, document_id=document_id)
    return {
        "query": query,
        "hybrid_reranked": [
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
        ],
    }
