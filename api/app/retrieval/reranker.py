# api/app/retrieval/reranker.py
"""
Reranker — lightweight version for production (no cross-encoder model).

In production (RERANKER=none), returns candidates sorted by RRF score as-is.
In dev (RERANKER=crossencoder), uses cross-encoder/ms-marco-MiniLM-L-6-v2.

The cross-encoder gives better precision but requires ~300MB RAM which exceeds
Render's free tier (512MB). RRF alone is still significantly better than
vector-only retrieval.

Set RERANKER=crossencoder in your local .env to enable it locally.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class RankedChunk:
    chunk_id: str
    document_id: str
    content: str
    section_title: str | None
    start_char: int
    end_char: int
    rerank_score: float


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[RankedChunk]:
    mode = os.environ.get("RERANKER", "none").lower()

    if mode == "crossencoder":
        return _rerank_crossencoder(query, candidates, top_k)
    else:
        return _rerank_passthrough(candidates, top_k)


def _rerank_passthrough(candidates: list[dict], top_k: int) -> list[RankedChunk]:
    """Return candidates as-is (already RRF-sorted), no model needed."""
    results = []
    for i, c in enumerate(candidates[:top_k]):
        results.append(RankedChunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            content=c["content"],
            section_title=c.get("section_title"),
            start_char=c["start_char"],
            end_char=c["end_char"],
            rerank_score=float(top_k - i),  # synthetic descending score
        ))
    return results


def _rerank_crossencoder(query: str, candidates: list[dict], top_k: int) -> list[RankedChunk]:
    from sentence_transformers import CrossEncoder  # type: ignore
    global _ce_model
    if _ce_model is None:
        print("[reranker] Loading cross-encoder model...")
        _ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(query, c["content"]) for c in candidates]
    scores: list[float] = _ce_model.predict(pairs).tolist()

    ranked = []
    for c, score in zip(candidates, scores):
        ranked.append(RankedChunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            content=c["content"],
            section_title=c.get("section_title"),
            start_char=c["start_char"],
            end_char=c["end_char"],
            rerank_score=score,
        ))
    ranked.sort(key=lambda r: r.rerank_score, reverse=True)
    return ranked[:top_k]


_ce_model = None
