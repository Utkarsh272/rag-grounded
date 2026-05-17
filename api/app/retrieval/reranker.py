"""
Cross-encoder reranker.

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face — a small
(~23 MB) model that scores (query, passage) pairs directly. Unlike bi-encoder
embeddings, the cross-encoder sees both texts at once, giving much better
relevance scores at the cost of being O(N) per query (not pre-computable).

sentence-transformers is already installed (used for embeddings in embedder.py),
so no new dependency is needed.

Model downloads to ~/.cache/huggingface on first run (~23 MB).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder  # type: ignore


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RankedChunk:
    chunk_id: str
    document_id: str
    content: str
    section_title: str | None
    start_char: int
    end_char: int
    rerank_score: float  # raw cross-encoder logit (higher = more relevant)


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    """
    Load the cross-encoder model once and cache it for the process lifetime.
    lru_cache on a module-level function means the model is loaded on the first
    call and reused on all subsequent calls — no repeated disk reads.
    """
    print(f"[reranker] Loading cross-encoder model: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query: str,
    candidates: list[dict],  # dicts with at minimum: chunk_id, content, + metadata
    top_k: int = 5,
) -> list[RankedChunk]:
    """
    Rerank a list of candidate chunks using the cross-encoder.

    Args:
        query: The user's question.
        candidates: List of dicts. Each must have keys:
            - chunk_id (str)
            - document_id (str)
            - content (str)
            - section_title (str | None)
            - start_char (int)
            - end_char (int)
        top_k: How many to return after reranking.

    Returns:
        top_k RankedChunk objects sorted by rerank_score descending.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()

    # Cross-encoder expects list of (query, passage) string pairs.
    pairs = [(query, c["content"]) for c in candidates]

    # predict() returns a numpy array of logit scores, one per pair.
    # Higher score = cross-encoder thinks the passage is more relevant.
    scores: list[float] = model.predict(pairs).tolist()

    ranked: list[RankedChunk] = []
    for chunk, score in zip(candidates, scores):
        ranked.append(
            RankedChunk(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                content=chunk["content"],
                section_title=chunk.get("section_title"),
                start_char=chunk["start_char"],
                end_char=chunk["end_char"],
                rerank_score=score,
            )
        )

    ranked.sort(key=lambda r: r.rerank_score, reverse=True)
    return ranked[:top_k]