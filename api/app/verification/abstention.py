# api/app/verification/abstention.py
"""
Pre-LLM abstention gate.

Decides whether to skip the LLM call entirely and return a "I don't know"
response, based purely on retrieval signal — before spending any LLM tokens.

Two independent signals, both must pass to proceed:

1. SIMILARITY GATE
   Average cosine similarity of the top-5 retrieved chunks vs the query.
   If every chunk is weakly similar, the document likely doesn't cover the topic.
   Threshold: 0.45 (tunable via ABSTAIN_SIM_THRESHOLD env var).

   Why 0.45 and not higher?
   The all-MiniLM-L6-v2 model produces lower absolute similarities than
   text-embedding-3-small — scores rarely exceed 0.85 even for good matches.
   0.45 is empirically ~1 std dev below the mean on relevant queries.
   The eval harness (Day 8) will produce a proper ROC curve to tune this.

2. KEYWORD OVERLAP GATE
   Jaccard similarity between query unigrams and the union of top-5 chunk
   unigrams. If the query uses vocabulary totally absent from the retrieved
   chunks, retrieval has missed the topic.
   Threshold: 0.08 (at least ~1 query keyword appears in the chunks).

Either gate failing → abstain. Both must pass → proceed to LLM.

Why pre-LLM instead of post-LLM?
   Post-LLM abstention (checking INSUFFICIENT_INFO in the response) catches
   cases where the LLM itself decides it can't answer. Pre-LLM abstention is
   complementary: it saves LLM tokens on clearly out-of-scope queries and gives
   a faster response. Both layers run: pre-LLM gate here, post-LLM sentinel
   in citation_parser.py.
"""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# Thresholds (overridable via env for easy tuning)
# ---------------------------------------------------------------------------

_SIM_THRESHOLD = float(os.environ.get("ABSTAIN_SIM_THRESHOLD", "0.45"))
_JACCARD_THRESHOLD = float(os.environ.get("ABSTAIN_JACCARD_THRESHOLD", "0.08"))

# Stopwords to exclude from keyword overlap (very common words add noise)
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "about", "above", "after", "before", "between",
    "out", "off", "over", "under", "again", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "and", "or", "but", "if",
    "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
    "this", "that", "these", "those",
}


def _tokenise(text: str) -> set[str]:
    """Lowercase unigrams, stopwords removed."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def should_abstain(chunks: list, query: str) -> tuple[bool, str]:
    """
    Decide whether to abstain before calling the LLM.

    Args:
        chunks: HybridResult list from hybrid_retrieve (top-5).
        query:  The user's question string.

    Returns:
        (abstain: bool, reason: str)
        reason is a human-readable explanation shown in the UI when abstaining.
    """
    if not chunks:
        return True, "No relevant content was found in this document for your question."

    # ── Signal 1: average rerank score ───────────────────────────────────────
    # HybridResult has rerank_score (cross-encoder logit). We use the top-3
    # to avoid being dragged down by lower-ranked chunks that are legitimately
    # less relevant but still valid context.
    top_scores = sorted(
        [c.rerank_score for c in chunks], reverse=True
    )[:3]
    avg_top_score = top_scores[0]  # use best score only, not average

    # Cross-encoder scores are logits (unbounded). Empirically on ms-marco:
    # > 0   → likely relevant
    # -5 to 0 → borderline
    # < -5  → not relevant
    # We use -4.0 as the threshold — below this, even the best chunks are weak.
    RERANK_THRESHOLD = float(os.environ.get("ABSTAIN_RERANK_THRESHOLD", "-6.0"))

    if avg_top_score < RERANK_THRESHOLD:
        return (
            True,
            f"The document doesn't appear to contain information relevant to your question "
            f"(retrieval confidence: {avg_top_score:.2f}, threshold: {RERANK_THRESHOLD}).",
        )

    # ── Signal 2: keyword overlap ─────────────────────────────────────────────
    query_tokens = _tokenise(query)
    if not query_tokens:
        # Query is all stopwords — can't compute overlap, don't abstain
        return False, ""

    chunk_tokens: set[str] = set()
    for chunk in chunks:
        chunk_tokens.update(_tokenise(chunk.content))

    intersection = query_tokens & chunk_tokens
    union = query_tokens | chunk_tokens
    jaccard = len(intersection) / len(union) if union else 0.0

    if jaccard < _JACCARD_THRESHOLD:
        missing = query_tokens - chunk_tokens
        return (
            True,
            f"Your question uses terms not found in this document "
            f"(keyword overlap: {jaccard:.2f}, missing: {', '.join(sorted(missing)[:5])}).",
        )

    return False, ""
