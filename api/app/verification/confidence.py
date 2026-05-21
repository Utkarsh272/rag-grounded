# api/app/verification/confidence.py
"""
Per-claim confidence scoring.

After the LLM generates an answer, this module:
1. Splits the answer into atomic claims (one factual sentence each).
2. For each claim, embeds it and computes cosine similarity against every
   chunk that was cited in support of that claim.
3. Returns a confidence score per claim: max cosine similarity across cited chunks.
4. Flags any claim whose score falls below CLAIM_THRESHOLD as low-confidence.

Why max similarity instead of mean?
   A claim needs at least one supporting chunk to be valid. If any cited chunk
   strongly supports the claim (high similarity), the claim is well-grounded.
   Mean would penalise claims that cite multiple chunks for completeness even
   though one chunk fully supports them.

Why cosine similarity of embeddings instead of NLI?
   NLI (natural language inference) models give true entailment scores, which
   are more accurate. But they require another model download and add 200–500ms
   per claim. Cosine similarity with the same embedder (already loaded) is fast
   and good enough for a portfolio-grade confidence signal. The DESIGN.md will
   note this as a known trade-off.

Claim splitting approach:
   We use a simple LLM call with a structured prompt rather than sentence
   tokenisation (spaCy/NLTK). Reasons:
   - Claim splitting isn't the same as sentence splitting: "The sky is blue.
     It is also vast." is two sentences but one claim about the sky.
   - LLM splitting handles complex nested sentences correctly.
   - We reuse the same LLM client already initialised, so no new dependency.
   The call uses max_tokens=256 and a strict "one claim per line" format,
   making it fast and cheap (~0.01s on Groq).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from app.generation.llm import generate
from app.ingestion.embedder import embed_texts

# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

CLAIM_THRESHOLD = float(os.environ.get("CLAIM_CONFIDENCE_THRESHOLD", "0.50"))


@dataclass
class ClaimScore:
    claim: str
    score: float          # 0.0–1.0 cosine similarity against best cited chunk
    low_confidence: bool  # True if score < CLAIM_THRESHOLD
    cited_chunk_ids: list[str]  # chunk IDs cited inline in this claim


# ---------------------------------------------------------------------------
# Claim splitting
# ---------------------------------------------------------------------------

_SPLIT_SYSTEM = (
    "You are a precise text analyser. "
    "Split the following answer into atomic factual claims, one per line. "
    "Each line must be a single standalone factual statement. "
    "Do not include citations like [1] or [SOURCE_1] in the output. "
    "Do not add numbering, bullets, or any other formatting. "
    "Output ONLY the claims, one per line, nothing else."
)


def _split_into_claims(answer: str) -> list[str]:
    """
    Use the LLM to split an answer into atomic factual claims.
    Returns a list of claim strings, stripped of citation markers.
    """
    # Strip citation markers before sending to splitter
    clean = re.sub(r"\[\d+\]", "", answer).strip()

    if not clean:
        return []

    try:
        raw = generate(system=_SPLIT_SYSTEM, user=clean)
        claims = [line.strip() for line in raw.splitlines() if line.strip()]
        return claims
    except Exception as exc:
        print(f"[confidence] WARNING: claim splitting failed — {exc}")
        # Fallback: split by sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        return [s.strip() for s in sentences if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Cosine similarity (pure Python, no extra deps)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Citation-to-chunk mapping
# ---------------------------------------------------------------------------

def _citation_number_pattern() -> re.Pattern:
    return re.compile(r"\[(\d+)\]")


def _find_cited_numbers(text: str) -> list[int]:
    """Extract [1], [2] citation numbers from a text snippet."""
    return [int(m) for m in _citation_number_pattern().findall(text)]


def _map_citations_to_chunks(
    answer: str,
    claim: str,
    citations: list,  # list of Citation objects from citation_parser
) -> list[str]:
    """
    For a given claim string, find which citation numbers appear near it
    in the full answer, then return the corresponding chunk IDs.

    Strategy: find the claim text in the answer, extract [N] markers in a
    ±200 char window around it, map to chunk_ids via the citations list.
    """
    # Build citation_id → chunk_id map
    id_to_chunk: dict[int, str] = {c.id: c.chunk_id for c in citations}

    # Find claim position in the answer (stripped of citations for matching)
    clean_answer = re.sub(r"\[\d+\]", "", answer)
    clean_claim = re.sub(r"\[\d+\]", "", claim)

    pos = clean_answer.find(clean_claim[:40])  # use first 40 chars for matching
    if pos == -1:
        # Can't locate the claim — use all cited chunks as a fallback
        return [c.chunk_id for c in citations]

    # Extract window around the claim in the original answer
    window_start = max(0, pos - 50)
    window_end = min(len(answer), pos + len(clean_claim) + 100)
    window = answer[window_start:window_end]

    cited_numbers = _find_cited_numbers(window)
    chunk_ids = [id_to_chunk[n] for n in cited_numbers if n in id_to_chunk]

    # If no citations found in window, fall back to all citations
    return chunk_ids if chunk_ids else [c.chunk_id for c in citations]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_claims(
    answer: str,
    citations: list,  # list of Citation from citation_parser
    chunks: list,     # list of HybridResult / SearchResult passed to LLM
) -> list[ClaimScore]:
    """
    Split answer into claims and score each one against its cited chunks.

    Args:
        answer:    Clean answer text with [1], [2] citation markers.
        citations: Citation objects from parse_citations().
        chunks:    The retrieved chunks that were passed to the LLM.

    Returns:
        List of ClaimScore, one per atomic claim in the answer.
        Empty list if answer is very short or claim splitting fails.
    """
    if not answer or not chunks:
        return []

    # Build chunk_id → embedding lookup (embed all chunks at once — one batch)
    chunk_contents = [c.content for c in chunks]
    chunk_id_list = [c.chunk_id for c in chunks]

    try:
        chunk_embeddings = embed_texts(chunk_contents)
    except Exception as exc:
        print(f"[confidence] WARNING: could not embed chunks — {exc}")
        return []

    chunk_emb_map: dict[str, list[float]] = dict(zip(chunk_id_list, chunk_embeddings))

    # Split answer into claims
    claims = _split_into_claims(answer)
    if not claims:
        return []

    # Embed all claims in one batch
    try:
        claim_embeddings = embed_texts(claims)
    except Exception as exc:
        print(f"[confidence] WARNING: could not embed claims — {exc}")
        return []

    # Score each claim
    results: list[ClaimScore] = []

    for claim_text, claim_emb in zip(claims, claim_embeddings):
        cited_chunk_ids = _map_citations_to_chunks(answer, claim_text, citations)

        if not cited_chunk_ids:
            # Claim has no detectable citation — score against all chunks
            cited_chunk_ids = chunk_id_list

        # Max similarity across all cited chunks
        similarities = []
        for cid in cited_chunk_ids:
            if cid in chunk_emb_map:
                sim = _cosine(claim_emb, chunk_emb_map[cid])
                similarities.append(sim)

        score = max(similarities) if similarities else 0.0

        results.append(
            ClaimScore(
                claim=claim_text,
                score=round(score, 4),
                low_confidence=score < CLAIM_THRESHOLD,
                cited_chunk_ids=cited_chunk_ids,
            )
        )

    return results
