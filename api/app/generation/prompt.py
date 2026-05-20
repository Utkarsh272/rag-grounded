# api/app/generation/prompt.py
"""
Prompt templates for citation-grounded answer generation.

Design:
- The system prompt instructs the LLM to emit [SOURCE_X] tokens inline after
  every claim it makes. These tokens are parsed out by citation_parser.py.
- Sources are numbered SOURCE_1 through SOURCE_N, matching the order of chunks
  passed in. The LLM never sees chunk UUIDs — only human-readable SOURCE numbers.
- If no source supports the answer, the LLM must respond with the exact sentinel
  string INSUFFICIENT_INFO. This is the abstention trigger checked in Day 5.

The prompt is deliberately strict about output format because citation_parser.py
uses a simple regex — no fuzzy matching.
"""

from __future__ import annotations

from app.retrieval.hybrid import HybridResult
from app.retrieval.vector import SearchResult


# Accepts either HybridResult (Day 3+) or SearchResult (vector-only fallback)
ChunkLike = HybridResult | SearchResult


def build_system_prompt(chunks: list[ChunkLike]) -> str:
    """
    Build the system prompt with source chunks embedded.

    Each chunk becomes a numbered SOURCE block. The LLM is instructed to:
    1. Only make claims supported by these sources.
    2. Cite inline with [SOURCE_X] immediately after the supported claim.
    3. Return INSUFFICIENT_INFO if sources don't answer the question.
    """
    source_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        section = getattr(chunk, "section_title", "") or "Unknown section"
        source_blocks.append(
            f"[SOURCE_{i}] (section: \"{section}\"):\n{chunk.content}"
        )

    sources_text = "\n\n".join(source_blocks)

    return f"""You are a research assistant that answers questions strictly from provided source documents.

RULES:
1. Answer using ONLY the information in the sources below. Do not use outside knowledge.
2. After every factual claim you make, immediately cite the source with [SOURCE_X] — where X is the source number. Place the citation directly after the claim, before any punctuation.
3. Every sentence in your answer that makes a factual claim MUST have at least one citation.
4. If multiple sources support a claim, cite all of them: [SOURCE_1][SOURCE_2].
5. If the sources do not contain enough information to answer the question, respond with EXACTLY this string and nothing else:
   INSUFFICIENT_INFO
6. Do not say "according to the sources" or "based on SOURCE_1" — embed citations inline only.
7. Write in clear, concise prose. Do not use bullet points unless the question specifically asks for a list.

SOURCES:
{sources_text}"""


def build_user_prompt(question: str) -> str:
    """The user turn — just the question, cleanly formatted."""
    return f"Question: {question}"
