# api/app/generation/citation_parser.py
"""
Parse [SOURCE_X] tokens from LLM output and map them to chunk metadata.

Input:  Raw LLM text like:
        "Backpropagation adjusts weights iteratively [SOURCE_1].
         This minimises the loss function [SOURCE_1][SOURCE_2]."

Output: Clean answer text with [1], [2] markers, plus a citations list
        mapping each number to the source chunk's ID and span offsets.

Design decisions:
- We replace [SOURCE_X] with [X] in the answer so the frontend can render
  clickable superscript numbers without knowing about SOURCE numbering.
- Deduplication: if SOURCE_1 is cited multiple times, it gets one entry in
  citations with id=1. The frontend uses the id to look up the full chunk.
- We validate that cited source numbers are within the range of provided
  chunks, discarding hallucinated citation numbers silently (and logging them).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Matches [SOURCE_1], [SOURCE_12], etc.
SOURCE_PATTERN = re.compile(r"\[SOURCE_(\d+)\]")


@dataclass
class Citation:
    id: int            # 1-based citation number shown in the answer text ([1], [2])
    chunk_id: str      # UUID of the source chunk in the DB
    source_number: int # The SOURCE_X number from the LLM (same as id in our impl)
    section_title: str | None
    start_char: int
    end_char: int
    text: str          # first 300 chars of chunk content — for the source panel preview


@dataclass
class ParsedAnswer:
    answer: str              # cleaned answer with [1], [2] markers
    citations: list[Citation]
    abstained: bool          # True if LLM returned INSUFFICIENT_INFO
    raw_llm_output: str      # original unmodified LLM text, for debugging


def parse_citations(raw_output: str, chunks: list) -> ParsedAnswer:
    """
    Parse LLM output, extract citations, return clean answer + citation list.

    Args:
        raw_output: Raw text from the LLM, may contain [SOURCE_X] tokens.
        chunks:     Ordered list of chunks (HybridResult or SearchResult) that
                    were passed to the LLM. Index 0 = SOURCE_1, etc.

    Returns:
        ParsedAnswer with clean answer text and structured citations list.
    """
    # Check for abstention sentinel first
    if raw_output.strip() == "INSUFFICIENT_INFO":
        return ParsedAnswer(
            answer="I don't have enough information in the provided documents to answer this question confidently.",
            citations=[],
            abstained=True,
            raw_llm_output=raw_output,
        )

    # Find all unique SOURCE numbers cited in the output
    cited_source_numbers = sorted(
        set(int(m) for m in SOURCE_PATTERN.findall(raw_output))
    )

    # Build citation objects, validating each source number
    citation_map: dict[int, Citation] = {}  # source_number → Citation
    next_citation_id = 1

    for source_num in cited_source_numbers:
        chunk_index = source_num - 1  # SOURCE_1 → chunks[0]

        if chunk_index < 0 or chunk_index >= len(chunks):
            # LLM hallucinated a source number that doesn't exist — skip silently
            print(
                f"[citation_parser] WARNING: LLM cited SOURCE_{source_num} "
                f"but only {len(chunks)} chunks were provided. Discarding."
            )
            continue

        chunk = chunks[chunk_index]
        citation_map[source_num] = Citation(
            id=next_citation_id,
            chunk_id=chunk.chunk_id,
            source_number=source_num,
            section_title=getattr(chunk, "section_title", None),
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            text=chunk.content[:300],
        )
        next_citation_id += 1

    # Replace [SOURCE_X] tokens in the answer text with [citation_id]
    def replace_token(match: re.Match) -> str:
        source_num = int(match.group(1))
        citation = citation_map.get(source_num)
        if citation is None:
            return ""  # drop hallucinated citations from display text
        return f"[{citation.id}]"

    clean_answer = SOURCE_PATTERN.sub(replace_token, raw_output).strip()

    # Collapse any double spaces left by removed tokens
    clean_answer = re.sub(r"  +", " ", clean_answer)

    return ParsedAnswer(
        answer=clean_answer,
        citations=list(citation_map.values()),
        abstained=False,
        raw_llm_output=raw_output,
    )
