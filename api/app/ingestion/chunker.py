# api/app/ingestion/chunker.py
from dataclasses import dataclass
from typing import List
import re

MAX_CHUNK = 1000   # max characters per chunk
MIN_CHUNK = 20     # drop chunks shorter than this (stray captions etc)
OVERLAP   = 100    # characters of overlap between consecutive chunks

@dataclass
class Chunk:
    content: str
    start_char: int   # offset in the ORIGINAL full text
    end_char: int
    section_title: str
    chunk_index: int


def _split_into_sections(text: str) -> List[dict]:
    """
    Split raw text into sections based on markdown headings.
    Returns list of {title, content, start_char} dicts.
    Works on both markdown (# Heading) and plain text (double newline blocks).
    """
    sections = []
    # Match markdown headings: lines starting with 1-3 # chars
    heading_pattern = re.compile(r'^(#{1,3})\s+(.+)', re.MULTILINE)

    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No markdown headings — treat entire text as one section
        return [{"title": "Document", "content": text, "start_char": 0}]

    # Text before the first heading
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if preamble:
            sections.append({
                "title": "Introduction",
                "content": preamble,
                "start_char": 0,
            })

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        # Content runs from end of this heading line to start of next heading (or EOF)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        sections.append({
            "title": title,
            "content": content,
            # start_char points to where this section's content begins in original text
            "start_char": text.index(content, content_start) if content else content_start,
        })

    return sections


def _split_section(content: str, section_start: int, section_title: str,
                   start_index: int) -> List[Chunk]:
    """
    Split a single section into MAX_CHUNK-sized chunks with OVERLAP.
    Tries to split on paragraph boundaries first, then sentences.
    """
    chunks = []
    chunk_index = start_index

    if len(content) <= MAX_CHUNK:
        if len(content) >= MIN_CHUNK:
            chunks.append(Chunk(
                content=content,
                start_char=section_start,
                end_char=section_start + len(content),
                section_title=section_title,
                chunk_index=chunk_index,
            ))
        return chunks

    # Split by paragraphs (double newline)
    paragraphs = re.split(r'\n\n+', content)
    current_chunk = ""
    current_start = section_start

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= MAX_CHUNK:
            # Fits in current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
                # Find where this paragraph starts in original text
                idx = content.find(para)
                if idx != -1:
                    current_start = section_start + idx
        else:
            # Flush current chunk
            if current_chunk and len(current_chunk) >= MIN_CHUNK:
                chunks.append(Chunk(
                    content=current_chunk,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    section_title=section_title,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

            # Start new chunk with OVERLAP from end of previous
            overlap_text = current_chunk[-OVERLAP:] if current_chunk else ""
            current_chunk = (overlap_text + "\n\n" + para).strip() if overlap_text else para
            idx = content.find(para)
            current_start = section_start + idx if idx != -1 else current_start

    # Flush remaining
    if current_chunk and len(current_chunk) >= MIN_CHUNK:
        chunks.append(Chunk(
            content=current_chunk,
            start_char=current_start,
            end_char=current_start + len(current_chunk),
            section_title=section_title,
            chunk_index=chunk_index,
        ))

    return chunks


def chunk_text(text: str) -> List[Chunk]:
    """
    Main entry point. Splits text into structure-aware chunks.
    Each chunk tracks its exact character offsets in the original text.
    """
    sections = _split_into_sections(text)
    all_chunks = []
    chunk_index = 0

    for section in sections:
        new_chunks = _split_section(
            content=section["content"],
            section_start=section["start_char"],
            section_title=section["title"],
            start_index=chunk_index,
        )
        all_chunks.extend(new_chunks)
        chunk_index += len(new_chunks)

    return all_chunks