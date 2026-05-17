"""
BM25 keyword search using Postgres tsvector / tsrank.
The ts_vector column and chunks_ts_idx GIN index were created on Day 1,
so no schema changes are needed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.db.client import get_supabase


@dataclass
class BM25Result:
    chunk_id: str
    document_id: str
    content: str
    section_title: str | None
    start_char: int
    end_char: int
    rank: float  # ts_rank score (higher = more relevant)


def bm25_search(
    query: str,
    top_k: int = 20,
    document_id: str | None = None,
) -> list[BM25Result]:
    """
    Full-text search using Postgres tsvector + plainto_tsquery.

    plainto_tsquery is used instead of to_tsquery because it handles
    arbitrary user input without requiring the user to write tsquery syntax
    (e.g., 'neural & networks'). It tokenises and ANDs terms automatically.

    ts_rank_cd (cover density ranking) weights results by how close matched
    terms are to each other — better than ts_rank for short chunks.

    Returns up to top_k results ordered by descending ts_rank score.
    """
    if not query.strip():
        return []

    supabase = get_supabase()

    # Build the SQL manually via rpc so we get ts_rank in the response.
    # Supabase's Python client doesn't support full-text filter expressions
    # (textSearch) combined with computed columns, so we use a raw RPC call
    # backed by a Postgres function (defined below in the docstring).
    #
    # IMPORTANT: the `bm25_search` Postgres function must be created in Supabase
    # before this code is called. See the SQL block in the module docstring below.

    params: dict = {
        "query_text": query,
        "match_count": top_k,
    }
    if document_id:
        params["filter_document_id"] = document_id

    try:
        response = supabase.rpc("bm25_search", params).execute()
    except Exception as exc:
        # Graceful degradation: if BM25 fails (e.g. function not yet created),
        # log and return empty so hybrid retrieval still works with vector-only.
        print(f"[bm25] WARNING: bm25_search RPC failed — {exc}")
        return []

    if not response.data:
        return []

    results: list[BM25Result] = []
    for row in response.data:
        results.append(
            BM25Result(
                chunk_id=row["id"],
                document_id=row["document_id"],
                content=row["content"],
                section_title=row.get("section_title"),
                start_char=row["start_char"],
                end_char=row["end_char"],
                rank=float(row["rank"]),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Postgres function to create in Supabase SQL editor (run once):
# ---------------------------------------------------------------------------
#
# CREATE OR REPLACE FUNCTION bm25_search(
#   query_text         text,
#   match_count        int,
#   filter_document_id uuid DEFAULT NULL
# )
# RETURNS TABLE (
#   id            uuid,
#   document_id   uuid,
#   content       text,
#   section_title text,
#   start_char    int,
#   end_char      int,
#   rank          float
# )
# LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
# AS $$
#   SELECT
#     c.id,
#     c.document_id,
#     c.content,
#     c.section_title,
#     c.start_char,
#     c.end_char,
#     ts_rank_cd(c.ts_vector, plainto_tsquery('english', query_text)) AS rank
#   FROM chunks c
#   WHERE
#     c.ts_vector @@ plainto_tsquery('english', query_text)
#     AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
#   ORDER BY rank DESC
#   LIMIT match_count;
# $$;