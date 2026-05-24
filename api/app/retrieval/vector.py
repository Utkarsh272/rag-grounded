# api/app/retrieval/vector.py
from dataclasses import dataclass
from app.db.client import get_supabase
from app.ingestion.embedder import embed_query


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    section_title: str
    start_char: int
    end_char: int
    similarity: float


def vector_search(query: str, document_id: str = None, top_k: int = 20) -> list[SearchResult]:
    """Embed query with retrieval.query task and find top_k similar chunks."""
    query_embedding = embed_query(query)
    sb = get_supabase()

    # Send as string — Supabase RPC can't auto-cast Python list to pgvector type
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    params = {"query_embedding": embedding_str, "match_count": top_k}
    if document_id:
        params["filter_document_id"] = document_id

    result = sb.rpc("match_chunks", params).execute()

    return [
        SearchResult(
            chunk_id=row["id"],
            document_id=row["document_id"],
            content=row["content"],
            section_title=row["section_title"] or "",
            start_char=row["start_char"],
            end_char=row["end_char"],
            similarity=row["similarity"],
        )
        for row in result.data
    ]
