# api/app/retrieval/vector.py
from app.db.client import get_supabase
from app.ingestion.embedder import embed_texts
from dataclasses import dataclass

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
    """Embed query and find top_k most similar chunks via cosine similarity."""
    query_embedding = embed_texts([query])[0]
    sb = get_supabase()

    # Must send as string — Supabase RPC can't auto-cast Python list to pgvector type
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