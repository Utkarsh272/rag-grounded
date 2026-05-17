# api/app/routes/search.py
from fastapi import APIRouter, Query
from app.retrieval.vector import vector_search

router = APIRouter()

@router.get("/v1/search")
def search(
    q: str = Query(..., description="The question or query"),
    document_id: str = Query(None, description="Optional: limit to one document"),
    top_k: int = Query(5, ge=1, le=20),
):
    """Search chunks by semantic similarity. Use this to verify retrieval before building generation."""
    results = vector_search(query=q, document_id=document_id, top_k=top_k)
    return [
        {
            "rank": i + 1,
            "similarity": round(r.similarity, 4),
            "section": r.section_title,
            "content": r.content,
            "chunk_id": r.chunk_id,
            "start_char": r.start_char,
            "end_char": r.end_char,
        }
        for i, r in enumerate(results)
    ]