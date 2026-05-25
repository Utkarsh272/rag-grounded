# api/app/routes/documents.py
import os
import time
import traceback
import uuid

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.db.client import get_supabase
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_texts
from app.telemetry.metrics import record_ingestion, time_embedding

router = APIRouter()


async def process_document(document_id: str, raw_text: str) -> None:
    """Background task: chunk → embed (Jina API) → store."""
    sb = get_supabase()
    ingestion_start = time.perf_counter()

    try:
        sb.table("documents").update({"status": "processing"}).eq("id", document_id).execute()

        chunks = chunk_text(raw_text)
        print(f"📄 {len(chunks)} chunks created for document {document_id}")

        if not chunks:
            raise ValueError("Chunker returned 0 chunks — check the input text")

        batch_size = 20
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"🔢 Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")

            with time_embedding("passage"):
                embeddings = embed_texts([c.content for c in batch])

            rows = [
                {
                    "document_id": document_id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "section_title": c.section_title,
                    "embedding": emb,
                }
                for c, emb in zip(batch, embeddings)
            ]
            sb.table("chunks").insert(rows).execute()
            print(f"✅ Batch {i // batch_size + 1} inserted")

        sb.table("documents").update({"status": "complete"}).eq("id", document_id).execute()
        print(f"✅ Document {document_id} fully processed: {len(chunks)} chunks")

        record_ingestion(
            status="success",
            duration_seconds=time.perf_counter() - ingestion_start,
            num_chunks=len(chunks),
        )

    except Exception as e:
        print(f"❌ Document {document_id} failed:")
        traceback.print_exc()
        record_ingestion(
            status="error",
            duration_seconds=time.perf_counter() - ingestion_start,
        )
        try:
            sb.table("documents").update({
                "status": "failed",
                "error_message": str(e),
            }).eq("id", document_id).execute()
        except Exception as db_err:
            print(f"❌ Also failed to update error status in DB: {db_err}")


@router.post("/v1/documents")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".md", ".txt")):
        raise HTTPException(400, "Only PDF, Markdown, and text files are supported")

    contents = await file.read()

    if file.filename.endswith(".pdf"):
        import tempfile
        from unstructured.partition.auto import partition
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        elements = partition(filename=tmp_path)
        raw_text = "\n\n".join([str(e) for e in elements])
        os.unlink(tmp_path)
    else:
        raw_text = contents.decode("utf-8")

    sb = get_supabase()
    document_id = str(uuid.uuid4())
    sb.table("documents").insert({
        "id": document_id,
        "title": file.filename,
        "source_type": file.filename.split(".")[-1],
        "status": "pending",
    }).execute()

    background_tasks.add_task(process_document, document_id, raw_text)
    return {"document_id": document_id, "status": "processing"}


@router.get("/v1/documents")
def list_documents():
    sb = get_supabase()
    result = sb.table("documents").select("*").order("created_at", desc=True).execute()
    return result.data


@router.get("/v1/documents/{document_id}/status")
def get_status(document_id: str):
    sb = get_supabase()
    result = (
        sb.table("documents")
        .select("id, status, error_message")
        .eq("id", document_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(404, "Document not found")
    return result.data[0]
