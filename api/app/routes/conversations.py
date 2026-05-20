# api/app/routes/conversations.py
"""
Conversation management endpoints.

POST /v1/conversations          Create a new conversation tied to a document.
GET  /v1/conversations          List all conversations.
GET  /v1/conversations/{id}     Get a conversation with its messages.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.client import get_supabase

router = APIRouter()


class CreateConversationRequest(BaseModel):
    document_id: str
    title: str | None = None


@router.post("/v1/conversations")
def create_conversation(body: CreateConversationRequest):
    sb = get_supabase()

    # Verify the document exists and is fully processed
    doc = (
        sb.table("documents")
        .select("id, title, status")
        .eq("id", body.document_id)
        .execute()
    )
    if not doc.data:
        raise HTTPException(status_code=404, detail="Document not found")

    doc_row = doc.data[0]
    if doc_row["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Document is not ready (status: {doc_row['status']}). Wait for ingestion to complete.",
        )

    title = body.title or f"Chat about {doc_row['title']}"

    result = (
        sb.table("conversations")
        .insert({"document_id": body.document_id, "title": title})
        .execute()
    )
    return result.data[0]


@router.get("/v1/conversations")
def list_conversations():
    sb = get_supabase()
    result = (
        sb.table("conversations")
        .select("*, documents(title)")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


@router.get("/v1/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    sb = get_supabase()

    conv = (
        sb.table("conversations")
        .select("*, documents(title)")
        .eq("id", conversation_id)
        .execute()
    )
    if not conv.data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        sb.table("messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at")
        .execute()
    )

    return {**conv.data[0], "messages": messages.data}
