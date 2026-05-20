# api/app/routes/messages.py
"""
Message endpoint with Server-Sent Events streaming.

POST /v1/conversations/{id}/messages
  Body: {"question": "..."}

SSE stream format:
  event: token
  data: {"text": "..."}        <- one or more word chunks while generating

  event: citation
  data: {"id": 1, "chunk_id": "...", "section": "...", "start_char": 0, "end_char": 100}

  event: complete
  data: {
    "message_id": "...",
    "answer": "Full answer with [1] markers",
    "citations": [...],
    "abstained": false,
    "retrieval_meta": {"top5_avg_similarity": 0.72, "chunk_count": 5}
  }

  event: error
  data: {"detail": "..."}

Why SSE over WebSocket?
  SSE is unidirectional (server → client), which is all we need for streaming
  answers. It's simpler than WebSocket: no upgrade handshake, works over HTTP/1.1,
  and FastAPI + httpx handle it cleanly via StreamingResponse.

Why not stream token-by-token from the LLM?
  Groq and Anthropic both support streaming, but integrating streaming LLM output
  with citation parsing (which needs the full response to resolve [SOURCE_X] tokens)
  is complex. Day 4 uses a simpler two-phase approach:
    1. Call LLM (blocking, <3s on Groq).
    2. Parse citations from the full response.
    3. Stream the answer word-by-word over SSE (simulates token streaming for the UI).
  True token-by-token LLM streaming is a stretch goal after Day 6.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db.client import get_supabase
from app.generation.citation_parser import parse_citations
from app.generation.llm import generate
from app.generation.prompt import build_system_prompt, build_user_prompt
from app.retrieval.hybrid import hybrid_retrieve

router = APIRouter()


class AskRequest(BaseModel):
    question: str


def _sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/v1/conversations/{conversation_id}/messages")
async def ask(conversation_id: str, body: AskRequest):
    """
    Ask a question in a conversation. Returns an SSE stream.
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be blank.")

    sb = get_supabase()

    # Verify conversation exists and get document_id
    conv = (
        sb.table("conversations")
        .select("id, document_id")
        .eq("id", conversation_id)
        .execute()
    )
    if not conv.data:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    document_id = conv.data[0]["document_id"]

    async def stream():
        try:
            # ── 1. Save user message ─────────────────────────────────────────
            sb.table("messages").insert({
                "conversation_id": conversation_id,
                "role": "user",
                "content": body.question,
            }).execute()

            # ── 2. Retrieve relevant chunks ──────────────────────────────────
            chunks = hybrid_retrieve(
                query=body.question,
                top_k=5,
                document_id=document_id,
            )

            if not chunks:
                # No chunks at all — document may be empty or not yet indexed
                yield _sse("error", {"detail": "No relevant content found in this document."})
                return

            # Compute retrieval metadata for the complete event
            avg_similarity = None
            if hasattr(chunks[0], "rerank_score"):
                # HybridResult — use rerank scores as a proxy for relevance
                avg_similarity = round(
                    sum(c.rerank_score for c in chunks) / len(chunks), 4
                )

            retrieval_meta = {
                "chunk_count": len(chunks),
                "top5_avg_rerank_score": avg_similarity,
            }

            # ── 3. Build prompt and call LLM ─────────────────────────────────
            system_prompt = build_system_prompt(chunks)
            user_prompt = build_user_prompt(body.question)

            # This is blocking but fast (<3s on Groq free tier).
            raw_output = generate(system=system_prompt, user=user_prompt)

            # ── 4. Parse citations ────────────────────────────────────────────
            parsed = parse_citations(raw_output, chunks)

            # ── 5. Stream answer word-by-word ─────────────────────────────────
            # Split on spaces to simulate token streaming. The frontend renders
            # words as they arrive, giving a live "typing" effect.
            words = parsed.answer.split(" ")
            for word in words:
                yield _sse("token", {"text": word + " "})
                time.sleep(0.02)  # 20ms between words ≈ ~50 words/sec

            # ── 6. Stream citation metadata ───────────────────────────────────
            for citation in parsed.citations:
                yield _sse("citation", {
                    "id": citation.id,
                    "chunk_id": citation.chunk_id,
                    "section": citation.section_title,
                    "start_char": citation.start_char,
                    "end_char": citation.end_char,
                    "text": citation.text,
                })

            # ── 7. Save assistant message to DB ──────────────────────────────
            citations_json = [
                {
                    "id": c.id,
                    "chunk_id": c.chunk_id,
                    "section_title": c.section_title,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "text": c.text,
                }
                for c in parsed.citations
            ]

            saved = sb.table("messages").insert({
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": parsed.answer,
                "citations": citations_json,
                "abstained": parsed.abstained,
                "retrieval_meta": retrieval_meta,
            }).execute()

            message_id = saved.data[0]["id"] if saved.data else None

            # ── 8. Send complete event ────────────────────────────────────────
            yield _sse("complete", {
                "message_id": message_id,
                "answer": parsed.answer,
                "citations": citations_json,
                "abstained": parsed.abstained,
                "retrieval_meta": retrieval_meta,
            })

        except Exception as exc:
            import traceback
            traceback.print_exc()
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            # Disable buffering so tokens reach the client immediately
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
