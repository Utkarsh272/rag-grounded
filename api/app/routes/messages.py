# api/app/routes/messages.py
"""
Message endpoint with Server-Sent Events streaming.

POST /v1/conversations/{id}/messages
  Body: {"question": "..."}

Pipeline (Day 5 update):
  1. Save user message
  2. Hybrid retrieve top-5 chunks
  3. Pre-LLM abstention check (verification/abstention.py)
     → if weak retrieval: stream abstain event, save message, done
  4. Build prompt + call LLM
  5. Parse [SOURCE_X] citations (generation/citation_parser.py)
     → if LLM returned INSUFFICIENT_INFO: abstain path
  6. Score claims for confidence (verification/confidence.py)
  7. Stream answer word-by-word (token events)
  8. Stream citation metadata (citation events)
  9. Save assistant message with citations + claim_scores + abstained flag
  10. Stream complete event

SSE stream format:
  event: token      data: {"text": "..."}
  event: citation   data: {"id", "chunk_id", "section", "start_char", "end_char", "text"}
  event: complete   data: {"message_id", "answer", "citations", "claim_scores",
                           "abstained", "retrieval_meta"}
  event: error      data: {"detail": "..."}
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
from app.verification.abstention import should_abstain
from app.verification.confidence import score_claims

router = APIRouter()


class AskRequest(BaseModel):
    question: str


def _sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/v1/conversations/{conversation_id}/messages")
async def ask(conversation_id: str, body: AskRequest):
    """Ask a question in a conversation. Returns an SSE stream."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be blank.")

    sb = get_supabase()

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
                yield _sse("error", {"detail": "No relevant content found in this document."})
                return

            retrieval_meta = {
                "chunk_count": len(chunks),
                "top3_avg_rerank_score": round(
                    sum(sorted([c.rerank_score for c in chunks], reverse=True)[:3])
                    / min(3, len(chunks)),
                    4,
                ),
            }

            # ── 3. Pre-LLM abstention check ───────────────────────────────────
            abstain, reason = should_abstain(chunks=chunks, query=body.question)

            if abstain:
                abstain_answer = (
                    "I don't have enough information in this document to answer "
                    f"your question confidently. {reason}"
                )
                saved = sb.table("messages").insert({
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": abstain_answer,
                    "citations": [],
                    "claim_scores": [],
                    "abstained": True,
                    "retrieval_meta": retrieval_meta,
                }).execute()

                message_id = saved.data[0]["id"] if saved.data else None

                # Stream the abstain answer word-by-word so the UI looks consistent
                for word in abstain_answer.split(" "):
                    yield _sse("token", {"text": word + " "})
                    time.sleep(0.02)

                yield _sse("complete", {
                    "message_id": message_id,
                    "answer": abstain_answer,
                    "citations": [],
                    "claim_scores": [],
                    "abstained": True,
                    "retrieval_meta": retrieval_meta,
                })
                return

            # ── 4. Build prompt + call LLM ────────────────────────────────────
            system_prompt = build_system_prompt(chunks)
            user_prompt = build_user_prompt(body.question)
            raw_output = generate(system=system_prompt, user=user_prompt)

            # ── 5. Parse citations (also handles post-LLM INSUFFICIENT_INFO) ──
            parsed = parse_citations(raw_output, chunks)

            # ── 6. Score claims for confidence ────────────────────────────────
            # Skip scoring if the LLM abstained — there are no claims to score.
            claim_scores = []
            if not parsed.abstained:
                scored = score_claims(
                    answer=parsed.answer,
                    citations=parsed.citations,
                    chunks=chunks,
                )
                claim_scores = [
                    {
                        "claim": cs.claim,
                        "score": cs.score,
                        "low_confidence": cs.low_confidence,
                    }
                    for cs in scored
                ]

            # ── 7. Stream answer word-by-word ─────────────────────────────────
            for word in parsed.answer.split(" "):
                yield _sse("token", {"text": word + " "})
                time.sleep(0.02)

            # ── 8. Stream citation metadata ───────────────────────────────────
            for citation in parsed.citations:
                yield _sse("citation", {
                    "id": citation.id,
                    "chunk_id": citation.chunk_id,
                    "section": citation.section_title,
                    "start_char": citation.start_char,
                    "end_char": citation.end_char,
                    "text": citation.text,
                })

            # ── 9. Save assistant message ─────────────────────────────────────
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
                "claim_scores": claim_scores,
                "abstained": parsed.abstained,
                "retrieval_meta": retrieval_meta,
            }).execute()

            message_id = saved.data[0]["id"] if saved.data else None

            # ── 10. Send complete event ───────────────────────────────────────
            yield _sse("complete", {
                "message_id": message_id,
                "answer": parsed.answer,
                "citations": citations_json,
                "claim_scores": claim_scores,
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
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
