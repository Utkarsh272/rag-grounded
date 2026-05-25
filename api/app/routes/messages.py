# api/app/routes/messages.py
"""
Message endpoint with Server-Sent Events streaming.

POST /v1/conversations/{id}/messages
  Body: {"question": "..."}

Pipeline:
  1. Save user message
  2. Hybrid retrieve top-5 chunks       ← timed: time_retrieval("hybrid")
  3. Pre-LLM abstention check           ← records: record_abstention(reason)
  4. Build prompt + call LLM            ← timed: time_llm(provider, "answer")
  5. Parse [SOURCE_X] citations
  6. Score claims for confidence        ← timed: time_llm(provider, "claims") + time_embedding("claims")
  7. Stream answer word-by-word
  8. Stream citation metadata
  9. Save assistant message
  10. Stream complete event             ← records: record_request(status, elapsed)
"""

from __future__ import annotations

import json
import os
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.db.client import get_supabase
from app.generation.citation_parser import parse_citations
from app.generation.llm import generate
from app.generation.prompt import build_system_prompt, build_user_prompt
from app.retrieval.hybrid import hybrid_retrieve
from app.telemetry.metrics import (
    record_abstention,
    record_request,
    time_embedding,
    time_llm,
    time_retrieval,
)
from app.verification.abstention import should_abstain
from app.verification.confidence import score_claims

router = APIRouter()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")


class AskRequest(BaseModel):
    question: str


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/v1/conversations/{conversation_id}/messages")
async def ask(conversation_id: str, body: AskRequest):
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
        request_start = time.perf_counter()
        final_status = "success"

        try:
            # ── 1. Save user message ──────────────────────────────────────────
            sb.table("messages").insert({
                "conversation_id": conversation_id,
                "role": "user",
                "content": body.question,
            }).execute()

            # ── 2. Retrieve relevant chunks ───────────────────────────────────
            with time_retrieval("hybrid"):
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
                record_abstention("rerank_threshold")
                final_status = "abstained"

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

            with time_llm(LLM_PROVIDER, "answer"):
                raw_output = generate(system=system_prompt, user=user_prompt)

            # ── 5. Parse citations ────────────────────────────────────────────
            parsed = parse_citations(raw_output, chunks)

            if parsed.abstained:
                record_abstention("insufficient_info")
                final_status = "abstained"

            # ── 6. Score claims ───────────────────────────────────────────────
            claim_scores = []
            if not parsed.abstained:
                with time_embedding("claims"):
                    with time_llm(LLM_PROVIDER, "claims"):
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

            # ── 10. Stream complete event ─────────────────────────────────────
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
            final_status = "error"
            yield _sse("error", {"detail": str(exc)})

        finally:
            record_request(final_status, time.perf_counter() - request_start)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
