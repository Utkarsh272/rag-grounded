"""
Evaluation harness for RAG with Grounded Citations.

Measures three metrics across two retrieval modes:
  - recall@5: does the expected section appear in top-5 retrieved chunks?
  - answer_accuracy: LLM-as-judge scores generated answer vs ground truth (0 / 0.5 / 1)
  - citation_precision: fraction of cited chunks whose section matches expected section

Usage:
  cd ~/Documents/rag-grounded/api
  uv run python tests/eval/run_eval.py \\
    --eval-set tests/eval/eval_set.json \\
    --document-id <uuid-of-ingested-eval-doc> \\
    --output tests/eval/results.json

Prerequisites:
  1. Upload distributed_systems_rag_eval.md via the UI (or POST /v1/documents)
  2. Wait for status == "complete"
  3. Copy the document_id from GET /v1/documents and pass as --document-id
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = os.getenv("EVAL_API_URL", "http://localhost:8000")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "llama-3.3-70b-versatile"

RETRIEVAL_MODES = ["vector", "hybrid"]
TOP_K = 5
REQUEST_TIMEOUT = 60.0
SLEEP_BETWEEN_REQUESTS = 1.5  # seconds — avoids Groq rate limits


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def search(query: str, document_id: str, mode: str, top_k: int = TOP_K) -> list[dict]:
    """Call GET /v1/search and return list of chunks."""
    resp = httpx.get(
        f"{BASE_URL}/v1/search",
        params={"q": query, "document_id": document_id, "mode": mode, "top_k": top_k},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    # Route returns a bare list, not {"results": [...]}
    return data if isinstance(data, list) else data.get("results", [])


def ask_llm_judge(question: str, generated: str, ground_truth: str) -> float:
    """
    LLM-as-judge via Groq. Returns 1.0 / 0.5 / 0.0.
    Uses a direct Groq call to avoid polluting the pipeline's conversation history.
    """
    import groq as groq_lib

    client = groq_lib.Groq(api_key=GROQ_API_KEY)
    system = (
        "You are an evaluation judge for a question-answering system. "
        "Score the generated answer compared to the ground truth.\n"
        "Reply with ONLY one of: 1, 0.5, or 0\n"
        "1   = fully correct, covers all key points\n"
        "0.5 = partially correct, captures some points but misses others\n"
        "0   = incorrect, irrelevant, or refuses to answer\n"
        "No explanation. Just the score."
    )
    user = (
        f"QUESTION: {question}\n\n"
        f"GROUND TRUTH: {ground_truth}\n\n"
        f"GENERATED ANSWER: {generated}"
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=16,
        )
        return float(resp.choices[0].message.content.strip())
    except (ValueError, Exception) as e:
        log(f"    Judge error: {e}")
        return 0.0


def create_conversation(document_id: str) -> str:
    resp = httpx.post(
        f"{BASE_URL}/v1/conversations",
        json={"document_id": document_id},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def ask_pipeline(conversation_id: str, question: str) -> dict[str, Any]:
    """
    POST /v1/conversations/{id}/messages, consume SSE stream.
    Returns the payload from the 'complete' event.
    """
    url = f"{BASE_URL}/v1/conversations/{conversation_id}/messages"
    result: dict[str, Any] = {}

    with httpx.stream(
        "POST",
        url,
        json={"question": question},
        timeout=REQUEST_TIMEOUT,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        event_type = ""
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:") and event_type == "complete":
                result = json.loads(line[len("data:"):].strip())
                break
    return result


# ── Metric calculators ────────────────────────────────────────────────────────

def compute_recall_at_k(chunks: list[dict], expected_section: str) -> float:
    """1.0 if any chunk's section matches expected_section, else 0.0."""
    expected_lower = expected_section.lower()
    for chunk in chunks:
        section = (chunk.get("section") or chunk.get("section_title") or "").lower()
        if expected_lower in section or section in expected_lower:
            return 1.0
    return 0.0


def compute_citation_precision(citations: list[dict], expected_section: str) -> float | None:
    """
    Fraction of cited chunks whose section matches expected_section.
    Returns None if no citations (abstained or none generated).
    """
    if not citations:
        return None
    expected_lower = expected_section.lower()
    supporting = sum(
        1 for c in citations
        if expected_lower in (c.get("section_title") or c.get("section") or "").lower()
    )
    return supporting / len(citations)


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(eval_set_path: str, document_id: str, output_path: str) -> None:
    with open(eval_set_path) as f:
        eval_set = json.load(f)

    questions = eval_set["questions"]
    log(f"Loaded {len(questions)} questions from {eval_set_path}")
    log(f"Document ID : {document_id}")
    log(f"API base URL: {BASE_URL}")
    log("")

    results: dict[str, Any] = {
        "document_id": document_id,
        "eval_set": eval_set["document_title"],
        "run_at": datetime.now().isoformat(),
        "modes": {},
    }

    # ── Phase 1: Recall@5 per retrieval mode ──────────────────────────────────
    log("=" * 60)
    log("PHASE 1: Recall@5 (retrieval only, no LLM)")
    log("=" * 60)

    recall_results: dict[str, list[float]] = {m: [] for m in RETRIEVAL_MODES}

    for mode in RETRIEVAL_MODES:
        log(f"\nMode: {mode}")
        for q in questions:
            try:
                chunks = search(q["question"], document_id, mode, TOP_K)
                recall = compute_recall_at_k(chunks, q["expected_section"])
                recall_results[mode].append(recall)
                mark = "✓" if recall == 1.0 else "✗"
                log(f"  {mark} [{q['id']}] {q['question'][:65]}")
            except Exception as e:
                log(f"  ERR [{q['id']}]: {e}")
                recall_results[mode].append(0.0)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        avg = sum(recall_results[mode]) / len(recall_results[mode])
        log(f"\n  → Recall@{TOP_K} ({mode}): {avg:.3f}")
        results["modes"][mode] = {"recall_at_5": round(avg, 4)}

    # ── Phase 2: Full pipeline — accuracy + citation precision ────────────────
    log("")
    log("=" * 60)
    log("PHASE 2: Answer accuracy + Citation precision (hybrid pipeline)")
    log("=" * 60)

    accuracy_scores: list[float] = []
    cite_precision_scores: list[float] = []
    pipeline_details: list[dict] = []

    for q in questions:
        log(f"\n[{q['id']}] {q['question'][:70]}")
        try:
            conv_id = create_conversation(document_id)
            pr = ask_pipeline(conv_id, q["question"])

            generated = pr.get("answer", "")
            citations = pr.get("citations", [])
            abstained = pr.get("abstained", False)

            # Answer accuracy via LLM judge
            if abstained or not generated.strip():
                accuracy = 0.0
            else:
                accuracy = ask_llm_judge(q["question"], generated, q["expected_answer"])
            accuracy_scores.append(accuracy)

            # Citation precision
            cp = compute_citation_precision(citations, q["expected_section"])
            if cp is not None:
                cite_precision_scores.append(cp)

            log(f"  abstained       : {abstained}")
            log(f"  answer_accuracy : {accuracy}")
            log(f"  cite_precision  : {cp}")
            log(f"  generated       : {generated[:120]}...")

            pipeline_details.append({
                "id": q["id"],
                "question": q["question"],
                "expected_section": q["expected_section"],
                "difficulty": q["difficulty"],
                "abstained": abstained,
                "accuracy": accuracy,
                "citation_precision": cp,
                "num_citations": len(citations),
                "generated_preview": generated[:250],
            })

        except Exception as e:
            log(f"  ERROR: {e}")
            accuracy_scores.append(0.0)
            pipeline_details.append({"id": q["id"], "error": str(e)})

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # ── Aggregate + print summary ─────────────────────────────────────────────
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
    avg_cite_prec = (
        sum(cite_precision_scores) / len(cite_precision_scores)
        if cite_precision_scores else 0.0
    )
    abstention_rate = sum(1 for d in pipeline_details if d.get("abstained")) / len(pipeline_details)

    results["modes"]["hybrid"]["answer_accuracy"] = round(avg_accuracy, 4)
    results["modes"]["hybrid"]["citation_precision"] = round(avg_cite_prec, 4)
    results["abstention_rate"] = round(abstention_rate, 4)
    results["pipeline_details"] = pipeline_details

    log("")
    log("=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"{'Metric':<35} {'vector':>8} {'hybrid':>8}")
    log("-" * 53)
    log(f"{'Recall@5':<35} {results['modes']['vector']['recall_at_5']:>8.3f} {results['modes']['hybrid']['recall_at_5']:>8.3f}")
    log(f"{'Answer accuracy (hybrid)':<35} {'—':>8} {avg_accuracy:>8.3f}")
    log(f"{'Citation precision (hybrid)':<35} {'—':>8} {avg_cite_prec:>8.3f}")
    log(f"{'Abstention rate':<35} {'—':>8} {abstention_rate:>8.1%}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved → {output_path}")
    log("Check live: curl http://localhost:8000/v1/eval/results")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation harness")
    parser.add_argument("--eval-set", default="tests/eval/eval_set.json")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", default="tests/eval/results.json")
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    run_eval(args.eval_set, args.document_id, args.output)
