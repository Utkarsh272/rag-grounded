"""
Prometheus metrics for the RAG pipeline (RED method per stage).

Exposes GET /metrics — Prometheus scrape endpoint.

Import the context managers and helpers from routes/messages.py
and routes/documents.py to instrument the pipeline.

Usage example in messages.py:
    from app.telemetry.metrics import time_retrieval, time_llm, record_request

    with time_retrieval("hybrid"):
        chunks = hybrid_retrieve(...)

    with time_llm("groq", "answer"):
        raw_output = generate(...)

    record_request("success", elapsed_seconds)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

# ── Counters ──────────────────────────────────────────────────────────────────

requests_total = Counter(
    "rag_requests_total",
    "Total chat requests processed",
    ["status"],  # success | error | abstained
)

retrieval_total = Counter(
    "rag_retrieval_total",
    "Total retrieval operations",
    ["mode"],  # vector | hybrid
)

llm_calls_total = Counter(
    "rag_llm_calls_total",
    "Total LLM API calls",
    ["provider", "purpose"],  # groq/anthropic × answer/claims/judge
)

abstentions_total = Counter(
    "rag_abstentions_total",
    "Total abstentions (pre-LLM gate or post-LLM INSUFFICIENT_INFO)",
    ["reason"],  # rerank_threshold | jaccard_threshold | insufficient_info
)

ingestion_total = Counter(
    "rag_ingestion_total",
    "Total document ingestion jobs",
    ["status"],  # success | error
)

chunks_created_total = Counter(
    "rag_chunks_created_total",
    "Total chunks created during ingestion",
)

# ── Histograms ────────────────────────────────────────────────────────────────

request_duration_seconds = Histogram(
    "rag_request_duration_seconds",
    "End-to-end latency of chat requests",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Latency of the retrieval stage",
    ["mode"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

embedding_duration_seconds = Histogram(
    "rag_embedding_duration_seconds",
    "Latency of Jina embedding API calls",
    ["purpose"],  # query | passage | claims
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

llm_duration_seconds = Histogram(
    "rag_llm_duration_seconds",
    "Latency of LLM API calls",
    ["provider", "purpose"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
)

ingestion_duration_seconds = Histogram(
    "rag_ingestion_duration_seconds",
    "End-to-end latency of document ingestion",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

active_requests = Gauge(
    "rag_active_requests",
    "Number of chat requests currently being processed",
)


# ── Context managers for easy instrumentation ─────────────────────────────────

@contextmanager
def time_retrieval(mode: str) -> Generator[None, None, None]:
    retrieval_total.labels(mode=mode).inc()
    start = time.perf_counter()
    try:
        yield
    finally:
        retrieval_duration_seconds.labels(mode=mode).observe(time.perf_counter() - start)


@contextmanager
def time_embedding(purpose: str) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        embedding_duration_seconds.labels(purpose=purpose).observe(time.perf_counter() - start)


@contextmanager
def time_llm(provider: str, purpose: str) -> Generator[None, None, None]:
    llm_calls_total.labels(provider=provider, purpose=purpose).inc()
    start = time.perf_counter()
    try:
        yield
    finally:
        llm_duration_seconds.labels(provider=provider, purpose=purpose).observe(
            time.perf_counter() - start
        )


# ── Convenience helpers ───────────────────────────────────────────────────────

def record_request(status: str, duration_seconds: float) -> None:
    """status: success | error | abstained"""
    requests_total.labels(status=status).inc()
    request_duration_seconds.observe(duration_seconds)


def record_abstention(reason: str) -> None:
    """reason: rerank_threshold | jaccard_threshold | insufficient_info"""
    abstentions_total.labels(reason=reason).inc()


def record_ingestion(status: str, duration_seconds: float, num_chunks: int = 0) -> None:
    ingestion_total.labels(status=status).inc()
    ingestion_duration_seconds.observe(duration_seconds)
    if num_chunks:
        chunks_created_total.inc(num_chunks)


# ── /metrics endpoint ─────────────────────────────────────────────────────────

async def metrics_endpoint(request: Request) -> Response:
    """Mount this at GET /metrics in main.py."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
