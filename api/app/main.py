# api/app/main.py
"""
FastAPI application entry point.

Changes in Day 8/9:
  - GET /metrics  → Prometheus scrape endpoint
  - GET /readyz   → readiness probe (checks Supabase + Jina)
  - GET /v1/eval/results → eval harness results (served from results.json)
  - configure_tracing() called at startup
  - HTTP middleware tracks active requests + request duration
"""

from __future__ import annotations

import os
import time

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.routes.conversations import router as conversations_router
from app.routes.documents import router as documents_router
from app.routes.eval import router as eval_router
from app.routes.messages import router as messages_router
from app.routes.search import router as search_router
from app.telemetry.metrics import active_requests, metrics_endpoint, record_request
from app.telemetry.otel import configure_tracing

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG with Grounded Citations",
    version="1.0.0",
    description="Production-grade document Q&A with inline citations and confidence scoring.",
)

# ── CORS (preserve existing logic exactly) ────────────────────────────────────

_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    configure_tracing()


# ── Middleware: active request gauge + message-endpoint timing ─────────────────

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    skip = request.url.path in ("/metrics", "/healthz", "/readyz")
    if not skip:
        active_requests.inc()

    start = time.perf_counter()
    status = "success"
    try:
        response = await call_next(request)
        if response.status_code >= 500:
            status = "error"
        return response
    except Exception:
        status = "error"
        raise
    finally:
        if not skip:
            active_requests.dec()
            # Only charge the full request duration to the messages endpoint
            if "/messages" in request.url.path and request.method == "POST":
                record_request(status, time.perf_counter() - start)


# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(conversations_router)
app.include_router(messages_router)
app.include_router(eval_router)


# ── Observability endpoints ───────────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request):
    """Prometheus scrape endpoint."""
    return await metrics_endpoint(request)


@app.get("/healthz", tags=["ops"])
def health():
    """Liveness probe — always 200 if the process is up."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
async def readyz():
    """
    Readiness probe — checks Supabase and Jina API are reachable.
    Returns 200 when ready, 503 when degraded.
    """
    checks: dict[str, str] = {}

    # Supabase
    try:
        from app.db.client import get_supabase
        sb = get_supabase()
        sb.table("documents").select("id").limit(1).execute()
        checks["supabase"] = "ok"
    except Exception as e:
        checks["supabase"] = f"error: {e}"

    # Jina AI
    try:
        jina_key = os.getenv("JINA_API_KEY", "")
        if not jina_key:
            checks["jina"] = "error: JINA_API_KEY not set"
        else:
            resp = httpx.post(
                "https://api.jina.ai/v1/embeddings",
                headers={"Authorization": f"Bearer {jina_key}"},
                json={
                    "model": "jina-embeddings-v3",
                    "input": ["ping"],
                    "task": "retrieval.query",
                },
                timeout=10.0,
            )
            checks["jina"] = "ok" if resp.status_code == 200 else f"error: HTTP {resp.status_code}"
    except Exception as e:
        checks["jina"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
        status_code=200 if all_ok else 503,
    )
