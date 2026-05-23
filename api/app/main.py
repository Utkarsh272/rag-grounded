# api/app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.documents import router as documents_router
from app.routes.search import router as search_router
from app.routes.conversations import router as conversations_router
from app.routes.messages import router as messages_router

app = FastAPI(title="RAG with Grounded Citations")

# ALLOWED_ORIGINS env var: comma-separated list of allowed origins.
# Defaults to localhost for local dev; set to your Vercel URL in production.
_raw_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000",
)
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(conversations_router)
app.include_router(messages_router)


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def ready():
    """Readiness probe — checks DB and embedder are reachable."""
    from app.db.client import get_supabase
    from app.ingestion.embedder import embed_texts

    errors = []

    try:
        sb = get_supabase()
        sb.table("documents").select("id").limit(1).execute()
    except Exception as e:
        errors.append(f"db: {e}")

    try:
        embed_texts(["ping"])
    except Exception as e:
        errors.append(f"embedder: {e}")

    if errors:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail={"errors": errors})

    return {"status": "ready"}
