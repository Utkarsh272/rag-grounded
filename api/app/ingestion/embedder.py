# api/app/ingestion/embedder.py
"""
Embedder — local sentence-transformers, loaded in a background thread at startup.

The model loads AFTER uvicorn binds to the port, so Render's health check
passes immediately. The first actual embedding request waits for the model
to finish loading (via threading.Event), then proceeds normally.
"""
from __future__ import annotations

import threading
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_ready = threading.Event()


def _load():
    global _model
    print("[embedder] Loading all-MiniLM-L6-v2 in background...")
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _ready.set()
    print("[embedder] Model ready.")


def start_background_load():
    """Call once at app startup. Returns immediately; model loads in background."""
    t = threading.Thread(target=_load, daemon=True)
    t.start()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts. Blocks until model is ready if still loading."""
    _ready.wait()  # waits here only if model hasn't finished loading yet
    return _model.encode(texts, convert_to_numpy=True).tolist()
