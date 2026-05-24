# api/app/ingestion/embedder.py
"""
Embedder — Jina AI jina-embeddings-v3 API.
1024-dim vectors, free tier (1M tokens on signup), no local model needed.
Requires JINA_API_KEY env var.
"""
from __future__ import annotations

import os
import httpx


JINA_MODEL = "jina-embeddings-v3"
JINA_URL = "https://api.jina.ai/v1/embeddings"


def embed_texts(texts: list[str]) -> list[list[float]]:
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY not set in environment")

    response = httpx.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_MODEL,
            "input": texts,
            "task": "retrieval.passage",  # optimised for document retrieval
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    # Sort by index to guarantee order matches input
    embeddings = sorted(data["data"], key=lambda x: x["index"])
    return [e["embedding"] for e in embeddings]


def embed_query(text: str) -> list[float]:
    """Embed a single query string with the retrieval.query task type."""
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY not set in environment")

    response = httpx.post(
        JINA_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_MODEL,
            "input": [text],
            "task": "retrieval.query",  # different task type for queries
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]
