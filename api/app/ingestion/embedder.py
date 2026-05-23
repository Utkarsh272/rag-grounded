# api/app/ingestion/embedder.py
"""
Embedder — uses OpenAI text-embedding-3-small in production,
falls back to local sentence-transformers in dev if OPENAI_API_KEY is not set.

Dimension: 1536 (OpenAI) or 384 (local fallback).

IMPORTANT: if you switch providers, you must re-embed all existing chunks
because the vector dimensions are different. Run a full re-ingestion.
"""
from __future__ import annotations

import os


def _use_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def embed_texts(texts: list[str]) -> list[list[float]]:
    if _use_openai():
        return _embed_openai(texts)
    return _embed_local(texts)


def _embed_openai(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def _embed_local(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    global _local_model
    if _local_model is None:
        print("[embedder] Loading local model (dev only)...")
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model.encode(texts, convert_to_numpy=True).tolist()


_local_model = None
