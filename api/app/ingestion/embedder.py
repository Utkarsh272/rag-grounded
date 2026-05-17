# api/app/ingestion/embedder.py
# Using local sentence-transformers model — free, no API key needed.
# Model: all-MiniLM-L6-v2 → 384-dim vectors, fast on CPU, good quality.
# NOTE: First run downloads ~90MB model to ~/.cache/huggingface — normal.
from sentence_transformers import SentenceTransformer

_model = None

def _get_model() -> SentenceTransformer:
    """Lazy-load the model once and reuse it."""

    global _model
    if _model is None:
        print("🔄 Loading embedding model (first time only, ~90MB download)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of 384-dim vectors."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()