# RAG with Grounded Citations

> Production-grade document Q&A: every answer cites exact source spans, every claim carries a confidence score, and the system says "I don't know" when retrieval is weak.

**Status**: 🚧 In progress — Days 1–3 complete (ingestion + vector search + hybrid retrieval), Days 4–10 in progress.

---

## What makes this different from generic RAG

Most RAG demos retrieve chunks and dump them into an LLM prompt. This project goes further:

- **Inline citations** — every claim in the answer links back to the exact source span in the original document, not just the document name
- **Confidence scoring** — each claim is scored independently; low-confidence claims are flagged in the UI
- **Abstention** — when retrieval similarity is uniformly weak, the system says "I don't have enough information" instead of hallucinating
- **Hybrid retrieval** — vector search + BM25 keyword search fused via Reciprocal Rank Fusion, then reranked with a cross-encoder

---

## Architecture

```mermaid
graph LR
    User[User] --> UI[Next.js + TS UI]
    UI -->|upload| API[FastAPI Backend]
    UI -->|/ask| API
    API --> Parser[unstructured Parser]
    Parser --> Chunker[Structure-aware Chunker]
    Chunker --> Embedder[all-MiniLM-L6-v2 Local Embedder]
    Embedder --> PG[(Supabase Postgres + pgvector)]
    API --> Retriever[Hybrid Retriever]
    Retriever --> PG
    Retriever --> Reranker[Cross-Encoder Reranker]
    Reranker --> LLM[LLM Generation]
    LLM --> Verifier[Claim Verifier]
    Verifier --> Response[Answer + Citations + Confidence]
    Response --> UI
```

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 14 + TypeScript + Tailwind + shadcn/ui | Modern, type-safe |
| Backend | Python 3.12 + FastAPI + Pydantic | Best AI/ML ecosystem |
| Vector DB | pgvector on Supabase (free) | No Pinecone cost |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers | Free, runs on CPU |
| LLM | Groq Llama 3.3 70B (dev) + Claude Sonnet (final demo) | Free dev path |
| Doc parsing | `unstructured` | Production-grade PDF/MD parsing |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, runs on CPU |
| Hosting | Vercel (frontend) + Render (backend) | Free tier |

---

## Project Structure

```
/rag-grounded
├── /api                          # Python 3.12 + FastAPI backend
│   └── /app
│       ├── main.py               # FastAPI app + route wiring
│       ├── /routes
│       │   ├── documents.py      # Upload, list, status endpoints
│       │   └── search.py         # Search (?mode=vector|hybrid|compare)
│       ├── /ingestion
│       │   ├── chunker.py        # Structure-aware chunker with char offsets
│       │   └── embedder.py       # Local sentence-transformers embedder
│       ├── /retrieval
│       │   ├── vector.py         # pgvector cosine similarity search
│       │   ├── bm25.py           # Postgres tsvector full-text search
│       │   ├── reranker.py       # Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
│       │   └── hybrid.py         # RRF fusion: vector + BM25 → rerank → top-5
│       ├── /generation           # (Day 4) LLM + citation injection
│       ├── /verification         # (Day 5) Claim scoring + abstention
│       └── /db
│           └── client.py         # Supabase client
└── /web                          # Next.js 14 frontend (Day 6)
```

---

## API Endpoints

```
POST   /v1/documents               Upload PDF/MD/TXT → {document_id, status}
GET    /v1/documents               List all documents
GET    /v1/documents/{id}/status   Poll ingestion status
GET    /v1/search?q=...            Semantic search across chunks
         &top_k=5                    Number of results (default 5, max 20)
         &document_id=<uuid>         Scope to one document (optional)
         &mode=hybrid                vector | hybrid | compare (default: hybrid)
GET    /healthz                    Liveness check
```

---

## Key Design Decisions

### Structure-aware chunking over fixed-window chunking

Most tutorials chunk at every N tokens blindly. This project splits by markdown headings first, then by paragraph within each section. Every chunk stores `start_char` and `end_char` offsets into the original document — these are what power citation highlighting later. Blind fixed-window chunking breaks across section boundaries and makes citations meaningless.

### Local embeddings over OpenAI API

Using `all-MiniLM-L6-v2` via `sentence-transformers` instead of `text-embedding-3-small`. Reasons: zero API cost during development, no network latency, 384-dim vectors are fast to index and query. Trade-off: slightly lower retrieval quality than `text-embedding-3-small` on complex technical documents. Will benchmark both in the eval harness (Day 8).

### pgvector + SECURITY DEFINER function

Supabase's Row Level Security blocks functions from seeing rows unless the function runs with elevated permissions. The `match_chunks` Postgres function uses `SECURITY DEFINER` so it runs as its owner (postgres) and bypasses RLS. The Python client sends embeddings as text strings (`"[0.1,0.2,...]"`) rather than arrays because the Supabase client can't auto-cast Python lists to the `vector` type.

### Hybrid retrieval: vector + BM25 + RRF + cross-encoder

Vector search alone misses exact keyword matches ("backpropagation", "RLHF", proper nouns). BM25 via Postgres `tsvector` catches these for free — no extra infrastructure, no Elasticsearch. Results from both are fused with Reciprocal Rank Fusion (RRF, k=60 from Cormack et al. 2009): a chunk appearing in both lists gets a combined score even if it wasn't #1 in either. RRF requires no score normalisation across retrieval methods, making it robust without tuning.

The top-10 RRF candidates then go through a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~23 MB, runs on CPU). Unlike bi-encoder embeddings, the cross-encoder sees both query and passage together, giving much sharper relevance scores at the cost of being non-pre-computable. Running it only on the top-10 RRF candidates keeps latency acceptable.

The `/v1/search?mode=compare` endpoint returns all three modes side-by-side — this feeds directly into the Day 8 eval harness for the recall@5 and citation precision numbers in the table below.

---

## Setup

### Prerequisites

- Python 3.12+
- `uv` (Python package manager)
- A free [Supabase](https://supabase.com) account

### Database setup

In Supabase SQL Editor, run these in order:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title         TEXT NOT NULL,
  source_type   TEXT NOT NULL,
  status        TEXT DEFAULT 'pending',
  error_message TEXT,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE chunks (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id   UUID REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index   INTEGER NOT NULL,
  content       TEXT NOT NULL,
  start_char    INTEGER NOT NULL,
  end_char      INTEGER NOT NULL,
  section_title TEXT,
  embedding     vector(384),
  ts_vector     tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX chunks_embedding_idx ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX chunks_ts_idx ON chunks USING gin(ts_vector);
```

Vector similarity search function:

```sql
CREATE FUNCTION match_chunks(
  query_embedding    text,
  match_count        int,
  filter_document_id uuid DEFAULT NULL
)
RETURNS TABLE (
  id uuid, document_id uuid, content text,
  section_title text, start_char int, end_char int, similarity float
)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT c.id, c.document_id, c.content, c.section_title, c.start_char, c.end_char,
         1 - (c.embedding <=> query_embedding::vector(384)) AS similarity
  FROM chunks c
  WHERE c.embedding IS NOT NULL
    AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
  ORDER BY c.embedding <=> query_embedding::vector(384) ASC
  LIMIT match_count;
$$;
```

BM25 full-text search function:

```sql
CREATE FUNCTION bm25_search(
  query_text         text,
  match_count        int,
  filter_document_id uuid DEFAULT NULL
)
RETURNS TABLE (
  id uuid, document_id uuid, content text,
  section_title text, start_char int, end_char int, rank float
)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
  SELECT c.id, c.document_id, c.content, c.section_title, c.start_char, c.end_char,
         ts_rank_cd(c.ts_vector, plainto_tsquery('english', query_text))::float AS rank
  FROM chunks c
  WHERE c.ts_vector @@ plainto_tsquery('english', query_text)
    AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
  ORDER BY rank DESC
  LIMIT match_count;
$$;
```

### Backend

```bash
cd api
cp .env.example .env   # fill in your Supabase URL + service key
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

---

## Eval Results

*(Day 8 — to be filled in after eval harness is built)*

| Metric | Vector-only | Hybrid | Hybrid + Rerank |
|---|---|---|---|
| Recall@5 | — | — | — |
| Answer accuracy | — | — | — |
| Citation precision | — | — | — |

---

## Roadmap

- [x] Day 1 — Scaffolding, PDF/MD ingestion, document storage
- [x] Day 2 — Structure-aware chunking, local embeddings, vector search
- [x] Day 3 — BM25 keyword search + hybrid RRF fusion + cross-encoder reranking
- [ ] Day 4 — LLM answer generation with inline citation injection
- [ ] Day 5 — Claim verification, confidence scoring, abstention
- [ ] Day 6 — Frontend: chat UI, citation highlighting, file uploader
- [ ] Day 7 — Auth (Supabase), multi-tenancy, deployment
- [ ] Day 8 — Evaluation harness (20 Q&A ground truth set)
- [ ] Day 9 — OpenTelemetry tracing, Prometheus metrics, Grafana dashboard
- [ ] Day 10 — README polish, DESIGN.md, demo video

---

## Cost

| Component | Service | Cost |
|---|---|---|
| Embeddings | Local `all-MiniLM-L6-v2` | $0 |
| Reranker | Local `ms-marco-MiniLM-L-6-v2` | $0 |
| LLM (dev) | Groq Cloud | $0 |
| LLM (final demo) | Anthropic Claude Sonnet | ~$5 |
| Database | Supabase free tier | $0 |
| Hosting | Vercel + Render free tier | $0 |
| **Total** | | **~$5** |
