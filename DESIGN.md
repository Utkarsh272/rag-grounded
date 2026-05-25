# DESIGN.md — RAG with Grounded Citations

Architecture decisions, trade-offs, and what I'd do differently at production scale.

---

## 1. Why structure-aware chunking instead of fixed-window chunking

Most RAG tutorials chunk documents at every N tokens blindly. I split by markdown/document headings first, then by paragraph within each section.

**The problem with fixed-window chunking:** a 1024-token window bisects sentences and crosses section boundaries. A chunk might start mid-sentence from one section and end mid-sentence into the next. This produces three failure modes:
- Retrieval returns a chunk whose content is semantically incoherent
- The citation points to a span that includes irrelevant neighbouring text
- The LLM receives confused context and either hallucinates or hedges

**My approach:** `chunker.py` uses `re.finditer` on heading patterns to identify section boundaries, then splits each section by paragraph. Every chunk stores `start_char` and `end_char` — byte offsets into the original document. These offsets are what makes citation highlighting work: the UI can highlight exactly the passage the LLM cited, not just "somewhere in section 3."

Overlap (100 chars at chunk boundaries) is added to preserve cross-boundary context without duplicating full paragraphs.

**Trade-off:** structure-aware chunking assumes the document has headings. For flat prose (legal contracts, continuous narrative) it degrades to paragraph splitting, which is still better than fixed-window but loses the section-title signal. Production fix: detect document type first and choose the chunking strategy accordingly.

---

## 2. Why Jina AI embeddings instead of a local model

The original implementation used `sentence-transformers` with `all-MiniLM-L6-v2` (384-dim) locally. This worked perfectly on my MacBook. It broke immediately on Render's free tier.

**The failure:** Render's free web service has 512 MB RAM. `sentence-transformers` downloads ~90 MB of model weights and loads ~300 MB into RAM on first use. The container OOM-killed before the first request completed.

**Solutions considered:**

| Option | Problem |
|---|---|
| Pre-download model in Dockerfile | Still OOM on load — the RAM doesn't change |
| Lazy load in background thread | Race condition; still OOM on the first real request |
| Upgrade Render plan | Cost — defeats the $0 hosting goal |
| OpenAI `text-embedding-3-small` | No free tier; account had no credits |
| Jina AI `jina-embeddings-v3` | Free API, 1M tokens on signup, 1024-dim, zero RAM |

**Jina's asymmetric retrieval:** `jina-embeddings-v3` supports task types — `retrieval.passage` for documents at ingestion time, `retrieval.query` for user questions at search time. This is genuine asymmetric embedding: the model is fine-tuned to maximise similarity between a query representation and a passage representation, rather than treating both identically. Using the wrong task type (e.g. encoding queries as passages) degrades retrieval quality measurably.

**Consequence of switching:** the pgvector column type changed from `vector(384)` to `vector(1024)`. All existing chunks had to be deleted and re-ingested. The `match_chunks` Postgres function was rewritten to cast `text → vector(1024)` internally, because the Supabase Python client cannot auto-cast Python `list[float]` to the pgvector type.

**What I'd do at scale:** use a dedicated embedding service (Jina, Voyage, or Cohere) regardless of infrastructure, not a locally loaded model. The RAM and startup-time costs of local models don't fit well with autoscaling backends. The network latency (~50–100ms) is acceptable given the retrieval step isn't on the critical streaming path.

---

## 3. Why hybrid retrieval (vector + BM25 + RRF + reranker) instead of vector-only

Vector search alone has a well-known weakness: it fails on exact keyword matches. If a user asks "what is PBFT?" and the document contains "Practical Byzantine Fault Tolerance (PBFT)", the embedding model may or may not catch this depending on how rare the acronym is in its training data. BM25 catches it deterministically.

**The pipeline:**

```
vector search (top 20) ─┐
                         ├─ RRF fusion → top 10 → cross-encoder rerank → top 5
BM25 search   (top 20) ─┘
```

**Why Reciprocal Rank Fusion:** RRF combines ranked lists without needing score normalisation. A chunk appearing at rank 3 in vector search and rank 7 in BM25 scores higher than one appearing at rank 1 in only one list. The formula `1 / (k + rank)` with k=60 is robust across different score distributions. k=60 is the standard default from the original Cormack et al. paper — I didn't tune it.

**Why a cross-encoder reranker on top:** bi-encoder embeddings (like Jina's) encode query and passage independently and compare them via cosine similarity. This is fast but loses the interaction signal — the model never sees query and passage together. A cross-encoder (ms-marco-MiniLM-L-6-v2) takes query + passage concatenated and outputs a relevance score. It's 5–10× more accurate than bi-encoder ranking but 100× slower, so I only run it on the top-10 RRF candidates to reorder them.

**Production caveat:** the cross-encoder runs on CPU. At scale (>10 QPS), this would be a bottleneck. Mitigation: cache rerank results for identical (query, chunk_id) pairs with a short TTL, or move to a managed reranking API (Cohere Rerank, Jina Reranker API).

**Eval results:** both vector-only and hybrid achieved Recall@5 = 1.0 on my 20-question eval set. This is expected — the eval document has clean, well-separated sections that either retrieval mode handles well. The gap between vector-only and hybrid would widen on:
- Acronym-heavy documents (BM25 advantage)
- Long documents with many similar sections (reranker advantage)
- Documents where query vocabulary differs from passage vocabulary (vector advantage)

---

## 4. Why two-layer abstention and how the thresholds were tuned

The system has two independent abstention layers:

**Layer 1 — Pre-LLM gate** (`verification/abstention.py`): fires before any LLM call, saving tokens and latency.
- Signal 1: top-1 cross-encoder rerank score. Cross-encoder logits are unbounded; empirically on ms-marco, scores below -5 indicate non-relevant pairs. I set `ABSTAIN_RERANK_THRESHOLD=-8.0` after production testing showed -6.0 was too aggressive for some valid queries.
- Signal 2: Jaccard overlap between query unigrams and chunk unigrams. `ABSTAIN_JACCARD_THRESHOLD=0.0` (effectively disabled) — Jaccard kills meta-queries like "summarise the key points" because they share no vocabulary with the passage content even when retrieval is strong.

**Layer 2 — Post-LLM gate** (`generation/citation_parser.py`): the LLM is instructed to respond with the literal string `INSUFFICIENT_INFO` when sources don't support the answer. The citation parser checks for this sentinel before regex processing. This catches cases that pass Layer 1 (retrieval looks plausible) but the LLM determines it cannot answer from the provided context.

**Why two layers:** they catch different failure modes. Layer 1 is cheaper (no LLM call) and catches clearly out-of-scope queries. Layer 2 catches in-scope queries where the LLM's internal reasoning determines the retrieved chunks don't actually answer the question. Running Layer 1 without Layer 2 means the LLM sometimes hallucinates plausible-sounding answers when context is borderline. Running Layer 2 without Layer 1 wastes LLM tokens on clearly irrelevant queries.

**The production problem I hit:** deploying with `ABSTAIN_JACCARD_THRESHOLD=0.08` caused "summarise the key points" to always abstain in production. The Jaccard overlap of "summarise", "key", "points" against any technical document is near zero — those words simply don't appear in the document. Disabling Jaccard (`0.0`) fixed it. The lesson: keyword overlap is a reasonable heuristic for factual queries but a bad one for meta-instructions. A production system should classify query type first (factual vs. meta) and apply different thresholds.

---

## 5. Citation injection via [SOURCE_X] tokens instead of structured output

I considered two approaches for getting citations out of the LLM:

**Option A — JSON structured output:** prompt the LLM to return `{"answer": "...", "citations": [{"claim": "...", "source": 1}]}`. Reliable format, easy to parse.

**Option B — Inline token injection:** prompt the LLM to emit `[SOURCE_X]` after every claim inline in natural prose.

I chose Option B for a specific reason: inline citations are what the UI needs. A claim and its citation appear adjacent in the text stream, which means the streaming UI can link citation numbers to source spans as the answer arrives, rather than waiting for the complete JSON payload.

**The parsing risk:** the LLM occasionally emits `[SOURCE_7]` when only 5 sources were provided. `citation_parser.py` handles this with bounds checking — any citation number outside the valid range is silently dropped and logged. In practice this happens on <2% of responses with Groq Llama 3.3 70B.

**What I'd do at production scale:** use Anthropic's tool use / function calling to enforce a structured citation schema at generation time, then post-process into inline markers for the UI. This eliminates hallucinated citation numbers entirely and gives a JSON-typed response. The trade-off is that streaming becomes more complex (tool use responses don't stream cleanly in all clients).

---

## 6. Per-claim confidence scoring via embedding cosine similarity

After generation, the system splits the answer into atomic claims (via a small LLM call), embeds all claims in a single Jina batch, and computes cosine similarity between each claim embedding and its cited chunk embeddings. Score = max similarity across cited chunks for that claim.

**Why not NLI (Natural Language Inference):** NLI models (e.g. `cross-encoder/nli-deberta-v3-small`) give true entailment scores — they directly model whether a premise entails a hypothesis. This is more semantically accurate than cosine similarity. I didn't use NLI because:
1. It requires another model download (~300 MB) — back to the RAM problem on Render
2. Inference adds 200–500ms per claim on CPU
3. For a UI confidence badge, "medium confidence" vs. "high confidence" is sufficient signal — precise entailment probability isn't needed

**Why max similarity instead of mean:** a claim is well-grounded if at least one cited chunk strongly supports it. Mean similarity penalises claims that cite multiple chunks for completeness (e.g. citing both a definition chunk and an example chunk), even if one of them fully supports the claim.

**Known limitation:** cosine similarity between embeddings conflates topical similarity with entailment. A claim about "distributed consensus" will have high cosine similarity to any chunk in the consensus section even if the chunk doesn't entail the specific claim. This means confidence scores are "is this claim on-topic" not "is this claim factually supported." For a portfolio demo this is acceptable; for a production hallucination detector, NLI or LLM-as-judge per-claim is the right approach.

---

## 7. Observability design

**Metrics** follow the RED method per pipeline stage:
- Rate: `rag_requests_total`, `rag_retrieval_total`, `rag_llm_calls_total`
- Errors: `status=error` label on `rag_requests_total`, `status=error` on `rag_ingestion_total`
- Duration: `rag_request_duration_seconds`, `rag_retrieval_duration_seconds{mode}`, `rag_embedding_duration_seconds{purpose}`, `rag_llm_duration_seconds{provider,purpose}`

Additional signals: `rag_abstentions_total{reason}` (tracks which abstention layer fires and why), `rag_active_requests` gauge (concurrent request count), `rag_chunks_created_total` (ingestion volume).

**Why context managers for instrumentation:** wrapping pipeline stages in `with time_retrieval("hybrid"):` keeps instrumentation out of business logic. The context manager always records the duration even if the stage raises an exception, which matters for error rate tracking.

**OpenTelemetry:** console exporter locally (no Docker dependency for dev), OTLP-ready for production via `OTEL_EXPORTER=otlp` + `OTEL_EXPORTER_OTLP_ENDPOINT`. No code change needed to switch exporters.

**The `host.docker.internal` problem:** Docker Desktop on Mac doesn't always resolve `host.docker.internal` to the Mac's LAN IP. The workaround is to run uvicorn with `--host 0.0.0.0` and point Prometheus at the Mac's LAN IP (`192.168.x.x`) directly. This is documented in the `prometheus/prometheus.yml` comment.

---

## 8. What I'd do differently at production scale

**Multi-tenancy and auth:** currently there's no auth — anyone with the API URL can read all documents. The original plan included Supabase Auth (Google OAuth) with Row Level Security scoping documents per user. This was cut to hit the deployment deadline. Adding it requires: Supabase Auth setup, JWT verification middleware on FastAPI, and RLS policies on all tables. The schema already has the structure for it.

**True streaming:** the LLM response is currently buffered fully before streaming begins. This is because `citation_parser.py` needs the complete response to map `[SOURCE_X]` tokens to chunk IDs. True token-by-token streaming would require streaming citation resolution — knowing which `[SOURCE_X]` corresponds to which chunk as each token arrives. This is solvable: pass chunk IDs in the order they were presented to the LLM (SOURCE_1 = chunks[0], etc.) and resolve eagerly rather than post-hoc.

**Reranker in production:** the cross-encoder runs on CPU locally. At >10 QPS this would be the bottleneck. Options: Jina Reranker API (same provider as embeddings, simple HTTP call), Cohere Rerank v3, or a GPU-backed endpoint. All are ~$0.001 per query which is acceptable at product scale.

**Chunking for non-Markdown documents:** PDFs without heading structure (e.g. scanned academic papers, legal contracts) would chunk poorly with the current heading-based approach. Production fix: use `unstructured`'s semantic chunking mode which uses layout analysis and font size to infer section boundaries even without explicit markdown headings.

**Eval coverage:** the 20-question eval set covers a single document type (technical reference material with clear section headings). Production eval would need: multi-document queries, ambiguous questions with partial answers, adversarial queries designed to trigger hallucination, and documents in multiple languages.
