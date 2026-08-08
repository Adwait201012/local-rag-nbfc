# Local RAG stack — RTX 5070 (12 GB) / 32 GB RAM / 1 TB SSD

Fully local, fully open. No API keys, nothing leaves the machine.

```
Docling → bge-m3 → Postgres+pgvector → bge-reranker-v2-m3 → Qwen3 8B (Ollama)
 parse    embed     hybrid retrieval        rerank              generate
```

Every component is Apache-2.0 / MIT / PostgreSQL-licensed. Nothing here restricts
commercial or multi-tenant use, so this can become a product later.

## The VRAM budget — this drives every model choice

12 GB is plenty for good RAG, but not for a large LLM *and* the retrieval models
at once. The split below is why the stack is shaped the way it is:

| Container | On GPU | VRAM |
|---|---|---|
| `ollama` | qwen3:8b Q4_K_M | ~5.2 GB |
| `embeddings` | bge-m3 + bge-reranker-v2-m3, both fp16 | ~2.5 GB |
| `ingest` | nothing — Docling runs on CPU deliberately | 0 GB |
| | **total** | **~7.7 GB** |

That leaves ~4 GB of headroom for KV cache growth and long-context queries. Two
consequences worth knowing before you change anything:

- **Don't jump to a 14B model.** At Q4 it's ~9 GB on its own and will evict the
  embedder. If you want 14B, set the embeddings service to CPU first.
- **Ingestion is CPU-only on purpose.** Docling's layout models would otherwise
  take ~2 GB while you're serving. On CPU, parsing is slower but you can index a
  corpus while still answering queries. 32 GB RAM handles this comfortably.

Disk: ~15 GB for model weights, plus roughly 1 GB of Postgres per ~250k chunks
including the HNSW index. A 1 TB SSD is not close to a constraint here.

## The Blackwell gotcha — read before installing

The RTX 5070 is compute capability **sm_120**. CUDA 12.8 was the first release to
add the Blackwell targets, and PyTorch 2.7.0 was the first stable build shipping
sm_120 wheels — anything older refuses to run on the card. The embeddings
Dockerfile therefore pins `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`. Don't
downgrade that base image.

You also need the **NVIDIA Container Toolkit** so Docker can see the GPU:

```bash
nvidia-smi                                             # driver present?
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

If the second command fails, install the container toolkit before going further.
Linux or WSL2 both work; WSL2 is the smoother path on Windows.

## Setup

```bash
cp .env.example .env          # defaults are fine to start
make up                       # build + start db, embeddings, ollama, api
make model                    # pull qwen3:8b into Ollama (~5 GB)
make health                   # confirm all four components are talking

cp ~/your-docs/*.pdf corpus/
make ingest                   # parse, chunk, embed, store
```

Then open **http://localhost:8080** for the chat UI, or:

```bash
curl -s localhost:8080/api/search -H 'content-type: application/json' \
  -d '{"query":"what is the warranty period?"}' | python3 -m json.tool
```

First boot downloads ~2.5 GB of embedding weights and ~5 GB of model — the
`embeddings` healthcheck allows for that, so give it a few minutes.

## Why each piece is there

**Docling for parsing, not PyPDF.** This is where home-built RAG quietly fails: a
table flattened into a wall of numbers is unretrievable no matter how good your
embeddings are. Docling preserves layout, reading order, headings and tables. If
your documents are scanned or heavily CJK, swap in MinerU.

**Structure-aware chunking at 512 tokens.** Chunking is the highest-leverage
index-time decision — wrong chunk shape degrades retrieval more than almost any
other knob. Fixed-size splitting cuts sentences and tables mid-thought.

**Hybrid retrieval, fused in SQL.** Dense search cannot handle
`ERR_SSL_PROTOCOL_ERROR` or part number `WX-4200` — semantic similarity is
meaningless for an identifier. Postgres full-text catches those; the vector index
catches paraphrase. The `hybrid_search()` function in `db/init.sql` merges both
with reciprocal rank fusion, which needs no score normalisation between two very
different scales.

**Rerank 40 down to 6.** The cross-encoder sees query and passage *together*,
which is why it ranks far better than the bi-encoder that produced the embeddings.
This is usually the single biggest quality jump available without changing models.
Passing raw top-5 vector hits straight to the LLM is the most common way to leave
quality on the table.

**Only 6 passages reach the LLM.** Stuffing in 20 dilutes attention and makes
answers worse, not better.

**An OpenAI-compatible endpoint** at `/v1/chat/completions`, so you can point Open
WebUI, LibreChat, or any OpenAI SDK at this and get grounded answers with
citations instead of raw model output.

## When answers are bad, debug in this order

Retrieval is the culprit far more often than generation:

1. `python eval/evaluate.py` — if hit@5 is low, the answer never reached the LLM
   and no prompt engineering will fix it.
2. **Bad hit@5** → parsing or chunking. Call `/api/search` and read the actual
   chunks. Are tables intact? Are chunks cut mid-sentence?
3. **Good hit@5, bad hit@1** → ranking. Raise `TOP_K_CANDIDATES`, or raise
   `hnsw.ef_search` if recall looks thin.
4. **Good retrieval, bad answer** → *now* look at the prompt or a bigger model.

Write `eval/questions.jsonl` (20–30 questions with known answer locations) on day
one. Without it, every "improvement" is a coin flip.

## Growing out of this

- **>10M chunks** → move to Qdrant. Below that, keeping documents, metadata and
  vectors in one Postgres is a real operational advantage.
- **Many concurrent users** → replace Ollama with vLLM. But vLLM pre-allocates
  ~90% of VRAM for its KV cache at startup, so on 12 GB it will fight the
  embedder; you'd move retrieval models to CPU or a second box.
- **A polished UI** → point Open WebUI or AnythingLLM at the `/v1` endpoint.
  Avoid Dify if you ever intend to sell this: its licence bars multi-tenant use
  without written permission from LangGenius.
- **Tracing** → add Langfuse (MIT core) and instrument `api/app.py`.

## Layout

```
db/init.sql              schema + hybrid_search() RRF function
embeddings/server.py     bge-m3 + reranker held resident on GPU (:8081)
ingest/ingest.py         Docling parse → chunk → embed → store (CPU, one-shot)
api/app.py               retrieval + generation + OpenAI-compatible API (:8080)
api/index.html           minimal chat UI
eval/evaluate.py         retrieval regression harness
```
