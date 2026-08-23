# Local RAG over Indian NBFC regulation

A fully self-hosted retrieval-augmented generation system. Every component is
open source and runs on one consumer GPU. No API keys, no cloud, no data leaving
the machine.

Built and measured on the RBI's consolidated 2025 NBFC Master Directions —
46 documents, 2,181 passages.

```
documents ──► Docling ──► chunks ──► bge-m3 ──┐
   (CPU)      parse + chunk         embed     │
                                              ▼
              query ──► bge-m3 ──►  Postgres + pgvector
                                    HNSW  ∪  full-text  ──► RRF fusion (top 40)
                                              │
                                              ▼
                                    bge-reranker-v2-m3  ──► top 6
                                              │
                                              ▼
                                    Ollama / Qwen3-8B  ──► answer with [n] citations
```

## Measured results

Twenty questions with hand-verified answers, drawn from the indexed Master
Directions. `eval/questions.jsonl` and `eval/evaluate.py` are in this repo.

| corpus | retrieval | recall@6 | MRR |
|---|---|---|---|
| 2,181 chunks | hybrid only | 0.95 | 0.787 |
| 2,181 chunks | hybrid + rerank | **1.00** | **0.912** |

**Read this honestly.** Twenty questions is a small set, they were written by me
against a corpus I chose, and the pass condition is "a passage containing the
answer appeared in the top 6" — not "the generated answer was correct." The
number says the retriever is finding the right material; it does not say the
system is 100% accurate. See *Open problems* below.

## Stack

| Layer | Choice | Licence |
|---|---|---|
| Parsing & chunking | Docling (HybridChunker), RapidOCR fallback | MIT |
| Embeddings | BAAI/bge-m3, 1024-dim, 100+ languages | MIT |
| Store & retrieval | Postgres 17 + pgvector HNSW + tsvector, fused with RRF | PostgreSQL / MIT |
| Reranking | BAAI/bge-reranker-v2-m3 cross-encoder | Apache-2.0 |
| Generation | Ollama serving Qwen3-8B Q4_K_M | MIT / Apache-2.0 |
| API & UI | FastAPI + one HTML page | this repo |

All permissively licensed. Nothing here restricts commercial use.

## Design decisions, and why

**Hybrid search, not pure vector.** Dense embeddings are strong on paraphrase and
useless on exact tokens — regulation is full of paragraph numbers, thresholds and
identifiers that carry no semantic meaning. Postgres full-text search catches
those. Both run, and results are merged.

**RRF, not weighted score addition.** Cosine similarity and BM25 produce scores on
incomparable scales; adding them is adding rupees to kilometres. Reciprocal Rank
Fusion discards the scores and keeps only ranks: each result scores
`1/(60 + rank)` from each list. A passage both retrievers merely like can beat one
a single retriever loves.

**Retrieve 40, rerank to 6.** bge-m3 is a bi-encoder — question and passage are
embedded separately, so every passage is pre-computed and search is fast but
approximate. The reranker is a cross-encoder that reads question and passage
together: far more accurate, far too slow for a whole corpus. Wide cheap net, then
careful judgement. Measured contribution of the reranker at 2,181 chunks:
recall +0.05, MRR +0.125.

**Postgres rather than a dedicated vector database.** Under ~10M vectors pgvector
is simpler: one container, ordinary SQL joins against document metadata, and
transactional consistency between documents and chunks.

**Parsing on CPU, retrieval on GPU.** The 12 GB budget is ~8.6 GB resident —
Qwen3-8B with a q8_0 KV cache, plus bge-m3 and the reranker in fp16. Docling would
want several GB more and would evict the language model mid-conversation, so it
runs on system RAM instead.

## Experiments that failed

Recorded because the failures were more informative than the successes.

**1. Chunk size 512 → 1024.** Hypothesis: the failures were lists split across
chunk boundaries, so bigger chunks would keep them whole.
Result: identical after reranking (0.80 → 0.80), and hybrid-only recall *dropped*
0.75 → 0.70. Longer chunks average more topics into one vector, so first-stage
retrieval got worse and the reranker absorbed the damage. Chunk size was not the
bottleneck.

**2. Splitting oversized chunks at enumeration markers.** 231 of 615 chunks
exceeded 2,000 characters — heterogeneous definition sections where the answer sat
at, in one measured case, character 2,842 of 3,101. Splitting at `(i)`, `(1)`, `(a)`
boundaries reduced oversized chunks to 137 and doubled the count to 1,195.
Result: recall fell 0.85 → 0.80, and a previously passing question broke. In
regulatory text sub-clauses are not independent — clause (1) reads "in the case
of…" and means nothing without its parent. The chunks got smaller and dumber.
Reverted.

**3. Assuming low chunk counts meant truncated parsing.** Three documents produced
8–16 chunks where others produced 40–83. Checked by comparing Docling's extracted
character count against the sum of stored chunk lengths: 6,907 vs 6,926. Nothing
was lost — those Master Directions are genuinely short. A false alarm resolved by
measurement rather than assumption.

## The eval bug

The most useful finding in the project.

Retrieval scored 0.80 while the system visibly answered several "failed" questions
correctly. The cause was in `evaluate.py`: a question counted as a hit only if one
specific expected string appeared in a retrieved chunk. Once the corpus held 46
overlapping regulations, the same rule appeared in several places — the retriever
returned a perfectly good passage from a different Master Direction and the eval
scored it a miss.

Three separate versions of this bug turned up:

- **Wrong strings.** `Net Owned Fund` matched four chunks, none of which was the
  deposit-ceiling rule — it appears in definitions and capital sections too.
- **Notation mismatch.** Searching for `Tier I` returned zero results; the
  documents write `Tier 1` with a digit. A real answer, invisible to the test.
- **Questions the corpus could not answer.** Some expected facts came from a web
  summary of all 34 Master Directions, not the subset actually indexed.

Fixing the ruler moved recall 0.80 → 0.85 → 1.00 without a single change to the
retrieval pipeline.

The general lesson: **an eval that checks for retrieval of one specific chunk is
not measuring answer correctness**, and the two diverge as a corpus grows. Tuning
a retriever against a broken eval is a way to spend weeks going nowhere.

## Open problems

Known limitations, none of them solved:

- **No contradiction detection.** When two retrieved passages disagree — a rule
  superseded by a later direction — the system picks one and sounds certain. In a
  compliance context that is a wrong answer with consequences.
- **No cross-chunk reasoning.** The cross-encoder scores each passage in
  isolation, so it cannot recognise that two chunks are only useful together.
- **Silent parsing loss.** Five chunks contain `formula-not-decoded` where Docling
  dropped content it could not parse. Nothing warns you; retrieval simply cannot
  find text that was never extracted.
- **No per-user isolation.** A single shared password, one document pool. Two
  users would see each other's files.
- **OCR is Chinese/English-trained.** Fine on the RBI's digital PDFs, where it
  barely fires. Untested and probably poor on scanned Devanagari, which is the
  case that actually matters for Indian document work.

## Running it

Requires Docker with the NVIDIA Container Toolkit, an NVIDIA driver of 580 or
newer, and roughly 12 GB of VRAM.

```bash
cp .env.example .env     # set APP_PASSWORD and SESSION_SECRET
make up                  # build and start db, embeddings, ollama, api, ingest
make model               # pull qwen3:8b into Ollama (~5 GB)
make health              # all four services should report ok
```

Then open `http://localhost:8080`, drag documents onto the left panel, and ask
questions once they finish indexing.

First start downloads ~10 GB of images and model weights. Everything is cached
afterwards, so later starts take seconds.

### Evaluating

```bash
python3 -m venv .venv && .venv/bin/pip install httpx
.venv/bin/python eval/evaluate.py eval/questions.jsonl
```

Prints recall and MRR with and without reranking. Write your own questions before
trusting the numbers in this README — and check that every expected phrase
actually appears in your corpus, or you will measure the ruler instead of the
system.

### Notes on the build

Blackwell GPUs (RTX 50-series, sm_120) need CUDA 12.8 or newer, which is why the
embeddings image is pinned to `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`.

Ingestion runs as a serial queue backed by the `jobs` table rather than in-process
background tasks. Parsing is CPU-bound, so concurrent jobs only thrash; and jobs
that survive in the database can be resumed after a restart instead of being lost.

RapidOCR downloads its weights to a mounted directory rather than into the
container, so a rebuild does not force a re-download from a slow remote host.
