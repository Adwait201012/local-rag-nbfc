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
These are historical English results; rerun after changing OCR or evaluation
labels. They have not been remeasured for this Hindi/Hinglish addition.

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
| Parsing & chunking | Docling (HybridChunker), Tesseract Hindi/English OCR | MIT / Apache-2.0 |
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
- **Scanned Hindi quality is not yet measured.** Hindi/English Tesseract OCR is
  configured, but the English retrieval results above do not establish Hindi or
  Hinglish quality. Use the separate OCR and retrieval evaluations below with
  real, manually labelled scans.

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

The ingestion image installs Tesseract's Hindi (`hin`) and English (`eng`) language
packs. The legacy RapidOCR cache mount is retained but is not used by this OCR setup.

## Hindi, Hinglish, and scanned circulars

The existing bge-m3 embedder and multilingual reranker are retained. Questions go
to them in their original language; no cloud translation service is added.
Use Hindi (`इस परिपत्र की समय सीमा क्या है?`) or Hinglish
(`Is circular ki deadline kya hai?`) in the existing input box. Select **हिन्दी**,
**Hinglish**, **English**, or **Same as question** for the answer. Hinglish means
Hindi in Latin script mixed with English; it is not a separate OCR language pack.
Prompt instructions preserve citations, figures, dates, and identifiers.
Romanized Hindi spelling varies, so retrieval quality must be checked on your data.
The existing English full-text search remains; cross-script matching relies on
the multilingual dense retriever and reranker, not transliteration-aware SQL.

Both `/api/chat` and `/v1/chat/completions` accept an optional `language` field:
`auto` (default), `hi`, `hinglish`, or `en`. The latter field is a custom extension
to the OpenAI-compatible endpoint. Automatic empty-result messages detect Hindi
script; select Hinglish explicitly to get a Hinglish empty-result message.
Hindi filenames are preserved on upload.

### Enable OCR and reindex

From the repository root, rebuild the two changed services:

```bash
docker compose up -d --build api ingest
docker compose run --rm --no-deps --entrypoint tesseract ingest --list-langs
make reindex
```

The language list should include `hin` and `eng`. `make reindex` is necessary for
already indexed scans: the original file hash has not changed, so normal ingestion
would otherwise skip them. This reparses documents and recomputes their embeddings.
Default `OCR_LANGUAGES=hin,eng` applies to PDF and image ingestion. Normal digital
PDFs retain native text extraction. If a scan has a broken embedded text layer,
set `OCR_FORCE_FULL_PAGE=true` in `.env`, recreate the ingest service, and reindex.
Full-page OCR costs more CPU time; use the same setting during evaluation.

### Measure scanned Hindi performance

Use real scanned Hindi circulars with varied resolution, skew, tables, and mixed
Hindi/English content. Keep tuning and held-out evaluation circulars separate.
Keep the current English questions as a regression set. Templates below contain
**no benchmark evidence or measured scores** and deliberately refuse to run until
you replace their placeholders and remove `template:true`.

**1. OCR extraction quality.** Put PDFs in `corpus/`. For each labelled page, type
and manually verify its full text in reading order in a UTF-8 file under
`eval/references/`. Include headers, footers, punctuation, and table content as
represented in Docling's plain-text export; do not use the OCR output as ground truth.
Copy `eval/ocr.hindi.example.jsonl` to `eval/ocr.hindi.jsonl` and fill in actual PDF
paths, 1-based page numbers, and reference paths. Docker sees PDFs under `/data/`
and the evaluation directory under `/eval/`; relative reference paths are resolved
from the manifest directory.

```bash
docker compose run --rm --no-deps --entrypoint python ingest \
  /eval/evaluate_ocr.py /eval/ocr.hindi.jsonl --output /eval/ocr-results.json
```

The script uses the same Docling/Tesseract converter as ingestion and reports:

| Metric | Meaning |
|---|---|
| CER | Character edit distance / reference characters; lower is better |
| WER | Word edit distance / reference words; lower is better |
| Extraction seconds | Whole-document conversion time, including layout and OCR |

CER is Unicode-code-point based after NFC normalization and whitespace collapse,
not grapheme based. Hindi matras, nukta, digits, case, and punctuation are preserved.
Corpus CER/WER use total errors divided by total reference length; rates can exceed
1 when OCR inserts text. These measure Docling's extracted reading-order text, so
layout/serialization errors also count. The report includes predictions per page
for inspecting failures. Timings include first-use model loading; repeat warm runs
separately and record hardware, software versions, scan quality, and OCR settings.

**2. Retrieval quality in each language.** Copy
`eval/questions.hindi.example.jsonl` to `eval/questions.hindi.jsonl`. Write equivalent
English, Hindi, and Hinglish questions for each fact, labelled `en`, `hi`, and
`hinglish`. Use verified Hindi evidence in `expect_text` or `expect_any`, plus the
source filename. Optional `expect_page` matches the chunk's **starting page**; omit
it for cross-page chunks whose start differs from the evidence page.

```bash
python3 -m venv .venv
.venv/bin/pip install httpx
.venv/bin/python eval/evaluate.py eval/questions.hindi.jsonl --output eval/retrieval-results.json
.venv/bin/python eval/evaluate.py eval/questions.jsonl --output eval/english-regression.json
```

Reports include recall@k (the fraction of questions with at least one relevant
hit, also called hit@k), MRR, mean/p95 request latency, misses, and separate results
for every language, with and without reranking. All supplied text/source/page
constraints must hold in the same result. Old source-only labels still work but
measure document retrieval, not answer evidence. Phrase matching can penalize
OCR spelling errors; manually inspect misses against the scans before changing
labels. Do not silently weaken ground truth to match broken OCR.

**3. Answer quality.** Retrieval scores are not answer accuracy. Review generated
answers separately for factual correctness against the scan, supported citations,
correct numbers/dates, requested language/script, and abstention on unanswerable
questions. No automatic answer-quality score is claimed by these scripts.

No Hindi/Hinglish benchmark scores are published until real scans and checked
labels have been run through the stack. Do not reuse the earlier English scores.

### Focused checks

```bash
.venv/bin/pip install -r api/requirements.txt pytest
.venv/bin/pytest -q
```

These checks exercise language handling and streaming with mocked retrieval/model
responses, Unicode OCR metrics, and evaluation labels. They do not replace a live
Docker/GPU run or measure OCR/model quality.
