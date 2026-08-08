"""RAG API.

Query path:
    embed query
      -> hybrid_search()  : pgvector HNSW + Postgres full-text, fused with RRF  (top 40)
      -> cross-encoder rerank                                                    (top 6)
      -> grounded generation via Ollama, with inline [n] citations

The two-stage retrieve-then-rerank shape is deliberate: dense search alone misses
exact tokens (error codes, IDs, names), lexical search alone misses paraphrase, and
neither orders the survivors well enough for a small context window.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_URL = os.getenv("EMBED_URL", "http://embeddings:8081")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "40"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "6"))

SYSTEM_PROMPT = """You answer questions using only the numbered context passages provided.

Rules:
- Ground every factual claim in the passages. Cite them inline as [1], [2], and so on.
- If the passages do not contain the answer, say so plainly. Do not fill the gap from
  general knowledge, and do not guess.
- Quote exact figures, names and identifiers from the passages rather than paraphrasing them.
- Be concise. No preamble."""

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = AsyncConnectionPool(DATABASE_URL, min_size=1, max_size=8, open=False)
    await pool.open(wait=True, timeout=60)
    state["pool"] = pool
    state["http"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    yield
    await state["http"].aclose()
    await pool.close()


app = FastAPI(title="rag-api", lifespan=lifespan)


# ---------------------------------------------------------------- models

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=TOP_K_FINAL, ge=1, le=50)
    rerank: bool = True


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(default=TOP_K_FINAL, ge=1, le=20)
    temperature: float = 0.2


# ---------------------------------------------------------------- retrieval

async def embed_query(text: str) -> list[float]:
    r = await state["http"].post(f"{EMBED_URL}/embed", json={"texts": [text]})
    r.raise_for_status()
    return r.json()["embeddings"][0]


async def rerank(query: str, docs: list[str], top_k: int) -> list[dict]:
    r = await state["http"].post(
        f"{EMBED_URL}/rerank", json={"query": query, "documents": docs, "top_k": top_k}
    )
    r.raise_for_status()
    return r.json()["results"]


async def retrieve(query: str, top_k: int, use_rerank: bool = True) -> list[dict]:
    vec = await embed_query(query)
    async with state["pool"].connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, doc_id, source, heading, page, text, score "
                "FROM hybrid_search(%s::vector, %s, %s)",
                (str(vec), query, TOP_K_CANDIDATES),
            )
            rows = await cur.fetchall()

    candidates = [
        {
            "chunk_id": r[0], "doc_id": r[1], "source": r[2], "heading": r[3],
            "page": r[4], "text": r[5], "fusion_score": float(r[6]),
        }
        for r in rows
    ]
    if not candidates:
        return []
    if not use_rerank:
        return candidates[:top_k]

    ranked = await rerank(query, [c["text"] for c in candidates], top_k)
    out = []
    for item in ranked:
        c = dict(candidates[item["index"]])
        c["rerank_score"] = item["score"]
        out.append(c)
    return out


def build_context(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        loc = os.path.basename(p["source"])
        if p.get("page"):
            loc += f" p.{p['page']}"
        if p.get("heading"):
            loc += f" — {p['heading']}"
        blocks.append(f"[{i}] ({loc})\n{p['text']}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- generation

async def ollama_chat(messages: list[dict], temperature: float, stream: bool):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX},
    }
    if stream:
        return payload
    r = await state["http"].post(f"{OLLAMA_URL}/api/chat", json=payload)
    r.raise_for_status()
    return r.json()["message"]["content"]


async def stream_answer(query: str, passages: list[dict], temperature: float) -> AsyncIterator[str]:
    yield f"event: sources\ndata: {json.dumps([{k: v for k, v in p.items() if k != 'text'} for p in passages])}\n\n"

    if not passages:
        yield f"event: token\ndata: {json.dumps('Nothing in the indexed corpus matches that query.')}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {query}"},
    ]
    payload = await ollama_chat(messages, temperature, stream=True)
    async with state["http"].stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield f"event: token\ndata: {json.dumps(piece)}\n\n"
            if chunk.get("done"):
                break
    yield "event: done\ndata: {}\n\n"


# ---------------------------------------------------------------- endpoints

@app.get("/health")
async def health() -> dict:
    out = {"api": "ok"}
    try:
        async with state["pool"].connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT count(*) FROM documents")
                docs = (await cur.fetchone())[0]
                await cur.execute("SELECT count(*) FROM chunks")
                chunks = (await cur.fetchone())[0]
        out["db"] = {"documents": docs, "chunks": chunks}
    except Exception as exc:
        out["db"] = f"error: {exc}"
    for name, url in (("embeddings", f"{EMBED_URL}/health"), ("ollama", f"{OLLAMA_URL}/api/tags")):
        try:
            r = await state["http"].get(url, timeout=5)
            out[name] = "ok" if r.status_code == 200 else f"http {r.status_code}"
        except Exception as exc:
            out[name] = f"error: {type(exc).__name__}"
    return out


@app.post("/api/search")
async def api_search(req: SearchRequest) -> dict:
    t0 = time.time()
    results = await retrieve(req.query, req.top_k, req.rerank)
    return {"query": req.query, "took_ms": int((time.time() - t0) * 1000), "results": results}


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> StreamingResponse:
    passages = await retrieve(req.query, req.top_k)
    return StreamingResponse(
        stream_answer(req.query, passages, req.temperature),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# OpenAI-compatible surface, so Open WebUI / LibreChat / any SDK can point at this
# service and get RAG answers instead of raw model answers.
@app.get("/v1/models")
async def list_models() -> dict:
    return {"object": "list", "data": [{"id": "rag", "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
async def openai_chat(body: dict) -> dict:
    messages = body.get("messages") or []
    user_turns = [m for m in messages if m.get("role") == "user"]
    if not user_turns:
        raise HTTPException(400, "no user message")
    query = user_turns[-1]["content"]

    passages = await retrieve(query, TOP_K_FINAL)
    prompt = f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {query}"
    answer = await ollama_chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        float(body.get("temperature", 0.2)),
        stream=False,
    )
    if passages:
        cited = "\n".join(
            f"[{i}] {os.path.basename(p['source'])}" + (f" p.{p['page']}" if p.get("page") else "")
            for i, p in enumerate(passages, 1)
        )
        answer = f"{answer}\n\n---\n{cited}"
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "rag",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
    }


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    with open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8") as fh:
        return fh.read()
