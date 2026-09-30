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

import hashlib
import hmac
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import httpx
import re
import shutil

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (HTMLResponse, JSONResponse,
                               RedirectResponse, StreamingResponse)
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
- Understand English, Hindi (Devanagari), and Hinglish (Hindi written in Latin script,
  possibly mixed with English). Passages may be in a different language from the question.
- Unless a response language is specified, answer in the question's language and script:
  Hindi in Devanagari, Hinglish in Latin script, and English in English.
- Preserve the original numbers, dates, regulation identifiers, and [n] citations in every language.
- Be concise. No preamble."""

# v2 replaces the final "Be concise" rule. Compliance answers are judged on what
# they leave out: an answer that states the general rule but drops an exception
# is wrong in practice even when every sentence in it is true. So v2 asks for
# every condition and exception the passages give, kept short point by point.
COMPLETENESS_RULES = """- If the passages answer the question, start with the direct answer in one sentence.
- If they do not answer it, say so in the first sentence, before anything else. Never
  open with "yes" or "no" unless the passages directly settle that exact question.
- Then state every condition, exception, limit, threshold and carve-out that the
  passages give on this question, including ones that seem minor. When a rule has
  numbered sub-clauses, provisos or exceptions, cover each one. Never drop one to
  make the answer shorter.
- When the passages treat different cases differently (types of company, deposit,
  customer, amount), cover each case separately.
- Keep each point short. No preamble, and nothing the passages do not support."""

PROMPTS = {
    "v1": SYSTEM_PROMPT,
    "v2": SYSTEM_PROMPT.replace("- Be concise. No preamble.", COMPLETENESS_RULES),
}
assert PROMPTS["v2"] != PROMPTS["v1"], "v2 rule replacement failed"
PromptVersion = Literal["v1", "v2"]
DEFAULT_PROMPT = os.getenv("PROMPT_VERSION", "v1")

ResponseLanguage = Literal["auto", "en", "hi", "hinglish"]
LANGUAGE_RULES = {
    "auto": "Follow the question's language and script.",
    "en": "Write the answer in English.",
    "hi": ("Write the answer in Hindi using Devanagari script, but keep every regulatory "
           "term, condition, figure and name in English exactly as the passages state them "
           "(for example renew, fresh deposit, matured deposit, depositor's consent, NOF). "
           "Use Hindi for the explanation around those terms. Never change a 'cannot' into a 'can'."),
    "hinglish": ("Write the answer in Hinglish using only Latin (English) letters, never Devanagari. "
                 "Use Hindi only for connecting words (hai, ka, ke liye, nahi, lekin, agar, jab tak). "
                 "Keep every regulatory term, condition, figure and name in English exactly as the "
                 "passages state them. If unsure how to say something in Hinglish, say it in English."),
}


def answer_prompt(language: str = "auto", version: str | None = None) -> str:
    if not isinstance(language, str) or language not in LANGUAGE_RULES:
        raise HTTPException(422, "language must be auto, en, hi, or hinglish")
    version = version or DEFAULT_PROMPT
    if version not in PROMPTS:
        raise HTTPException(422, "prompt_version must be v1 or v2")
    return PROMPTS[version] + "\nResponse language: " + LANGUAGE_RULES[language]


# Hindi words as people type them in English letters. Hinglish looks like English to
# a model reading the script alone, and with English passages it tends to answer in
# English, so "auto" is resolved in code before the model sees the question.
# Words that are also common English ("to", "me", "hi", "par", "so", "the") are left
# out, so an English question is not mistaken for Hinglish.
HINGLISH_WORDS = {
    "hai", "hain", "ho", "hoga", "hogi", "honge", "hona", "hota", "hoti", "hote", "tha", "thi", "the_",
    "kya", "kyun", "kyon", "kaise", "kab", "kahan", "kaun", "kitna", "kitni", "kitne", "konsa", "kaunsa",
    "ka", "ki", "ke", "ko", "se", "mein", "mai", "pe", "tak", "liye", "wala", "wali", "wale",
    "aur", "ya", "bhi", "nahi", "nahin", "mat", "sirf", "bas", "agar", "toh", "phir", "lekin", "par_",
    "chahiye", "sakta", "sakti", "sakte", "karna", "karni", "karne", "karein", "karo", "kare", "kar",
    "lena", "leni", "lene", "dena", "deni", "dene", "milta", "milti", "milega", "raha", "rahi", "rahe",
    "hum", "humein", "hamara", "hamari", "aap", "aapka", "apna", "apne", "apni", "unka", "uska", "iska",
    "yeh", "ye", "woh", "wo", "koi", "kuch", "sab", "zyada", "kam", "pehle", "baad", "abhi", "matlab",
    "paisa", "paise", "saal", "mahine", "karod", "wapas", "nikal", "jama", "mukable",
}
HINGLISH_WORDS = {w for w in HINGLISH_WORDS if not w.endswith("_")}


def detect_language(query: str) -> str:
    """'hi' for Devanagari, 'hinglish' for Hindi in Latin letters, otherwise 'en'."""
    if re.search(r"[\u0900-\u097f]", query):
        return "hi"
    words = re.findall(r"[a-z]+", query.lower())
    if not words:
        return "en"
    hits = sum(1 for w in words if w in HINGLISH_WORDS)
    # Two Hindi words, or a clear share of a short question, is enough; one stray
    # match in a long English question is not.
    if hits >= 2 and hits / len(words) >= 0.15:
        return "hinglish"
    return "en"


def resolve_language(query: str, language: str) -> str:
    return detect_language(query) if language == "auto" else language


def answer_language(query: str, language: str) -> str:
    """The language the answer is written in.

    Detected Hindi and Hinglish are answered in English. Tested on this model,
    writing regulatory answers in either damaged the meaning: a Hinglish answer
    reversed a rule ("cannot accept fresh deposits" became "can"), and a Hindi
    answer turned "sixty months" into a wrong word that reads as a different
    figure. The model understands both languages well; it is writing them that
    fails. The regulations exist only in English, so a correct English answer
    beats a damaged translation. Choosing Hindi or Hinglish explicitly still
    asks for it, with a note to verify against the English source.
    """
    resolved = resolve_language(query, language)
    return "en" if language == "auto" and resolved in ("hi", "hinglish") else resolved


# Fixed, pre-written notes. They are shown as written and never pass through the
# model, so they cannot be mistranslated.
ENGLISH_ON_PURPOSE = {
    "hi": "(यह उत्तर अंग्रेज़ी में दिया गया है, क्योंकि RBI के मूल नियम अंग्रेज़ी में हैं "
          "और अनुवाद से अर्थ बदल सकता है।)\n\n",
    "hinglish": "(Yeh answer English mein hai, kyunki RBI ke rules English mein hain "
                "aur translation se meaning badal sakta hai.)\n\n",
}
VERIFY_TRANSLATION = {
    "hi": "\n\n(यह अनुवाद मशीन द्वारा किया गया है। कृपया उद्धृत अंग्रेज़ी स्रोत से इसकी पुष्टि करें।)",
    "hinglish": "\n\n(Yeh translation machine ka hai. Please cited English source se verify karein.)",
}


def no_match(query: str, language: str) -> str:
    language = resolve_language(query, language)
    if language == "hi":
        return "इंडेक्स किए गए दस्तावेज़ों में इस प्रश्न से संबंधित जानकारी नहीं मिली।"
    if language == "hinglish":
        return "Indexed documents mein is sawaal se judi jaankari nahi mili."
    return "Nothing in the indexed corpus matches that query."

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
    area: str | None = None


class ChatRequest(BaseModel):
    query: str
    top_k: int = Field(default=TOP_K_FINAL, ge=1, le=20)
    temperature: float = 0.2
    language: ResponseLanguage = "auto"
    area: str | None = None
    prompt_version: PromptVersion | None = None
    think: bool | None = None


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


DEFAULT_AREA = os.getenv("DEFAULT_AREA", "rbi")
AREA_NAME = re.compile(r"[^a-z0-9_]+")


def clean_area(area: str | None) -> str | None:
    """None, "" and "all" mean every area. Anything else becomes a safe folder name."""
    if not area or area.strip().lower() == "all":
        return None
    return AREA_NAME.sub("_", area.strip().lower()).strip("_") or None


async def _candidates(q: str, area: str | None = None) -> list[dict]:
    vec = await embed_query(q)
    async with state["pool"].connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT chunk_id, doc_id, source, heading, page, text, score "
                "FROM hybrid_search(%s::vector, %s, %s, 60, %s::text)",
                (str(vec), q, TOP_K_CANDIDATES, clean_area(area)),
            )
            rows = await cur.fetchall()
    return [{"chunk_id": r[0], "doc_id": r[1], "source": r[2], "heading": r[3],
             "page": r[4], "text": r[5], "fusion_score": float(r[6])} for r in rows]


async def retrieve(query: str, top_k: int, use_rerank: bool = True,
                   area: str | None = None) -> list[dict]:
    # The question is used exactly as typed. Layer abbreviations are normalised
    # in the documents at index time instead (see ingest.normalize_layers), because
    # every attempt to patch the question helped one kind of page and hurt another.
    cands = await _candidates(query, area)
    if not cands:
        return []
    if not use_rerank:
        return cands[:top_k]
    ranked = await rerank(query, [c["text"] for c in cands], top_k)
    out = []
    for item in ranked:
        c = dict(cands[item["index"]])
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

# Qwen3 reasons silently before answering unless told not to. Measured on this
# machine, that hidden reasoning was most of every answer's wait (~2,000 characters
# for a one-sentence question). Whether it improves regulatory answers enough to be
# worth the time is measured, not assumed, so it is switchable per request.
LLM_THINK = os.getenv("LLM_THINK", "true").lower() in ("1", "true", "yes")
# Hard ceiling on generated tokens, hidden reasoning included. Without it a model
# stuck repeating itself runs until the client gives up, which is what produced the
# five-minute timeouts. Real answers here use a few hundred tokens.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "3072"))
CUT_SHORT = "\n\n[Answer cut short at the length limit. Try a narrower question.]"


async def ollama_chat(messages: list[dict], temperature: float, stream: bool,
                      think: bool | None = None):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": stream,
        "think": LLM_THINK if think is None else think,
        "options": {"temperature": temperature, "num_ctx": NUM_CTX,
                    "num_predict": LLM_MAX_TOKENS},
    }
    if stream:
        return payload
    r = await state["http"].post(f"{OLLAMA_URL}/api/chat", json=payload)
    r.raise_for_status()
    return r.json()["message"]["content"]


async def stream_answer(query: str, passages: list[dict], temperature: float, language: str = "auto",
                        prompt_version: str | None = None,
                        think: bool | None = None) -> AsyncIterator[str]:
    yield f"event: sources\ndata: {json.dumps([{k: v for k, v in p.items() if k != 'text'} for p in passages])}\n\n"

    if not passages:
        yield f"event: token\ndata: {json.dumps(no_match(query, language))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    asked_in = resolve_language(query, language)
    written_in = answer_language(query, language)
    # Held back until the answer's first words, so the page keeps showing its
    # "Reasoning..." status while the model thinks instead of clearing it early.
    pending_note = ENGLISH_ON_PURPOSE[asked_in] if written_in == "en" and asked_in in ENGLISH_ON_PURPOSE else ""
    messages = [
        {"role": "system", "content": answer_prompt(written_in, prompt_version)},
        {"role": "user", "content": f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {query}"},
    ]
    payload = await ollama_chat(messages, temperature, stream=True, think=think)
    async with state["http"].stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
        resp.raise_for_status()
        announced = False
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            message = chunk.get("message", {})
            # The model's hidden reasoning streams first, with no answer text.
            # Tell the page once, so the wait reads as work rather than a hang.
            if message.get("thinking") and not announced:
                announced = True
                yield f"event: status\ndata: {json.dumps('thinking')}\n\n"
            piece = message.get("content", "")
            if piece:
                if pending_note:
                    yield f"event: token\ndata: {json.dumps(pending_note)}\n\n"
                    pending_note = ""
                yield f"event: token\ndata: {json.dumps(piece)}\n\n"
            if chunk.get("done"):
                # Say so when the length ceiling stopped the answer, rather than
                # letting a truncated answer look complete.
                if chunk.get("done_reason") == "length":
                    yield f"event: token\ndata: {json.dumps(CUT_SHORT)}\n\n"
                break
    if written_in in VERIFY_TRANSLATION:
        yield f"event: token\ndata: {json.dumps(VERIFY_TRANSLATION[written_in])}\n\n"
    yield "event: done\ndata: {}\n\n"


# ---------------------------------------------------------------- endpoints

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
SESSION_SECRET = os.getenv("SESSION_SECRET", "rag-dev-secret")
OPEN_PATHS = {"/login", "/api/login", "/health", "/api/search"}


def session_token() -> str:
    return hmac.new(SESSION_SECRET.encode(), b"session-v1", hashlib.sha256).hexdigest()


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    # No password set = wide open. That is fine on your own machine and
    # unacceptable the moment this is reachable from the internet.
    if not APP_PASSWORD:
        return await call_next(request)
    if request.url.path in OPEN_PATHS:
        return await call_next(request)
    if hmac.compare_digest(request.cookies.get("rag_session", ""), session_token()):
        return await call_next(request)
    if request.url.path.startswith(("/api", "/v1")):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return RedirectResponse("/login", status_code=302)


@app.post("/api/login")
async def login(body: dict) -> JSONResponse:
    if not APP_PASSWORD or not hmac.compare_digest(
            str(body.get("password", "")), APP_PASSWORD):
        return JSONResponse({"ok": False}, status_code=401)
    resp = JSONResponse({"ok": True})
    resp.set_cookie("rag_session", session_token(), httponly=True,
                    samesite="lax", max_age=60 * 60 * 24 * 30)
    return resp


@app.get("/login", response_class=HTMLResponse)
async def login_page() -> str:
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Sign in</title>
<style>
body{margin:0;height:100vh;display:grid;place-items:center;background:#161a20;color:#dde3ea;
     font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.box{width:300px;text-align:center}
h1{font:600 13px/1 ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase;
   color:#8b9aa8;margin:0 0 20px}
input{width:100%;background:#1e242c;border:1px solid #2c343f;border-radius:6px;color:#dde3ea;
      padding:12px 14px;font-size:15px;margin-bottom:10px}
input:focus{outline:2px solid #5ec8c0;outline-offset:1px}
button{width:100%;background:#5ec8c0;color:#0d1114;border:0;border-radius:6px;padding:12px;
       font:600 14px sans-serif;cursor:pointer}
#err{color:#e07a6a;font-size:13px;height:18px;margin-top:10px}
</style></head><body><div class="box">
<h1>Document intelligence</h1>
<input type="password" id="p" placeholder="Password" autofocus>
<button onclick="go()">Sign in</button>
<div id="err"></div>
</div><script>
async function go(){
  const r = await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({password:document.getElementById('p').value})});
  if(r.ok) location.href='/'; else document.getElementById('err').textContent='Wrong password';
}
document.getElementById('p').addEventListener('keydown',e=>{if(e.key==='Enter')go()});
</script></body></html>"""


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
    results = await retrieve(req.query, req.top_k, req.rerank, req.area)
    return {"query": req.query, "took_ms": int((time.time() - t0) * 1000), "results": results}


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> StreamingResponse:
    passages = await retrieve(req.query, req.top_k, area=req.area)
    return StreamingResponse(
        stream_answer(req.query, passages, req.temperature, req.language, req.prompt_version, req.think),
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
    language = body.get("language", "auto")
    system_prompt = answer_prompt(answer_language(query, language))

    passages = await retrieve(query, TOP_K_FINAL)
    prompt = f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {query}"
    answer = await ollama_chat(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        float(body.get("temperature", 0.2)),
        stream=False,
    ) if passages else no_match(query, language)
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


UPLOAD_DIR = "/data"
INGEST_URL = os.getenv("INGEST_URL", "http://ingest:8082")
SAFE_NAME = re.compile(r"[^A-Za-z0-9\u0900-\u097f._ -]")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), area: str = Form(DEFAULT_AREA)) -> dict:
    name = SAFE_NAME.sub("_", os.path.basename(file.filename or "upload"))
    area = clean_area(area) or DEFAULT_AREA
    # Default-area files stay in the corpus root, where the existing library
    # already lives; every other area gets its own folder.
    rel = name if area == DEFAULT_AREA else f"{area}/{name}"
    os.makedirs(os.path.dirname(os.path.join(UPLOAD_DIR, rel)), exist_ok=True)
    with open(os.path.join(UPLOAD_DIR, rel), "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        r = await state["http"].post(f"{INGEST_URL}/ingest",
                                     json={"filename": rel}, timeout=30)
        return {"filename": rel, "area": area, **r.json()}
    except Exception as exc:
        return {"filename": rel, "area": area, "error": f"saved, but indexing failed: {exc}"}


@app.get("/api/areas")
async def areas() -> dict:
    async with state["pool"].connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT area, count(*), coalesce(sum(n_chunks), 0) "
                              "FROM documents GROUP BY area ORDER BY area")
            rows = await cur.fetchall()
    return {"default": DEFAULT_AREA,
            "areas": [{"area": r[0], "documents": r[1], "passages": r[2]} for r in rows]}


@app.get("/api/jobs")
async def jobs() -> dict:
    async with state["pool"].connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, filename, status, message, n_chunks "
                              "FROM jobs ORDER BY id DESC LIMIT 20")
            rows = await cur.fetchall()
    keys = ("id", "filename", "status", "message", "n_chunks")
    return {"jobs": [dict(zip(keys, r)) for r in rows]}


@app.delete("/api/jobs/{job_id}")
async def dismiss_job(job_id: int) -> dict:
    # Only finished or failed jobs can be dismissed. A queued or running job is
    # still owned by the worker, and removing it would silently drop the file.
    async with state["pool"].connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM jobs WHERE id=%s AND status IN ('error','done') "
                              "RETURNING id", (job_id,))
            row = await cur.fetchone()
    if not row:
        raise HTTPException(409, "job is still running or does not exist")
    return {"dismissed": job_id}


@app.get("/api/documents")
async def documents() -> dict:
    async with state["pool"].connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, source_path, n_chunks, ingested_at, area "
                              "FROM documents ORDER BY id DESC")
            rows = await cur.fetchall()
    return {"documents": [{"id": r[0], "name": os.path.basename(r[1]),
                           "chunks": r[2], "added": r[3].strftime("%d %b %H:%M"),
                           "area": r[4]}
                          for r in rows]}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int) -> dict:
    async with state["pool"].connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM documents WHERE id=%s RETURNING source_path",
                              (doc_id,))
            row = await cur.fetchone()
    if not row:
        raise HTTPException(404, "no such document")
    try:
        os.remove(row[0])
    except OSError:
        pass
    return {"deleted": doc_id}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    with open(os.path.join(os.path.dirname(__file__), "index.html"), encoding="utf-8") as fh:
        return fh.read()
