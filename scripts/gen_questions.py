"""Generate practitioner-style questions for every indexed passage.

For each passage, the local model writes a few questions a compliance officer or
banker would ask that the passage answers, in everyday industry wording rather
than the regulation's formal phrasing. Their embeddings go into chunk_questions,
which hybrid_search uses as a third retrieval list.

Runs from the host against the running stack. Resumable: passages that already
have questions are skipped, so it is safe to stop with Ctrl-C and start again.

    .venv/bin/pip install "psycopg[binary]" httpx
    .venv/bin/python scripts/gen_questions.py --limit 5     # try a few first
    .venv/bin/python scripts/gen_questions.py               # then all

Re-indexing documents deletes their questions automatically; run this again after.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx
import psycopg

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED = os.getenv("EMBED_URL", "http://localhost:8081")
PER_CHUNK = 3
MIN_CHARS = 60

# The prompt deliberately names no specific jargon. Listing words taken from
# failing test questions would teach the index the answers to the test.
PROMPT = """You are helping index Indian laws and regulations so that practitioners can search them.

The passage below comes from {law}, in the document titled: {title}

Write {n} short questions that {askers} might realistically ask, which this passage directly answers.

Rules:
- Each question must make sense on its own. Name the specific subject (the topic, type of company or product), never "these directions" or "this passage".
- Phrase them the way practitioners talk in everyday work. Their wording often differs from the regulation's formal language, so do not simply copy the passage's phrases.
- Only ask questions the passage actually answers. Do not invent facts.
- Keep each question under 25 words.
- Output only the questions, one per line, with no numbering and no other text.

Passage:
\"\"\"{text}\"\"\""""


def database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env = {}
    for line in Path(".env").read_text().splitlines() if Path(".env").exists() else []:
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    user = env.get("POSTGRES_USER", "rag")
    pw = env.get("POSTGRES_PASSWORD", "ragpass")
    db = env.get("POSTGRES_DB", "rag")
    return f"postgresql://{user}:{pw}@localhost:5433/{db}"


def worth_asking(text: str) -> bool:
    """Skip fragments that cannot answer anything: headings, tables of contents."""
    if len(text) < MIN_CHARS:
        return False
    if "Table of Contents" in text or text.count("....") >= 3:
        return False
    return True


# Who realistically asks about each body of law, and what to call it. The
# generated questions should sound like the people who will search that area:
# a CA asks about income tax differently from an NBFC compliance officer.
AREAS = {
    "rbi": ("RBI regulations", "a compliance officer, banker or company secretary"),
    "income_tax": ("Indian income-tax law", "a chartered accountant, tax practitioner or taxpayer"),
    "gst": ("Indian GST law", "a chartered accountant, GST practitioner or business owner"),
    "companies_act": ("Indian company law", "a company secretary, chartered accountant or company director"),
    "llp": ("Indian LLP law", "a chartered accountant or an LLP partner"),
}


def area_context(area: str | None) -> tuple[str, str]:
    if area in AREAS:
        return AREAS[area]
    law = f"Indian {area.replace('_', ' ')} rules" if area else "Indian regulations"
    return (law, "a compliance professional, chartered accountant or company secretary")


def document_title(source_path: str) -> str:
    name = Path(source_path).stem
    name = re.sub(r"^\d+[_ -]*", "", name)
    name = re.sub(r"^SUP[-_ ]", "", name)
    return re.sub(r"[-_]+", " ", name).strip() or "NBFC regulations"


def parse_questions(raw: str, n: int = PER_CHUNK) -> list[str]:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    out = []
    for line in raw.splitlines():
        q = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)]|Q\d*[:.)])\s*", "", line).strip().strip('"')
        if len(q) < 12 or len(q.split()) > 40:
            continue
        if not q.endswith("?"):
            continue
        if q.lower() not in (x.lower() for x in out):
            out.append(q)
    return out[:n]


def generate(client: httpx.Client, model: str, text: str, title: str, area: str | None = None) -> list[str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(n=PER_CHUNK, text=text[:3500], title=title,
                                                       law=area_context(area)[0], askers=area_context(area)[1])}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_ctx": 4096},
    }
    r = client.post(f"{OLLAMA}/api/chat", json=body, timeout=300)
    if r.status_code == 400:
        # Older Ollama builds reject the "think" field; fall back to the inline switch.
        body.pop("think")
        body["messages"][0]["content"] = "/no_think\n" + body["messages"][0]["content"]
        r = client.post(f"{OLLAMA}/api/chat", json=body, timeout=300)
    r.raise_for_status()
    return parse_questions(r.json()["message"]["content"])


def embed(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    r = client.post(f"{EMBED}/embed", json={"texts": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only N passages")
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen3:8b"))
    ap.add_argument("--show", action="store_true", help="print every generated question")
    ap.add_argument("--sample", type=int, default=None, help="process N random passages")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--area", default=None, help="only passages from this area, e.g. income_tax")
    args = ap.parse_args()

    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.text, d.source_path, d.area FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                WHERE NOT EXISTS (SELECT 1 FROM chunk_questions q WHERE q.chunk_id = c.id)
                  AND (%s::text IS NULL OR d.area = %s::text)
                ORDER BY c.id
            """, (args.area, args.area))
            todo = [(i, t, document_title(sp), a) for i, t, sp, a in cur.fetchall() if worth_asking(t)]
        if args.sample:
            import random
            todo = random.Random(args.seed).sample(todo, min(args.sample, len(todo)))
        if args.limit:
            todo = todo[: args.limit]
        print(f"{len(todo)} passages to process with {args.model}\n")
        if not todo:
            return 0

        client = httpx.Client()
        started, made, failed = time.time(), 0, 0
        for n, (chunk_id, text, title, area) in enumerate(todo, 1):
            try:
                qs = generate(client, args.model, text, title, area)
                if qs:
                    vecs = embed(client, qs)
                    with conn.cursor() as cur:
                        cur.executemany(
                            "INSERT INTO chunk_questions (chunk_id, question, embedding) "
                            "VALUES (%s, %s, %s)",
                            [(chunk_id, q, str(v)) for q, v in zip(qs, vecs)],
                        )
                    conn.commit()
                    made += len(qs)
                    if args.show:
                        print(f"  chunk {chunk_id} [{title}]:")
                        print(f"    page: {' '.join(text.split())[:160]}...")
                        for q in qs:
                            print(f"    - {q}")
            except Exception as exc:
                conn.rollback()
                failed += 1
                print(f"  chunk {chunk_id} failed: {type(exc).__name__}: {exc}")

            if n % 25 == 0 or n == len(todo):
                rate = (time.time() - started) / n
                left = rate * (len(todo) - n)
                print(f"[{n}/{len(todo)}] {made} questions, {failed} failed, "
                      f"{rate:.1f}s/passage, ~{left/60:.0f} min left", flush=True)

    print(f"\ndone: {made} questions for {len(todo) - failed} passages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
