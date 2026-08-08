"""Ingestion worker: parse -> chunk -> embed -> store.

Usage (from the repo root, stack already up):
    docker compose run --rm ingest /data
    docker compose run --rm ingest /data --reindex     # force re-parse everything

Parsing uses Docling, which preserves layout, reading order, headings and tables
instead of flattening the page. Chunking is Docling's HybridChunker, which splits
on document structure and then packs to a token budget, so chunks stop at real
boundaries rather than mid-sentence. Chunking is the highest-leverage index-time
decision in the pipeline; if retrieval is bad, look here first.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path

import httpx
import psycopg
from psycopg.types.json import Json  # noqa: F401  (kept for future metadata use)

DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_URL = os.getenv("EMBED_URL", "http://embeddings:8081")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))

DOCLING_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".png", ".jpg", ".jpeg", ".tiff"}
PLAIN_EXT = {".md", ".markdown", ".txt", ".rst", ".csv", ".json", ".py", ".js", ".ts", ".go", ".java", ".sql"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plain_chunks(text: str, max_chars: int = 1800, overlap: int = 200) -> list[dict]:
    """Fallback splitter for plain text/markdown/code. Paragraph-aware."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 > max_chars and buf:
            out.append(buf)
            buf = buf[-overlap:] + "\n\n" + p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        out.append(buf)
    return [{"text": c, "heading": None, "page": None} for c in out]


LIST_MARKER = re.compile(r"^\s*[-*]?\s*\((?:[ivxlcdm]{1,7}|\d{1,3}|[a-zA-Z])\)\s")


def split_enumerated(text: str, max_chars: int = 1500, min_piece: int = 250) -> list[str]:
    """Split oversized chunks at their internal enumeration boundaries.

    Docling's HybridChunker keeps a structural unit whole, so definition sections
    and long enumerated clauses arrive as single 3000+ char chunks covering a dozen
    unrelated items. One vector then has to represent all of them, and the meaning
    of any single item is averaged away. Splitting at (i)/(1)/(a) markers gives each
    item its own sharply-focused embedding.
    """
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    pieces, buf = [], []
    for line in lines:
        if LIST_MARKER.match(line) and sum(len(x) + 1 for x in buf) >= min_piece:
            pieces.append("\n".join(buf))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        pieces.append("\n".join(buf))
    out = [p.strip() for p in pieces if p.strip()]
    return out if len(out) > 1 else [text]


def docling_chunks(path: Path) -> list[dict]:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    doc = converter.convert(str(path)).document
    chunker = HybridChunker(tokenizer="BAAI/bge-m3", max_tokens=MAX_TOKENS, merge_peers=True)

    chunks = []
    for ch in chunker.chunk(doc):
        # contextualize() prepends the heading path, which measurably improves
        # retrieval: an isolated chunk loses the section it belonged to.
        text = chunker.contextualize(chunk=ch)
        headings = getattr(ch.meta, "headings", None) or []
        page = None
        for item in getattr(ch.meta, "doc_items", []) or []:
            for prov in getattr(item, "prov", []) or []:
                if getattr(prov, "page_no", None):
                    page = prov.page_no
                    break
            if page:
                break
        chunks.append({"text": text, "heading": " > ".join(headings) or None, "page": page})
    return chunks


def extract(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext in PLAIN_EXT:
        return plain_chunks(path.read_text(encoding="utf-8", errors="ignore"))
    if ext in DOCLING_EXT:
        return docling_chunks(path)
    return []


def embed(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        r = client.post("/embed", json={"texts": batch, "batch_size": EMBED_BATCH}, timeout=600)
        r.raise_for_status()
        vectors.extend(r.json()["embeddings"])
    return vectors


def wait_for_embeddings(client: httpx.Client, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if client.get("/health", timeout=10).json().get("status") == "ok":
                return
        except Exception:
            pass
        print("[ingest] waiting for embedding service (first run downloads ~2.5 GB of weights)...")
        time.sleep(10)
    raise SystemExit("embedding service never became ready")


def ingest_file(conn: psycopg.Connection, client: httpx.Client, path: Path, reindex: bool) -> str:
    digest = sha256_file(path)
    with conn.cursor() as cur:
        cur.execute("SELECT id, sha256 FROM documents WHERE source_path = %s", (str(path),))
        row = cur.fetchone()
    if row and row[1] == digest and not reindex:
        return "skip"

    chunks = extract(path)
    if not chunks:
        return "empty"

    vectors = embed(client, [c["text"] for c in chunks])

    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE source_path = %s", (str(path),))
        cur.execute(
            "INSERT INTO documents (source_path, title, sha256, n_chunks) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (str(path), path.stem, digest, len(chunks)),
        )
        doc_id = cur.fetchone()[0]
        cur.executemany(
            "INSERT INTO chunks (doc_id, ord, heading, page, text, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [
                (doc_id, i, c["heading"], c["page"], c["text"], str(v))
                for i, (c, v) in enumerate(zip(chunks, vectors))
            ],
        )
    conn.commit()
    return f"ok ({len(chunks)} chunks)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="/data")
    ap.add_argument("--reindex", action="store_true", help="re-parse files even if unchanged")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not files:
        print(f"[ingest] no files under {root}")
        return 1

    client = httpx.Client(base_url=EMBED_URL)
    wait_for_embeddings(client)

    with psycopg.connect(DATABASE_URL) as conn:
        for n, path in enumerate(files, 1):
            started = time.time()
            try:
                status = ingest_file(conn, client, path, args.reindex)
            except Exception as exc:  # keep going; one bad PDF shouldn't kill the run
                conn.rollback()
                status = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[{n}/{len(files)}] {path.name} -> {status} ({time.time() - started:.1f}s)", flush=True)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents")
            docs = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunks")
            n_chunks = cur.fetchone()[0]
    print(f"[ingest] done. corpus: {docs} documents / {n_chunks} chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
