"""Ingestion worker with a serial job queue.

Jobs live in the database, not in memory. A single background thread claims one
job at a time and runs it to completion before touching the next. Two consequences:
parsing never competes with itself for CPU, and a container restart no longer
loses the queue -- anything left mid-flight is picked up again.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from pathlib import Path

import httpx
import psycopg
from fastapi import FastAPI
from pydantic import BaseModel

from ingest import DATABASE_URL, EMBED_URL, ingest_file, wait_for_embeddings

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
POLL_SECONDS = 3
app = FastAPI(title="rag-ingest-worker")


class IngestRequest(BaseModel):
    filename: str
    reindex: bool = False


def set_job(job_id, status, message=None, n_chunks=0):
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("UPDATE jobs SET status=%s, message=%s, n_chunks=%s, "
                    "updated_at=now() WHERE id=%s", (status, message, n_chunks, job_id))
        conn.commit()


def claim_next():
    """Atomically take the oldest queued job."""
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("""
            UPDATE jobs SET status='parsing', updated_at=now()
            WHERE id = (SELECT id FROM jobs WHERE status='queued'
                        ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED)
            RETURNING id, filename, reindex
        """)
        row = cur.fetchone()
        conn.commit()
    return row


def process(job_id, filename, reindex):
    try:
        client = httpx.Client(base_url=EMBED_URL)
        wait_for_embeddings(client)
        with psycopg.connect(DATABASE_URL) as conn:
            result = ingest_file(conn, client, DATA_DIR / filename, reindex)
        n = 0
        if "(" in result:
            try:
                n = int(result.split("(")[1].split()[0])
            except (IndexError, ValueError):
                n = 0
        if result.startswith("ok"):
            set_job(job_id, "done", f"indexed {n} passages", n)
        elif result == "skip":
            set_job(job_id, "done", "already indexed, unchanged")
        elif result == "empty":
            set_job(job_id, "error", "no readable text found")
        else:
            set_job(job_id, "error", result[:400])
    except Exception as exc:
        traceback.print_exc()
        set_job(job_id, "error", f"{type(exc).__name__}: {exc}"[:400])


def worker_loop():
    print("[worker] serial queue started", flush=True)
    while True:
        try:
            job = claim_next()
        except Exception as exc:
            print(f"[worker] cannot reach db: {exc}", flush=True)
            time.sleep(POLL_SECONDS); continue
        if not job:
            time.sleep(POLL_SECONDS); continue
        job_id, filename, reindex = job
        started = time.time()
        print(f"[worker] start {filename}", flush=True)
        process(job_id, filename, bool(reindex))
        print(f"[worker] done {filename} in {time.time()-started:.1f}s", flush=True)


@app.on_event("startup")
def startup():
    try:
        with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            cur.execute("UPDATE jobs SET status='queued' WHERE status='parsing'")
            conn.commit()
    except Exception as exc:
        print(f"[worker] requeue failed: {exc}", flush=True)
    threading.Thread(target=worker_loop, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/queue")
def queue():
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT status, count(*) FROM jobs GROUP BY status")
        return {"queue": dict(cur.fetchall())}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if not (DATA_DIR / req.filename).exists():
        return {"error": f"{req.filename} not found"}
    with psycopg.connect(DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO jobs (filename, status, reindex) "
                    "VALUES (%s,'queued',%s) RETURNING id", (req.filename, req.reindex))
        job_id = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM jobs WHERE status='queued' AND id < %s", (job_id,))
        ahead = cur.fetchone()[0]
        conn.commit()
    return {"job_id": job_id, "status": "queued", "ahead_in_queue": ahead}
