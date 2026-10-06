"""Index every document in one area folder.

    .venv/bin/python scripts/import_area.py income_tax          # queue and return
    .venv/bin/python scripts/import_area.py income_tax --wait   # queue, watch, report

Put the files in corpus/<area>/ first, e.g. corpus/income_tax/. The default area
(rbi) lives in the corpus root, so `import_area.py rbi` queues only root files.

Files already indexed and unchanged are skipped by the worker in a second, so
running this again after adding a few new files is safe and quick. Afterwards,
generate search questions for the new passages:

    .venv/bin/python scripts/gen_questions.py --area income_tax
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx

WORKER = os.getenv("INGEST_URL", "http://localhost:8082")
DEFAULT_AREA = os.getenv("DEFAULT_AREA", "rbi")
SUPPORTED = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt",
             ".png", ".jpg", ".jpeg", ".tiff"}


def clean_area(area: str) -> str:
    """Same rule the API uses, so the folder name and the stored area agree."""
    return re.sub(r"[^a-z0-9_]+", "_", area.strip().lower()).strip("_")


def list_files(corpus: Path, area: str) -> list[str]:
    """Worker-relative filenames for every supported file in the area."""
    folder = corpus if area == DEFAULT_AREA else corpus / area
    if not folder.is_dir():
        sys.exit(f"no folder {folder} - create it and put the documents there first")
    names = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED and not p.name.startswith("."):
            names.append(p.name if area == DEFAULT_AREA else f"{area}/{p.name}")
    return names


def queue(client: httpx.Client, names: list[str]) -> list[int]:
    ids = []
    for name in names:
        r = client.post(f"{WORKER}/ingest", json={"filename": name}, timeout=30)
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            print(f"  skipped {name}: {body['error']}")
            continue
        ids.append(body["job_id"])
        print(f"  queued  {name}")
    return ids


def wait(client: httpx.Client, every: float = 15.0) -> None:
    started = time.time()
    while True:
        q = client.get(f"{WORKER}/queue", timeout=30).json()["queue"]
        busy = q.get("queued", 0) + q.get("parsing", 0)
        mins = (time.time() - started) / 60
        print(f"  [{mins:5.1f} min] queued {q.get('queued', 0)}, reading {q.get('parsing', 0)}, "
              f"done {q.get('done', 0)}, errors {q.get('error', 0)}", flush=True)
        if busy == 0:
            return
        time.sleep(every)


def database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    env = {}
    if Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return (f"postgresql://{env.get('POSTGRES_USER', 'rag')}:{env.get('POSTGRES_PASSWORD', 'ragpass')}"
            f"@localhost:5433/{env.get('POSTGRES_DB', 'rag')}")


def report(ids: list[int]) -> None:
    """What happened to each job queued in this run."""
    try:
        import psycopg
        with psycopg.connect(database_url()) as conn, conn.cursor() as cur:
            cur.execute("SELECT filename, status, message FROM jobs WHERE id = ANY(%s) ORDER BY id", (ids,))
            rows = cur.fetchall()
    except Exception as exc:
        print(f"\n(could not read job results: {exc})")
        return
    ok = [r for r in rows if r[1] == "done"]
    bad = [r for r in rows if r[1] != "done"]
    print(f"\n{len(ok)} indexed, {len(bad)} not indexed")
    for f, status, msg in bad:
        print(f"  {status:6} {f}: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("area", help="area folder under corpus/, e.g. income_tax, gst, companies_act")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--wait", action="store_true", help="watch until done, then report each file")
    args = ap.parse_args()

    area = clean_area(args.area)
    if area != args.area:
        sys.exit(f"use the folder name '{area}' (lowercase letters, digits and underscores)")
    names = list_files(Path(args.corpus), area)
    if not names:
        sys.exit(f"no supported files found for area '{area}'")
    print(f"{len(names)} file(s) for area '{area}':")
    with httpx.Client() as client:
        ids = queue(client, names)
        if args.wait and ids:
            print("\nwatching the queue (large Acts can take a while; safe to Ctrl-C, work continues):")
            wait(client)
            report(ids)
    print(f"\nnext: .venv/bin/python scripts/gen_questions.py --area {area}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
