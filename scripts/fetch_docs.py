"""Download the documents listed in docs_links.txt into their area folders.

    .venv/bin/python scripts/fetch_docs.py                # download only
    .venv/bin/python scripts/fetch_docs.py --index        # download, index, generate questions

Every download is checked before it is kept: government sites often answer a
script with an HTML error or login page instead of the PDF, and indexing that
would put a web page's text into Vidhi. Anything that is not a real PDF is
discarded and reported, with what to do instead.

Files already present are left alone, so the script is safe to re-run after
filling in more links.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
           "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
MIN_BYTES = 20_000          # an Act is far larger; anything smaller is an error page
AREA_NAME = re.compile(r"^[a-z0-9_]+$")


def read_links(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3:
            sys.exit(f"{path}:{n}: expected 'area | filename | url', got: {raw}")
        area, name, url = parts
        if not AREA_NAME.match(area):
            sys.exit(f"{path}:{n}: area '{area}' must be lowercase letters, digits, underscores")
        if "/" in name or "\\" in name or not name.lower().endswith(".pdf"):
            sys.exit(f"{path}:{n}: filename '{name}' must be a plain name ending in .pdf")
        rows.append((area, name, url))
    return rows


def looks_like_pdf(path: Path) -> str | None:
    """None if it is a usable PDF, otherwise the reason it is not."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(1024)
    if not head.startswith(b"%PDF"):
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            return "the site returned a web page, not the PDF (often a block or login page)"
        return "the file is not a PDF"
    if size < MIN_BYTES:
        return f"only {size} bytes - too small to be a full Act"
    return None


def download(client: httpx.Client, url: str, dest: Path) -> str | None:
    """Save url to dest if it is a real PDF. Returns None on success, else the reason."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        last = "unknown error"
        for attempt in range(3):
            try:
                with client.stream("GET", url) as r:
                    if r.status_code != 200:
                        last = f"HTTP {r.status_code}"
                        continue
                    with tmp_path.open("wb") as fh:
                        for chunk in r.iter_bytes(1 << 16):
                            fh.write(chunk)
                problem = looks_like_pdf(tmp_path)
                if problem:
                    return problem
                tmp_path.replace(dest)
                return None
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
        return last
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", default="docs_links.txt")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--index", action="store_true",
                    help="after downloading, index each area and generate search questions")
    args = ap.parse_args()

    rows = read_links(Path(args.links))
    corpus = Path(args.corpus)
    got, have, todo, failed = [], [], [], []
    client = httpx.Client(headers={"User-Agent": BROWSER}, follow_redirects=True,
                          timeout=httpx.Timeout(180.0, connect=30.0))
    for area, name, url in rows:
        dest = corpus / area / name
        if dest.exists() and not looks_like_pdf(dest):
            have.append((area, name)); print(f"  have     {area}/{name}")
            continue
        if not url:
            todo.append((area, name)); print(f"  no link  {area}/{name}")
            continue
        print(f"  fetching {area}/{name} ...", flush=True)
        problem = download(client, url, dest)
        if problem:
            failed.append((area, name, problem)); print(f"  FAILED   {area}/{name}: {problem}")
        else:
            got.append((area, name)); print(f"  saved    {area}/{name} ({dest.stat().st_size // 1024} KB)")

    print(f"\n{len(got)} downloaded, {len(have)} already present, "
          f"{len(todo)} without a link, {len(failed)} failed")
    if failed:
        print("\nFor each failure: open the link in your browser, download the PDF yourself,")
        print("and save it with exactly the filename shown into the area folder, e.g.")
        area, name, _ = failed[0]
        print(f"    corpus/{area}/{name}")
        print("then re-run this script; it will find the file and skip the download.")

    if args.index:
        areas = sorted({a for a, _ in got + have})
        for area in areas:
            print(f"\n===== indexing {area} =====", flush=True)
            subprocess.run([sys.executable, "scripts/import_area.py", area, "--wait"], check=False)
            subprocess.run([sys.executable, "scripts/gen_questions.py", "--area", area], check=False)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
