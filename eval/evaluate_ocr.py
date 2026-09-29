"""Measure page-level OCR extraction through the same Docling setup as ingestion.

Run inside the ingest container; no database or embedding service is used.
Manifest paths are relative to the manifest. Page numbers start at 1.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    # Preserve matras, nukta, punctuation and digits. Only normalize canonical
    # Unicode forms and whitespace; never strip non-ASCII characters.
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(reference, prediction) -> int:
    previous = list(range(len(prediction) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def error_counts(reference: str, prediction: str) -> dict:
    ref, pred = normalize(reference), normalize(prediction)
    if not ref:
        raise ValueError("Reference transcription must not be empty")
    chars, words = len(ref), len(ref.split())
    ce = edit_distance(ref, pred)
    we = edit_distance(ref.split(), pred.split())
    return {"char_errors": ce, "reference_chars": chars, "cer": ce / chars,
            "word_errors": we, "reference_words": words, "wer": we / words}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--force-full-page", action="store_true", default=None,
                    help="Force OCR even if the PDF already has a text layer")
    args = ap.parse_args()
    cases = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        ap.error("Manifest is empty")
    grouped = {}
    seen = set()
    for case in cases:
        if case.get("template"):
            ap.error("Replace template paths and labels, then remove template:true")
        page = case.get("page")
        if type(page) is not int or page < 1:
            ap.error("Each case needs a positive, 1-based page number")
        source = (args.manifest.parent / case["source"]).resolve()
        reference = (args.manifest.parent / case["reference"]).read_text(encoding="utf-8")
        if not source.is_file() or not normalize(reference):
            ap.error(f"Missing source or empty reference: {source}")
        if (source, page) in seen:
            ap.error(f"Duplicate page: {source}, {page}")
        seen.add((source, page))
        grouped.setdefault(source, []).append((page, reference))

    # /app in Docker; sibling ingest/ for a local Python environment.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))
    if Path("/app/ocr.py").exists():
        sys.path.insert(0, "/app")
    from ocr import convert_document, make_converter

    started = time.perf_counter()
    converter = make_converter(args.force_full_page)
    rows, documents = [], []
    for source, pages in grouped.items():
        t0 = time.perf_counter()
        doc = convert_document(source, converter)
        documents.append({"source": str(source), "extraction_seconds": time.perf_counter() - t0,
                          "document_pages": len(doc.pages)})
        for page, reference in pages:
            if page not in doc.pages:
                ap.error(f"Page {page} does not exist in {source}")
            kwargs = {"page_no": page}
            if "traverse_pictures" in inspect.signature(doc.export_to_text).parameters:
                kwargs["traverse_pictures"] = True
            prediction = doc.export_to_text(**kwargs)
            rows.append({"source": str(source), "page": page, "prediction": prediction,
                         **error_counts(reference, prediction)})
    char_errors = sum(row["char_errors"] for row in rows)
    word_errors = sum(row["word_errors"] for row in rows)
    report = {
        "ocr_languages": os.getenv("OCR_LANGUAGES", "hin,eng"),
        "force_full_page": args.force_full_page or os.getenv("OCR_FORCE_FULL_PAGE", "false").lower() in {"1", "true", "yes"},
        "evaluated_pages": len(rows),
        "cer": char_errors / sum(row["reference_chars"] for row in rows),
        "wer": word_errors / sum(row["reference_words"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "documents": documents, "pages": rows,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} labelled pages: CER={report['cer']:.4f}, WER={report['wer']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
