"""Measure retrieval quality so tuning is evidence-based rather than vibes-based.

Retrieval, not generation, is where most RAG systems fail. This scores the retriever
directly: for each question, did the passage that contains the answer make the cut,
and at what position?

    pip install httpx
    python eval/evaluate.py eval/questions.jsonl --api http://localhost:8080

Each line of the questions file is JSON:
    {"question": "What is the refund window?",
     "expect_source": "policy.pdf",            # optional: filename substring
     "expect_text": "within 30 days"}          # optional: substring in the chunk

Run it before and after every change to chunk size, embedder, or top-k. A change
that does not move these numbers is not an improvement.
"""
from __future__ import annotations

import argparse
import json
import sys
import math
import time
import unicodedata
from pathlib import Path

import httpx


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def hit(result: dict, case: dict) -> bool:
    # A question can be answered from more than one passage once the corpus holds
    # several regulations restating the same rule. Accept any listed phrase rather
    # than insisting on one blessed chunk.
    wanted = case.get("expect_any") or ([case["expect_text"]] if "expect_text" in case else [])
    checks = []
    if wanted:
        checks.append(any(normalize(p) in normalize(result["text"]) for p in wanted))
    if case.get("expect_source"):
        checks.append(normalize(case["expect_source"]) in normalize(result["source"]))
    if "expect_page" in case:
        checks.append(result.get("page") == case["expect_page"])
    # All supplied constraints must hold in the SAME passage.
    return bool(checks) and all(checks)


def validate_cases(cases: list[dict]) -> None:
    if not cases:
        raise ValueError("The question file is empty")
    for n, case in enumerate(cases, 1):
        if case.get("template"):
            raise ValueError(f"Case {n}: replace template labels with verified evidence and remove template:true")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {n}: question must be nonempty")
        phrases = case.get("expect_any", [])
        if not isinstance(phrases, list) or any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise ValueError(f"Case {n}: expect_any must be a list of nonempty strings")
        for key in ("expect_text", "expect_source"):
            if key in case and (not isinstance(case[key], str) or not case[key].strip()):
                raise ValueError(f"Case {n}: {key} must be nonempty")
        if not (phrases or case.get("expect_text") or case.get("expect_source")):
            raise ValueError(f"Case {n}: supply expected text or source")
        if "expect_page" in case and (type(case["expect_page"]) is not int or case["expect_page"] < 1):
            raise ValueError(f"Case {n}: expect_page must be a positive, 1-based page number")


def summarize(rows: list[dict]) -> dict:
    times = sorted(row["latency_ms"] for row in rows)
    return {
        "n": len(rows),
        "recall": sum(row["rank"] is not None for row in rows) / len(rows),
        "mrr": sum(1 / row["rank"] if row["rank"] else 0 for row in rows) / len(rows),
        "mean_ms": sum(times) / len(times),
        "p95_ms": times[math.ceil(0.95 * len(times)) - 1],
    }


def score(api: str, cases: list[dict], top_k: int, rerank: bool) -> dict:
    validate_cases(cases)
    rows = []
    with httpx.Client(base_url=api, timeout=300) as client:
        for case in cases:
            started = time.perf_counter()
            r = client.post(
                "/api/search",
                json={"query": case["question"], "top_k": top_k, "rerank": rerank},
            )
            r.raise_for_status()
            latency_ms = (time.perf_counter() - started) * 1000
            results = r.json()["results"][:top_k]
            rank = next((i for i, res in enumerate(results, 1) if hit(res, case)), None)
            rows.append({"question": case["question"], "language": case.get("language", "unspecified"),
                         "rank": rank, "latency_ms": latency_ms})
    return {
        **summarize(rows),
        "missed": [row["question"] for row in rows if row["rank"] is None],
        "by_language": {lang: summarize([row for row in rows if row["language"] == lang])
                        for lang in sorted({row["language"] for row in rows})},
        "cases": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--output", type=Path, help="Save a UTF-8 JSON report")
    args = ap.parse_args()

    if not 1 <= args.top_k <= 50:
        ap.error("--top-k must be between 1 and 50")
    cases = [json.loads(line) for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        validate_cases(cases)
    except ValueError as exc:
        ap.error(str(exc))
    print(f"{len(cases)} questions, top_k={args.top_k}\n")

    report = {"top_k": args.top_k, "questions": args.questions, "runs": {}}
    for label, use_rerank in (("hybrid only", False), ("hybrid + rerank", True)):
        s = score(args.api, cases, args.top_k, use_rerank)
        report["runs"][label] = s
        print(f"{label:>16}   recall@{args.top_k} {s['recall']:.2f}   MRR {s['mrr']:.3f}")
        for lang, group in s["by_language"].items():
            print(f"  {lang}: n={group['n']} recall={group['recall']:.3f} MRR={group['mrr']:.3f} "
                  f"mean={group['mean_ms']:.0f}ms p95={group['p95_ms']:.0f}ms")
        if use_rerank and s["missed"]:
            print("\nMissed (inspect OCR, labels, retrieval, and reranking):")
            for q in s["missed"]:
                print(f"  - {q}")
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
