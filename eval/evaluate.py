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

import httpx


def hit(result: dict, case: dict) -> bool:
    # A question can be answered from more than one passage once the corpus holds
    # several regulations restating the same rule. Accept any listed phrase rather
    # than insisting on one blessed chunk.
    wanted = case.get("expect_any") or ([case["expect_text"]] if "expect_text" in case else [])
    for phrase in wanted:
        if phrase.lower() in result["text"].lower():
            return True
    if "expect_source" in case:
        if case["expect_source"].lower() in result["source"].lower():
            return True
    return False


def score(api: str, cases: list[dict], top_k: int, rerank: bool) -> dict:
    hits, rr, missed = 0, 0.0, []
    with httpx.Client(base_url=api, timeout=300) as client:
        for case in cases:
            r = client.post(
                "/api/search",
                json={"query": case["question"], "top_k": top_k, "rerank": rerank},
            )
            r.raise_for_status()
            results = r.json()["results"]
            rank = next((i for i, res in enumerate(results, 1) if hit(res, case)), None)
            if rank:
                hits += 1
                rr += 1 / rank
            else:
                missed.append(case["question"])
    n = len(cases)
    return {
        "recall": hits / n,
        "mrr": rr / n,
        "missed": missed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()

    cases = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    print(f"{len(cases)} questions, top_k={args.top_k}\n")

    for label, use_rerank in (("hybrid only", False), ("hybrid + rerank", True)):
        s = score(args.api, cases, args.top_k, use_rerank)
        print(f"{label:>16}   recall@{args.top_k} {s['recall']:.2f}   MRR {s['mrr']:.3f}")
        if use_rerank and s["missed"]:
            print("\nStill missed (these are your chunking or parsing problems, not model problems):")
            for q in s["missed"]:
                print(f"  - {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
