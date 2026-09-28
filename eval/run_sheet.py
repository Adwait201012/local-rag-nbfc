"""Run an evaluation spreadsheet against the running RAG system.

Reads a workbook of questions, asks each one, and writes back what the system
actually answered alongside the passages it used. It does not decide whether an
answer is right -- it flags the cases worth your attention and leaves the verdict
to you, because an automated judge here would repeat the exact mistake that made
the last evaluation worthless.

    .venv/bin/python eval/run_sheet.py eval/questions.xlsx --password YOURPASS

Writes <name>_results.xlsx next to the input.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

import httpx
import pandas as pd

REFUSAL_MARKERS = [
    "do not mention", "does not mention", "not mention",
    "do not specify", "does not specify", "not specify",
    "do not contain", "does not contain", "not contain",
    "do not address", "does not address", "not addressed",
    "do not include", "does not include",
    "do not discuss", "does not discuss", "not discussed",
    "do not state", "does not state",
    "no information", "no explicit", "no details", "no reference to",
    "no passage", "not in the", "not present", "not available",
    "cannot find", "can't find", "cannot be determined", "not covered",
    "not mentioned", "not provided", "not specified", "not found",
    "unable to", "nothing in", "outside the scope", "silent on",
]

# Numbers, percentages, rupee amounts and crore figures: the load-bearing details
# in a regulatory answer. If these survive into the generated answer, the answer
# is probably grounded in the right passage.
FACT = re.compile(r"₹?\s?\d[\d,]*\.?\d*\s?(?:per cent|%|crore|lakh|days?|months?|years?|times)?", re.I)


def login(client: httpx.Client, base: str, password: str | None) -> None:
    if not password:
        return
    r = client.post(f"{base}/api/login", json={"password": password})
    if r.status_code != 200:
        sys.exit("login failed: check --password against APP_PASSWORD in .env")
    print("[auth] signed in")


def ask(client: httpx.Client, base: str, question: str) -> tuple[str, list[dict]]:
    """Stream one answer out of /api/chat, collecting text and cited sources."""
    answer, sources = [], []
    with client.stream("POST", f"{base}/api/chat",
                       json={"query": question}, timeout=300) as resp:
        resp.raise_for_status()
        event = ""
        for line in resp.iter_lines():
            if not line.strip():
                continue
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "sources":
                    sources = data
                elif event == "token":
                    answer.append(data)
    return "".join(answer).strip(), sources


def key_facts(text: str) -> list[str]:
    """Distinctive figures from the expected answer, normalised for comparison."""
    out = []
    for m in FACT.findall(str(text)):
        f = m.strip().replace(" ", "").replace(",", "").replace("₹", "").lower()
        f = f.replace("percent", "%").replace("per cent", "%")
        if f and not f.isalpha() and len(f) > 1:
            out.append(f)
    return list(dict.fromkeys(out))[:6]


def looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def assess(row, answer: str, sources: list[dict]) -> tuple[str, str]:
    """Return (flag, note). The flag says what to look at, never what is correct."""
    qtype = str(row.get("type", "")).strip().lower()
    refused = looks_like_refusal(answer)

    if qtype == "out_of_scope":
        if refused:
            return "PASS", "correctly declined to answer"
        return "REVIEW", "answered a question the corpus should not cover - check for invention"

    if refused:
        return "REVIEW", "declined to answer an in-scope question - retrieval likely missed"

    wanted = key_facts(row.get("expected_answer", ""))
    if not wanted:
        return "CHECK", "no numeric facts to compare - read the answer yourself"

    norm = answer.lower().replace(",", "").replace("₹", "").replace(" ", "")
    norm = norm.replace("percent", "%").replace("per cent", "%")
    hit = [f for f in wanted if f in norm]
    if len(hit) == len(wanted):
        return "PASS", f"all key figures present ({', '.join(hit)})"
    if hit:
        missing = [f for f in wanted if f not in hit]
        return "CHECK", f"found {', '.join(hit)}; missing {', '.join(missing)}"
    return "REVIEW", f"none of the expected figures appear ({', '.join(wanted)})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--password", default=None)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N rows")
    args = ap.parse_args()

    df = pd.read_excel(args.sheet)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} questions against {args.api}\n")

    actual, cited, flags, notes, timings = [], [], [], [], []

    with httpx.Client(follow_redirects=True) as client:
        login(client, args.api, args.password)

        for i, row in df.iterrows():
            q = str(row["question"])
            t0 = time.time()
            try:
                answer, sources = ask(client, args.api, q)
                flag, note = assess(row, answer, sources)
            except Exception as exc:
                answer, sources = "", []
                flag, note = "ERROR", f"{type(exc).__name__}: {exc}"
            took = time.time() - t0

            names = []
            for s in sources:
                n = str(s.get("source", "")).split("/")[-1]
                if s.get("page"):
                    n += f" p.{s['page']}"
                names.append(n)

            actual.append(answer)
            cited.append(" | ".join(names))
            flags.append(flag)
            notes.append(note)
            timings.append(round(took, 1))

            mark = {"PASS": "ok  ", "CHECK": "?   ", "REVIEW": "!!  ", "ERROR": "ERR "}[flag]
            print(f"{mark}[{i+1}/{len(df)}] {took:5.1f}s  {q[:70]}")
            if flag != "PASS":
                print(f"          {note}")

    df["actual_answer"] = actual
    df["sources_used"] = cited
    df["auto_flag"] = flags
    df["auto_note"] = notes
    df["seconds"] = timings

    # Put the columns you fill in by hand at the end, after the evidence.
    manual = [c for c in df.columns
              if c.split(" (")[0].strip().lower() in ("verified", "corrected_answer", "notes")]
    for name in ("verified", "corrected_answer", "notes"):
        if not any(c.split(" (")[0].strip().lower() == name for c in manual):
            df[name] = None
            manual.append(name)
    order = [c for c in df.columns if c not in manual]
    df = df[order + manual]

    out = args.sheet.rsplit(".", 1)[0] + "_results.xlsx"
    df.to_excel(out, index=False)

    print("\n" + "-" * 60)
    counts = pd.Series(flags).value_counts()
    for k in ("PASS", "CHECK", "REVIEW", "ERROR"):
        if k in counts:
            print(f"{k:8} {counts[k]:3}")
    print(f"\nby type:")
    tmp = df.copy()
    tmp["flag"] = flags
    print(tmp.groupby(["type", "flag"]).size().to_string())
    print(f"\nwritten: {out}")
    print("Fill in 'verified' for every row not marked PASS. The auto flag is a "
          "prompt to look, not a verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
