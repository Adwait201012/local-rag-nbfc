"""Compare two run_sheet result files question by question.

    .venv/bin/python eval/compare_runs.py eval/X_results_v1.xlsx eval/X_results_v2.xlsx

Reports what moved, not just the totals: which questions got better, which got
worse, whether out-of-scope refusals held, and what the change cost in answer
length and time. A change that lifts the total while breaking refusals, or that
doubles the wait for every answer, is not an improvement.

Remember the auto flags are prompts to look, not verdicts. The rows listed as
worse are the ones to read with your own eyes before deciding anything.
"""
from __future__ import annotations

import argparse
import sys
import textwrap

import pandas as pd

RANK = {"ERROR": -1, "REVIEW": 0, "CHECK": 1, "PASS": 2}


def load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = {"question", "type", "actual_answer", "auto_flag"} - set(df.columns)
    if missing:
        sys.exit(f"{path} is missing columns {sorted(missing)} - is it a run_sheet results file?")
    return df


def summary(df: pd.DataFrame) -> dict:
    oos = df[df["type"] == "out_of_scope"]
    return {
        "flags": df["auto_flag"].value_counts().to_dict(),
        "refusals": f"{(oos['auto_flag'] == 'PASS').sum()}/{len(oos)}",
        "chars": int(df["actual_answer"].fillna("").str.len().mean()),
        "seconds": round(float(df["seconds"].mean()), 1) if "seconds" in df else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--show", type=int, default=5, help="how many changed rows to print per direction")
    args = ap.parse_args()

    a, b = load(args.before), load(args.after)
    m = a.merge(b, on="question", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        print(f"warning: only {len(m)} questions appear in both files "
              f"({len(a)} before, {len(b)} after)\n")

    sa, sb = summary(a), summary(b)
    print(f"{'':22}{'before':>12}{'after':>12}")
    for flag in ("PASS", "CHECK", "REVIEW", "ERROR"):
        x, y = sa["flags"].get(flag, 0), sb["flags"].get(flag, 0)
        if x or y:
            print(f"{flag:22}{x:>12}{y:>12}")
    print(f"{'out-of-scope refused':22}{sa['refusals']:>12}{sb['refusals']:>12}")
    print(f"{'avg answer length':22}{sa['chars']:>11}c{sb['chars']:>11}c")
    if sa["seconds"] is not None and sb["seconds"] is not None:
        print(f"{'avg seconds':22}{sa['seconds']:>12}{sb['seconds']:>12}")

    m["delta"] = m["auto_flag_b"].map(RANK) - m["auto_flag_a"].map(RANK)
    better, worse = m[m["delta"] > 0], m[m["delta"] < 0]
    print(f"\nby question: {len(better)} better, {len(worse)} worse, "
          f"{len(m) - len(better) - len(worse)} unchanged")

    if "type_a" in m:
        print("\nnet change by type (positive = improved):")
        for t, g in m.groupby("type_a"):
            print(f"  {t:14} {int(g['delta'].sum()):+d}")

    for label, rows in (("WORSE", worse), ("BETTER", better)):
        if rows.empty:
            continue
        print(f"\n===== {label} (showing up to {args.show}) =====")
        for _, r in rows.head(args.show).iterrows():
            print(f"\n[{r['auto_flag_a']} -> {r['auto_flag_b']}] {r['question'][:100]}")
            print(textwrap.indent(textwrap.fill("before: " + str(r["actual_answer_a"])[:300], 96), "  "))
            print(textwrap.indent(textwrap.fill("after:  " + str(r["actual_answer_b"])[:300], 96), "  "))

    if sa["refusals"] != sb["refusals"]:
        print("\n!! out-of-scope refusals changed - read those rows before trusting anything else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
