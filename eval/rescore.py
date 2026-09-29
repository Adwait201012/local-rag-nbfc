"""Re-apply the current checker to saved results, without asking the model again.

    .venv/bin/python eval/rescore.py eval/X_results_v1.xlsx [more files...]

Use this after changing how answers are judged. The answers themselves are
untouched; only auto_flag and auto_note are recomputed, so older runs can be
compared fairly against new ones judged by the same rules.
"""
import sys

import pandas as pd

from run_sheet import assess


def rescore(path: str) -> None:
    df = pd.read_excel(path)
    before = df["auto_flag"].value_counts().to_dict()
    flags, notes = [], []
    for _, row in df.iterrows():
        answer = row.get("actual_answer")
        if not isinstance(answer, str) or not answer.strip():
            flags.append("ERROR"); notes.append(row.get("auto_note", "no answer recorded"))
            continue
        f, n = assess(row, answer, [])
        flags.append(f); notes.append(n)
    df["auto_flag"], df["auto_note"] = flags, notes
    df.to_excel(path, index=False)
    print(f"{path}\n  before: {before}\n  after:  {df['auto_flag'].value_counts().to_dict()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        rescore(p)
