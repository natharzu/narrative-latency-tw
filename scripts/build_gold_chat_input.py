"""build_gold_chat_input.py — build the blind {id, text} JSON for in-chat gold rating (§2c).

Joins gold_cohort_sample.csv (from scripts/14_gold_sample.py) to the labeled pilot
CSVs on articleId to recover full message text, then writes a blind rater input
(gold_cohort_chat_input.json) with only id + text (no rule/llm labels leaked).
Attach that file to each chat model with prompts/gold_chat_prompt.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

try:
    from narrative_latency import PROC
except ModuleNotFoundError:  # sandbox / standalone preview
    PROC = Path(".")

ELECTIONS = ["pilot500_labeled_elections.csv", "sample500_elections.csv"]
RECENT = ["pilot500_labeled_recent.csv", "pilot_recent_labeled.csv", "sample500_recent.csv"]


def first_existing(names: list[str]) -> Path:
    for n in names:
        p = PROC / n
        if p.exists():
            return p
    raise SystemExit(f"None of {names} found under {PROC}/.")


def main() -> None:
    gold = pd.read_csv(PROC / "gold_cohort_sample.csv")
    frames = [pd.read_csv(first_existing(ELECTIONS)), pd.read_csv(first_existing(RECENT))]
    pool = pd.concat(frames, ignore_index=True)[["articleId", "text"]].drop_duplicates("articleId")

    g = gold.merge(pool, on="articleId", how="left")
    missing = int(g["text"].isna().sum())
    if missing:
        print(f"WARNING: {missing} rows missing full text (falling back to text_preview)")
        g["text"] = g["text"].fillna(g["text_preview"])

    items = [{"id": int(r.gold_id), "text": str(r.text)} for r in g.itertuples()]
    out = PROC / "gold_cohort_chat_input.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"wrote {out} ({len(items)} items)")


if __name__ == "__main__":
    main()
