"""14_gold_sample.py — draw a cohort-stratified gold subset for per-cohort validation (§2c).

The 40-row cross-validation gold (ref_labels.csv / gold_labels.csv) is a random
draw from the full corpus and lands ~0 rows inside the election / recent pilot
cohorts, so it can't give a per-cohort validity or memorization number. This
script instead draws the gold subset DIRECTLY from the labeled pilot samples, so
it joins by construction (on articleId) and already carries DeepSeek's llm_topic.

Draws N per cohort (2020 / 2024 / recent), stratified by rule_topic proportional
to each cohort's topic mix with a floor per present topic, seed=7. The output is
the row set to run the remaining raters over (scripts/15_gold_annotate.py) before
taking a majority-vote consensus gold.

Reads:  data/processed/pilot500_labeled_elections.csv  (else sample500_elections.csv)
        data/processed/pilot500_labeled_recent.csv      (else pilot_recent_labeled.csv / sample500_recent.csv)

Writes: data/processed/gold_cohort_sample.csv

Usage:
    uv run python scripts/14_gold_sample.py
    uv run python scripts/14_gold_sample.py --per-cohort 40 --floor 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

try:
    from narrative_latency import PROC
except ModuleNotFoundError:  # sandbox / standalone preview
    PROC = Path(".")

SEED = 7
TOPICS = ["scam", "political", "health", "other"]
KEEP = [
    "articleId", "cohort", "text_preview", "latency_hours", "reply_type",
    "rule_topic", "llm_topic", "article_createdAt",
]

ELECTIONS = ["pilot500_labeled_elections.csv", "sample500_elections.csv"]
RECENT = ["pilot500_labeled_recent.csv", "pilot_recent_labeled.csv", "sample500_recent.csv"]


def first_existing(names: list[str]) -> Path:
    for n in names:
        p = PROC / n
        if p.exists():
            return p
    raise SystemExit(f"None of {names} found under {PROC}/. Run scripts/11 (+13) first.")


def allocate(counts: pd.Series, n: int, floor: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of n across topics, with a floor."""
    present = counts[counts > 0]
    total = present.sum()
    raw = {t: n * c / total for t, c in present.items()}
    alloc = {t: min(int(present[t]), max(floor, int(v))) for t, v in raw.items()}
    # adjust to hit n as closely as availability allows
    def cur():
        return sum(alloc.values())
    # bump up by largest fractional remainder while under n and rows remain
    rema = sorted(raw, key=lambda t: raw[t] - int(raw[t]), reverse=True)
    i = 0
    while cur() < n and any(alloc[t] < int(present[t]) for t in present.index):
        t = rema[i % len(rema)]
        if alloc[t] < int(present[t]):
            alloc[t] += 1
        i += 1
        if i > 10000:
            break
    return alloc


def draw_cohort(df: pd.DataFrame, cohort, n: int, floor: int) -> pd.DataFrame:
    counts = df["rule_topic"].value_counts().reindex(TOPICS).fillna(0).astype(int)
    alloc = allocate(counts, n, floor)
    parts = []
    for t, k in alloc.items():
        pool = df[df["rule_topic"] == t]
        parts.append(pool.sample(min(k, len(pool)), random_state=SEED))
    out = pd.concat(parts, ignore_index=True)
    out["cohort"] = cohort
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cohort-stratified gold subset (§2c).")
    ap.add_argument("--per-cohort", type=int, default=40)
    ap.add_argument("--floor", type=int, default=5)
    args = ap.parse_args()

    ele = pd.read_csv(first_existing(ELECTIONS))
    rec = pd.read_csv(first_existing(RECENT))
    rec["cohort"] = "recent"
    ele["cohort"] = ele["cohort"].astype(str)

    frames = []
    for cohort in ["2020", "2024"]:
        sub = ele[ele["cohort"] == cohort]
        if len(sub):
            frames.append(draw_cohort(sub, cohort, args.per_cohort, args.floor))
    frames.append(draw_cohort(rec, "recent", args.per_cohort, args.floor))

    gold = pd.concat(frames, ignore_index=True)
    for c in KEEP:
        if c not in gold.columns:
            gold[c] = pd.NA
    gold = gold[KEEP]
    gold.insert(0, "gold_id", range(1, len(gold) + 1))

    out = PROC / "gold_cohort_sample.csv"
    gold.to_csv(out, index=False, encoding="utf-8")

    print("gold cohort sample:", len(gold), "rows")
    print(pd.crosstab(gold["cohort"], gold["rule_topic"]).reindex(columns=TOPICS, fill_value=0))
    print("\nunique articleId:", gold["articleId"].nunique(), "/", len(gold))
    print("wrote:", out)


if __name__ == "__main__":
    main()
