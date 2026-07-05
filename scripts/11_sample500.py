"""11_sample500.py — deterministic pilot sampler (two selectable cohorts).

The §1 step of the 500-row Cofacts pilot. Emits one of two disjoint samples,
selected with --mode, so the election-aligned analysis sample and the fresh
post-cutoff contamination-control sample are both reproducible:

  --mode elections   (default)
      Draw from the +/-90d windows around the 2020 and 2024 Taiwan presidential
      elections (E2020/E2024/WIN). Balanced 250/250 across cycles, proportional
      by rule_topic within each cycle, 30-per-cell floor. Entirely PRE the model
      cutoff -> this is the analysis sample and the one at risk of memorization.
      -> data/processed/sample500_elections.csv

  --mode recent
      Draw from a trailing ~365d window ending 30d before the data snapshot,
      two-way stratified by year x rule_topic with a 60-per-class floor.
      Entirely POST the mid-2024 model cutoff -> unseen by the model, so it
      serves as a contamination control.
      -> data/processed/sample500_recent.csv

Both outputs carry a `cohort` column (election_cycle value, or "recent") and a
`sample_mode` column, so they can be concatenated and grouped for comparison
(see scripts/12_compare_samples.py).

Reads:  data/processed/cofacts_latency_topic.csv
        (regenerate from committed data via scripts/06b_topic_label_lite.py)

Usage:
    uv run python scripts/11_sample500.py --mode elections
    uv run python scripts/11_sample500.py --mode recent
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import PROC, E2020, E2024, WIN, parse_dates_safe

IN = PROC / "cofacts_latency_topic.csv"
OUT = {
    "elections": PROC / "sample500_elections.csv",
    "recent": PROC / "sample500_recent.csv",
}

N_TARGET = 500
SEED = 7

# --- elections mode ---
PER_CYCLE = N_TARGET // 2   # 250 each -> equal power on the 2020-vs-2024 axis
MIN_PER_CELL = 30           # floor per (election_cycle x rule_topic) cell

# --- recent mode ---
MATURITY = pd.Timedelta(days=30)    # give each item >=30d to receive a reply
RECENCY = pd.Timedelta(days=365)    # sample from the last ~year before that
MIN_PER_CLASS = 60                  # floor per rule_topic


def load_clean() -> pd.DataFrame:
    if not IN.exists():
        raise SystemExit(
            f"Missing {IN}.\n"
            "Generate it first from the committed data:\n"
            "    uv run python scripts/06b_topic_label_lite.py"
        )
    df = pd.read_csv(IN)
    df["article_createdAt"] = parse_dates_safe(df["article_createdAt"])
    df = df.dropna(subset=["article_createdAt"]).copy()
    df = df[df["text_preview"].astype(str).str.len() > 15].copy()
    df = df.rename(columns={"topic": "rule_topic"})
    return df


def sample_elections(df: pd.DataFrame) -> pd.DataFrame:
    in_2020 = (df["article_createdAt"] - E2020).abs() <= WIN
    in_2024 = (df["article_createdAt"] - E2024).abs() <= WIN
    df = df[in_2020 | in_2024].copy()
    # windows are 4 years apart, so membership is mutually exclusive
    df["election_cycle"] = "2020"
    df.loc[(df["article_createdAt"] - E2024).abs() <= WIN, "election_cycle"] = "2024"

    df = df[~df["text_preview"].duplicated(keep="first")].copy()

    picks, diag = [], []
    for cycle, cdf in df.groupby("election_cycle"):
        total = len(cdf)
        for topic, g in cdf.groupby("rule_topic"):
            share = len(g) / total if total else 0
            target = max(MIN_PER_CELL, round(PER_CYCLE * share))
            take = min(target, len(g))
            picks.append(g.sample(take, random_state=SEED))
            diag.append((cycle, topic, len(g), take))
    sample = pd.concat(picks)

    lo20, hi20 = (E2020 - WIN).date(), (E2020 + WIN).date()
    lo24, hi24 = (E2024 - WIN).date(), (E2024 + WIN).date()
    print(f"windows: 2020 = {lo20}..{hi20}   2024 = {lo24}..{hi24}")
    print(f"{'cycle':<6} {'rule_topic':<10} {'pool':>6} {'drawn':>6}  flag")
    for cycle, topic, pool, take in sorted(diag):
        flag = "UNDERPOWERED" if take < MIN_PER_CELL else ""
        print(f"{cycle:<6} {topic:<10} {pool:>6} {take:>6}  {flag}")

    sample["cohort"] = sample["election_cycle"]
    return sample


def sample_recent(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["article_createdAt"].max()
    hi = snapshot - MATURITY
    lo = hi - RECENCY
    win = df[(df["article_createdAt"] >= lo) & (df["article_createdAt"] <= hi)].copy()
    win = win[~win["text_preview"].duplicated(keep="first")].copy()
    win["year"] = win["article_createdAt"].dt.year
    win["stratum"] = win["year"].astype(str) + "|" + win["rule_topic"].astype(str)

    picks = []
    for _stratum, g in win.groupby("stratum"):
        share = len(g) / len(win)
        n = min(len(g), max(1, round(N_TARGET * share)))
        picks.append(g.sample(n, random_state=SEED))
    sample = pd.concat(picks)

    for _topic, g in win.groupby("rule_topic"):
        have = (sample["rule_topic"] == _topic).sum()
        if have < MIN_PER_CLASS:
            pool = g.drop(sample.index, errors="ignore")
            take = min(MIN_PER_CLASS - have, len(pool))
            if take > 0:
                sample = pd.concat([sample, pool.sample(take, random_state=SEED)])

    print(f"window: {lo.date()} -> {hi.date()}")
    print(sample.groupby(["year", "rule_topic"]).size())

    sample["cohort"] = "recent"
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw a 500-row Cofacts pilot sample (see module docstring)."
    )
    parser.add_argument(
        "--mode", choices=["elections", "recent"], default="elections"
    )
    args = parser.parse_args()

    df = load_clean()
    sample = sample_elections(df) if args.mode == "elections" else sample_recent(df)

    sample = (
        sample.drop_duplicates(subset=["text_preview"])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    sample.index = sample.index + 1
    sample.index.name = "id"
    sample["sample_mode"] = args.mode
    out = OUT[args.mode]
    sample.to_csv(out, encoding="utf-8")

    print()
    print("cohort counts:", sample["cohort"].value_counts().to_dict())
    print("total:", len(sample))
    print("wrote:", out)


if __name__ == "__main__":
    main()
