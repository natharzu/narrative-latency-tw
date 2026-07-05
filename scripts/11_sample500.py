"""11_sample500.py — deterministic stratified pilot sample across election cycles.

The §1 step of the 500-row Cofacts pilot, aligned with the repo's headline
2020-vs-2024 comparison. Instead of a trailing-365-day recency window (which
would exclude both election cycles), this draws from the +/-90d windows around
the 2020 and 2024 Taiwan presidential elections, using the same E2020/E2024/WIN
anchors as scripts/03_election_windows.py.

Design:
  * membership: |article_createdAt - E{2020,2024}| <= WIN  (90 days)
  * balanced 250/250 across the two cycles, so the comparison axis has equal
    annotation power on both sides;
  * within each cycle, proportional by rule_topic (preserves that cycle's mix);
  * a 30-per-cell floor keeps rare topics (political/health) modelable;
  * if a cell's pool is smaller than its target, we take all of it and flag it
    UNDERPOWERED in the printout;
  * deterministic (random_state=7).

Reads:  data/processed/cofacts_latency_topic.csv
        (regenerate from committed data via scripts/06b_topic_label_lite.py)
Writes: data/processed/sample500.csv   (adds an election_cycle column)

Usage:
    uv run python scripts/06b_topic_label_lite.py   # if the topic file is absent
    uv run python scripts/11_sample500.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import PROC, E2020, E2024, WIN, parse_dates_safe

IN = PROC / "cofacts_latency_topic.csv"
OUT = PROC / "sample500.csv"

N_TARGET = 500
PER_CYCLE = N_TARGET // 2   # 250 each -> equal power on the 2020-vs-2024 axis
MIN_PER_CELL = 30           # floor per (election_cycle x rule_topic) cell
SEED = 7


def main() -> None:
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

    # --- (a) keep only rows inside an election window; label the cycle -------
    in_2020 = (df["article_createdAt"] - E2020).abs() <= WIN
    in_2024 = (df["article_createdAt"] - E2024).abs() <= WIN
    df = df[in_2020 | in_2024].copy()
    # the two windows are 4 years apart, so membership is mutually exclusive
    df["election_cycle"] = "2020"
    df.loc[(df["article_createdAt"] - E2024).abs() <= WIN, "election_cycle"] = "2024"

    # --- (b) first-occurrence only (drop reused/duplicate articles) ---------
    df = df.rename(columns={"topic": "rule_topic"})
    df = df[~df["text_preview"].duplicated(keep="first")].copy()

    # --- (c) balanced across cycles, proportional by rule_topic within -----
    picks = []
    diag = []   # (cycle, topic, pool, drawn)
    for cycle, cdf in df.groupby("election_cycle"):
        total = len(cdf)
        for topic, g in cdf.groupby("rule_topic"):
            share = len(g) / total if total else 0
            target = max(MIN_PER_CELL, round(PER_CYCLE * share))
            take = min(target, len(g))
            picks.append(g.sample(take, random_state=SEED))
            diag.append((cycle, topic, len(g), take))
    sample = pd.concat(picks)

    sample = (
        sample.drop_duplicates(subset=["text_preview"])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    sample.index = sample.index + 1
    sample.index.name = "id"
    sample.to_csv(OUT, encoding="utf-8")

    # --- diagnostics --------------------------------------------------------
    lo20, hi20 = (E2020 - WIN).date(), (E2020 + WIN).date()
    lo24, hi24 = (E2024 - WIN).date(), (E2024 + WIN).date()
    print(f"windows: 2020 = {lo20}..{hi20}   2024 = {lo24}..{hi24}")
    print(f"{'cycle':<6} {'rule_topic':<10} {'pool':>6} {'drawn':>6}  flag")
    for cycle, topic, pool, take in sorted(diag):
        flag = "UNDERPOWERED" if take < MIN_PER_CELL else ""
        print(f"{cycle:<6} {topic:<10} {pool:>6} {take:>6}  {flag}")
    print()
    print(sample.groupby(["election_cycle", "rule_topic"]).size())
    print("per cycle:", sample["election_cycle"].value_counts().to_dict())
    print("total:", len(sample))
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
