"""11_sample500.py — deterministic stratified 500-row pilot sample.

Draws a reproducible (random_state=7), two-way (year x rule_topic) stratified
sample of ~500 first-occurrence articles from a recent, maturity-buffered
window of the topic-labelled latency table. This is the §1 step of the
500-row Cofacts pilot, packaged to match the repo's uv/script conventions.

Reads:  data/processed/cofacts_latency_topic.csv
        (regenerate from committed data via scripts/06b_topic_label_lite.py)
Writes: data/processed/sample500.csv

Usage:
    uv sync
    uv run python scripts/06b_topic_label_lite.py   # if the topic file is absent
    uv run python scripts/11_sample500.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import PROC

IN = PROC / "cofacts_latency_topic.csv"
OUT = PROC / "sample500.csv"


def main() -> None:
    if not IN.exists():
        raise SystemExit(
            f"Missing {IN}.\n"
            "Generate it first from the committed data:\n"
            "    uv run python scripts/06b_topic_label_lite.py"
        )

    df = pd.read_csv(IN)
    df["article_createdAt"] = pd.to_datetime(
        df["article_createdAt"], format="mixed", utc=True, errors="coerce"
    )
    df = df[df["text_preview"].astype(str).str.len() > 15].copy()
    df = df.dropna(subset=["article_createdAt"])

    # --- (a) recency window + maturity buffer ----------------------------
    # Fresh enough to matter (and to sit past most LLM training cutoffs), but
    # old enough that the reply has usually arrived -> limits right-censoring.
    snapshot = df["article_createdAt"].max()
    maturity = pd.Timedelta(days=30)   # give each item >=30d to receive a reply
    recency = pd.Timedelta(days=365)   # sample from the last ~year before that
    hi = snapshot - maturity
    lo = hi - recency
    win = df[(df["article_createdAt"] >= lo) & (df["article_createdAt"] <= hi)].copy()

    # --- (b) first-occurrence only (drop reused/duplicate articles) ------
    win = win[~win["text_preview"].duplicated(keep="first")].copy()

    # --- (c) two-way stratify: year x rule_topic ------------------------
    win["year"] = win["article_createdAt"].dt.year
    win = win.rename(columns={"topic": "rule_topic"})
    win["stratum"] = win["year"].astype(str) + "|" + win["rule_topic"].astype(str)

    n_target = 500
    min_per_class = 60          # floor per rule_topic so scam/political stay modelable

    # proportional cell targets across year x topic strata
    groups = []
    for _stratum, g in win.groupby("stratum"):
        share = len(g) / len(win)
        n = min(len(g), max(1, round(n_target * share)))
        groups.append(g.sample(n, random_state=7))
    sample = pd.concat(groups)

    # top up any rule_topic that fell under the floor
    for _topic, g in win.groupby("rule_topic"):
        have = (sample["rule_topic"] == _topic).sum()
        if have < min_per_class:
            pool = g.drop(sample.index, errors="ignore")
            take = min(min_per_class - have, len(pool))
            if take > 0:
                sample = pd.concat([sample, pool.sample(take, random_state=7)])

    sample = (
        sample.drop_duplicates(subset=["text_preview"])
        .sample(frac=1, random_state=7)
        .reset_index(drop=True)
    )
    sample.index = sample.index + 1
    sample.index.name = "id"
    sample.to_csv(OUT, encoding="utf-8")

    print("window:", lo.date(), "->", hi.date())
    print(sample.groupby(["year", "rule_topic"]).size())
    print("total:", len(sample))
    print("wrote:", OUT)


if __name__ == "__main__":
    main()
