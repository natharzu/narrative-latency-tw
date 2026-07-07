#!/usr/bin/env python3
"""
scripts/17_popularity.py — reply-request & view popularity layer.

Builds a per-article popularity table from the RAW Cofacts open-data tables the
latency pipeline never loads (reply_requests, analytics), joins them onto the
processed latency/topic rows by articleId, and reports popularity by narrative.
Optionally re-fits the Cox model with request volume as a covariate.

Usage:
    uv run python scripts/17_popularity.py                 # build + attach + summarise
    uv run python scripts/17_popularity.py --cox           # also fit the Cox model
    uv run python scripts/17_popularity.py --raw data/raw/cofacts

Outputs:
    data/processed/cofacts_popularity.csv
    data/processed/cofacts_latency_topic_pop.csv   (if the latency/topic file exists)

Caveats (see the Notion \"Popularity layer\" page):
    * reply_requests accumulate only BEFORE an article's first reply; use views
      (analytics) for post-reply attention, and prefer an early-window request
      count as a latency covariate to avoid endogeneity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Shared repo constants via the scripts/utils.py shim (bootstraps src/ onto
# sys.path and re-exports narrative_latency.*), matching 09_electoral_timing.py.
from utils import ROOT, RAW, PROC

RAW_DEFAULT = RAW / "cofacts"
LATENCY_TOPIC = PROC / "cofacts_latency_topic.csv"
POP_OUT = PROC / "cofacts_popularity.csv"
JOINED_OUT = PROC / "cofacts_latency_topic_pop.csv"


# --------------------------------------------------------------------------- #
# Pure transforms (unit-tested in tests/test_popularity.py)
# --------------------------------------------------------------------------- #
def build_request_counts(rr: pd.DataFrame) -> pd.DataFrame:
    """Distinct askers per article from a reply_requests frame."""
    rr = rr[rr["status"].astype(str).str.upper() == "NORMAL"].copy()
    # Force NumPy-backed dtypes. pandas' default PyArrow-backed strings send
    # groupby nunique/min/max down a pure-python fallback that is orders of
    # magnitude slower on a multi-million-row table (it looks like a hang).
    rr["articleId"] = rr["articleId"].astype("object")
    rr["userIdsha256"] = rr["userIdsha256"].astype("object")
    rr["createdAt"] = pd.to_datetime(rr["createdAt"], errors="coerce", utc=True)
    return (rr.groupby("articleId")
              .agg(request_count=("userIdsha256", "nunique"),
                   first_request=("createdAt", "min"),
                   last_request=("createdAt", "max"))
              .reset_index())


def build_view_counts(an: pd.DataFrame) -> pd.DataFrame:
    """Lifetime visit/user totals per article from an analytics frame."""
    an = an[an["type"].astype(str) == "article"].copy()
    an["docId"] = an["docId"].astype("object")
    for _c in ["lineVisit", "lineUser", "webVisit", "webUser"]:
        an[_c] = pd.to_numeric(an[_c], errors="coerce").fillna(0)
    return (an.groupby("docId")
              .agg(line_visits=("lineVisit", "sum"),
                   line_users=("lineUser", "sum"),
                   web_visits=("webVisit", "sum"),
                   web_users=("webUser", "sum"))
              .reset_index()
              .rename(columns={"docId": "articleId"}))


def merge_popularity(req: pd.DataFrame, views: pd.DataFrame) -> pd.DataFrame:
    pop = req.merge(views, on="articleId", how="outer")
    view_cols = ["line_visits", "line_users", "web_visits", "web_users"]
    for c in view_cols:
        if c not in pop:
            pop[c] = 0
    pop[view_cols] = pop[view_cols].fillna(0)
    # every article carries >=1 implicit request (its own submission)
    pop["request_count"] = pop["request_count"].fillna(1).astype(int)
    return pop


def popularity_by_narrative(d: pd.DataFrame, topic_col: str) -> pd.DataFrame:
    return (d.groupby(topic_col)
              .agg(n=("articleId", "size"),
                   median_requests=("request_count", "median"),
                   total_requests=("request_count", "sum"),
                   median_web_visits=("web_visits", "median"))
              .sort_values("total_requests", ascending=False))


# --------------------------------------------------------------------------- #
# IO wrappers
# --------------------------------------------------------------------------- #
def _read_raw(raw: Path, name: str) -> pd.DataFrame:
    """Read <name>.csv.zip (repo convention, cf. 10_survival.py) or <name>.csv."""
    for fn in (f"{name}.csv.zip", f"{name}.csv"):
        p = raw / fn
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"neither {name}.csv.zip nor {name}.csv in {raw} — download the "
        "Cofacts reply_requests / analytics open-data dumps first."
    )


def build_popularity_table(raw: Path) -> pd.DataFrame:
    rr = _read_raw(raw, "reply_requests")
    an = _read_raw(raw, "analytics")
    pop = merge_popularity(build_request_counts(rr), build_view_counts(an))
    PROC.mkdir(parents=True, exist_ok=True)
    pop.to_csv(POP_OUT, index=False, encoding="utf-8")
    print(f"wrote {POP_OUT.relative_to(ROOT)}  rows={len(pop):,}")
    print(pop[["request_count", "line_visits", "web_visits"]].describe())
    return pop


def attach_to_latency(pop: pd.DataFrame) -> pd.DataFrame | None:
    if not LATENCY_TOPIC.exists():
        print(f"skip attach: {LATENCY_TOPIC.relative_to(ROOT)} not found")
        return None
    d = pd.read_csv(LATENCY_TOPIC)
    key = "articleId" if "articleId" in d.columns else ("id" if "id" in d.columns else None)
    if key is None:
        raise KeyError(
            "cofacts_latency_topic.csv has neither 'articleId' nor 'id'. "
            "Carry the raw article.id through 01_clean.py before this join."
        )
    d = d.merge(pop, left_on=key, right_on="articleId", how="left")
    d["request_count"] = d["request_count"].fillna(1).astype(int)
    for c in ["line_visits", "line_users", "web_visits", "web_users"]:
        if c in d:
            d[c] = d[c].fillna(0)
    d.to_csv(JOINED_OUT, index=False, encoding="utf-8")
    print(f"wrote {JOINED_OUT.relative_to(ROOT)}  rows={len(d):,}")
    for col in ("rule_topic", "llm_topic"):
        if col in d.columns:
            print(f"\n--- popularity by {col} ---")
            print(popularity_by_narrative(d, col))
    return d


def fit_cox(d: pd.DataFrame) -> None:
    from lifelines import CoxPHFitter
    need = {"duration", "event"}
    missing = need - set(d.columns)
    if missing:
        print(f"skip cox: missing columns {sorted(missing)}")
        return
    d = d.copy()
    d["log_requests"] = np.log1p(d["request_count"])
    cols = ["llm_topic", "valence", "intensity", "novel", "submit_volume",
            "log_requests", "hour", "dow", "text_len", "year", "duration", "event"]
    cols = [c for c in cols if c in d.columns]
    dummy = ["llm_topic"] if "llm_topic" in cols else []
    X = pd.get_dummies(d[cols].dropna(), columns=dummy, drop_first=True)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(X, duration_col="duration", event_col="event")
    cph.print_summary()
    # HR(log_requests) > 1  => more-requested rumors are replied to faster.
    # Watch whether adding log_requests moves the topic/affect HRs.


def main() -> None:
    ap = argparse.ArgumentParser(description="Cofacts popularity layer")
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT,
                    help="dir with reply_requests.csv / analytics.csv")
    ap.add_argument("--cox", action="store_true", help="also fit the Cox model")
    args = ap.parse_args()

    pop = build_popularity_table(args.raw)
    d = attach_to_latency(pop)
    if args.cox and d is not None:
        fit_cox(d)


if __name__ == "__main__":
    main()
