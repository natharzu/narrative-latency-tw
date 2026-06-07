"""08_cluster_latency_hdbscan.py -- crack open the keyword "Other" bucket.

Phase 2 (scripts/07_cluster_latency.py) tags articles with the hand-written
keyword taxonomy (``CLUSTERS`` / ``tag``), which leaves ~88% of volume in a
single "Other" bucket. scripts/05_cluster_articles.py already discovered
data-driven semantic clusters and saved them to ``cofacts_clustered.csv`` +
``cluster_profiles.csv`` -- but that output was never wired into the latency
analysis.

This script closes that gap. It reuses the SAME tested helpers
(``per_year_median`` / ``within_year_election_contrast``) but groups by the
HDBSCAN ``cluster_id`` instead of the keyword tag, and -- the headline -- zooms
into the rows the keyword tagger calls "Other" to show what actually lives
inside that bucket. Large clusters there are named topics the taxonomy is
currently missing (candidates to promote into ``CLUSTERS``).

Prereq: run scripts/05_cluster_articles.py first (produces the two CSVs).

Run:
    uv run python scripts/08_cluster_latency_hdbscan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import (
    PROC,
    E2024,
    parse_dates_safe,
    reconstruct_article_dates,
    tag,
    per_year_median,
    within_year_election_contrast,
)

CLUSTERED = PROC / "cofacts_clustered.csv"
PROFILES = PROC / "cluster_profiles.csv"
# Fallback text column for CSVs that predate the full-text column (Option 0).
TEXT_COL = "text_preview"
TOP_N = 20


def _round(x, n=1):
    return round(float(x), n) if pd.notna(x) else None


def _labels() -> dict:
    """cluster_id -> human label from cluster_profiles.csv (if available)."""
    if not PROFILES.exists():
        return {}
    prof = pd.read_csv(PROFILES)
    if "cluster_id" not in prof.columns or "label" not in prof.columns:
        return {}
    out = {}
    for r in prof.itertuples():
        try:
            out[int(r.cluster_id)] = str(r.label)
        except (TypeError, ValueError):
            continue
    return out


def label_for(cid, labels) -> str:
    if cid == -2:
        return "URL-only"
    if cid == -1:
        return "noise / unclustered"
    return labels.get(cid, f"cluster {cid}")


def load() -> pd.DataFrame:
    if not CLUSTERED.exists():
        sys.exit(
            f"Missing {CLUSTERED}; run scripts/05_cluster_articles.py first "
            "to generate the semantic clusters."
        )
    df = pd.read_csv(CLUSTERED)
    df["article_createdAt"] = parse_dates_safe(df["article_createdAt"])
    if "reply_createdAt" in df.columns:
        df["reply_createdAt"] = parse_dates_safe(df["reply_createdAt"])
        missing = df["article_createdAt"].isna() & df["reply_createdAt"].notna()
        if missing.any():
            df.loc[missing, "article_createdAt"] = reconstruct_article_dates(
                df.loc[missing]
            )
    df = df.dropna(subset=["article_createdAt"])
    if "cluster_id" not in df.columns:
        sys.exit(
            f"{CLUSTERED} has no 'cluster_id' column; re-run "
            "scripts/05_cluster_articles.py."
        )
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)
    # Reproduce the keyword bucketing so we can isolate "Other".
    # Prefer FULL text (Option 0); fall back to the preview for older CSVs.
    text_col = "text" if "text" in df.columns else TEXT_COL
    df["narrative"] = df[text_col].apply(tag)
    return df


def _cluster_row(sub, cid, labels, total):
    pym = per_year_median(sub)
    m2020 = pym.get(2020, float("nan"))
    m2024 = pym.get(2024, float("nan"))
    ratio = (m2024 / m2020) if (pd.notna(m2020) and m2020) else float("nan")
    c24 = within_year_election_contrast(sub, E2024)
    return {
        "cluster_id": cid,
        "label": label_for(cid, labels)[:45],
        "n": len(sub),
        "share_%": _round(100 * len(sub) / total) if total else None,
        "median_h": _round(sub["latency_hours"].median()),
        "med_2020_h": _round(m2020),
        "med_2024_h": _round(m2024),
        "yr_ratio_24/20": _round(ratio, 2),
        "2024_win/base": _round(c24["window_over_baseline"], 2),
        "n_2024_win": c24["n_window"],
    }


def main() -> None:
    labels = _labels()
    df = load()
    total = len(df)
    print(f"Loaded {total:,} clustered rows.\n")

    real = sorted(c for c in df["cluster_id"].unique() if c >= 0)
    print(
        f"{len(real)} real clusters | "
        f"noise(-1): {(df['cluster_id'] == -1).sum():,} | "
        f"URL-only(-2): {(df['cluster_id'] == -2).sum():,}\n"
    )

    other = df[df["narrative"] == "Other"]
    n_other = len(other)
    pct = (100 * n_other / total) if total else 0.0
    print(
        f"== Keyword 'Other' = {n_other:,} rows ({pct:.1f}%). "
        "Decomposing it by HDBSCAN cluster ==\n"
    )

    # Headline: largest semantic clusters INSIDE the keyword-'Other' bucket.
    rows = []
    for cid in real:
        sub = other[other["cluster_id"] == cid]
        if len(sub) == 0:
            continue
        rows.append(_cluster_row(sub, cid, labels, n_other))
    if rows:
        inside = (
            pd.DataFrame(rows).sort_values("n", ascending=False).head(TOP_N)
        )
        print(f"-- Top {TOP_N} semantic clusters within keyword-'Other' (by size) --")
        print(inside.to_string(index=False))
        print()

        slow = (
            inside[inside["n"] >= 100]
            .sort_values("median_h", ascending=False, na_position="last")
            .head(10)
        )
        if len(slow):
            print("-- Slowest clusters within 'Other' (>=100 rows, by median latency) --")
            print(slow.to_string(index=False))
            print()
    else:
        print("(No real HDBSCAN clusters intersect the keyword-'Other' rows.)\n")

    # Reference: full-dataset per-cluster overview (not just Other).
    allrows = [
        _cluster_row(df[df["cluster_id"] == cid], cid, labels, total)
        for cid in real
    ]
    if allrows:
        allk = (
            pd.DataFrame(allrows).sort_values("n", ascending=False).head(TOP_N)
        )
        print(f"== For reference: top {TOP_N} clusters across the FULL dataset ==")
        print(allk.to_string(index=False))
        print()

    print(
        "Reading guide:\n"
        "  yr_ratio_24/20 -> how much this topic participated in the secular "
        "slowdown (higher = slowed more).\n"
        "  2024_win/base  -> <1 means fact-checked FASTER during the 2024 "
        "election window than the rest of 2024.\n"
        "  Large clusters inside 'Other' are named topics the keyword taxonomy "
        "is currently missing -- candidates to promote into CLUSTERS (option 4)."
    )


if __name__ == "__main__":
    main()
