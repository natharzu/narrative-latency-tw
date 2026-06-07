"""07_cluster_latency.py -- narrative-cluster latency deep-dive.

Now that we know 2024 was a slow *year* (not a slow *election*), this script
breaks the pattern down by narrative. For each keyword-defined cluster
(``CLUSTERS`` / ``tag``) it reports:

  * overall N + median latency, and the 2020 vs 2024 yearly medians
    (how much each narrative participated in the platform-wide slowdown);
  * the within-year election-window vs same-year-baseline contrast
    (whether the election-window speed-up was uniform across narratives or
    concentrated in a few).

Run:
    uv run python scripts/07_cluster_latency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import (
    PROC,
    E2020,
    E2024,
    parse_dates_safe,
    reconstruct_article_dates,
    tag,
    per_year_median,
    within_year_election_contrast,
)

CSV = PROC / "cofacts_latency.csv"
# Fallback text column for CSVs that predate the full-text column (Option 0).
TEXT_COL = "text_preview"


def _round(x, n=1):
    return round(float(x), n) if pd.notna(x) else None


def load() -> pd.DataFrame:
    if not CSV.exists():
        sys.exit(f"Missing {CSV}; run scripts/01_clean.py first.")
    df = pd.read_csv(CSV)
    df["article_createdAt"] = parse_dates_safe(df["article_createdAt"])
    df["reply_createdAt"] = parse_dates_safe(df["reply_createdAt"])
    missing = df["article_createdAt"].isna() & df["reply_createdAt"].notna()
    if missing.any():
        df.loc[missing, "article_createdAt"] = reconstruct_article_dates(df.loc[missing])
    df = df.dropna(subset=["article_createdAt"])
    # Tag on FULL text when available (Option 0); fall back to the preview.
    text_col = "text" if "text" in df.columns else TEXT_COL
    df["narrative"] = df[text_col].apply(tag)
    return df


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows; tagged into narratives.\n")

    order = df["narrative"].value_counts().index.tolist()

    print("== Per-narrative overview (volume + secular slowdown) ==")
    rows = []
    for k in order:
        sub = df[df["narrative"] == k]
        pym = per_year_median(sub)
        m2020 = pym.get(2020, float("nan"))
        m2024 = pym.get(2024, float("nan"))
        ratio = (m2024 / m2020) if (pd.notna(m2020) and m2020) else float("nan")
        rows.append(
            {
                "narrative": k,
                "n": len(sub),
                "share_%": _round(100 * len(sub) / len(df)),
                "median_h": _round(sub["latency_hours"].median()),
                "med_2020_h": _round(m2020),
                "med_2024_h": _round(m2024),
                "yr_ratio_24/20": _round(ratio, 2),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("== Within-year election-window contrast, by narrative ==")
    for anchor, lbl in ((E2020, "2020"), (E2024, "2024")):
        print(f"\n-- {lbl} window vs same-year baseline --")
        crows = []
        for k in order:
            sub = df[df["narrative"] == k]
            c = within_year_election_contrast(sub, anchor)
            crows.append(
                {
                    "narrative": k,
                    "n_window": c["n_window"],
                    "median_window_h": _round(c["median_window_h"]),
                    "median_baseline_h": _round(c["median_baseline_h"]),
                    "window/baseline": _round(c["window_over_baseline"], 2),
                }
            )
        print(pd.DataFrame(crows).to_string(index=False))
    print()
    print(
        "Reading: window/baseline < 1 => that narrative was fact-checked FASTER "
        "during the election window than during the rest of that same year."
    )


if __name__ == "__main__":
    main()
