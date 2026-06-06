"""06_robustness.py -- confound + window-sensitivity robustness checks.

Reads ``data/processed/cofacts_latency.csv`` and reports:
  1. Window sensitivity: 2024/2020 median-latency ratio across window sizes.
  2. Secular trend: median latency per calendar year.
  3. Within-year election contrast: each election window vs its own year's
     out-of-window baseline.
  4. Log-linear model (linear year trend): election effects net of trend.
  5. Log-linear model (YEAR FIXED EFFECTS): within-year election effects.

Run:
    uv run python scripts/06_robustness.py
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
    window_sensitivity,
    per_year_median,
    within_year_election_contrast,
    loglinear_election_effect,
    loglinear_election_effect_year_fe,
)

CSV = PROC / "cofacts_latency.csv"
WIN_DAYS = [30, 45, 60, 90, 120, 180]


def load() -> pd.DataFrame:
    if not CSV.exists():
        sys.exit(f"Missing {CSV}; run scripts/01_clean.py first.")
    df = pd.read_csv(CSV)
    df["article_createdAt"] = parse_dates_safe(df["article_createdAt"])
    df["reply_createdAt"] = parse_dates_safe(df["reply_createdAt"])
    missing = df["article_createdAt"].isna() & df["reply_createdAt"].notna()
    if missing.any():
        df.loc[missing, "article_createdAt"] = reconstruct_article_dates(df.loc[missing])
    return df.dropna(subset=["article_createdAt"])


def main() -> None:
    df = load()
    print(f"Loaded {len(df):,} rows with usable article timestamps.\n")

    print("== 1. Window sensitivity (2024 vs 2020 median latency) ==")
    sens = window_sensitivity(df, WIN_DAYS)
    print(sens.to_string(index=False))
    print()

    print("== 2. Secular trend: median latency per year ==")
    for yr, med in per_year_median(df).items():
        print(f"  {int(yr)}: {med:8.1f} h")
    print()

    print("== 3. Within-year election contrast (window vs same-year baseline) ==")
    for anchor in (E2020, E2024):
        c = within_year_election_contrast(df, anchor)
        print(
            f"  {c['year']}: window median {c['median_window_h']:.1f} h "
            f"(N={c['n_window']:,}) vs baseline {c['median_baseline_h']:.1f} h "
            f"(N={c['n_baseline']:,}) -> {c['window_over_baseline']:.2f}x"
        )
    print()

    print("== 4. Log-linear model (LINEAR year trend) ==")
    m = loglinear_election_effect(df)
    print(
        f"  Secular trend:      x{m['year_trend_mult_per_yr']:.3f} per year "
        f"({m['year_trend_dex_per_yr']:+.3f} dex/yr)"
    )
    print(f"  2020 window effect: x{m['early_multiplier']:.3f} ({m['early_effect_dex']:+.3f} dex)")
    print(f"  2024 window effect: x{m['late_multiplier']:.3f} ({m['late_effect_dex']:+.3f} dex)")
    print("  (A linear year term underfits the non-monotonic trend -- see #5.)")
    print()

    print("== 5. Log-linear model (YEAR FIXED EFFECTS) ==")
    fe = loglinear_election_effect_year_fe(df)
    print(f"  Year dummies: {fe['n_year_dummies']}")
    print(
        f"  2020 window effect: x{fe['early_multiplier']:.3f} "
        f"({fe['early_effect_dex']:+.3f} dex), within-year"
    )
    print(
        f"  2024 window effect: x{fe['late_multiplier']:.3f} "
        f"({fe['late_effect_dex']:+.3f} dex), within-year"
    )
    print()
    print(
        "Interpretation: with year fixed effects absorbing the secular level, "
        "both election-window multipliers should fall below 1 -- i.e. faster "
        "than their own year -- reconciling the regression with check #3."
    )


if __name__ == "__main__":
    main()
