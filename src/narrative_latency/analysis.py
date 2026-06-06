"""Robustness + confound analysis helpers for the narrative-latency study.

Pure, side-effect-free functions used by ``scripts/06_robustness.py`` and
exercised by ``tests/test_analysis.py`` with synthetic data (no CSV required).

The central question these answer: the headline says the 2024 election window
is ~10x slower than the 2020 window, but reply latency also drifts upward over
time. Are we measuring an *election* effect or just the *secular* slowdown?
These helpers separate the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import E2020, E2024, WIN

DATE_COL = "article_createdAt"
VAL_COL = "latency_hours"
# Hours. Matches the dashboard's clip so log10 stays finite for ~0 latencies
# without dropping rows.
_LOG_FLOOR = 0.01


def in_window(dates, anchor, win=WIN):
    """Boolean mask for rows whose date is within +/- ``win`` of ``anchor``."""
    return (dates - anchor).abs() <= win


def window_latencies(df, anchor, win=WIN, date_col=DATE_COL, val_col=VAL_COL):
    """Latency values for rows inside the +/- ``win`` window around ``anchor``."""
    return df.loc[in_window(df[date_col], anchor, win), val_col]


def window_ratio(df, win=WIN, early=E2020, late=E2024, date_col=DATE_COL, val_col=VAL_COL):
    """Median latency in each election window and the late/early ratio."""
    e = window_latencies(df, early, win, date_col, val_col)
    l = window_latencies(df, late, win, date_col, val_col)
    me, ml = e.median(), l.median()
    return {
        "win_days": int(win.days),
        "n_early": int(e.shape[0]),
        "n_late": int(l.shape[0]),
        "median_early_h": float(me) if pd.notna(me) else np.nan,
        "median_late_h": float(ml) if pd.notna(ml) else np.nan,
        "ratio_late_over_early": float(ml / me) if me else np.nan,
    }


def window_sensitivity(df, win_days, early=E2020, late=E2024, date_col=DATE_COL, val_col=VAL_COL):
    """``window_ratio`` across a list of window sizes (in days) -> DataFrame."""
    rows = [
        window_ratio(df, pd.Timedelta(days=d), early, late, date_col, val_col)
        for d in win_days
    ]
    return pd.DataFrame(rows)


def per_year_median(df, date_col=DATE_COL, val_col=VAL_COL):
    """Median latency per calendar year (the secular trend)."""
    years = df[date_col].dt.year
    return df.assign(_year=years).groupby("_year")[val_col].median()


def within_year_election_contrast(df, anchor, win=WIN, date_col=DATE_COL, val_col=VAL_COL):
    """Election-window median vs the SAME calendar year's out-of-window median.

    Controls for the secular trend by comparing each election window only to
    its own year, isolating an election-specific effect from year-over-year
    drift.
    """
    year = int(anchor.year)
    yr = df[df[date_col].dt.year == year]
    mask = in_window(yr[date_col], anchor, win)
    win_med = yr.loc[mask, val_col].median()
    base_med = yr.loc[~mask, val_col].median()
    return {
        "year": year,
        "n_window": int(mask.sum()),
        "n_baseline": int((~mask).sum()),
        "median_window_h": float(win_med) if pd.notna(win_med) else np.nan,
        "median_baseline_h": float(base_med) if pd.notna(base_med) else np.nan,
        "window_over_baseline": float(win_med / base_med) if base_med else np.nan,
    }


def loglinear_election_effect(df, early=E2020, late=E2024, win=WIN, date_col=DATE_COL, val_col=VAL_COL):
    """OLS of log10(latency) on a centered year trend + per-election indicators.

    Disentangles the secular slowdown (year trend) from an election-window
    effect. Returns each election's multiplicative effect on latency
    (``10**coef``) net of the trend, plus the per-year trend multiplier.
    """
    d = df[[date_col, val_col]].copy()
    d = d.dropna(subset=[date_col])
    lat = pd.to_numeric(d[val_col], errors="coerce")
    d = d.assign(_lat=lat).dropna(subset=["_lat"])
    if d.empty:
        raise ValueError("No usable rows for loglinear_election_effect.")
    y = np.log10(d["_lat"].clip(lower=_LOG_FLOOR).to_numpy())
    year = d[date_col].dt.year.to_numpy().astype(float)
    year_c = year - year.mean()
    in_e = in_window(d[date_col], early, win).to_numpy().astype(float)
    in_l = in_window(d[date_col], late, win).to_numpy().astype(float)
    X = np.column_stack([np.ones_like(year_c), year_c, in_e, in_l])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {
        "n": int(d.shape[0]),
        "year_trend_dex_per_yr": float(coef[1]),
        "year_trend_mult_per_yr": float(10 ** coef[1]),
        "early_effect_dex": float(coef[2]),
        "early_multiplier": float(10 ** coef[2]),
        "late_effect_dex": float(coef[3]),
        "late_multiplier": float(10 ** coef[3]),
    }
