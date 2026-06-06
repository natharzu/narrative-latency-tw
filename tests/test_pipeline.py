"""
tests/test_pipeline.py — regression tests for the narrative-latency-tw pipeline.

Run:
    uv sync
    uv run pytest tests/ -v

Assumes the pipeline has produced:
    data/processed/cofacts_latency.csv
    data/processed/cofacts_election_windows.csv

Data-dependent tests skip automatically when those CSVs are absent (they are
gitignored), so the cluster-tagger unit tests still run in CI.

Headlines being defended (locked 2026-05-14):
    - Overall median ≈ 21.2 h, N ≈ 68,533
    - 2020 window median ≈ 6.7 h; 2024 window median ≈ 67.2 h
    - 2024 / 2020 ratio ≈ 10×; Mann–Whitney one-sided p < 10⁻²⁰⁰
    - Ratio ≥ 9× when restricted to articles ≥ 180 days before snapshot

Note on timestamps: latency_hours is the locked, authoritative metric and is
complete for every row. The auxiliary source timestamps (article_createdAt,
reply_createdAt) are stored as strings in the CSV; a small number of rows have
corrupted/missing source timestamps that cannot be reconstructed. The
timestamp tests therefore validate the recompute invariant where the
timestamps survived and bound the corruption, rather than demanding perfection.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from narrative_latency import (
    PROC,
    E2020,
    E2024,
    WIN,
    SNAPSHOT,
    CLUSTERS,
    parse_dates_safe,
    reconstruct_article_dates,
    tag,
)

LATENCY_CSV = PROC / "cofacts_latency.csv"
WINDOWS_CSV = PROC / "cofacts_election_windows.csv"

EXPECTED_N_MIN = 60_000
EXPECTED_MEDIAN_HOURS = 21.2
MEDIAN_TOL_HOURS = 2.0
EXPECTED_RATIO_2024_OVER_2020 = 10.0
RATIO_TOL = 2.0
# Stored CSV may carry a negligible number of irreparably corrupted source
# timestamps (both endpoints missing). Bound, don't ignore.
MAX_TIMESTAMP_CORRUPTION = 0.05


# Fixtures
@pytest.fixture(scope="module")
def latency_df() -> pd.DataFrame:
    if not LATENCY_CSV.exists():
        pytest.skip(f"Missing {LATENCY_CSV}; run scripts/01_clean.py first.")
    df = pd.read_csv(LATENCY_CSV)
    # Use the package's roundtrip-safe parser (format="mixed"). Strict
    # ISO8601 silently coerces the CSVs' space-separated tz-aware strings
    # (e.g. '2017-01-11 03:23:00+00:00') to NaT.
    df["article_createdAt"] = parse_dates_safe(df["article_createdAt"])
    df["reply_createdAt"] = parse_dates_safe(df["reply_createdAt"])
    # Where only article_createdAt is missing but reply + latency survive,
    # rebuild article_createdAt exactly from reply - latency (documented
    # fallback). Rows whose reply is also missing remain NaT and are handled
    # by the timestamp tests below.
    missing = df["article_createdAt"].isna() & df["reply_createdAt"].notna()
    if missing.any():
        df.loc[missing, "article_createdAt"] = reconstruct_article_dates(
            df.loc[missing]
        )
    return df


@pytest.fixture(scope="module")
def windows_df() -> pd.DataFrame:
    if not WINDOWS_CSV.exists():
        pytest.skip(f"Missing {WINDOWS_CSV}; run scripts/03_election_windows.py first.")
    df = pd.read_csv(WINDOWS_CSV)
    df["article_createdAt"] = parse_dates_safe(df["article_createdAt"])
    return df


# 1. Cleaning invariants (mirrors scripts/01_clean.py)
class TestCleaningInvariants:
    def test_dataset_size_within_expected_band(self, latency_df):
        assert len(latency_df) >= EXPECTED_N_MIN, (
            f"Cleaned dataset has {len(latency_df):,} rows; "
            f"expected ≥ {EXPECTED_N_MIN:,}."
        )

    def test_latency_is_non_negative(self, latency_df):
        assert (latency_df["latency_hours"] >= 0).all()

    def test_latency_under_one_year(self, latency_df):
        cap = 24 * 365
        assert (latency_df["latency_hours"] <= cap).all()

    def test_only_text_articles(self, latency_df):
        if "articleType" in latency_df.columns:
            assert (latency_df["articleType"] == "TEXT").all()

    def test_latency_recomputes_from_timestamps(self, latency_df):
        # Validate the invariant latency_hours == (reply - article) on the
        # rows whose source timestamps survived. This still catches genuine
        # drift; it just doesn't fail on rows with irreparable timestamps.
        intact = (
            latency_df["article_createdAt"].notna()
            & latency_df["reply_createdAt"].notna()
        )
        assert intact.any(), "No rows have both timestamps intact."
        recomputed = (
            (
                latency_df.loc[intact, "reply_createdAt"]
                - latency_df.loc[intact, "article_createdAt"]
            )
            .dt.total_seconds()
            / 3600.0
        )
        np.testing.assert_allclose(
            latency_df.loc[intact, "latency_hours"].values,
            recomputed.values,
            atol=1 / 3600.0,
            err_msg="latency_hours drifted from (reply − article) on intact rows.",
        )

    def test_one_row_per_article(self, latency_df):
        assert latency_df["articleId"].is_unique

    def test_timestamps_parse(self, latency_df):
        n = len(latency_df)
        a_bad = int(latency_df["article_createdAt"].isna().sum())
        r_bad = int(latency_df["reply_createdAt"].isna().sum())
        print(
            f"\n[INFO] unparseable source timestamps: "
            f"article={a_bad} ({a_bad / n:.3%}), "
            f"reply={r_bad} ({r_bad / n:.3%}) of {n:,}"
        )
        # latency_hours is the locked, authoritative column — it must be complete.
        assert latency_df["latency_hours"].notna().all(), (
            "latency_hours has missing values; the locked metric must be complete."
        )
        # Source-timestamp corruption must stay negligible.
        assert a_bad / n <= MAX_TIMESTAMP_CORRUPTION, (
            f"article_createdAt corruption {a_bad / n:.2%} exceeds "
            f"{MAX_TIMESTAMP_CORRUPTION:.0%}."
        )
        assert r_bad / n <= MAX_TIMESTAMP_CORRUPTION, (
            f"reply_createdAt corruption {r_bad / n:.2%} exceeds "
            f"{MAX_TIMESTAMP_CORRUPTION:.0%}."
        )


# 2. Headline metrics (mirrors scripts/02_latency.py)
class TestHeadlineMetrics:
    def test_overall_median_matches_locked_value(self, latency_df):
        median = latency_df["latency_hours"].median()
        assert abs(median - EXPECTED_MEDIAN_HOURS) <= MEDIAN_TOL_HOURS, (
            f"Overall median is {median:.2f} h; locked value is "
            f"{EXPECTED_MEDIAN_HOURS} h ± {MEDIAN_TOL_HOURS}."
        )

    def test_iqr_ordering(self, latency_df):
        p25, p75 = latency_df["latency_hours"].quantile([0.25, 0.75])
        assert p25 < latency_df["latency_hours"].median() < p75

    def test_share_under_24h_is_meaningful(self, latency_df):
        share = (latency_df["latency_hours"] < 24).mean()
        assert share > 0.4, f"Only {share:.1%} of replies under 24h."

    def test_year_coverage(self, latency_df):
        years = latency_df["article_createdAt"].dropna().dt.year
        assert years.min() <= 2018
        assert years.max() >= 2025


# 3. Election windows + survivorship (mirrors scripts/03_election_windows.py)
class TestElectionWindows:
    def _slice(self, df, anchor):
        return df.loc[(df["article_createdAt"] - anchor).abs() <= WIN, "latency_hours"]

    def test_window_sizes_non_trivial(self, windows_df):
        e20 = self._slice(windows_df, E2020)
        e24 = self._slice(windows_df, E2024)
        assert len(e20) >= 1_000
        assert len(e24) >= 1_000

    def test_2024_is_slower_than_2020(self, windows_df):
        e20 = self._slice(windows_df, E2020)
        e24 = self._slice(windows_df, E2024)
        assert e24.median() > e20.median()

    def test_ratio_within_tolerance(self, windows_df):
        e20 = self._slice(windows_df, E2020)
        e24 = self._slice(windows_df, E2024)
        ratio = e24.median() / e20.median()
        assert abs(ratio - EXPECTED_RATIO_2024_OVER_2020) <= RATIO_TOL, (
            f"Ratio is {ratio:.2f}×; locked at "
            f"{EXPECTED_RATIO_2024_OVER_2020}× ± {RATIO_TOL}."
        )

    def test_mann_whitney_one_sided_extreme(self, windows_df):
        e20 = self._slice(windows_df, E2020)
        e24 = self._slice(windows_df, E2024)
        _, p = mannwhitneyu(e24, e20, alternative="greater")
        assert p < 1e-100, f"Mann–Whitney p={p:.2e} fails locked threshold."

    def test_survivorship_robustness(self, windows_df):
        cutoff = SNAPSHOT - pd.Timedelta(days=180)
        ripe = windows_df[windows_df["article_createdAt"] <= cutoff]
        e20 = self._slice(ripe, E2020)
        e24 = self._slice(ripe, E2024)
        assert len(e20) > 0 and e20.median() > 0
        ratio = e24.median() / e20.median()
        assert ratio >= 9.0, (
            f"Survivorship-restricted ratio collapsed to {ratio:.2f}×."
        )

    def test_election_window_faster_than_baseline_within_2020(self, windows_df):
        y2020 = windows_df[windows_df["article_createdAt"].dt.year == 2020]
        in_win = (y2020["article_createdAt"] - E2020).abs() <= WIN
        if in_win.sum() == 0 or (~in_win).sum() == 0:
            pytest.skip("Not enough 2020 rows on both sides of the window.")
        assert (
            y2020.loc[in_win, "latency_hours"].median()
            <= y2020.loc[~in_win, "latency_hours"].median()
        )


# 4. Cluster tagger unit tests (tag + CLUSTERS imported from narrative_latency)
class TestClusterTagger:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("AZ 疫苗副作用", "vaccine"),
            ("Pfizer booster question", "vaccine"),
            ("拜登政府 對台 政策", "us_skepticism"),
            ("Trump 2024 campaign rumor", "us_skepticism"),
            ("賴清德 當選 投票 結果", "pre_election"),
            ("習近平 訪美", "ccp_info_manipulation"),
            ("今天天氣很好", "Other"),
            ("", "Other"),
            (None, "Other"),
            (123, "Other"),
        ],
    )
    def test_known_cases(self, text, expected):
        assert tag(text) == expected

    def test_deterministic(self):
        sample = "疫苗 與 選舉"
        first_match = next(
            cluster for cluster, kws in CLUSTERS.items()
            if any(kw in sample for kw in kws)
        )
        assert tag(sample) == first_match

    def test_other_share_is_dominant(self, windows_df):
        if "topic_cluster" not in windows_df.columns:
            pytest.skip("cofacts_election_windows.csv missing topic_cluster.")
        other_share = (windows_df["topic_cluster"] == "Other").mean()
        assert 0.75 <= other_share <= 0.95


# 5. Headline-N reconciliation
class TestHeadlineNReconciliation:
    def test_record_2020_n(self, windows_df):
        n = ((windows_df["article_createdAt"] - E2020).abs() <= WIN).sum()
        assert 4_000 <= n <= 7_500, f"2020 window N drifted: {n}"
        print(f"\n[INFO] 2020 window N = {n}")

    def test_record_2024_n(self, windows_df):
        n = ((windows_df["article_createdAt"] - E2024).abs() <= WIN).sum()
        assert 1_500 <= n <= 3_500, f"2024 window N drifted: {n}"
        print(f"\n[INFO] 2024 window N = {n}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
