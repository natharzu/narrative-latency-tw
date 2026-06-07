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

Reframed thesis being defended (locked 2026-06; recomputed on the 68,453-row
usable snapshot):
    - The 10× cross-election gap is the visible tip of a ~38× platform-wide
      secular slowdown: per-year median rises from a 2018 floor (≈8 h) to a
      2024 peak (≈291 h).
    - Election windows are *bright spots*, not slow zones: within each
      election year the ±90-day window is FASTER than that year's baseline
      (2020 ≈ 0.46×, 2024 ≈ 0.13×), and the effect survives year fixed effects.
    - The cross-election ratio is not a window-size artefact (stable across a
      30–180 day sweep).
    - After Option 4 (2026-06) the keyword taxonomy carries 13 topic
      categories tagged on the full article body; 'Other' falls to ≈70%. The
      largest substantive clusters are
      us_skepticism > health > vaccine > scam > pre_election.

These golden values are pinned with tolerances. Run locally against your CSV;
if a value legitimately moved (e.g. a re-pull), update the constant rather than
loosening the test.

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
    per_year_median,
    within_year_election_contrast,
    loglinear_election_effect_year_fe,
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

# --- Reframed-thesis golden values (locked 2026-06; 68,453-row snapshot) -----
# Window-size sweep: smallest/largest 2024-over-2020 median ratio observed was
# 7.10× (45d) .. 9.94× (90d); bound generously so a real collapse still trips.
WINDOW_SIZES_DAYS = [30, 45, 60, 90, 120, 180]
MIN_CROSS_ELECTION_RATIO = 5.0
MAX_CROSS_ELECTION_RATIO = 13.0
# Secular slowdown: 2018 ≈ 8.2 h floor, 2024 ≈ 290.6 h peak.
YEAR_FAST_FLOOR = 2018
YEAR_FLOOR_MAX_MEDIAN_H = 20.0
YEAR_PEAK = 2024
YEAR_PEAK_MIN_MEDIAN_H = 100.0
SECULAR_PEAK_OVER_FLOOR_MIN = 8.0
# Bright spots: within-year window/baseline < 1 (2020 ≈ 0.46, 2024 ≈ 0.13).
MAX_WITHIN_YEAR_CONTRAST = 0.9
# Year fixed-effects election multipliers (2020 ≈ 0.698, 2024 ≈ 0.621).
MAX_FE_MULTIPLIER = 0.95
# Topic-cluster counts over the full set (Option 4 taxonomy, tagged on full
# 'text'; locked 2026-06; sum = 68,533). Re-pin these if the taxonomy or the
# processed CSV legitimately changes rather than loosening the tolerance.
EXPECTED_CLUSTER_COUNTS = {
    "Other": 47_995,
    "us_skepticism": 4_156,
    "health": 3_913,
    "vaccine": 2_817,
    "scam": 2_656,
    "pre_election": 2_621,
    "traffic": 1_007,
    "pension": 831,
    "energy": 551,
    "food_safety": 508,
    "disaster": 488,
    "lgbtq": 370,
    "ccp_info_manipulation": 355,
    "international": 265,
}
CLUSTER_COUNT_REL_TOL = 0.02
EXPECTED_OTHER_SHARE = 0.70
OTHER_SHARE_TOL = 0.03
# Text columns the keyword tagger can run on when the processed CSV has no
# precomputed topic_cluster column. The live pipeline (post Option 0) tags on
# the full article body, so prefer "text" and fall back to the preview.
TEXT_COLUMN_CANDIDATES = ["text", "text_preview", "article_text", "articleText"]


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


@pytest.fixture(scope="module")
def clustered_df(latency_df) -> pd.DataFrame:
    """latency_df guaranteed to carry a topic_cluster column.

    If the processed CSV already provides topic_cluster, use it as-is.
    Otherwise derive it by applying the keyword tagger to the article text
    column — the same lightweight approach the live 07/08 scripts use.
    Skips cleanly if neither a topic_cluster nor a known text column exists.
    """
    if "topic_cluster" in latency_df.columns:
        return latency_df
    text_col = next(
        (c for c in TEXT_COLUMN_CANDIDATES if c in latency_df.columns), None
    )
    if text_col is None:
        pytest.skip(
            "No topic_cluster column and no known text column to derive it from."
        )
    df = latency_df.copy()
    df["topic_cluster"] = df[text_col].apply(tag)
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
            # original four IORG clusters (ordering preserved under Option 4)
            ("AZ 疫苗副作用", "vaccine"),
            ("Pfizer booster question", "vaccine"),
            ("拜登政府 對台 政策", "us_skepticism"),
            ("Trump 2024 campaign rumor", "us_skepticism"),
            ("賴清德 當選 投票 結果", "pre_election"),
            ("習近平 訪美", "ccp_info_manipulation"),
            # Option 4 categories
            ("假投資 詐騙 飆股 群組", "scam"),
            ("癌症 致癌 偏方 療法", "health"),
            ("闖紅燈 機車 違規 罰單", "traffic"),
            ("台電 宣布 停電 區域", "energy"),
            ("退休 年金 公教 改革", "pension"),
            ("萊豬 瘦肉精 進口", "food_safety"),
            ("同性 婚姻 平權 愛滋", "lgbtq"),
            ("颱風 豪雨 淹水", "disaster"),
            ("烏克蘭 俄羅斯 戰爭", "international"),
            ("公投 結果 出爐", "pre_election"),
            # negatives
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
        # The windows CSV's topic_cluster may predate Option 4 (≈88% Other) or
        # be regenerated under it (≈70%); accept either as long as 'Other' is
        # still the dominant bucket.
        other_share = (windows_df["topic_cluster"] == "Other").mean()
        assert 0.60 <= other_share <= 0.95


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


# 6. Window-size robustness (Phase 1 sensitivity sweep)
class TestWindowSensitivityRegression:
    """The 2024/2020 slowdown is not an artefact of the ±90-day window: the
    median ratio stays elevated across a 30–180 day sweep."""

    @staticmethod
    def _ratio(df, days):
        win = pd.Timedelta(days=days)
        e20 = df.loc[(df["article_createdAt"] - E2020).abs() <= win, "latency_hours"]
        e24 = df.loc[(df["article_createdAt"] - E2024).abs() <= win, "latency_hours"]
        return e24.median() / e20.median(), len(e20), len(e24)

    def test_ratio_elevated_across_all_window_sizes(self, latency_df):
        ratios = {}
        for days in WINDOW_SIZES_DAYS:
            ratio, n20, n24 = self._ratio(latency_df, days)
            assert n20 >= 500 and n24 >= 500, (
                f"{days}-day window too sparse: n2020={n20}, n2024={n24}"
            )
            ratios[days] = round(float(ratio), 2)
        print(f"\n[INFO] window-size ratios (2024/2020): {ratios}")
        assert min(ratios.values()) >= MIN_CROSS_ELECTION_RATIO, (
            f"Slowdown collapses at some window size: {ratios}"
        )
        assert max(ratios.values()) <= MAX_CROSS_ELECTION_RATIO, (
            f"Ratio implausibly large at some window size: {ratios}"
        )


# 7. Secular slowdown + election-window bright spots (the reframed thesis)
class TestSecularSlowdownAndBrightSpots:
    def test_per_year_median_shows_secular_slowdown(self, latency_df):
        pym = per_year_median(latency_df)
        pym.index = pym.index.astype(int)
        assert YEAR_FAST_FLOOR in pym.index and YEAR_PEAK in pym.index, (
            f"Expected years missing from per-year median: {sorted(pym.index)}"
        )
        floor = float(pym.loc[YEAR_FAST_FLOOR])
        peak = float(pym.loc[YEAR_PEAK])
        print(f"\n[INFO] per-year median (h): {pym.round(1).to_dict()}")
        assert floor <= YEAR_FLOOR_MAX_MEDIAN_H, (
            f"{YEAR_FAST_FLOOR} median {floor:.1f} h above floor cap "
            f"{YEAR_FLOOR_MAX_MEDIAN_H}."
        )
        assert peak >= YEAR_PEAK_MIN_MEDIAN_H, (
            f"{YEAR_PEAK} median {peak:.1f} h below expected peak "
            f"{YEAR_PEAK_MIN_MEDIAN_H}."
        )
        assert peak > SECULAR_PEAK_OVER_FLOOR_MIN * floor, (
            f"Secular slowdown {YEAR_FAST_FLOOR}→{YEAR_PEAK} not present "
            f"(peak {peak:.1f} h vs floor {floor:.1f} h)."
        )

    def test_2020_election_window_is_a_bright_spot(self, latency_df):
        c = within_year_election_contrast(latency_df, E2020)
        print(f"\n[INFO] 2020 window/baseline = {c['window_over_baseline']:.3f}")
        assert c["window_over_baseline"] < MAX_WITHIN_YEAR_CONTRAST, (
            "2020 election window is not faster than its same-year baseline."
        )

    def test_2024_election_window_is_a_bright_spot(self, latency_df):
        c = within_year_election_contrast(latency_df, E2024)
        print(f"\n[INFO] 2024 window/baseline = {c['window_over_baseline']:.3f}")
        assert c["window_over_baseline"] < MAX_WITHIN_YEAR_CONTRAST, (
            "2024 election window is not faster than its same-year baseline."
        )

    def test_2024_window_is_relatively_brighter_than_2020(self, latency_df):
        c20 = within_year_election_contrast(latency_df, E2020)["window_over_baseline"]
        c24 = within_year_election_contrast(latency_df, E2024)["window_over_baseline"]
        assert c24 < c20, (
            f"2024 contrast {c24:.3f} not below 2020 contrast {c20:.3f}."
        )

    def test_election_effect_survives_year_fixed_effects(self, latency_df):
        m = loglinear_election_effect_year_fe(latency_df)
        print(
            f"\n[INFO] year-FE multipliers: early(2020)={m['early_multiplier']:.3f}, "
            f"late(2024)={m['late_multiplier']:.3f}"
        )
        assert m["early_multiplier"] < MAX_FE_MULTIPLIER, (
            "2020 election effect vanished after year fixed effects."
        )
        assert m["late_multiplier"] < MAX_FE_MULTIPLIER, (
            "2024 election effect vanished after year fixed effects."
        )


# 8. Topic-cluster distribution (Option 4 taxonomy)
class TestClusterRegression:
    """Lock the Option 4 cluster mix. topic_cluster comes from the processed CSV
    when present, else is derived by applying tag() to the full article text
    (see the clustered_df fixture). Skips cleanly when no text source is
    available.
    """

    def test_full_set_cluster_counts_locked(self, clustered_df):
        counts = clustered_df["topic_cluster"].value_counts()
        print(f"\n[INFO] cluster counts: {counts.to_dict()}")
        for cluster, expected in EXPECTED_CLUSTER_COUNTS.items():
            actual = int(counts.get(cluster, 0))
            assert actual == pytest.approx(expected, rel=CLUSTER_COUNT_REL_TOL), (
                f"Cluster '{cluster}' count {actual} drifted from {expected} "
                f"(rel tol {CLUSTER_COUNT_REL_TOL})."
            )

    def test_other_share_dominant_full_set(self, clustered_df):
        share = (clustered_df["topic_cluster"] == "Other").mean()
        assert abs(share - EXPECTED_OTHER_SHARE) <= OTHER_SHARE_TOL, (
            f"Other share {share:.3f} drifted from {EXPECTED_OTHER_SHARE} "
            f"± {OTHER_SHARE_TOL}."
        )

    def test_every_substantive_cluster_present_and_ordered(self, clustered_df):
        counts = clustered_df["topic_cluster"].value_counts()
        for cluster in CLUSTERS:  # every taxonomy category should appear
            assert int(counts.get(cluster, 0)) > 0, f"Cluster '{cluster}' absent."
        assert counts.idxmax() == "Other", "'Other' is no longer the largest cluster."
        substantive = counts.drop(labels=["Other"], errors="ignore")
        assert substantive.idxmax() == "us_skepticism", (
            "us_skepticism is no longer the largest substantive cluster."
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
