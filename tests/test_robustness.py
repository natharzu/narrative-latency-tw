"""
tests/test_robustness.py - robustness & invariant tests for narrative_latency.

Where test_units.py / test_analysis.py check *correctness* on hand-built cases,
this module checks *robustness*: properties that must survive perturbation of
the inputs (row order, global rescaling, injected NaNs, far-out-of-window
rows), plus numerical and degenerate-input edge cases. All synthetic - no
dependency on the gitignored processed CSVs, so the whole file runs in CI on
every push.

Run:
    uv run pytest tests/test_robustness.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from narrative_latency import (
    E2020,
    E2024,
    WIN,
    in_window,
    window_ratio,
    window_sensitivity,
    per_year_median,
    within_year_election_contrast,
    loglinear_election_effect,
    loglinear_election_effect_year_fe,
    parse_dates_safe,
    reconstruct_article_dates,
    tag,
)


def _ts(*dates):
    return pd.to_datetime(list(dates), utc=True)


def _windows_df():
    """Two early-window rows (5h) and two late-window rows (50h); ratio = 10x."""
    return pd.DataFrame(
        {
            "article_createdAt": _ts(
                "2020-01-11", "2020-01-20", "2024-01-13", "2024-01-20"
            ),
            "latency_hours": [5.0, 5.0, 50.0, 50.0],
        }
    )


def _trend_df():
    """Flat 10h mid-year baseline 2018-2024; 2020 window faster (5h), 2024
    window slower (50h). Full-rank design for both regressions."""
    dates, lat = [], []
    for y in range(2018, 2025):
        for _ in range(100):
            dates.append(pd.Timestamp(f"{y}-06-01", tz="UTC"))
            lat.append(10.0)
    for _ in range(100):
        dates.append(pd.Timestamp("2020-01-11", tz="UTC"))
        lat.append(5.0)
    for _ in range(100):
        dates.append(pd.Timestamp("2024-01-13", tz="UTC"))
        lat.append(50.0)
    return pd.DataFrame({"article_createdAt": dates, "latency_hours": lat})


def _shuffle(df, seed=0):
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class TestOrderInvariance:
    """Aggregates must not depend on row order."""

    def test_window_ratio_invariant_to_shuffle(self):
        df = _windows_df()
        assert window_ratio(df) == window_ratio(_shuffle(df, 1))

    def test_per_year_median_invariant_to_shuffle(self):
        df = _trend_df()
        a = per_year_median(df).sort_index()
        b = per_year_median(_shuffle(df, 2)).sort_index()
        pd.testing.assert_series_equal(a, b)

    def test_loglinear_invariant_to_shuffle(self):
        df = _trend_df()
        a = loglinear_election_effect(df)
        b = loglinear_election_effect(_shuffle(df, 3))
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=1e-6, abs=1e-9)

    def test_loglinear_fe_invariant_to_shuffle(self):
        df = _trend_df()
        a = loglinear_election_effect_year_fe(df)
        b = loglinear_election_effect_year_fe(_shuffle(df, 4))
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=1e-6, abs=1e-9)


class TestScaleInvariance:
    """Multiplying every latency by k rescales medians by k but leaves
    dimensionless quantities (ratios, log-linear multipliers) unchanged."""

    K = 1000.0

    def test_window_ratio_is_scale_free(self):
        df = _windows_df()
        scaled = df.assign(latency_hours=df["latency_hours"] * self.K)
        r0, r1 = window_ratio(df), window_ratio(scaled)
        assert r1["ratio_late_over_early"] == pytest.approx(
            r0["ratio_late_over_early"]
        )
        assert r1["median_early_h"] == pytest.approx(r0["median_early_h"] * self.K)
        assert r1["median_late_h"] == pytest.approx(r0["median_late_h"] * self.K)

    def test_per_year_median_scales_linearly(self):
        df = _trend_df()
        scaled = df.assign(latency_hours=df["latency_hours"] * self.K)
        pd.testing.assert_series_equal(
            per_year_median(scaled).sort_index(),
            (per_year_median(df).sort_index() * self.K),
        )

    def test_loglinear_multipliers_are_scale_free(self):
        df = _trend_df()
        scaled = df.assign(latency_hours=df["latency_hours"] * self.K)
        a, b = loglinear_election_effect(df), loglinear_election_effect(scaled)
        for k in (
            "early_multiplier",
            "late_multiplier",
            "year_trend_mult_per_yr",
        ):
            assert b[k] == pytest.approx(a[k], rel=1e-6)

    def test_loglinear_fe_multipliers_are_scale_free(self):
        df = _trend_df()
        scaled = df.assign(latency_hours=df["latency_hours"] * self.K)
        a = loglinear_election_effect_year_fe(df)
        b = loglinear_election_effect_year_fe(scaled)
        assert b["early_multiplier"] == pytest.approx(a["early_multiplier"], rel=1e-6)
        assert b["late_multiplier"] == pytest.approx(a["late_multiplier"], rel=1e-6)


class TestNaNRobustness:
    """Injected NaNs (missing latency / unparseable dates) must be ignored,
    not crash or shift the estimates."""

    def test_window_ratio_ignores_nan_latency(self):
        df = _windows_df()
        noisy = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "article_createdAt": _ts("2020-01-12", "2024-01-14"),
                        "latency_hours": [np.nan, np.nan],
                    }
                ),
            ],
            ignore_index=True,
        )
        r0, r1 = window_ratio(df), window_ratio(noisy)
        assert r1["median_early_h"] == pytest.approx(r0["median_early_h"])
        assert r1["median_late_h"] == pytest.approx(r0["median_late_h"])
        assert r1["ratio_late_over_early"] == pytest.approx(
            r0["ratio_late_over_early"]
        )

    def test_loglinear_ignores_nan_rows(self):
        df = _trend_df()
        noisy = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "article_createdAt": [pd.NaT, pd.Timestamp("2022-06-01", tz="UTC")],
                        "latency_hours": [12.0, np.nan],
                    }
                ),
            ],
            ignore_index=True,
        )
        a, b = loglinear_election_effect(df), loglinear_election_effect(noisy)
        assert b["n"] == a["n"]  # both injected rows dropped
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12)

    def test_loglinear_fe_ignores_nan_rows(self):
        df = _trend_df()
        noisy = pd.concat(
            [df, pd.DataFrame({"article_createdAt": [pd.NaT], "latency_hours": [99.0]})],
            ignore_index=True,
        )
        a = loglinear_election_effect_year_fe(df)
        b = loglinear_election_effect_year_fe(noisy)
        assert b["n"] == a["n"]
        assert b["early_multiplier"] == pytest.approx(a["early_multiplier"], rel=1e-9)
        assert b["late_multiplier"] == pytest.approx(a["late_multiplier"], rel=1e-9)

    def test_per_year_median_drops_nat_year(self):
        df = _trend_df()
        noisy = pd.concat(
            [df, pd.DataFrame({"article_createdAt": [pd.NaT], "latency_hours": [99.0]})],
            ignore_index=True,
        )
        pym = per_year_median(noisy)
        ref = per_year_median(df)
        # The NaT row contributes no year bucket (no NaN in the index) and
        # leaves every real year's median untouched. Compare sorted values so
        # the assertion is robust to int-vs-float index dtype (a stray NaT
        # coerces the grouping key to float).
        assert not pym.index.isna().any()
        assert len(pym) == len(ref)
        np.testing.assert_allclose(
            pym.sort_index().to_numpy(), ref.sort_index().to_numpy()
        )


class TestLocality:
    """Rows far outside both election windows must not move window results."""

    def test_window_ratio_unaffected_by_far_rows(self):
        df = _windows_df()
        far = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "article_createdAt": _ts("2010-01-01", "2030-01-01"),
                        "latency_hours": [9999.0, 9999.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        assert window_ratio(df) == window_ratio(far)

    def test_per_year_median_year_independence(self):
        df = _trend_df()
        base = per_year_median(df).sort_index()
        extra = pd.concat(
            [df, pd.DataFrame({"article_createdAt": _ts("2030-06-01"), "latency_hours": [1.0]})],
            ignore_index=True,
        )
        after = per_year_median(extra).sort_index()
        # Pre-existing years are unchanged; only a new bucket appears.
        pd.testing.assert_series_equal(after.loc[base.index], base)
        assert 2030 in after.index


class TestDegenerateInputs:
    def test_window_ratio_empty_windows_is_nan(self):
        df = pd.DataFrame(
            {"article_createdAt": _ts("2010-01-01", "2030-01-01"), "latency_hours": [3.0, 4.0]}
        )
        r = window_ratio(df)
        assert r["n_early"] == 0 and r["n_late"] == 0
        assert np.isnan(r["median_early_h"]) and np.isnan(r["median_late_h"])
        assert np.isnan(r["ratio_late_over_early"])

    def test_within_year_contrast_no_baseline_is_nan(self):
        df = pd.DataFrame(
            {"article_createdAt": _ts("2024-01-13", "2024-01-20"), "latency_hours": [40.0, 60.0]}
        )
        c = within_year_election_contrast(df, E2024)
        assert c["n_baseline"] == 0
        assert np.isnan(c["median_baseline_h"])
        assert np.isnan(c["window_over_baseline"])

    def test_loglinear_raises_on_empty(self):
        df = pd.DataFrame({"article_createdAt": _ts(), "latency_hours": pd.Series([], dtype=float)})
        with pytest.raises(ValueError):
            loglinear_election_effect(df)

    def test_loglinear_raises_when_all_latency_nan(self):
        df = pd.DataFrame(
            {"article_createdAt": _ts("2020-06-01", "2024-06-01"), "latency_hours": [np.nan, np.nan]}
        )
        with pytest.raises(ValueError):
            loglinear_election_effect(df)

    def test_loglinear_fe_raises_on_empty(self):
        df = pd.DataFrame({"article_createdAt": _ts(), "latency_hours": pd.Series([], dtype=float)})
        with pytest.raises(ValueError):
            loglinear_election_effect_year_fe(df)


class TestLogFloorNumerics:
    """Zero / sub-floor latencies must clip (not -inf) so the fit stays finite."""

    def _df_with_zeros(self):
        df = _trend_df()
        zeros = pd.DataFrame(
            {
                "article_createdAt": [pd.Timestamp("2019-06-01", tz="UTC")] * 20,
                "latency_hours": [0.0] * 20,
            }
        )
        return pd.concat([df, zeros], ignore_index=True)

    def test_loglinear_finite_with_zero_latency(self):
        m = loglinear_election_effect(self._df_with_zeros())
        assert all(np.isfinite(v) for v in m.values())

    def test_loglinear_fe_finite_with_zero_latency(self):
        m = loglinear_election_effect_year_fe(self._df_with_zeros())
        assert all(np.isfinite(v) for v in m.values())


class TestWindowBoundary:
    """+/- WIN is inclusive at the edge and exclusive just past it."""

    def test_exactly_plus_window_is_inside(self):
        s = pd.Series([E2020 + WIN])
        assert in_window(s, E2020).tolist() == [True]

    def test_exactly_minus_window_is_inside(self):
        s = pd.Series([E2020 - WIN])
        assert in_window(s, E2020).tolist() == [True]

    def test_one_second_past_window_is_outside(self):
        s = pd.Series([E2020 + WIN + pd.Timedelta(seconds=1)])
        assert in_window(s, E2020).tolist() == [False]


class TestWindowSensitivityMonotonic:
    def test_counts_non_decreasing_in_window_size(self):
        df = pd.DataFrame(
            {
                "article_createdAt": _ts(
                    "2020-01-21", "2020-02-25", "2020-04-20", "2020-07-29",
                    "2024-01-23", "2024-02-27", "2024-04-22", "2024-07-31",
                ),
                "latency_hours": [1.0] * 8,
            }
        )
        out = window_sensitivity(df, [30, 90, 180, 365])
        assert list(out["win_days"]) == [30, 90, 180, 365]
        assert out["n_early"].is_monotonic_increasing
        assert out["n_late"].is_monotonic_increasing


class TestDeterminism:
    def test_loglinear_repeatable(self):
        df = _trend_df()
        assert loglinear_election_effect(df) == loglinear_election_effect(df)

    def test_loglinear_fe_repeatable(self):
        df = _trend_df()
        assert (
            loglinear_election_effect_year_fe(df)
            == loglinear_election_effect_year_fe(df)
        )


class TestDataioInvariants:
    def test_parse_dates_safe_is_idempotent(self):
        once = parse_dates_safe(
            pd.Series(["2020-01-11 00:00:00+00:00", "2024-01-13T05:30:00Z"])
        )
        twice = parse_dates_safe(once)
        pd.testing.assert_series_equal(once, twice)

    def test_parse_dates_safe_never_raises_on_garbage(self):
        out = parse_dates_safe(pd.Series(["", "not a date", None, "2020-13-99"]))
        assert out.isna().all()

    def test_reconstruct_then_recompute_roundtrips(self):
        reply = pd.Series(
            ["2020-01-10 06:00:00+00:00", "2024-02-01 18:45:00+00:00", "2019-12-31 23:59:00+00:00"]
        )
        latency = [36.5, 0.25, 250.0]
        df = pd.DataFrame({"reply_createdAt": reply, "latency_hours": latency})
        article = reconstruct_article_dates(df)
        recomputed = (parse_dates_safe(reply) - article).dt.total_seconds() / 3600.0
        np.testing.assert_allclose(recomputed.to_numpy(), latency, atol=1e-6)


class TestTagRobustness:
    def test_keyword_inside_longer_sentence(self):
        assert tag("請問這則 疫苗 訊息是真的嗎") == "vaccine"

    def test_english_keyword_with_surrounding_text(self):
        assert tag("just got my Pfizer booster today") == "vaccine"

    def test_first_match_precedence_is_stable(self):
        # Hits both 'vaccine' and 'pre_election' keywords; dict order wins.
        assert tag("疫苗 與 選舉 議題") == "vaccine"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
