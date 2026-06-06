"""Unit + regression tests for narrative_latency.analysis.

All synthetic: no dependency on the (gitignored) processed CSVs, so these run
in CI on every push.
"""

import numpy as np
import pandas as pd
import pytest

from narrative_latency import (
    E2020,
    E2024,
    in_window,
    window_ratio,
    window_sensitivity,
    per_year_median,
    within_year_election_contrast,
    loglinear_election_effect,
    loglinear_election_effect_year_fe,
)


def _ts(*dates):
    return pd.to_datetime(list(dates), utc=True)


class TestInWindow:
    def test_basic_membership(self):
        dates = pd.Series(_ts("2020-01-11", "2020-02-01", "2020-06-01"))
        mask = in_window(dates, E2020)
        assert mask.tolist() == [True, True, False]

    def test_symmetric_around_anchor(self):
        # +/- 90 days inclusive; 91 days out.
        dates = pd.Series(_ts("2023-10-15", "2024-04-12", "2024-04-13"))
        mask = in_window(dates, E2024)
        # 2023-10-15 is ~90d before, 2024-04-12 ~90d after -> in; 04-13 ~91d out
        assert mask.tolist() == [True, True, False]


class TestWindowRatio:
    def _df(self):
        return pd.DataFrame(
            {
                "article_createdAt": _ts(
                    "2020-01-11", "2020-01-20", "2024-01-13", "2024-01-20"
                ),
                "latency_hours": [5.0, 5.0, 50.0, 50.0],
            }
        )

    def test_ratio_and_counts(self):
        r = window_ratio(self._df())
        assert r["n_early"] == 2 and r["n_late"] == 2
        assert r["median_early_h"] == 5.0
        assert r["median_late_h"] == 50.0
        assert r["ratio_late_over_early"] == 10.0

    def test_empty_window_is_nan(self):
        df = pd.DataFrame(
            {"article_createdAt": _ts("2010-01-01"), "latency_hours": [3.0]}
        )
        r = window_ratio(df)
        assert r["n_early"] == 0 and r["n_late"] == 0
        assert np.isnan(r["ratio_late_over_early"])


class TestWindowSensitivity:
    def test_one_row_per_window(self):
        df = pd.DataFrame(
            {
                "article_createdAt": _ts("2020-01-11", "2024-01-13"),
                "latency_hours": [4.0, 40.0],
            }
        )
        out = window_sensitivity(df, [30, 90, 180])
        assert list(out["win_days"]) == [30, 90, 180]
        assert (out["ratio_late_over_early"] == 10.0).all()


class TestPerYearMedian:
    def test_medians_by_year(self):
        df = pd.DataFrame(
            {
                "article_createdAt": _ts(
                    "2020-06-01", "2020-07-01", "2024-06-01", "2024-07-01"
                ),
                "latency_hours": [4.0, 6.0, 40.0, 60.0],
            }
        )
        pym = per_year_median(df)
        assert pym.loc[2020] == 5.0
        assert pym.loc[2024] == 50.0


class TestWithinYearContrast:
    def test_window_vs_same_year_baseline(self):
        df = pd.DataFrame(
            {
                "article_createdAt": _ts(
                    "2024-01-13", "2024-01-20", "2024-07-01", "2024-08-01"
                ),
                "latency_hours": [60.0, 60.0, 20.0, 20.0],
            }
        )
        c = within_year_election_contrast(df, E2024)
        assert c["year"] == 2024
        assert c["n_window"] == 2 and c["n_baseline"] == 2
        assert c["median_window_h"] == 60.0
        assert c["median_baseline_h"] == 20.0
        assert c["window_over_baseline"] == 3.0


class TestLogLinearEffect:
    def _df(self):
        dates, lat = [], []
        # Flat baseline at 10h, mid-year (out of windows), 2018..2024.
        for y in range(2018, 2025):
            for _ in range(100):
                dates.append(pd.Timestamp(f"{y}-06-01", tz="UTC"))
                lat.append(10.0)
        # 2020 window faster (5h), 2024 window slower (50h).
        for _ in range(100):
            dates.append(pd.Timestamp("2020-01-11", tz="UTC"))
            lat.append(5.0)
        for _ in range(100):
            dates.append(pd.Timestamp("2024-01-13", tz="UTC"))
            lat.append(50.0)
        return pd.DataFrame({"article_createdAt": dates, "latency_hours": lat})

    def test_separates_trend_from_election(self):
        m = loglinear_election_effect(self._df())
        # Baseline is flat across years -> trend near zero.
        assert abs(m["year_trend_dex_per_yr"]) < 0.05
        # 2020 window faster than baseline, 2024 slower, net of trend.
        assert m["early_multiplier"] < 1.0
        assert m["late_multiplier"] > 1.0

    def test_recovers_secular_trend(self):
        # Pure trend: +0.1 dex/yr, no election rows in/out distinction.
        dates, lat = [], []
        for y in range(2018, 2025):
            level = 10 ** (1.0 + 0.1 * (y - 2021))
            for _ in range(50):
                dates.append(pd.Timestamp(f"{y}-06-01", tz="UTC"))
                lat.append(level)
        df = pd.DataFrame({"article_createdAt": dates, "latency_hours": lat})
        m = loglinear_election_effect(df)
        assert m["year_trend_mult_per_yr"] == pytest.approx(10 ** 0.1, rel=0.05)

    def test_raises_on_empty(self):
        df = pd.DataFrame({"article_createdAt": [], "latency_hours": []})
        df["article_createdAt"] = pd.to_datetime(df["article_createdAt"], utc=True)
        with pytest.raises(ValueError):
            loglinear_election_effect(df)


class TestLogLinearYearFE:
    def _df(self):
        # Rising baseline level by year; each election window is exactly half
        # of its OWN year's baseline (i.e. faster within-year).
        dates, lat = [], []
        levels = {y: 10.0 * (1.5 ** (y - 2018)) for y in range(2018, 2025)}
        for y in range(2018, 2025):
            for _ in range(100):
                dates.append(pd.Timestamp(f"{y}-06-01", tz="UTC"))
                lat.append(levels[y])
        for _ in range(100):
            dates.append(pd.Timestamp("2020-01-11", tz="UTC"))
            lat.append(levels[2020] * 0.5)
        for _ in range(100):
            dates.append(pd.Timestamp("2024-01-13", tz="UTC"))
            lat.append(levels[2024] * 0.5)
        return pd.DataFrame({"article_createdAt": dates, "latency_hours": lat})

    def test_fe_recovers_within_year_window_effect(self):
        fe = loglinear_election_effect_year_fe(self._df())
        # Both windows are half their own year -> multiplier ~0.5, < 1.
        assert fe["early_multiplier"] < 1.0
        assert fe["late_multiplier"] < 1.0
        assert fe["early_multiplier"] == pytest.approx(0.5, rel=0.05)
        assert fe["late_multiplier"] == pytest.approx(0.5, rel=0.05)

    def test_raises_on_empty(self):
        df = pd.DataFrame({"article_createdAt": [], "latency_hours": []})
        df["article_createdAt"] = pd.to_datetime(df["article_createdAt"], utc=True)
        with pytest.raises(ValueError):
            loglinear_election_effect_year_fe(df)
