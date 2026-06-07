"""
tests/test_units.py — pure unit + regression tests for narrative_latency helpers.

These exercise the package's pure functions and locked constants. They have NO
dependency on the (gitignored) processed CSVs, so they always run in CI and
guard the shared library used by every script and the dashboard.

Run:
    uv run pytest tests/test_units.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from narrative_latency import (
    ROOT,
    DATA,
    RAW,
    PROC,
    VIZ,
    WIN,
    E2020,
    E2024,
    ELECTIONS,
    SNAPSHOT,
    parse_dates_safe,
    reconstruct_article_dates,
    cast_bool_columns,
    assign_window,
    CLUSTERS,
    tag,
)


class TestParseDatesSafe:
    def test_space_separated_tz_aware(self):
        out = parse_dates_safe(pd.Series(["2017-01-11 03:23:00+00:00"]))
        assert out.notna().all()
        assert out.iloc[0] == pd.Timestamp("2017-01-11 03:23:00", tz="UTC")

    def test_iso8601_t_z(self):
        out = parse_dates_safe(pd.Series(["2020-01-11T00:00:00Z"]))
        assert out.iloc[0] == pd.Timestamp("2020-01-11 00:00:00", tz="UTC")

    def test_nanosecond_precision(self):
        out = parse_dates_safe(pd.Series(["2017-01-11 03:20:59.999999999+00:00"]))
        assert out.notna().all()

    def test_mixed_batch_and_garbage_coerces_to_nat(self):
        out = parse_dates_safe(
            pd.Series(
                [
                    "2020-01-01T00:00:00Z",
                    "2021-06-15 12:30:00+00:00",
                    "not a date",
                    "",
                    None,
                ]
            )
        )
        assert list(out.notna()) == [True, True, False, False, False]

    def test_result_is_utc_tz_aware(self):
        out = parse_dates_safe(pd.Series(["2020-01-01 00:00:00+00:00"]))
        assert str(out.dt.tz) == "UTC"

    def test_naive_input_localized_to_utc(self):
        out = parse_dates_safe(pd.Series(["2020-01-01 05:00:00"]))
        assert out.iloc[0] == pd.Timestamp("2020-01-01 05:00:00", tz="UTC")


class TestReconstructArticleDates:
    def test_exact_reconstruction(self):
        df = pd.DataFrame(
            {
                "reply_createdAt": ["2020-01-10 00:00:00+00:00"],
                "latency_hours": [24.0],
            }
        )
        out = reconstruct_article_dates(df)
        assert out.iloc[0] == pd.Timestamp("2020-01-09 00:00:00", tz="UTC")

    def test_roundtrip_identity(self):
        article = parse_dates_safe(
            pd.Series(["2019-05-01 08:00:00+00:00", "2023-11-20 23:30:00+00:00"])
        )
        reply = parse_dates_safe(
            pd.Series(["2019-05-02 08:00:00+00:00", "2023-11-21 11:30:00+00:00"])
        )
        latency = (reply - article).dt.total_seconds() / 3600.0
        df = pd.DataFrame(
            {
                "reply_createdAt": [
                    "2019-05-02 08:00:00+00:00",
                    "2023-11-21 11:30:00+00:00",
                ],
                "latency_hours": latency.values,
            }
        )
        out = reconstruct_article_dates(df)
        for got, want in zip(out, article):
            assert abs((got - want).total_seconds()) < 1e-3


class TestCastBoolColumns:
    def test_true_false_strings(self):
        df = pd.DataFrame({"flag": ["true", "false", "True", "FALSE"]})
        out = cast_bool_columns(df, ["flag"])
        assert out["flag"].tolist() == [True, False, True, False]
        assert out["flag"].dtype == bool

    def test_non_true_strings_become_false(self):
        df = pd.DataFrame({"flag": ["yes", "1", "0", "t"]})
        out = cast_bool_columns(df, ["flag"])
        assert out["flag"].tolist() == [False, False, False, False]

    def test_missing_column_is_ignored(self):
        df = pd.DataFrame({"a": [1, 2]})
        out = cast_bool_columns(df, ["nonexistent"])
        assert "nonexistent" not in out.columns
        assert out["a"].tolist() == [1, 2]

    def test_already_bool_untouched(self):
        df = pd.DataFrame({"flag": [True, False]})
        out = cast_bool_columns(df, ["flag"])
        assert out["flag"].tolist() == [True, False]
        assert out["flag"].dtype == bool


class TestAssignWindow:
    def test_labels_and_precedence(self):
        df = pd.DataFrame(
            {
                "in_2020_win": [True, False, False, True],
                "in_2024_win": [False, True, False, True],
            }
        )
        out = assign_window(df)
        # Last row has both flags -> 2024 wins (applied last in assign_window).
        assert out["window"].tolist() == ["2020", "2024", "off", "2024"]

    def test_input_not_mutated(self):
        df = pd.DataFrame({"in_2020_win": [True], "in_2024_win": [False]})
        assign_window(df)
        assert "window" not in df.columns


class TestClusterTaxonomy:
    def test_expected_cluster_keys(self):
        assert set(CLUSTERS) == {
            # original IORG clusters
            "vaccine",
            "us_skepticism",
            "pre_election",
            "ccp_info_manipulation",
            # Option 4 topic categories
            "scam",
            "health",
            "traffic",
            "energy",
            "pension",
            "food_safety",
            "lgbtq",
            "disaster",
            "international",
        }

    def test_no_empty_keyword_lists(self):
        for name, kws in CLUSTERS.items():
            assert isinstance(kws, list) and len(kws) > 0, name
            assert all(isinstance(k, str) and k for k in kws), name

    def test_first_match_ordering(self):
        # A string hitting multiple clusters returns the first by dict order.
        sample = "疫苗 與 選舉"  # vaccine keyword precedes election keyword
        first = next(
            c for c, kws in CLUSTERS.items() if any(k in sample for k in kws)
        )
        assert tag(sample) == first

    def test_non_string_is_other(self):
        assert tag(None) == "Other"
        assert tag(123) == "Other"
        assert tag(["疫苗"]) == "Other"


class TestConstants:
    def test_election_ordering(self):
        assert E2020 < E2024 < SNAPSHOT

    def test_window_is_90_days(self):
        assert WIN == pd.Timedelta(days=90)

    def test_anchors_are_utc(self):
        for ts in (E2020, E2024, SNAPSHOT):
            assert ts.tzinfo is not None
            assert str(ts.tz) == "UTC"

    def test_elections_mapping(self):
        assert ELECTIONS == {"2020": E2020, "2024": E2024}

    def test_paths_layout(self):
        assert DATA == ROOT / "data"
        assert RAW == DATA / "raw"
        assert PROC == DATA / "processed"
        assert VIZ == ROOT / "viz"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
