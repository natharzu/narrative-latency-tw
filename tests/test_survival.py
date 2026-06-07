"""
tests/test_survival.py - unit + regression tests for the Phase 5 survival frame.

Pure-function tests on synthetic data always run in CI. The data-dependent and
lifelines-dependent tests skip cleanly when the processed survival CSV or the
optional `survival` extra (lifelines) is absent.

Run:
    uv run --extra survival pytest tests/test_survival.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest

from narrative_latency import PROC
from narrative_latency.survival import build_survival_frame, first_reply_times

SURVIVAL_CSV = PROC / "cofacts_survival.csv"


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


@pytest.fixture
def synthetic():
    """Four articles exercising every event/censoring branch.

    A: quick RUMOR reply           -> any=1, subst=1
    B: only an OPINIONATED reply    -> any=1, subst=0 (censored for subst)
    C: never replied                -> any=0, subst=0 (censored both)
    D: OPINIONATED then later RUMOR -> any uses first (opinion) time,
                                       subst uses the later RUMOR time
    """
    snapshot = _ts("2024-01-31")  # 30 days = 720h after creation
    articles = pd.DataFrame({
        "articleId": ["A", "B", "C", "D"],
        "article_createdAt": [_ts("2024-01-01")] * 4,
        "text": ["a", "b", "c", "d"],
    })
    article_replies = pd.DataFrame({
        "articleId": ["A", "B", "D", "D"],
        "replyId": ["ra", "rb", "rd1", "rd2"],
        "replyType": ["RUMOR", "OPINIONATED", "OPINIONATED", "RUMOR"],
        "ar_createdAt": [_ts("2024-01-01 02:00"), _ts("2024-01-01 05:00"),
                         _ts("2024-01-02 00:00"), _ts("2024-01-03 00:00")],
    })
    replies = pd.DataFrame({
        "replyId": ["ra", "rb", "rd1", "rd2"],
        "reply_createdAt": [_ts("2024-01-01 02:00"), _ts("2024-01-01 05:00"),
                            _ts("2024-01-02 00:00"), _ts("2024-01-03 00:00")],
    })
    frame = build_survival_frame(articles, article_replies, replies,
                                 snapshot=snapshot)
    return frame.set_index("articleId")


class TestBuildSurvivalFrame:
    def test_event_flags(self, synthetic):
        assert synthetic.loc["A", "event_any"] == 1
        assert synthetic.loc["A", "event_subst"] == 1
        assert synthetic.loc["B", "event_any"] == 1
        assert synthetic.loc["B", "event_subst"] == 0
        assert synthetic.loc["C", "event_any"] == 0
        assert synthetic.loc["C", "event_subst"] == 0
        assert synthetic.loc["D", "event_any"] == 1
        assert synthetic.loc["D", "event_subst"] == 1

    def test_durations(self, synthetic):
        assert synthetic.loc["A", "duration_any_h"] == pytest.approx(2.0)
        assert synthetic.loc["A", "duration_subst_h"] == pytest.approx(2.0)
        assert synthetic.loc["C", "duration_any_h"] == pytest.approx(720.0)
        assert synthetic.loc["C", "duration_subst_h"] == pytest.approx(720.0)
        assert synthetic.loc["B", "duration_any_h"] == pytest.approx(5.0)
        assert synthetic.loc["B", "duration_subst_h"] == pytest.approx(720.0)
        assert synthetic.loc["D", "duration_any_h"] == pytest.approx(24.0)
        assert synthetic.loc["D", "duration_subst_h"] == pytest.approx(48.0)

    def test_monotonic_event_invariant(self, synthetic):
        assert (synthetic["event_subst"] <= synthetic["event_any"]).all()

    def test_article_after_snapshot_dropped(self):
        snapshot = _ts("2024-01-31")
        articles = pd.DataFrame({
            "articleId": ["late"],
            "article_createdAt": [_ts("2024-02-15")],
            "text": ["x"],
        })
        empty_ar = pd.DataFrame(
            columns=["articleId", "replyId", "replyType", "ar_createdAt"]
        )
        empty_r = pd.DataFrame(columns=["replyId", "reply_createdAt"])
        frame = build_survival_frame(articles, empty_ar, empty_r,
                                     snapshot=snapshot)
        assert len(frame) == 0

    def test_first_reply_times_picks_substantive(self):
        ar = pd.DataFrame({
            "articleId": ["D", "D"],
            "replyId": ["rd1", "rd2"],
            "replyType": ["OPINIONATED", "RUMOR"],
            "ar_createdAt": [_ts("2024-01-02"), _ts("2024-01-03")],
        })
        r = pd.DataFrame({
            "replyId": ["rd1", "rd2"],
            "reply_createdAt": [_ts("2024-01-02"), _ts("2024-01-03")],
        })
        out = first_reply_times(ar, r).set_index("articleId")
        assert out.loc["D", "any_reply_at"] == _ts("2024-01-02")
        assert out.loc["D", "subst_reply_at"] == _ts("2024-01-03")


@pytest.fixture(scope="module")
def survival_df():
    if not SURVIVAL_CSV.exists():
        pytest.skip(f"Missing {SURVIVAL_CSV}; run scripts/10_survival.py first.")
    return pd.read_csv(SURVIVAL_CSV)


class TestSurvivalFrameRegression:
    def test_has_censoring(self, survival_df):
        rate = survival_df["event_any"].mean()
        assert 0.0 < rate < 1.0, f"event rate {rate:.3f} leaves no censoring"

    def test_substantive_subset_of_any(self, survival_df):
        assert survival_df["event_subst"].sum() <= survival_df["event_any"].sum()

    def test_durations_non_negative(self, survival_df):
        assert (survival_df["duration_any_h"] >= 0).all()
        assert (survival_df["duration_subst_h"] >= 0).all()

    def test_expected_columns(self, survival_df):
        for col in ["duration_any_h", "event_any", "duration_subst_h",
                    "event_subst", "election_window", "year"]:
            assert col in survival_df.columns


class TestKaplanMeierDirection:
    def test_2024_window_slower_than_2020(self, survival_df):
        pytest.importorskip("lifelines")
        from lifelines import KaplanMeierFitter

        df = survival_df.copy()
        for c in ["in_2020_win", "in_2024_win"]:
            if df[c].dtype != bool:
                df[c] = df[c].astype(str).str.lower().eq("true")
        e20 = df[df["in_2020_win"]]
        e24 = df[df["in_2024_win"]]
        if len(e20) < 100 or len(e24) < 100:
            pytest.skip("election-window subsets too small")
        k20 = KaplanMeierFitter().fit(e20["duration_any_h"], e20["event_any"])
        k24 = KaplanMeierFitter().fit(e24["duration_any_h"], e24["event_any"])
        assert k24.median_survival_time_ >= k20.median_survival_time_


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
