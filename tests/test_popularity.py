"""
tests/test_popularity.py — unit tests for the popularity layer.

The core transforms are pure, so these run without any raw data. The final
check exercises the real processed output only if it already exists.

Run:
    python3 -m pytest tests/test_popularity.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def _load_module():
    # digit-prefixed filename -> load by path; scripts/ must be importable so
    # the module's `from utils import ...` resolves under pytest.
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "popularity", SCRIPTS / "17_popularity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


def _requests() -> pd.DataFrame:
    return pd.DataFrame({
        "articleId": ["a", "a", "a", "b", "b", "c"],
        "userIdsha256": ["u1", "u2", "u2", "u1", "u3", "u9"],
        "createdAt": ["2024-01-01T00:00:00Z", "2024-01-01T02:00:00Z",
                      "2024-01-01T03:00:00Z", "2024-01-02T00:00:00Z",
                      "2024-01-02T01:00:00Z", "2024-01-03T00:00:00Z"],
        "status": ["NORMAL", "NORMAL", "NORMAL", "NORMAL", "BLOCKED", "NORMAL"],
    })


def _analytics() -> pd.DataFrame:
    return pd.DataFrame({
        "type": ["article", "article", "reply", "article"],
        "docId": ["a", "a", "a", "b"],
        "lineVisit": [5, 3, 99, 2],
        "lineUser": [4, 2, 50, 1],
        "webVisit": [1, 0, 0, 7],
        "webUser": [1, 0, 0, 5],
    })


def test_request_count_is_distinct_users_and_respects_status():
    req = M.build_request_counts(_requests()).set_index("articleId")
    assert req.loc["a", "request_count"] == 2   # u1, u2 (u2 duplicate collapsed)
    assert req.loc["b", "request_count"] == 1   # u3 is BLOCKED, dropped
    assert req.loc["c", "request_count"] == 1
    assert req.loc["a", "first_request"] <= req.loc["a", "last_request"]


def test_view_counts_sum_only_article_rows():
    v = M.build_view_counts(_analytics()).set_index("articleId")
    assert v.loc["a", "line_visits"] == 8       # 5 + 3; reply row (99) excluded
    assert v.loc["a", "web_visits"] == 1
    assert v.loc["b", "line_visits"] == 2


def test_merge_fills_views_and_keeps_implicit_request():
    pop = M.merge_popularity(
        M.build_request_counts(_requests()),
        M.build_view_counts(_analytics()),
    ).set_index("articleId")
    assert pop.loc["c", "line_visits"] == 0     # c has no analytics row
    assert (pop["request_count"] >= 1).all()
    assert pop["request_count"].dtype.kind in "iu"


def test_popularity_by_narrative_sorts_by_total_requests():
    d = pd.DataFrame({
        "articleId": ["a", "b", "c", "d"],
        "rule_topic": ["scam", "scam", "political", "political"],
        "request_count": [10, 5, 1, 1],
        "web_visits": [2, 2, 0, 0],
    })
    out = M.popularity_by_narrative(d, "rule_topic")
    assert list(out.index) == ["scam", "political"]   # 15 > 2
    assert out.loc["scam", "total_requests"] == 15
    assert out.loc["scam", "n"] == 2


@pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "cofacts_popularity.csv").exists(),
    reason="popularity table not built yet")
def test_real_popularity_table_is_sane():
    pop = pd.read_csv(ROOT / "data" / "processed" / "cofacts_popularity.csv")
    assert (pop["request_count"] >= 1).all()
    assert pop["articleId"].is_unique
    for c in ["line_visits", "web_visits"]:
        assert (pop[c] >= 0).all()
