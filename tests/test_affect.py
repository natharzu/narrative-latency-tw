"""
tests/test_affect.py — unit tests for the affect annotation layer.

Pure functions + a fake in-memory LLM, so these run with no network and no data.

Run:
    python3 -m pytest tests/test_affect.py -v
"""
from __future__ import annotations

import importlib.util
import json
import re
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
        "affect", SCRIPTS / "18_affect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    # keep retry paths instant
    monkeypatch.setattr(M.time, "sleep", lambda *a, **k: None)


def test_coerce_clamps_and_computes_intensity():
    c = M.coerce_record({"articleId": 7, "valence": -9, "arousal": 4,
                         "urgency": 4, "threat": 2, "anger": 2})
    assert c["articleId"] == "7"
    assert c["valence"] == -2          # clamped from -9
    assert c["intensity"] == 3.0       # (4+4+2+2)/4
    assert M.coerce_record({"valence": 1}) is None          # no articleId
    assert M.coerce_record({"articleId": "x", "valence": "NA"}) is None


def test_extract_json_array_tolerates_fences_and_prose():
    assert M.extract_json_array('```json\n[{"a":1}]\n```') == [{"a": 1}]
    assert M.extract_json_array('sure! [{"a":1},{"b":2}] ok') == [{"a": 1}, {"b": 2}]
    assert M.extract_json_array("no json") == []
    assert M.extract_json_array("[broken") == []


def test_parse_response_keeps_only_requested_ids():
    resp = ('[{"articleId":"a","valence":1,"arousal":2,"urgency":1,"threat":0,"anger":0},'
            '{"articleId":"ZZZ","valence":0,"arousal":0,"urgency":0,"threat":0,"anger":0}]')
    p = M.parse_response(resp, ["a", "b"])
    assert set(p) == {"a"}
    assert p["a"]["intensity"] == 0.75


def test_remaining_ids_dedups_and_skips_done():
    assert M.remaining_ids(["a", "b", "a", "c"], {"b"}) == ["a", "c"]


def _fake_llm(prompt):
    ids = re.findall(r"id=(\S+) ::", prompt)
    if "bad" in ids:
        return "sorry, cannot comply"            # unparseable -> batch yields nothing
    recs = [{"articleId": i, "valence": 1, "arousal": 2, "urgency": 1,
             "threat": 1, "anger": 0} for i in ids]
    return "```json\n" + json.dumps(recs) + "\n```"


def test_run_checkpoint_resume(tmp_path):
    out = tmp_path / "cofacts_affect.csv"
    arts = [{"articleId": x, "text": f"msg {x}"} for x in ["a", "b", "c", "bad", "e"]]

    # batch_size=2 => (a,b)(c,bad)(e); the (c,bad) batch fails entirely
    n1 = M.run(arts, out, _fake_llm, batch_size=2)
    assert n1 == 3
    df1 = pd.read_csv(out, dtype={"articleId": str})
    assert set(df1["articleId"]) == {"a", "b", "e"}
    assert list(df1.columns) == ["articleId", "valence"] + M.SCALES + ["intensity"]

    def _fixed(prompt):
        ids = re.findall(r"id=(\S+) ::", prompt)
        return json.dumps([{"articleId": i, "valence": 0, "arousal": 3,
                            "urgency": 3, "threat": 3, "anger": 1} for i in ids])

    # resume: only c & bad remain; appended, not rewritten
    n2 = M.run(arts, out, _fixed, batch_size=2)
    assert n2 == 2
    df2 = pd.read_csv(out, dtype={"articleId": str})
    assert set(df2["articleId"]) == {"a", "b", "c", "bad", "e"}
    assert df2["articleId"].is_unique
    assert abs(df2.set_index("articleId").loc["c", "intensity"] - 2.5) < 1e-9

    # fully annotated -> no-op
    assert M.run(arts, out, _fixed, batch_size=2) == 0
