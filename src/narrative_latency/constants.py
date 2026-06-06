"""Repo paths and election anchors — single source of truth.

Values are preserved exactly from the original scripts/utils.py and the
inline definitions in scripts 03/05 and tests/test_pipeline.py.
"""
from pathlib import Path
import pandas as pd

# src/narrative_latency/constants.py -> parents[2] is the repo root.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
VIZ = ROOT / "viz"

# Election anchors (Taiwan presidential)
WIN = pd.Timedelta(days=90)
E2020 = pd.Timestamp("2020-01-11", tz="UTC")
E2024 = pd.Timestamp("2024-01-13", tz="UTC")
ELECTIONS = {"2020": E2020, "2024": E2024}

# Hugging Face Cofacts dump snapshot used for every reported headline number.
SNAPSHOT = pd.Timestamp("2026-05-10", tz="UTC")
