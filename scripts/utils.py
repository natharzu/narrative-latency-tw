"""Backward-compatible shim for the narrative-latency pipeline.

The canonical implementations now live in the ``narrative_latency`` package
under ``src/``. This module bootstraps ``src/`` onto ``sys.path`` and re-exports
the package API so existing ``from utils import ...`` calls (e.g. in
09_electoral_timing.py and repair_dates.py) keep working when scripts are run
directly, without requiring an editable install.

Prefer importing from ``narrative_latency`` directly in new code.
"""
import sys
from pathlib import Path

# Make src/ importable when running scripts directly (no install needed).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from narrative_latency.constants import (  # noqa: E402,F401
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
)
from narrative_latency.dataio import (  # noqa: E402,F401
    parse_dates_safe,
    reconstruct_article_dates,
    cast_bool_columns,
    assign_window,
)
from narrative_latency.clusters import CLUSTERS, tag  # noqa: E402,F401
from narrative_latency.plotting import set_plot_style  # noqa: E402,F401

__all__ = [
    "ROOT",
    "DATA",
    "RAW",
    "PROC",
    "VIZ",
    "WIN",
    "E2020",
    "E2024",
    "ELECTIONS",
    "SNAPSHOT",
    "parse_dates_safe",
    "reconstruct_article_dates",
    "cast_bool_columns",
    "assign_window",
    "CLUSTERS",
    "tag",
    "set_plot_style",
]
