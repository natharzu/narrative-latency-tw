"""narrative_latency — shared library for the Cofacts narrative-latency capstone.

Single source of truth for repo paths, election anchors, the topic-cluster
taxonomy, date-handling helpers, and matplotlib styling. Scripts import from
here (directly or via the scripts/utils.py compatibility shim).
"""
from .constants import (
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
from .dataio import (
    parse_dates_safe,
    reconstruct_article_dates,
    cast_bool_columns,
    assign_window,
)
from .clusters import CLUSTERS, tag
from .plotting import set_plot_style

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
