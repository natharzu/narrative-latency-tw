"""narrative_latency: shared library for the Cofacts reply-latency study."""

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
from .viz import set_plot_style
from .analysis import (
    in_window,
    window_latencies,
    window_ratio,
    window_sensitivity,
    per_year_median,
    within_year_election_contrast,
    loglinear_election_effect,
)

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
    "in_window",
    "window_latencies",
    "window_ratio",
    "window_sensitivity",
    "per_year_median",
    "within_year_election_contrast",
    "loglinear_election_effect",
]
