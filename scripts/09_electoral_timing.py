"""
09_electoral_timing.py - electoral fact-check latency vs submission timing.

Pure analysis. If article_createdAt is corrupted on load, reconstructs in
memory from reply_createdAt - latency_hours; does NOT write back to disk.
For on-disk repair, use scripts/repair_dates.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang TC", "Heiti TC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

from utils import (
    PROC, VIZ, ELECTIONS,
    parse_dates_safe, reconstruct_article_dates,
    cast_bool_columns, assign_window,
)

ps = pd.read_csv(PROC / "political_strict.csv")
ps = cast_bool_columns(ps, ["in_2020_win", "in_2024_win"])
ps["article_createdAt"] = parse_dates_safe(ps["article_createdAt"])

if ps["article_createdAt"].isna().mean() > 0.5:
    print("⚠ Most article_createdAt missing — reconstructing in memory.")
    ps["article_createdAt"] = reconstruct_article_dates(ps)

print(f"Article dates valid: {ps['article_createdAt'].notna().sum():,} of {len(ps):,}")

ps = assign_window(ps)
elec = ps[(ps["pol_sub_strict"] == "electoral") & ps["window"].isin(["2020", "2024"])].copy()
elec["dte"] = elec.apply(
    lambda r: (r["article_createdAt"] - ELECTIONS[r["window"]]).total_seconds() / 86400
              if pd.notna(r["article_createdAt"]) else np.nan,
    axis=1,
)
elec = elec.dropna(subset=["dte"]).copy()
elec["bucket"] = pd.cut(
    elec["dte"],
    bins=[-91, -45, -15, 0, 15, 45, 91],
    labels=["−90 to −45", "−45 to −15", "−15 to 0", "0 to +15", "+15 to +45", "+45 to +90"],
)

print("\nCount + median latency (h) by bucket:")
agg = elec.groupby(["window", "bucket"], observed=True)["latency_hours"].agg(["count", "median"]).round(1)
print(agg.to_string())

print("\nPre-election vs post-election (day 0 split):")
for win in ["2020", "2024"]:
    sub = elec[elec["window"] == win]
    pre, post = sub[sub["dte"] < 0]["latency_hours"], sub[sub["dte"] >= 0]["latency_hours"]
    pre_med = pre.median() if len(pre) else float("nan")
    post_med = post.median() if len(post) else float("nan")
    print(f"  {win}: pre  N={len(pre):>3} median {pre_med:6.1f}h")
    print(f"        post N={len(post):>3} median {post_med:6.1f}h")

fig, ax = plt.subplots(figsize=(11, 6))
for win, color in [("2020", "#3b82f6"), ("2024", "#f59e0b")]:
    sub = elec[elec["window"] == win]
    ax.scatter(sub["dte"], sub["latency_hours"], s=48, alpha=0.6, color=color,
               edgecolor="white", linewidth=0.5,
               label=f"{win}  (N={len(sub)}, median {sub['latency_hours'].median():.1f}h)")
ax.axvline(0, color="#0f172a", ls="--", alpha=0.55, lw=1.4)
ax.text(2, ax.get_ylim()[1] * 0.55, "election day", fontsize=11,
        color="#475569", fontweight="bold")
ax.axhline(1, color="#10b981", ls=":", alpha=0.5)
ax.axhline(24, color="#94a3b8", ls=":", alpha=0.5)
ax.text(-88, 1.15, "golden hour", fontsize=9, color="#10b981", fontweight="bold")
ax.text(-88, 27, "1 day", fontsize=9, color="#64748b")
ax.set_yscale("log")
ax.set_xlabel("days relative to election day", fontsize=12)
ax.set_ylabel("response latency (hours, log scale)", fontsize=12)
ax.set_title("Electoral fact-checks — latency vs submission timing",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "electoral_timing.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ viz/electoral_timing.png")

ps_dated = ps.dropna(subset=["article_createdAt"]).copy()
ps_dated["month"] = ps_dated["article_createdAt"].dt.tz_convert(None).dt.to_period("M")
monthly = ps_dated.groupby(["month", "pol_sub_strict"]).size().unstack(fill_value=0)
show_months = [pd.Period(p, "M") for p in
               ["2019-11", "2019-12", "2020-01", "2020-02", "2020-03",
                "2023-11", "2023-12", "2024-01", "2024-02", "2024-03"]]
print("\nMonthly political volume (election months):")
print(monthly.loc[monthly.index.isin(show_months)].to_string())

print("\n✅ Done.")
