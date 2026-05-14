"""scripts/m5_charts.py — final M5 chart pass.

Regenerates 3 existing charts with M5 improvements + adds Golden Hour CDF.

Run:
    python3 scripts/m5_charts.py

Inputs:
    data/processed/cofacts_m4.csv   (from m4_analysis.py)

Outputs (overwrites):
    viz/cofacts_latency_distribution.png   — log x + 1d/3d/1w markers
    viz/cofacts_latency_by_year.png        — log y + 1-day baseline
    viz/election_window_comparison.png     — SHARED x-axis, capped at 500h
    viz/golden_hour_cdf.png                — NEW
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIZ = ROOT / "viz"
VIZ.mkdir(exist_ok=True)

# ---------- load ----------
df = pd.read_csv(
    ROOT / "data/processed/cofacts_m4.csv",
    parse_dates=["article_createdAt"],
)
df["year"] = df["article_createdAt"].dt.year

# Derive election window from createdAt — self-sufficient regardless of m4 column naming.
ELECTIONS = {"2020": pd.Timestamp("2020-01-11"), "2024": pd.Timestamp("2024-01-13")}
WINDOW_DAYS = 90

def election_label(dt):
    for year, e_dt in ELECTIONS.items():
        if abs((dt - e_dt).days) <= WINDOW_DAYS:
            return year
    return None

df["e_window"] = df["article_createdAt"].apply(election_label)
e2020 = df.loc[df["e_window"] == "2020", "latency_hours"]
e2024 = df.loc[df["e_window"] == "2024", "latency_hours"]

BLUE, RED, GRAY = "#2A6FB0", "#C0392B", "#7F8C8D"

# ---------- 1. distribution (log x + real-world markers) ----------
fig, ax = plt.subplots(figsize=(9, 5))
clip = df.loc[df["latency_hours"].between(0.01, 365 * 24), "latency_hours"]
ax.hist(np.log10(clip), bins=60, color=BLUE, edgecolor="white")
ymax = ax.get_ylim()[1]
for hours, label in [(24, "1 day"), (72, "3 days"), (168, "1 week")]:
    ax.axvline(np.log10(hours), color=RED, linestyle="--", linewidth=1)
    ax.text(np.log10(hours), ymax * 0.95, f" {label}", color=RED, fontsize=9, va="top")
ax.set_xticks([np.log10(x) for x in [1, 10, 100, 1000, 10000]])
ax.set_xticklabels(["1h", "10h", "100h", "1,000h", "10,000h"])
ax.set_xlabel("latency (log scale)")
ax.set_ylabel("number of article-reply pairs")
ax.set_title(f"Cofacts response-latency distribution (N={len(clip):,}, median 21.2h)")
plt.tight_layout()
plt.savefig(VIZ / "cofacts_latency_distribution.png", dpi=150)
plt.close()
print("OK  viz/cofacts_latency_distribution.png")

# ---------- 2. by year (log y) ----------
year_med = df.groupby("year")["latency_hours"].median().reset_index()
year_med = year_med[year_med["year"].between(2017, 2026)]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(year_med["year"], year_med["latency_hours"], color=BLUE)
ax.set_yscale("log")
ax.set_ylabel("median latency (hours, log scale)")
ax.set_xlabel("year")
ax.set_title("Cofacts median response latency by year (2017–2026)")
for bar, val in zip(bars, year_med["latency_hours"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15, f"{val:.1f}h",
            ha="center", fontsize=8)
ax.axhline(24, color=RED, linestyle="--", linewidth=1)
ax.text(year_med["year"].max() + 0.3, 24, " 1 day", color=RED, fontsize=9, va="center")
plt.tight_layout()
plt.savefig(VIZ / "cofacts_latency_by_year.png", dpi=150)
plt.close()
print("OK  viz/cofacts_latency_by_year.png")

# ---------- 3. election window comparison (SHARED x-axis) ----------
CAP = 500  # hours; trims long right tail so both panels are readable
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, data, year_lbl, color, med in [
    (ax1, e2020, "2020", BLUE, 6.7),
    (ax2, e2024, "2024", RED, 67.2),
]:
    clipped = data[data <= CAP]
    n_clipped = (data > CAP).sum()
    ax.hist(clipped, bins=50, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(med, color="black", linestyle="--", linewidth=1.5)
    ymax = ax.get_ylim()[1]
    ax.text(med, ymax * 0.95, f" median {med}h", fontsize=10, va="top",
            fontweight="bold")
    ax.set_title(f"{year_lbl} election ±90 days  (N={len(data):,})")
    ax.set_xlabel(f"latency (hours, capped at {CAP}h; {n_clipped} pairs beyond)")
ax1.set_ylabel("count")
fig.suptitle(
    "Cofacts response latency: 2020 vs 2024 election windows — 10× slowdown, p < 10⁻²⁰⁰",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(VIZ / "election_window_comparison.png", dpi=150)
plt.close()
print("OK  viz/election_window_comparison.png")

# ---------- 4. Golden Hour CDF (NEW) ----------
fig, ax = plt.subplots(figsize=(9, 5))
for data, label, color in [
    (df["latency_hours"], "all years", GRAY),
    (e2020, "2020 election ±90d", BLUE),
    (e2024, "2024 election ±90d", RED),
]:
    s = np.sort(data.dropna().values)
    cdf = np.arange(1, len(s) + 1) / len(s) * 100
    ax.plot(s, cdf, label=f"{label} (N={len(s):,})", color=color, linewidth=2)
ax.set_xscale("log")
ax.set_xticks([1, 6, 12, 24, 72, 168, 720, 8760])
ax.set_xticklabels(["1h", "6h", "12h", "1d", "3d", "1w", "1mo", "1y"])
ax.set_xlim(0.5, 8760)
ax.set_ylim(0, 100)
ax.set_ylabel("% of rumors with at least one reply")
ax.set_xlabel("time since rumor reported (log scale)")
ax.set_title("Golden Hour: how fast does Cofacts respond?")
ax.grid(True, alpha=0.3)
for h in [12, 48]:
    ax.axvline(h, color="black", linestyle=":", linewidth=1, alpha=0.5)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(VIZ / "golden_hour_cdf.png", dpi=150)
plt.close()
print("OK  viz/golden_hour_cdf.png")

# ---------- print Golden-Hour milestones for the deck ----------
print("\nGolden-Hour milestones (all-years baseline, N=68,533):")
for h in [6, 12, 24, 48, 72, 168]:
    pct = (df["latency_hours"] <= h).mean() * 100
    print(f"  by {h:>4}h ({h/24:>4.1f}d): {pct:5.1f}% of rumors addressed")
print("\nGolden-Hour milestones (2024 election ±90d, N=2,191):")
for h in [6, 12, 24, 48, 72, 168]:
    pct = (e2024 <= h).mean() * 100
    print(f"  by {h:>4}h ({h/24:>4.1f}d): {pct:5.1f}% of rumors addressed")
