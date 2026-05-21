"""scripts/m5_charts.py — final M5 chart pass.

Regenerates 3 existing charts with M5 improvements + adds Golden Hour CDF.

Run:
    python3 scripts/m5_charts.py

Inputs:
    data/processed/cofacts_election_windows.csv   (from 03_election_windows.py)

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
df = pd.read_csv(ROOT / "data/processed/cofacts_election_windows.csv")
df["article_createdAt"] = pd.to_datetime(df["article_createdAt"], format="ISO8601", utc=True).dt.tz_localize(None)

df["year"] = df["article_createdAt"].dt.year

# Derive election window from createdAt — self-sufficient regardless of m4 column naming.
ELECTIONS = {"2020": pd.Timestamp("2020-01-11"), "2024": pd.Timestamp("2024-01-13")}
WINDOW_DAYS = 90

WIN = pd.Timedelta(days=90)
E2020 = pd.Timestamp("2020-01-11")
E2024 = pd.Timestamp("2024-01-13")
e2020 = df.loc[(df["article_createdAt"] - E2020).abs() <= WIN, "latency_hours"]
e2024 = df.loc[(df["article_createdAt"] - E2024).abs() <= WIN, "latency_hours"]

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
print(f"\nGolden-Hour milestones (2024 election ±90d, N={len(e2024):,}):")
for h in [6, 12, 24, 48, 72, 168]:
    pct = (e2024 <= h).mean() * 100
    print(f"  by {h:>4}h ({h/24:>4.1f}d): {pct:5.1f}% of rumors addressed")

# ---------- election windows: overlay on log-x ----------
import numpy as np
BLUE_OVR, AMBER_OVR = "#3b82f6", "#f59e0b"

e2020_log = np.log10(e2020.clip(lower=0.01))
e2024_log = np.log10(e2024.clip(lower=0.01))

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.hist(e2020_log, bins=40, alpha=0.65, color=BLUE_OVR,
        edgecolor="white", linewidth=0.6,
        label=f"2020 election ±90d  (N={len(e2020):,}, median {e2020.median():.1f}h)")
ax.hist(e2024_log, bins=40, alpha=0.65, color=AMBER_OVR,
        edgecolor="white", linewidth=0.6,
        label=f"2024 election ±90d  (N={len(e2024):,}, median {e2024.median():.1f}h)")

m2020 = np.log10(e2020.median())
m2024 = np.log10(e2024.median())
ax.axvline(m2020, color=BLUE_OVR, linestyle="--", linewidth=1.5)
ax.axvline(m2024, color=AMBER_OVR, linestyle="--", linewidth=1.5)

ymax = ax.get_ylim()[1] * 1.18
ax.set_ylim(0, ymax)
arrow_y = ymax * 0.92
ax.annotate("", xy=(m2024, arrow_y), xytext=(m2020, arrow_y),
            arrowprops=dict(arrowstyle="->", color="#1e293b", lw=2.2))
ratio = e2024.median() / e2020.median()
ax.text((m2020 + m2024) / 2, arrow_y * 1.05,
        f"{ratio:.1f}× slower",
        ha="center", fontsize=14, fontweight="bold", color="#1e293b")

ax.set_xticks([-1, 0, 1, 2, 3, 4])
ax.set_xticklabels(["0.1h", "1h", "10h", "100h", "1,000h", "10,000h"])
ax.set_xlabel("response latency (log scale)", fontsize=11)
ax.set_ylabel("number of article-reply pairs", fontsize=11)
ax.set_title("Cofacts response latency: 2020 vs 2024 election windows\n"
             "Mann–Whitney one-sided p < 10⁻²⁰⁰",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("viz/election_window_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("OK viz/election_window_overlay.png")

