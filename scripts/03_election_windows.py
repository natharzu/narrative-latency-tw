"""
Analysis: election-window comparison, cluster tagging, survivorship sensitivity.

Inputs:  data/processed/cofacts_latency.csv
Outputs: data/processed/cofacts_election_windows.csv (adds election_window + topic_cluster)
         viz/election_window_comparison.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from narrative_latency import PROC, VIZ, E2020, E2024, WIN, SNAPSHOT, tag, parse_dates_safe

IN = PROC / "cofacts_latency.csv"
OUT = PROC / "cofacts_election_windows.csv"
VIZ.mkdir(exist_ok=True)

df = pd.read_csv(IN)
df["article_createdAt"] = parse_dates_safe(df["article_createdAt"])
df = df.dropna(subset=["article_createdAt"]).copy()
df["year"] = df["article_createdAt"].dt.year

df["near_2020"] = (df["article_createdAt"] - E2020).abs() <= WIN
df["near_2024"] = (df["article_createdAt"] - E2024).abs() <= WIN
df["election_window"] = df["near_2020"] | df["near_2024"]

# ============================================================
# 1. ELECTION WINDOW vs BASELINE
# ============================================================
print("=" * 60)
print("ELECTION WINDOW (90d around 2020 or 2024) vs BASELINE")
print("=" * 60)
election = df[df["election_window"]]["latency_hours"]
baseline = df[~df["election_window"]]["latency_hours"]
print(f"Election:  N={len(election):>6,}  median={election.median():>7.1f}h")
print(f"Baseline:  N={len(baseline):>6,}  median={baseline.median():>7.1f}h")
u, p = mannwhitneyu(election, baseline, alternative="two-sided")
print(f"Mann-Whitney U: p = {p:.2e}  (two-sided)")

# ============================================================
# 2. 2020 vs 2024 HEAD-TO-HEAD
# ============================================================
print()
print("=" * 60)
print("2020 vs 2024 ELECTION CYCLES")
print("=" * 60)
e20 = df[df["near_2020"]]["latency_hours"]
e24 = df[df["near_2024"]]["latency_hours"]
print(f"2020 window:  N={len(e20):>6,}  median={e20.median():>7.1f}h")
print(f"2024 window:  N={len(e24):>6,}  median={e24.median():>7.1f}h")
print(f"Ratio:        2024 is {e24.median()/e20.median():.1f}x slower than 2020")
u, p = mannwhitneyu(e24, e20, alternative="greater")
print(f"Mann-Whitney U (one-sided 2024 > 2020): p = {p:.2e}")

# ============================================================
# 3. SURVIVORSHIP SENSITIVITY (does 38x hold if we restrict to early articles?)
# ============================================================
print()
print("=" * 60)
print("SURVIVORSHIP SENSITIVITY (≥6 months before snapshot)")
print("=" * 60)
ripe = df[df["article_createdAt"] <= SNAPSHOT - pd.Timedelta(days=180)]
e20_r = ripe[ripe["near_2020"]]["latency_hours"]
e24_r = ripe[ripe["near_2024"]]["latency_hours"]
print(f"2020 (ripe): N={len(e20_r):>6,}  median={e20_r.median():>7.1f}h")
print(f"2024 (ripe): N={len(e24_r):>6,}  median={e24_r.median():>7.1f}h")
if len(e24_r) > 0 and e20_r.median() > 0:
    print(f"Ratio:       {e24_r.median()/e20_r.median():.1f}x")
print("If ratio is close to the full-sample ratio, the slowdown is robust to survivorship.")

# ============================================================
# 4. CLUSTER TAGGING (keyword-based on text_preview)
# ============================================================
df["topic_cluster"] = df["text_preview"].apply(tag)

print()
print("=" * 60)
print("BY CLUSTER (keyword-tagged on text_preview)")
print("=" * 60)
by_cluster = df.groupby("topic_cluster")["latency_hours"].agg(["count", "median"]).round(1)
print(by_cluster.sort_values("median"))

# ============================================================
# 5. CHART: 2020 vs 2024 SIDE-BY-SIDE
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, data, label, color in [
    (axes[0], e20, "2020 election ±90d", "#3b82f6"),
    (axes[1], e24, "2024 election ±90d", "#ef4444"),
]:
    p95 = data.quantile(0.95)
    capped = data[data <= p95]
    ax.hist(capped, bins=60, color=color, alpha=0.8)
    ax.axvline(data.median(), color="black", linestyle="--",
               label=f"Median = {data.median():.1f}h")
    ax.set_xlabel("Hours: rumor report → fact-check reply")
    ax.set_title(f"{label}\nN={len(data):,}, capped at p95")
    ax.legend()
fig.suptitle("Cofacts latency: 2020 vs 2024 Taiwan presidential elections")
fig.tight_layout()
fig.savefig(VIZ / "election_window_comparison.png", dpi=120)
print(f"\nSaved {VIZ / 'election_window_comparison.png'}")

# Save augmented CSV
df.to_csv(OUT, index=False)
print(f"Wrote {OUT}")
