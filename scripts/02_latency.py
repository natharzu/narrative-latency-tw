"""
Compute Cofacts article->reply latency metrics.
Reads data/processed/cofacts_latency.csv (produced by clean.py).
Writes viz/cofacts_latency_distribution.png + viz/cofacts_latency_by_year.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import matplotlib.pyplot as plt

from narrative_latency import PROC, VIZ

IN = PROC / "cofacts_latency.csv"
VIZ.mkdir(exist_ok=True)

df = pd.read_csv(IN)
df["article_createdAt"] = pd.to_datetime(df["article_createdAt"], utc=True, format="ISO8601", errors="coerce")
df["year"] = df["article_createdAt"].dt.year

n = len(df)
median_h = df["latency_hours"].median()
p25, p75 = df["latency_hours"].quantile([0.25, 0.75])
p90 = df["latency_hours"].quantile(0.90)
pct_under_24h = (df["latency_hours"] < 24).mean() * 100
pct_under_week = (df["latency_hours"] < 24 * 7).mean() * 100

print("=" * 60)
print("HEADLINE")
print("=" * 60)
print(f"N = {n:,} article-reply pairs")
print(f"Median article->reply latency: {median_h:.1f} hours ({median_h/24:.2f} days)")
print(f"IQR: [{p25:.1f}, {p75:.1f}] hours")
print(f"P90: {p90:.1f} hours")
print(f"{pct_under_24h:.1f}% of rumors get a fact-check reply within 24h")
print(f"{pct_under_week:.1f}% within a week")
print()

print("=" * 60)
print("BY REPLY TYPE")
print("=" * 60)
by_type = df.groupby("reply_type")["latency_hours"].agg(["count", "median", "mean"]).round(1)
print(by_type)
print()

print("=" * 60)
print("BY YEAR")
print("=" * 60)
by_year = df.groupby("year")["latency_hours"].agg(["count", "median"]).round(1)
print(by_year)

# Chart 1: distribution (capped at p95 for readability)
p95 = df["latency_hours"].quantile(0.95)
fig, ax = plt.subplots(figsize=(10, 5))
df[df["latency_hours"] <= p95]["latency_hours"].hist(bins=60, ax=ax, color="#3b82f6", alpha=0.8)
ax.axvline(median_h, color="red", linestyle="--", label=f"Median = {median_h:.1f}h")
ax.set_xlabel("Hours from rumor report to fact-check reply")
ax.set_ylabel("Count")
ax.set_title(f"Cofacts response latency (N={n:,}, capped at p95)")
ax.legend()
fig.tight_layout()
fig.savefig(VIZ / "cofacts_latency_distribution.png", dpi=120)
print(f"\nSaved {VIZ / 'cofacts_latency_distribution.png'}")

# Chart 2: median by year
fig, ax = plt.subplots(figsize=(10, 5))
by_year["median"].plot(kind="bar", ax=ax, color="#10b981")
ax.set_xlabel("Year")
ax.set_ylabel("Median latency (hours)")
ax.set_title("Cofacts: median rumor->reply latency by year")
fig.tight_layout()
fig.savefig(VIZ / "cofacts_latency_by_year.png", dpi=120)
print(f"Saved {VIZ / 'cofacts_latency_by_year.png'}")
