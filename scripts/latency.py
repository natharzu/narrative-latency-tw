"""02_latency equivalent — compute Stage 1->4 deltas, summary stats, save chart."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("data/processed/narratives_clean.csv")
VIZ = Path("viz/latency_by_cluster.png")

df = pd.read_csv(PROCESSED)
print(f"Loaded {len(df)} narratives")

for col in ["stage_1_date", "stage_2_date", "stage_3_date", "stage_4_date"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df_clean = df.dropna(subset=["stage_1_date", "stage_4_date"]).copy()
print(f"With Stage 1 + Stage 4 dates: {len(df_clean)}")

if len(df_clean) == 0:
    print("\n  No narratives have both Stage 1 and Stage 4 dates yet.")
    print("  Per-stage transcription pending (M3 task, due May 15).")
    print("  Running pipeline on SYNTHETIC dates to validate end-to-end...\n")
    np.random.seed(42)
    base = pd.to_datetime("2021-01-01")
    df["stage_1_date"] = base + pd.to_timedelta(np.random.randint(0, 365, len(df)), unit="D")
    df["stage_4_date"] = df["stage_1_date"] + pd.to_timedelta(np.random.randint(1, 30, len(df)), unit="D")
    df_clean = df.copy()
    SYNTHETIC = True
else:
    SYNTHETIC = False

df_clean["latency_days"] = (df_clean["stage_4_date"] - df_clean["stage_1_date"]).dt.days

median = df_clean["latency_days"].median()
iqr = df_clean["latency_days"].quantile([0.25, 0.75]).values
mode_label = "SYNTHETIC" if SYNTHETIC else "REAL"
print(f"\nHeadline [{mode_label}]: Median Stage 1->4 latency = {median:.1f} days, IQR = [{iqr[0]:.0f}, {iqr[1]:.0f}], N={len(df_clean)}")

print("\nMedian latency by topic cluster:")
by_cluster = df_clean.groupby("topic_cluster")["latency_days"].agg(["median", "count"]).sort_values("median")
print(by_cluster.to_string())

fig, ax = plt.subplots(figsize=(8, 4))
by_cluster["median"].plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Median Stage 1->4 latency (days)")
ax.set_title(f"Narrative latency by topic cluster ({mode_label} data, N={len(df_clean)})")
ax.axvline(median, color="red", linestyle="--", alpha=0.5, label=f"Overall median: {median:.1f}d")
ax.legend()
plt.tight_layout()

VIZ.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(VIZ, dpi=120, bbox_inches="tight")
print(f"\nSaved chart to {VIZ}")
