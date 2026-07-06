#!/usr/bin/env python3
"""Plot valence / arousal / intensity distributions per rater from the enriched
label files (no page refetch needed)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RATERS = ["deepseek", "grok", "glm", "gemini", "gpt", "opus"]

def load(m):
    path = "deepseek_affect.json" if m == "deepseek" else f"labels_{m}_gold.json"
    return json.load(open(path))

data = {m: load(m) for m in RATERS}

fig, axes = plt.subplots(3, 6, figsize=(20, 9), sharey="row")
val_bins = [-2, -1, 0, 1, 2]
aro_bins = [1, 2, 3, 4, 5]
for j, m in enumerate(RATERS):
    rows = data[m]
    val = [r["valence"] for r in rows if r.get("valence") is not None]
    aro = [r["arousal"] for r in rows if r.get("arousal") is not None]
    inten = []
    for r in rows:
        four = [r[k] for k in ["arousal", "urgency", "threat", "anger"] if r.get(k) is not None]
        if len(four) == 4:
            inten.append(sum(four) / 4)
    axes[0, j].hist(val, bins=np.arange(-2.5, 3.5, 1), color="#3b6ea5", rwidth=0.85)
    axes[0, j].set_title(m, fontsize=13, fontweight="bold")
    axes[0, j].set_xticks(val_bins)
    axes[1, j].hist(aro, bins=np.arange(0.5, 6.5, 1), color="#c1440e", rwidth=0.85)
    axes[1, j].set_xticks(aro_bins)
    axes[2, j].hist(inten, bins=np.arange(1, 5.25, 0.5), color="#4a7c59", rwidth=0.85)
axes[0, 0].set_ylabel("Valence (-2..+2)", fontsize=12)
axes[1, 0].set_ylabel("Arousal (1..5)", fontsize=12)
axes[2, 0].set_ylabel("Intensity (mean of 4)", fontsize=12)
fig.suptitle("Affect-vector distributions by rater (gold cohort, n=120) \u2014 compression = mass piled in the middle, ends untouched",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("affect_distributions.png", dpi=130)
print("saved affect_distributions.png")
