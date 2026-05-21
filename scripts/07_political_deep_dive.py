"""
07_political_deep_dive.py — slice political into sub-themes, find slow tail.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang TC", "Heiti TC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "processed" / "cofacts_topic_classified.csv")

# Filter to political
pol = df[df["topic"] == "political"].copy()
print(f"Political articles: {len(pol):,}  (2020 win: {(pol['in_2020_win']==True).sum() or (pol['in_2020_win']=='True').sum()}, 2024 win: {(pol['in_2024_win']==True).sum() or (pol['in_2024_win']=='True').sum()})")

# Robust bool cast
for col in ["in_2020_win", "in_2024_win"]:
    if pol[col].dtype == object:
        pol[col] = pol[col].astype(str).str.lower().eq("true")

# Sub-theme keywords
SUBTHEMES = {
    "electoral":    ["投票", "票數", "選舉", "開票", "罷免", "公投", "選票", "計票", "做票"],
    "candidate":    ["賴清德", "蔡英文", "侯友宜", "柯文哲", "韓國瑜", "馬英九", "蔣萬安", "郭台銘", "朱立倫", "蕭美琴"],
    "cross_strait": ["中國", "中共", "兩岸", "統一", "台獨", "親中", "抗中", "一中", "統獨", "九二共識"],
    "party":        ["民進黨", "國民黨", "民眾黨", "時代力量", "綠營", "藍營", "白營"],
    "defense":      ["國防", "國安", "共軍", "飛彈", "軍演", "國軍", "戰爭", "解放軍"],
}

def hits(t, kws): return sum(1 for k in kws if k in t) if isinstance(t, str) else 0

text = pol["text_preview"].fillna("").astype(str)
for theme, kws in SUBTHEMES.items():
    pol[f"sub_{theme}"] = text.apply(lambda t: hits(t, kws))

def classify_sub(r):
    counts = {t: r[f"sub_{t}"] for t in SUBTHEMES}
    return "general" if max(counts.values()) == 0 else max(counts, key=counts.get)

pol["sub_theme"] = pol.apply(classify_sub, axis=1)

# Window labels
pol["window"] = "off"
pol.loc[pol["in_2020_win"], "window"] = "2020"
pol.loc[pol["in_2024_win"], "window"] = "2024"

# 1. Sub-theme mix per window
print("\n" + "=" * 80)
print("POLITICAL SUB-THEME MIX (%) BY WINDOW")
print("=" * 80)
mix = pol.groupby(["window", "sub_theme"]).size().unstack(fill_value=0)
mix_pct = mix.div(mix.sum(axis=1), axis=0).mul(100).round(1)
print(mix_pct.to_string())

print("\n" + "=" * 80)
print("MEDIAN LATENCY (h) BY WINDOW × SUB-THEME")
print("=" * 80)
print(pol.groupby(["window", "sub_theme"])["latency_hours"].median().round(1).unstack().to_string())

print("\n" + "=" * 80)
print("COUNT BY WINDOW × SUB-THEME")
print("=" * 80)
print(mix.to_string())

# 2. Stat test: 2020 political vs 2024 political
pol2020 = pol.loc[pol["window"] == "2020", "latency_hours"].dropna()
pol2024 = pol.loc[pol["window"] == "2024", "latency_hours"].dropna()
stat, p = mannwhitneyu(pol2024, pol2020, alternative="greater")
print(f"\nMann-Whitney (2024 > 2020 political latency): U={stat:.0f}, p={p:.3e}")
print(f"  2020 political: N={len(pol2020)}, median={pol2020.median():.1f}h, p75={pol2020.quantile(0.75):.1f}h, p95={pol2020.quantile(0.95):.1f}h")
print(f"  2024 political: N={len(pol2024)}, median={pol2024.median():.1f}h, p75={pol2024.quantile(0.75):.1f}h, p95={pol2024.quantile(0.95):.1f}h")
print(f"  Median ratio: {pol2024.median() / pol2020.median():.2f}×")
print(f"  p95 ratio:    {pol2024.quantile(0.95) / pol2020.quantile(0.95):.2f}×")

# 3. Slow tail in 2024 political
print("\n" + "=" * 80)
print("2024 POLITICAL — SLOW TAIL (latency > 7 days)")
print("=" * 80)
slow_pol = pol[(pol["window"] == "2024") & (pol["latency_hours"] > 168)]
print(f"  Count: {len(slow_pol)} of {len(pol2024)} ({len(slow_pol)/len(pol2024)*100:.1f}%)")
print(f"  Sub-theme breakdown:")
print(slow_pol["sub_theme"].value_counts().to_string())
print(f"\n  Sample slow political headlines (10 random):")
sample = slow_pol[["latency_hours", "sub_theme", "text_preview"]].sample(min(10, len(slow_pol)), random_state=42)
for _, r in sample.iterrows():
    text_short = (r["text_preview"][:80] + "…") if isinstance(r["text_preview"], str) and len(r["text_preview"]) > 80 else r["text_preview"]
    print(f"    [{r['latency_hours']:>6.0f}h | {r['sub_theme']:>13}] {text_short}")

# 4. Plot: 2020 vs 2024 political latency overlay (log-x)
fig, ax = plt.subplots(figsize=(10, 5.5))
BLUE, AMBER = "#3b82f6", "#f59e0b"
log_2020 = np.log10(pol2020.clip(lower=0.01))
log_2024 = np.log10(pol2024.clip(lower=0.01))
ax.hist(log_2020, bins=30, alpha=0.65, color=BLUE, edgecolor="white",
        label=f"2020 political  (N={len(pol2020):,}, median {pol2020.median():.1f}h)")
ax.hist(log_2024, bins=30, alpha=0.65, color=AMBER, edgecolor="white",
        label=f"2024 political  (N={len(pol2024):,}, median {pol2024.median():.1f}h)")
ax.axvline(np.log10(pol2020.median()), color=BLUE, linestyle="--", linewidth=1.5)
ax.axvline(np.log10(pol2024.median()), color=AMBER, linestyle="--", linewidth=1.5)
ax.set_xticks([-1, 0, 1, 2, 3, 4])
ax.set_xticklabels(["0.1h", "1h", "10h", "100h", "1,000h", "10,000h"])
ax.set_xlabel("response latency (log scale)")
ax.set_ylabel("count")
ax.set_title("Political-only latency: 2020 vs 2024 election windows", fontsize=13, fontweight="bold")
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "viz" / "political_latency_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ viz/political_latency_overlay.png")

# 5. Plot: latency by sub-theme in 2024
fig, ax = plt.subplots(figsize=(10, 5.5))
sub_in_2024 = pol[pol["window"] == "2024"]
themes_present = sub_in_2024["sub_theme"].value_counts().index.tolist()
data = [sub_in_2024.loc[sub_in_2024["sub_theme"] == t, "latency_hours"].values for t in themes_present]
labels = [f"{t}\n(N={len(d)})" for t, d in zip(themes_present, data)]
bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.6)
for patch, c in zip(bp["boxes"], ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#cbd5e1"]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_yscale("log")
ax.set_ylabel("latency (hours, log scale)")
ax.set_title("2024 political sub-themes — response latency", fontsize=13, fontweight="bold")
ax.axhline(1, color="#10b981", linestyle="--", alpha=0.6, label="1 hour (golden)")
ax.axhline(24, color="#94a3b8", linestyle="--", alpha=0.6, label="1 day")
ax.axhline(168, color="#ef4444", linestyle="--", alpha=0.6, label="1 week")
ax.legend(loc="upper right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(ROOT / "viz" / "political_subthemes_2024.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ viz/political_subthemes_2024.png")

# Save augmented data
pol.to_csv(ROOT / "data" / "processed" / "political_subthemes.csv", index=False)
print("\n✅ Done.")
