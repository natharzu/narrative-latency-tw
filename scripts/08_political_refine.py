"""
08_political_refine.py — three-part political deep dive:
  1. Tighter classifier (drop ambiguous tokens like 統一, require strong matches)
  2. Electoral drilldown — days-to-election timing analysis
  3. Cross-strait validation — sample random headlines from each window
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

# Robust bool cast
for col in ["in_2020_win", "in_2024_win", "is_url_only"]:
    if col in df.columns and df[col].dtype == object:
        df[col] = df[col].astype(str).str.lower().eq("true")

# Robust date parse
df["article_createdAt"] = pd.to_datetime(df["article_createdAt"], format="mixed", utc=True, errors="coerce")
print(f"Date parse failures: {df['article_createdAt'].isna().sum():,} of {len(df):,}")

# ============================================================================
# PART 1: TIGHTER POLITICAL CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: TIGHTER POLITICAL CLASSIFIER")
print("=" * 80)

# High-specificity (any one match → political)
STRONG = {
    "candidate":    ["賴清德", "蔡英文", "侯友宜", "柯文哲", "韓國瑜", "馬英九",
                     "蔣萬安", "郭台銘", "朱立倫", "蕭美琴"],
    "party":        ["民進黨", "國民黨", "民眾黨", "時代力量"],
    "electoral":    ["選舉", "罷免", "公投", "投票", "票數", "計票", "做票",
                     "選票", "開票"],
    "cross_strait": ["兩岸", "台獨", "中共", "共軍", "共產黨", "九二共識", "一中"],
    "defense":      ["國防部", "國軍", "解放軍", "軍演", "飛彈"],
}
# Ambiguous (need ≥2 to count, sub-theme = "general")
AMBIG = ["中國", "親中", "抗中", "國安", "總統", "縣長", "市長",
         "綠營", "藍營", "白營", "政府"]
# DROPPED entirely: 統一 (also means "uniform/standardize"), 國防 alone (too generic)

def strict_pol_subtheme(text):
    if not isinstance(text, str):
        return None
    counts = {sub: sum(1 for k in kws if k in text) for sub, kws in STRONG.items()}
    if max(counts.values()) > 0:
        return max(counts, key=counts.get)
    if sum(1 for k in AMBIG if k in text) >= 2:
        return "general"
    return None

text = df["text_preview"].fillna("").astype(str)
df["pol_sub_strict"] = text.apply(strict_pol_subtheme)
df["is_political_strict"] = df["pol_sub_strict"].notna()

print(f"  Loose political (script 06): {(df['topic'] == 'political').sum():,}")
print(f"  Strict political (this run): {df['is_political_strict'].sum():,}")
dropped = (df['topic'] == 'political').sum() - (df['is_political_strict'] & (df['topic'] == 'political')).sum()
print(f"  False positives dropped: {dropped:,}")
added = (df['is_political_strict'] & (df['topic'] != 'political')).sum()
print(f"  New strict-positives (were 'other'): {added:,}")

# Window analysis with strict
ps = df[df["is_political_strict"]].copy()
ps["window"] = "off"
ps.loc[ps["in_2020_win"], "window"] = "2020"
ps.loc[ps["in_2024_win"], "window"] = "2024"

print("\n  STRICT political — sub-theme mix (%) by window:")
mix = ps.groupby(["window", "pol_sub_strict"]).size().unstack(fill_value=0)
print(mix.div(mix.sum(axis=1), axis=0).mul(100).round(1).to_string())

print("\n  STRICT political — count by window:")
print(mix.to_string())

print("\n  STRICT political — median latency (h) by window × sub-theme:")
print(ps.groupby(["window", "pol_sub_strict"])["latency_hours"].median().round(1).unstack().to_string())

# Mann-Whitney with strict
p2020 = ps.loc[ps["window"] == "2020", "latency_hours"].dropna()
p2024 = ps.loc[ps["window"] == "2024", "latency_hours"].dropna()
stat, p = mannwhitneyu(p2024, p2020, alternative="greater")
print(f"\n  Mann-Whitney (strict 2024 > 2020 latency):")
print(f"    U={stat:.0f}, p={p:.3e}")
print(f"    2020: N={len(p2020)}, median={p2020.median():.1f}h, p75={p2020.quantile(0.75):.1f}h, p95={p2020.quantile(0.95):.1f}h")
print(f"    2024: N={len(p2024)}, median={p2024.median():.1f}h, p75={p2024.quantile(0.75):.1f}h, p95={p2024.quantile(0.95):.1f}h")
print(f"    Median ratio: {p2024.median() / p2020.median():.2f}×")

# ============================================================================
# PART 2: ELECTORAL DRILLDOWN
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: ELECTORAL DRILLDOWN — what's driving the 8× slowdown?")
print("=" * 80)

elec = ps[(ps["pol_sub_strict"] == "electoral") & ps["window"].isin(["2020", "2024"])].copy()
print(f"  Strict electoral in election windows: {len(elec):,}")

E_DATES = {"2020": pd.Timestamp("2020-01-11", tz="UTC"),
           "2024": pd.Timestamp("2024-01-13", tz="UTC")}
elec["days_to_election"] = elec.apply(
    lambda r: (r["article_createdAt"] - E_DATES[r["window"]]).total_seconds() / 86400
              if pd.notna(r["article_createdAt"]) else np.nan,
    axis=1,
)

# Buckets relative to election day
elec["bucket"] = pd.cut(
    elec["days_to_election"],
    bins=[-91, -45, -15, 0, 15, 45, 91],
    labels=["−90 to −45", "−45 to −15", "−15 to 0", "0 to +15", "+15 to +45", "+45 to +90"],
)
print("\n  Count + median latency by bucket (days from election day):")
agg = elec.groupby(["window", "bucket"], observed=True)["latency_hours"].agg(["count", "median"]).round(1)
print(agg.to_string())

# Scatter plot
fig, ax = plt.subplots(figsize=(10, 5.5))
for win, color in [("2020", "#3b82f6"), ("2024", "#f59e0b")]:
    sub = elec[elec["window"] == win].dropna(subset=["days_to_election"])
    ax.scatter(sub["days_to_election"], sub["latency_hours"], s=35, alpha=0.55,
               color=color,
               label=f"{win}  (N={len(sub)}, median {sub['latency_hours'].median():.1f}h)")
ax.axvline(0, color="#94a3b8", linestyle="--", alpha=0.7, linewidth=1.2)
ymax = ax.get_ylim()[1]
ax.text(2, ymax * 0.6, "election day →", fontsize=10, color="#64748b", fontweight="bold")
ax.axhline(1, color="#10b981", linestyle=":", alpha=0.5, label="1 hour (golden)")
ax.axhline(24, color="#94a3b8", linestyle=":", alpha=0.5, label="1 day")
ax.set_yscale("log")
ax.set_xlabel("days relative to election day")
ax.set_ylabel("response latency (hours, log)")
ax.set_title("Electoral fact-checks — latency vs submission timing",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "viz" / "electoral_timing.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ viz/electoral_timing.png")

# Slow electoral 2024 samples
slow_elec = elec[(elec["window"] == "2024") & (elec["latency_hours"] > 50)]
print(f"\n  Slow electoral 2024 (>50h): {len(slow_elec)} of {(elec['window']=='2024').sum()}")
if len(slow_elec) > 0:
    for _, r in slow_elec.head(8).iterrows():
        snippet = (r["text_preview"][:110] + "…") if isinstance(r["text_preview"], str) and len(r["text_preview"]) > 110 else r["text_preview"]
        d = r["days_to_election"]
        print(f"    [{r['latency_hours']:>6.0f}h, day {d:+5.1f}] {snippet}")

# ============================================================================
# PART 3: CROSS-STRAIT VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: CROSS-STRAIT VALIDATION (random samples)")
print("=" * 80)

cs = ps[ps["pol_sub_strict"] == "cross_strait"]
cs_2020_all = cs[cs["window"] == "2020"]
cs_2024_all = cs[cs["window"] == "2024"]
print(f"  Strict cross-strait in 2020 window: {len(cs_2020_all)}")
print(f"  Strict cross-strait in 2024 window: {len(cs_2024_all)}")

cs_2020 = cs_2020_all.sample(min(10, len(cs_2020_all)), random_state=42)
cs_2024 = cs_2024_all.sample(min(10, len(cs_2024_all)), random_state=42)

print(f"\n  --- 2020 cross-strait sample ---")
for _, r in cs_2020.iterrows():
    snippet = (r["text_preview"][:140] + "…") if isinstance(r["text_preview"], str) and len(r["text_preview"]) > 140 else r["text_preview"]
    print(f"    [{r['latency_hours']:>6.0f}h] {snippet}")

print(f"\n  --- 2024 cross-strait sample ---")
for _, r in cs_2024.iterrows():
    snippet = (r["text_preview"][:140] + "…") if isinstance(r["text_preview"], str) and len(r["text_preview"]) > 140 else r["text_preview"]
    print(f"    [{r['latency_hours']:>6.0f}h] {snippet}")

# Save outputs
ps.to_csv(ROOT / "data" / "processed" / "political_strict.csv", index=False)
print("\n✅ Done.")
