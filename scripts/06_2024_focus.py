"""06_2024_focus.py — classify scam vs political vs health vs other,
and re-cluster the 2024 election window alone.

Generates:
- viz/topic_drift.png       — stacked-area topic mix over years
- viz/clusters_slow.png     — annotated 2D UMAP of 2024 window with English labels
- data/processed/cluster_profiles_2024.csv  (includes label_en)
- data/processed/cofacts_topic_classified.csv
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang TC", "Heiti TC", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent

# English category labels for the 2024-window clusters (cluster_id -> label).
# Single source of truth; written to cluster_profiles_2024.csv and used to
# annotate viz/clusters_slow.png.
LABELS_2024 = {
    0:  "Electricity bills",
    1:  "Traffic enforcement cameras",
    2:  "Senior-citizen scam",
    3:  "Facebook account scam",
    4:  "Pension (military/civil)",
    5:  "Junk (icon/url fragments)",
    6:  "Weight loss & wellness",
    7:  "Gift voucher scam",
    8:  "Mycoplasma outbreak",
    9:  "Sudan-red food contamination",
    10: "Hsiao Bi-khim nationality",
    11: "Lai Ching-te (family)",
    12: "Taiwan–China identity",
    13: "Defense ministry & satellites",
    14: "Lai illegal construction",
    15: "Click-farm scam",
    16: "Junk (slide spam)",
    17: "iPASS transit card",
    18: "Election & voting",
    19: "Lai Ching-te (campaign)",
    20: "DPP / Ko Wen-je",
    21: "Real-name registration scam",
    22: "Loan scam (banks)",
    23: "LINE sticker scam",
    24: "Junk (eating spam)",
    25: "Job & salary scam",
    26: "Stock-market scam",
    27: "Account registration scam",
    28: "Secretary job ads",
    29: "Dash-formatted scam ads",
}

print("→ Loading clustered data…")
df = pd.read_csv(ROOT / "data" / "processed" / "cofacts_clustered.csv")
df["article_createdAt"] = pd.to_datetime(df["article_createdAt"], format="mixed", utc=True, errors="coerce")
df["year"] = df["article_createdAt"].dt.year
print(f"  Rows: {len(df):,}")

# 1. Keyword classifier
SCAM = [
    "詐騙","加賴","加LINE","代操","保證獲利","翻倍","穩赚","內線","主力","獲利",
    "投資","交易","帳戶","註冊","帳號","密碼","驗證","個資","身分證",
    "兼職","薪資","薪水","0元","免費","限時","中獎","領取","點擊","下單",
    "貸款","借款","銀行","房貸","信用卡","卡片","匯款","訂單","實名","款卡",
    "兼差","【工作】","正職","在家工作","日領","月領","高薪",
]
POLITICAL = [
    "選舉","投票","候選","民進黨","國民黨","民眾黨","時代力量",
    "蔡英文","賴清德","韓國瑜","侯友宜","柯文哲","馬英九","蔣萬安","郭台銘","朱立倫",
    "中國","中共","共產黨","國防","兩岸","統一","台獨","罷免","公投","立委","總統","縣長","市長",
    "親中","抗中","國安",
]
HEALTH = [
    "疫苗","接種","病毒","肺炎","新冠","COVID","Covid","covid","癌症","致癌",
    "健康","治療","醫師","醫院","藥物","副作用","感染","食品","營養","保健",
]

text = df["text_preview"].fillna("").astype(str)


def hits(t, kws): return sum(1 for k in kws if k in t)


df["scam_hits"]   = text.apply(lambda t: hits(t, SCAM))
df["pol_hits"]    = text.apply(lambda t: hits(t, POLITICAL))
df["health_hits"] = text.apply(lambda t: hits(t, HEALTH))


def classify(r):
    h = {"scam": r["scam_hits"], "political": r["pol_hits"], "health": r["health_hits"]}
    return "other" if max(h.values()) == 0 else max(h, key=h.get)


df["topic"] = df.apply(classify, axis=1)

# 2. Window assignment — reuse columns saved by script 05 (avoids tz parsing issues)
for col in ["in_2020_win", "in_2024_win", "is_url_only"]:
    if col in df.columns and df[col].dtype == object:
        df[col] = df[col].astype(str).str.lower().eq("true")

df["window"] = "off"
df.loc[df["in_2020_win"], "window"] = "2020"
df.loc[df["in_2024_win"], "window"] = "2024"
print(f"  Windows: " + ", ".join(f"{k}={v:,}" for k, v in df["window"].value_counts().items()))

# 3. Headlines
print("\n" + "=" * 80)
print("OVERALL TOPIC MIX")
print("=" * 80)
print((df["topic"].value_counts(normalize=True) * 100).round(1).to_string())

print("\n" + "=" * 80)
print("TOPIC MIX BY ELECTION WINDOW (%)")
print("=" * 80)
win_mix = df.groupby(["window", "topic"]).size().unstack(fill_value=0)
win_pct = win_mix.div(win_mix.sum(axis=1), axis=0).mul(100).round(1)
print(win_pct.to_string())

print("\n" + "=" * 80)
print("MEDIAN LATENCY (hours) BY WINDOW × TOPIC")
print("=" * 80)
print(df.groupby(["window", "topic"])["latency_hours"].median().round(1).unstack().to_string())

# 4. Yearly time series → stacked area
print("\n→ Plotting topic drift over time…")
mix = df.groupby(["year", "topic"]).size().unstack(fill_value=0)
mix_pct = mix.div(mix.sum(axis=1), axis=0).mul(100)
mix_pct = mix_pct.loc[(mix_pct.index >= 2017) & (mix_pct.index <= 2025)]

order = ["scam", "political", "health", "other"]
colors = {"scam": "#ef4444", "political": "#3b82f6", "health": "#10b981", "other": "#cbd5e1"}

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.stackplot(
    mix_pct.index,
    [mix_pct.get(c, pd.Series(0, index=mix_pct.index)) for c in order],
    labels=order, colors=[colors[c] for c in order], alpha=0.88,
)
ax.set_ylim(0, 100)
ax.set_xlim(mix_pct.index.min(), mix_pct.index.max())
ax.set_xlabel("year")
ax.set_ylabel("% of submissions")
ax.set_title("Cofacts content drift: political/health → scam triage",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper left", framealpha=0.95)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)
for yr, col in [(2020, "#3b82f6"), (2024, "#f59e0b")]:
    ax.axvline(yr + 0.03, color=col, linestyle="--", alpha=0.6, linewidth=1.5)
    ax.text(yr + 0.05, 95, f"{yr} election", color=col, fontsize=9, fontweight="bold")
plt.tight_layout()
plt.savefig(ROOT / "viz" / "topic_drift.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ viz/topic_drift.png")

# 5. Focused 2024-window clustering (UMAP 5D + HDBSCAN)
print("\n" + "=" * 80)
print("FOCUSED 2024-WINDOW CLUSTERING")
print("=" * 80)

emb = np.load(ROOT / "data" / "processed" / "embeddings.npy")
idx_full = np.load(ROOT / "data" / "processed" / "embeddings_idx.npy")
win2024_set = set(df.index[df["window"] == "2024"].tolist())
mask = np.array([i in win2024_set for i in idx_full])
emb_2024 = emb[mask]
idx_2024 = idx_full[mask]
print(f"  2024-window embeddable articles: {len(idx_2024):,}")

import umap, hdbscan
print("→ UMAP (cosine, 5D) for clustering…")
red_5d = umap.UMAP(n_components=5, n_neighbors=10, min_dist=0.0,
                   metric="cosine", random_state=42, verbose=False).fit_transform(emb_2024)

print("→ HDBSCAN (min_cluster_size=20)…")
labels_24 = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5, metric="euclidean").fit_predict(red_5d)
n2 = len(set(labels_24)) - (1 if -1 in labels_24 else 0)
print(f"  Clusters: {n2} | Noise: {(labels_24 == -1).sum():,} ({(labels_24 == -1).mean():.1%})")

sub = df.loc[idx_2024].copy()
sub["c24"] = labels_24

# c-TF-IDF on 2024 clusters
from sklearn.feature_extraction.text import CountVectorizer
cls = sorted(c for c in set(labels_24) if c >= 0)
docs = [" ".join(sub.loc[sub["c24"] == c, "text_preview"].fillna("").astype(str)) for c in cls]
vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=10000)
X = vec.fit_transform(docs).toarray()
terms = vec.get_feature_names_out()
dc = (X > 0).sum(axis=0)
idf = np.log(len(docs) / (dc + 1)) + 1
tn = X / (X.sum(axis=1, keepdims=True) + 1e-9)
ctfidf = tn * idf
top_terms = {c: [terms[j] for j in ctfidf[i].argsort()[::-1][:6] if not terms[j].isspace()]
             for i, c in enumerate(cls)}

prof = []
for c in cls:
    s = sub[sub["c24"] == c]
    prof.append({
        "c24": c,
        "size": len(s),
        "pct_of_2024_win": round(len(s) / len(sub) * 100, 1),
        "median_h": round(s["latency_hours"].median(), 1),
        "dominant_topic": s["topic"].mode().iloc[0] if len(s["topic"].mode()) else "—",
        "top_terms": ", ".join(top_terms[c][:5]),
    })

prof_df = pd.DataFrame(prof).sort_values("size", ascending=False)
prof_df["label_en"] = prof_df["c24"].map(LABELS_2024).fillna("(other)")
prof_df.to_csv(ROOT / "data" / "processed" / "cluster_profiles_2024.csv", index=False)
print("\nTOP 15 CLUSTERS WITHIN 2024 WINDOW (by size):")
print(prof_df.head(15).to_string(index=False))

# Save augmented full df
df.to_csv(ROOT / "data" / "processed" / "cofacts_topic_classified.csv", index=False)
print("\n→ Saved cluster_profiles_2024.csv (with label_en), cofacts_topic_classified.csv")

# 6. Bonus: scam-vs-political latency comparison in 2024 window
print("\n" + "=" * 80)
print("LATENCY BY TOPIC IN 2024 WINDOW")
print("=" * 80)
win2024 = df[df["window"] == "2024"]
print(win2024.groupby("topic")["latency_hours"].agg(["count", "median", "mean"]).round(1).to_string())

# 7. Annotated slow-clusters viz (2D UMAP + English labels)
#    Labels placed in the empty left margin with leader lines to clusters.
#    Axes hidden — raw UMAP coordinates have no interpretable units.
print("\n→ UMAP 2D projection for viz/clusters_slow.png…")
red_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.10,
                   metric="cosine", random_state=42, verbose=False).fit_transform(emb_2024)
sub["x"] = red_2d[:, 0]
sub["y"] = red_2d[:, 1]

SLOW_N = 8
slowest = (
    prof_df[(prof_df["size"] >= 30) & (~prof_df["label_en"].str.startswith("Junk"))]
    .sort_values("median_h", ascending=False)
    .head(SLOW_N)
)
slow_ids = set(slowest["c24"].tolist())

topic_colors = {"scam": "#ef4444", "political": "#3b82f6", "health": "#10b981", "other": "#94a3b8"}

# Robust data bounds for label placement (trim 2% outliers)
x_lo_r, x_hi_r = sub["x"].quantile(0.02), sub["x"].quantile(0.98)
y_lo_r, y_hi_r = sub["y"].quantile(0.02), sub["y"].quantile(0.98)
x_range = x_hi_r - x_lo_r
y_range = y_hi_r - y_lo_r
label_x = x_lo_r - x_range * 0.32   # column of labels in the empty left margin

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Background: all 2024-window points in very subtle gray
ax.scatter(sub["x"], sub["y"], s=6, c="#d1d5db", alpha=0.28, edgecolors="none", zorder=1)

# Foreground: highlight slow clusters by topic color
for c in slow_ids:
    mask_c = sub["c24"] == c
    topic = prof_df.loc[prof_df["c24"] == c, "dominant_topic"].iloc[0]
    ax.scatter(
        sub.loc[mask_c, "x"], sub.loc[mask_c, "y"],
        s=38, c=topic_colors.get(topic, "#94a3b8"),
        alpha=0.92, edgecolors="white", linewidth=0.7, zorder=2,
    )

# Compute cluster centroids and sort by y (top → bottom) so leader lines don't cross
centroids = []
for _, r in slowest.iterrows():
    mask_c = sub["c24"] == r["c24"]
    if not mask_c.any():
        continue
    cx = sub.loc[mask_c, "x"].median()
    cy = sub.loc[mask_c, "y"].median()
    centroids.append((r, cx, cy))
centroids.sort(key=lambda t: -t[2])

# Distribute label y positions evenly across the plot height
n_lbl = len(centroids)
label_ys = np.linspace(y_hi_r - y_range * 0.03, y_lo_r + y_range * 0.03, n_lbl)

# Draw labels on the LEFT with colored leader lines to each cluster
for (r, cx, cy), ly in zip(centroids, label_ys):
    edge = topic_colors.get(r["dominant_topic"], "#94a3b8")
    label = f"{r['label_en']}\n{r['median_h']:.0f}h  ·  n={int(r['size'])}"
    ax.annotate(
        label, xy=(cx, cy),
        xytext=(label_x, ly),
        fontsize=10.5, fontweight="bold",
        ha="right", va="center", color="#1f2937",
        arrowprops=dict(
            arrowstyle="-", color=edge, lw=1.3, alpha=0.85,
            connectionstyle="arc3,rad=0.15",
        ),
        bbox=dict(
            boxstyle="round,pad=0.55", fc="white",
            ec=edge, lw=1.4, alpha=0.98,
        ),
        zorder=3,
    )

# Topic legend (lower right, compact) — only show topics actually present
# among the highlighted slow clusters (e.g. if all 8 are scam+other, hide
# political/health entries instead of misleading the reader).
present_topics = set(prof_df.loc[prof_df["c24"].isin(slow_ids), "dominant_topic"])
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=t.capitalize(),
           markerfacecolor=col, markersize=11, markeredgecolor='white', markeredgewidth=0.6)
    for t, col in topic_colors.items() if t in present_topics
]
leg = ax.legend(
    handles=legend_elements, loc="lower right", title="Dominant topic",
    framealpha=0.97, fontsize=10.5, title_fontsize=10.5,
    borderpad=0.7, handletextpad=0.6,
)
leg.get_frame().set_edgecolor("#e5e7eb")

# Title + subtitle (left-aligned, modern hierarchy)
ax.set_title(
    "Slowest narrative clusters in the 2024 election window",
    fontsize=16, fontweight="bold", pad=18, loc="left", x=0.0,
)
ax.text(
    0.0, 1.012,
    f"{len(sub):,} articles · {n2} clusters · top {len(slow_ids)} by median fact-check latency · "
    "position = semantic similarity (UMAP)",
    transform=ax.transAxes, fontsize=10.5, color="#6b7280", va="bottom", ha="left",
)

# Clean axes: no spines, no ticks, no labels
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Plot bounds: left edge accommodates label column; right edge follows data
ax.set_xlim(label_x - x_range * 0.08, sub["x"].max() + x_range * 0.03)
ax.set_ylim(sub["y"].min() - y_range * 0.05, sub["y"].max() + y_range * 0.05)

plt.tight_layout()
plt.savefig(ROOT / "viz" / "clusters_slow.png", dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print("  ✓ viz/clusters_slow.png")

print("\n✅ Done.")
