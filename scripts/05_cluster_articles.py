"""
05_cluster_articles.py — semantic clustering of Cofacts articles.

Pipeline:
  1. Load all 68k pairs; strip URLs from text_preview
  2. Split: meaningful texts vs URL-only
  3. Embed meaningful texts with multilingual-MiniLM
  4. UMAP-reduce (10D for clustering, 2D for viz)
  5. HDBSCAN cluster
  6. c-TF-IDF top terms per cluster (char n-grams)
  7. Profile + save outputs
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer

from narrative_latency import PROC, VIZ, E2020, E2024, WIN, set_plot_style

set_plot_style()

SRC = PROC / "cofacts_latency.csv"
EMB_CACHE = PROC / "embeddings.npy"
IDX_CACHE = PROC / "embeddings_idx.npy"
VIZ.mkdir(exist_ok=True)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
def clean(s: object) -> str:
    if not isinstance(s, str):
        return ""
    return URL_RE.sub(" ", s).strip()

print("→ Loading data…")
df = pd.read_csv(SRC)
df["article_createdAt"] = pd.to_datetime(
    df["article_createdAt"], utc=True, format="ISO8601", errors="coerce"
)
df["year"] = df["article_createdAt"].dt.year
df["clean_text"] = df["text_preview"].apply(clean)
df["is_url_only"] = df["clean_text"].str.len() < 10
print(f"  Total: {len(df):,}")
print(f"  URL-only: {df['is_url_only'].sum():,} ({df['is_url_only'].mean():.1%})")
print(f"  Embeddable: {(~df['is_url_only']).sum():,}")

# Election windows
df["in_2020_win"] = (df["article_createdAt"] - E2020).abs() <= WIN
df["in_2024_win"] = (df["article_createdAt"] - E2024).abs() <= WIN

# 1. Embed
mask = ~df["is_url_only"]
texts = df.loc[mask, "clean_text"].tolist()
idx = df.loc[mask].index.to_numpy()

if EMB_CACHE.exists() and IDX_CACHE.exists():
    print("→ Loading cached embeddings…")
    embeddings = np.load(EMB_CACHE)
    cached_idx = np.load(IDX_CACHE)
    assert len(cached_idx) == len(idx) and (cached_idx == idx).all(), "Cache stale; delete *.npy"
else:
    print("→ Embedding (this takes a few minutes)…")
    # Lazy import on purpose: sentence-transformers + torch are heavy (hundreds
    # of MB) and only needed on a cache miss. They live in the `ml` optional
    # dependency group, not the core install.
    from sentence_transformers import SentenceTransformer
    import torch
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)
    embeddings = model.encode(
        texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    np.save(EMB_CACHE, embeddings)
    np.save(IDX_CACHE, idx)
    print(f"  Saved → {EMB_CACHE.name}")
print(f"  Embeddings: {embeddings.shape}")

# 2. UMAP
print("→ UMAP → 10D for clustering…")
emb_10d = umap.UMAP(
    n_components=10, n_neighbors=15, min_dist=0.0,
    metric="cosine", random_state=42, verbose=False,
).fit_transform(embeddings)

print("→ UMAP → 2D for viz…")
emb_2d = umap.UMAP(
    n_components=2, n_neighbors=15, min_dist=0.1,
    metric="cosine", random_state=42, verbose=False,
).fit_transform(embeddings)

# 3. HDBSCAN
print("→ HDBSCAN clustering…")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=80, min_samples=10, metric="euclidean",
)
labels = clusterer.fit_predict(emb_10d)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"  Clusters: {n_clusters}")
print(f"  Noise (-1): {(labels == -1).sum():,} ({(labels == -1).mean():.1%})")

# 4. Attach back
df["cluster_id"] = pd.NA
df.loc[idx, "cluster_id"] = labels
df.loc[df["is_url_only"], "cluster_id"] = -2   # explicit URL-only bucket
df["umap_x"] = np.nan
df["umap_y"] = np.nan
df.loc[idx, "umap_x"] = emb_2d[:, 0]
df.loc[idx, "umap_y"] = emb_2d[:, 1]

# 5. c-TF-IDF labels
print("→ c-TF-IDF labels…")
real_clusters = sorted(c for c in set(labels) if c >= 0)
docs = []
for cid in real_clusters:
    sub_texts = df.loc[df["cluster_id"] == cid, "clean_text"].tolist()
    docs.append(" ".join(sub_texts))
vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=15000)
X = vec.fit_transform(docs).toarray()
terms = vec.get_feature_names_out()
df_count = (X > 0).sum(axis=0)
idf = np.log(len(docs) / (df_count + 1)) + 1
tf_norm = X / (X.sum(axis=1, keepdims=True) + 1e-9)
ctfidf = tf_norm * idf
top_terms = {
    cid: [terms[j] for j in ctfidf[i].argsort()[::-1][:8]
          if not terms[j].isspace()]
    for i, cid in enumerate(real_clusters)
}

# 6. Profile
profiles = []
for cid in [-2] + real_clusters:
    sub = df[df["cluster_id"] == cid]
    profiles.append({
        "cluster_id": cid,
        "label": "URL-only" if cid == -2 else ", ".join(top_terms[cid][:5]),
        "size": len(sub),
        "median_latency_h": round(sub["latency_hours"].median(), 1),
        "pct_2020_win": round(sub["in_2020_win"].mean() * 100, 1),
        "pct_2024_win": round(sub["in_2024_win"].mean() * 100, 1),
    })
prof = pd.DataFrame(profiles)

# 7. Save outputs
prof.to_csv(PROC / "cluster_profiles.csv", index=False)
df.to_csv(PROC / "cofacts_clustered.csv", index=False)
print(f"\n→ Saved cluster_profiles.csv ({len(prof)} rows)")
print(f"→ Saved cofacts_clustered.csv")

# 8. Headline outputs
print("\n" + "=" * 90)
print("TOP 10 LARGEST CLUSTERS")
print("=" * 90)
print(prof.nlargest(10, "size").to_string(index=False))

print("\n" + "=" * 90)
print("TOP 10 SLOWEST CLUSTERS (median latency, ≥100 articles)")
print("=" * 90)
slow = prof[prof["size"] >= 100].nlargest(10, "median_latency_h")
print(slow.to_string(index=False))

print("\n" + "=" * 90)
print("TOP 10 CLUSTERS MOST CONCENTRATED IN 2024 WINDOW (≥50 articles)")
print("=" * 90)
hot2024 = prof[prof["size"] >= 50].nlargest(10, "pct_2024_win")
print(hot2024.to_string(index=False))

# 9. Visualizations
print("\n→ Plotting…")
fig, ax = plt.subplots(figsize=(11, 8))
emb = df[mask].copy()
emb["cluster_id"] = labels
noise = emb[emb["cluster_id"] == -1]
real  = emb[emb["cluster_id"] >= 0]
ax.scatter(noise["umap_x"], noise["umap_y"], s=1, c="#cbd5e1", alpha=0.3, label=f"noise (N={len(noise):,})")
sc = ax.scatter(real["umap_x"], real["umap_y"], s=2, c=real["cluster_id"], cmap="tab20", alpha=0.6)
ax.set_title(f"Cofacts articles, semantic clusters (HDBSCAN, K={n_clusters})", fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(VIZ / "clusters_umap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ viz/clusters_umap.png")

# Slow-cluster horizontal bar
top10 = slow.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(range(len(top10)), top10["median_latency_h"], color="#ef4444")
ax.set_yticks(range(len(top10)))
ax.set_yticklabels([f"[{r['cluster_id']:>2}] {r['label'][:60]}" for _, r in top10.iterrows()], fontsize=9)
ax.set_xlabel("median latency (hours)")
ax.set_title("Top 10 slowest clusters (≥100 articles)", fontsize=13, fontweight="bold")
ax.axvline(24, color="#94a3b8", linestyle="--", alpha=0.6, label="1 day")
ax.spines[["top", "right"]].set_visible(False)
ax.legend()
plt.tight_layout()
plt.savefig(VIZ / "clusters_slow.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ viz/clusters_slow.png")

print("\n✅ Done.")
