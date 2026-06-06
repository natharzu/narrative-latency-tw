"""
Cofacts data loader and cleaner — local CSV mode.

Reads zipped CSVs from data/raw/cofacts/ (not in git — re-download from
https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw).
Joins each article to its FIRST normal reply, computes article->reply latency.
Output: data/processed/cofacts_latency.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import RAW, PROC

COFACTS = RAW / "cofacts"
OUT = PROC
OUT.mkdir(parents=True, exist_ok=True)

print("Loading Cofacts tables from local CSV zips...")
articles = pd.read_csv(COFACTS / "articles.csv.zip")
replies = pd.read_csv(COFACTS / "replies.csv.zip")
article_replies = pd.read_csv(COFACTS / "article_replies.csv.zip")

print(f"  articles:        {len(articles):>8,} rows")
print(f"  replies:         {len(replies):>8,} rows")
print(f"  article_replies: {len(article_replies):>8,} rows")

# Keep only NORMAL status (drop DELETED + BLOCKED)
articles = articles[articles["status"] == "NORMAL"].copy()
article_replies = article_replies[article_replies["status"] == "NORMAL"].copy()

# Parse timestamps to UTC datetime
for df, col in [(articles, "createdAt"),
                (replies, "createdAt"),
                (article_replies, "createdAt")]:
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

# For each article, take its FIRST article_reply (chronologically)
first_ar = (article_replies
            .sort_values("createdAt")
            .groupby("articleId", as_index=False)
            .first())

# Join: articles -> first article_reply -> replies
joined = (articles[["id", "createdAt", "articleType", "text"]]
          .rename(columns={"id": "articleId", "createdAt": "article_createdAt"})
          .merge(first_ar[["articleId", "replyId", "replyType"]], on="articleId", how="inner")
          .merge(replies[["id", "createdAt", "type"]]
                 .rename(columns={"id": "replyId",
                                  "createdAt": "reply_createdAt",
                                  "type": "reply_type"}),
                 on="replyId", how="inner"))

# Latency in hours; drop invalid (<0 or >1 year)
joined["latency_hours"] = (
    (joined["reply_createdAt"] - joined["article_createdAt"]).dt.total_seconds() / 3600.0
)
before = len(joined)
joined = joined[(joined["latency_hours"] >= 0) & (joined["latency_hours"] <= 24 * 365)]
print(f"Dropped {before - len(joined):,} rows with invalid latency (<0 or >1yr)")

# Focus on TEXT articles
joined = joined[joined["articleType"] == "TEXT"].copy()

# Trim text for storage (full text not needed downstream)
joined["text_preview"] = joined["text"].astype(str).str.slice(0, 200)
joined = joined.drop(columns=["text"])

print(f"Final dataset: {len(joined):,} article-reply pairs")
print(f"  Date range: {joined['article_createdAt'].min()} -> {joined['article_createdAt'].max()}")
print(f"  Median latency: {joined['latency_hours'].median():.1f} hours")

out_path = OUT / "cofacts_latency.csv"
joined.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
