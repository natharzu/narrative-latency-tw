"""
repair_dates.py - one-shot recovery tool.

Run only if article_createdAt has been corrupted in the processed CSVs.
Reconstructs article_createdAt = reply_createdAt - latency_hours and writes
back in place. NOT part of the canonical pipeline - a clean run from raw
data (scripts 02 -> 09) will never need it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from utils import PROC, reconstruct_article_dates

TARGETS = [
    "cofacts_latency.csv",
    "cofacts_election_windows.csv",
    "cofacts_clustered.csv",
    "cofacts_topic_classified.csv",
    "political_strict.csv",
    "political_subthemes.csv",
]

print("Reconstructing article_createdAt = reply_createdAt - latency_hours\n")
for fname in TARGETS:
    p = PROC / fname
    if not p.exists():
        print(f"  {fname}: missing, skip")
        continue
    df = pd.read_csv(p)
    if "reply_createdAt" not in df.columns or "latency_hours" not in df.columns:
        print(f"  {fname}: no reply_createdAt/latency_hours, skip")
        continue
    df["article_createdAt"] = reconstruct_article_dates(df)
    valid = df["article_createdAt"].notna().sum()
    df.to_csv(p, index=False)
    print(f"  ✓ {fname}: {valid:,}/{len(df):,} valid ({valid/len(df)*100:.1f}%)")
print("\n✅ Done.")
