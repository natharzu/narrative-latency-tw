"""01_clean equivalent — load raw IORG CSV, validate schema, save processed."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/iorg_narratives_scraped.csv")
OUT = Path("data/processed/narratives_clean.csv")

df = pd.read_csv(RAW)
print(f"Loaded {len(df)} narratives from {RAW}")

expected = {"narrative_id", "case_id", "case_name", "narrative_text",
            "topic_cluster", "time_frame_start", "time_frame_end",
            "stage_1_date", "stage_2_date", "stage_3_date", "stage_4_date",
            "election_window", "source_url", "retrieved_at"}
missing = expected - set(df.columns)
assert not missing, f"Missing columns: {missing}"
print(f"Schema OK ({len(df.columns)} columns)")

print("\nNarratives by topic cluster:")
print(df["topic_cluster"].value_counts().to_string())

print(f"\nUnique cases: {df['case_id'].nunique()}")

print("\nStage date completeness:")
for col in ["stage_1_date", "stage_2_date", "stage_3_date", "stage_4_date"]:
    filled = df[col].notna().sum()
    print(f"  {col}: {filled}/{len(df)} filled ({filled/len(df)*100:.0f}%)")

clean = df.dropna(subset=["stage_1_date", "stage_4_date"])
print(f"\nAfter drop rule (Stage 1 + Stage 4 required): {len(clean)}/{len(df)} narratives retained")

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")
