#!/usr/bin/env python3
"""Build a human spot-check sheet of rows where DeepSeek disagrees with the panel
consensus, split pre/post cutoff. Reviewers adjudicate the label to upgrade the
memorization claim from 'weak multi-model signal' to human-validated.
"""
import json, glob, os
import pandas as pd

gold = pd.read_csv("gold_cohort_sample.csv").set_index("gold_id")
con = pd.read_csv("gold_cohort_consensus.csv")
raters = {}
for path in sorted(glob.glob("labels_*_gold.json")):
    m = os.path.basename(path)[len("labels_"):-len("_gold.json")]
    d = json.load(open(path, encoding="utf-8"))
    raters[m] = {int(x["id"]): str(x["topic"]).strip().lower() for x in d if x.get("topic")}

rows = []
for _, r in con.iterrows():
    gid = int(r["gold_id"])
    if str(r["deepseek"]) == str(r["consensus"]):
        continue  # keep only DeepSeek-vs-consensus divergences
    era = "pre" if str(r["cohort"]) in ("2020", "2024") else "post"
    row = {
        "gold_id": gid, "cohort": r["cohort"], "era": era,
        "articleId": gold.loc[gid, "articleId"],
        "latency_hours": gold.loc[gid, "latency_hours"],
        "deepseek": r["deepseek"], "consensus": r["consensus"],
        "others_consensus": r["others_consensus"],
    }
    for m in sorted(raters):
        row[f"vote_{m}"] = raters[m].get(gid)
    row["text_preview"] = gold.loc[gid, "text_preview"]
    row["human_label"] = ""
    row["human_notes"] = ""
    rows.append(row)

df = pd.DataFrame(rows).sort_values(["era", "cohort", "gold_id"])
df.to_csv("spotcheck_disagreements.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
print(f"DeepSeek-vs-consensus disagreement rows: {len(df)}")
print("\nby era x cohort:")
print(df.groupby(["era", "cohort"]).size().to_string())
print("\nrows to review:")
print(df[["gold_id", "cohort", "era", "deepseek", "consensus", "text_preview"]].to_string(index=False))
