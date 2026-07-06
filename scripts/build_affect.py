#!/usr/bin/env python3
"""Extend per-rater gold label files with full affect vectors and quantify
the affect-compression contrast (DeepSeek central-tendency bias vs frontier panel).

- 5 chat raters (grok, glm, gemini, gpt, opus): pulled straight from their Notion
  pages via `ntn` (clean, unescaped JSON) -> enrich existing labels_<m>_gold.json IN PLACE
  (topic kept exactly as-is so 16_gold_compare.py results are unchanged; affect fields added).
- deepseek: affect joined from the pipeline pilot CSVs via articleId (topic = gold llm_topic).
  Stored separately (deepseek_affect.json) so it is NOT double-counted by the compare glob.
"""
import json, re, subprocess
import pandas as pd

PAGES = {
    "grok":   "39575f3013aa80068898ef79975a3886",
    "glm":    "39575f3013aa80a0a10cc45906118445",
    "gemini": "39575f3013aa80b3a068cf2caa898fca",
    "gpt":    "39575f3013aa8006a9d1ddcdda37a0ae",
    "opus":   "39575f3013aa80c3b9e5cbc1d802753e",
}
DIMS = ["valence", "arousal", "urgency", "threat", "anger"]


def fetch_page_text(page_id):
    parts, cursor = [], None
    while True:
        path = f"v1/blocks/{page_id}/children?page_size=100"
        if cursor:
            path += f"&start_cursor={cursor}"
        out = subprocess.run(["ntn", "api", path], capture_output=True, text=True)
        data = json.loads(out.stdout)
        for b in data["results"]:
            rt = (b.get("paragraph", {}).get("rich_text")
                  or b.get("code", {}).get("rich_text") or [])
            parts.append("".join(r["plain_text"] for r in rt))
        if data.get("has_more"):
            cursor = data["next_cursor"]
        else:
            break
    return "".join(parts)


def clean_parse(text):
    t = text.replace("<br>", "").replace("<empty-block/>", "")
    t = re.sub(r"\\([\\{}\[\]$])", r"\1", t)  # strip markdown escapes if any
    return json.loads(t[t.index("["): t.rindex("]") + 1])


affect = {}  # model -> {id -> {dim: val, confidence}}

# --- 5 chat raters from pages ---
for name, pid in PAGES.items():
    arr = clean_parse(fetch_page_text(pid))
    assert len(arr) == 120, f"{name}: expected 120 got {len(arr)}"
    affect[name] = {int(d["id"]): d for d in arr}

    # enrich existing label file IN PLACE (keep topic exactly, add affect)
    path = f"labels_{name}_gold.json"
    existing = json.load(open(path))
    page_by_id = affect[name]
    mismatches = 0
    for row in existing:
        i = int(row["id"])
        src = page_by_id[i]
        if str(row["topic"]).strip().lower() != str(src["topic"]).strip().lower():
            mismatches += 1
        for dim in DIMS:
            row[dim] = src[dim]
        row["confidence"] = src.get("confidence")
        row["political_subtheme"] = src.get("political_subtheme")
    json.dump(existing, open(path, "w"), ensure_ascii=False)
    print(f"enriched {path}: 120 rows, topic mismatches vs page = {mismatches}")

# --- deepseek affect from pipeline pilots via articleId join ---
gold = pd.read_csv("gold_cohort_sample.csv")
el = pd.read_csv("pilot500_labeled_elections.csv")
rc = pd.read_csv("pilot_recent_labeled.csv")
cols = ["articleId"] + DIMS + ["confidence", "llm_topic"]
pool = pd.concat([el[cols], rc[cols]]).drop_duplicates("articleId").set_index("articleId")
ds = {}
for _, g in gold.iterrows():
    row = pool.loc[g["articleId"]]
    ds[int(g["gold_id"])] = {
        "id": int(g["gold_id"]),
        "topic": str(g["llm_topic"]).strip().lower(),
        **{dim: (None if pd.isna(row[dim]) else (float(row[dim]) if dim == "valence" or True else row[dim])) for dim in DIMS},
        "confidence": None if pd.isna(row["confidence"]) else float(row["confidence"]),
    }
    # keep ints for the 1-5 / -2..2 scales
    for dim in DIMS:
        v = row[dim]
        ds[int(g["gold_id"])][dim] = None if pd.isna(v) else int(round(v))
affect["deepseek"] = ds
json.dump(list(ds.values()), open("deepseek_affect.json", "w"), ensure_ascii=False)
print(f"wrote deepseek_affect.json: {len(ds)} rows")

# ---------------- affect-compression analysis ----------------
import statistics as st

def vals(model, dim):
    return [affect[model][i][dim] for i in sorted(affect[model]) if affect[model][i][dim] is not None]

RATERS = ["deepseek", "grok", "glm", "gemini", "gpt", "opus"]
rows = []
for m in RATERS:
    n = len(affect[m])
    val = vals(m, "valence"); aro = vals(m, "arousal")
    intens = []
    for i in sorted(affect[m]):
        d = affect[m][i]
        four = [d[k] for k in ["arousal", "urgency", "threat", "anger"] if d[k] is not None]
        if len(four) == 4:
            intens.append(sum(four) / 4)
    rows.append({
        "rater": m, "n": n,
        "val_mean": round(st.mean(val), 2), "val_sd": round(st.pstdev(val), 2),
        "val_min": min(val), "val_max": max(val),
        "pct_val_+2": round(100 * sum(v == 2 for v in val) / len(val), 1),
        "pct_val_-2": round(100 * sum(v == -2 for v in val) / len(val), 1),
        "aro_mean": round(st.mean(aro), 2), "aro_sd": round(st.pstdev(aro), 2),
        "aro_max": max(aro),
        "pct_aro_5": round(100 * sum(a == 5 for a in aro) / len(aro), 1),
        "pct_aro_ge4": round(100 * sum(a >= 4 for a in aro) / len(aro), 1),
        "intensity_mean": round(st.mean(intens), 2), "intensity_sd": round(st.pstdev(intens), 2),
    })
summary = pd.DataFrame(rows)
summary.to_csv("affect_compression_summary.csv", index=False)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
print("\n================ AFFECT-COMPRESSION SUMMARY (n=120 each) ================")
print(summary.to_string(index=False))

print("\n---- extreme-value usage (does the rater ever touch the ends of the scale?) ----")
for m in RATERS:
    used5 = {dim: max(vals(m, dim)) for dim in ["arousal", "urgency", "threat", "anger"]}
    print(f"  {m:9s} valence[min={min(vals(m,'valence'))},max={max(vals(m,'valence'))}]  "
          f"max(arousal,urgency,threat,anger)=" + ",".join(f"{k[:3]}{used5[k]}" for k in used5))

print("\n---- per-dimension SD (compression = lower spread) ----")
sd_tbl = pd.DataFrame({m: {dim: round(st.pstdev(vals(m, dim)), 2) for dim in DIMS} for m in RATERS}).T
print(sd_tbl.to_string())
sd_tbl.to_csv("affect_sd_by_dim.csv")
print("\nDONE")
