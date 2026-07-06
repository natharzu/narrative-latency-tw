#!/usr/bin/env python3
"""Compare Qwen3.7-Plus vs Qwen3.7-Max as candidate raters for the gold-cohort panel.

Goal: decide whether to use Plus, Max, or both. Judge each candidate against the
ESTABLISHED 6-rater panel consensus (deepseek + grok/glm/gemini/gpt/opus), measure
mutual redundancy, independence from the panel, and affect-range usage. Does NOT
write the canonical gold_cohort_consensus.csv.
"""
import json, re, subprocess, glob, os, statistics as st
from collections import Counter
from itertools import combinations
import pandas as pd

PLUS = "39575f3013aa80a4bde0e4da08ca63d9"
MAX  = "39575f3013aa80abbba5d0c49d43cac3"
DIMS = ["valence", "arousal", "urgency", "threat", "anger"]


def fetch(page_id):
    parts, cursor = [], None
    while True:
        path = f"v1/blocks/{page_id}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        data = json.loads(subprocess.run(["ntn", "api", path], capture_output=True, text=True).stdout)
        for b in data["results"]:
            rt = (b.get("paragraph", {}).get("rich_text") or b.get("code", {}).get("rich_text") or [])
            parts.append("".join(r["plain_text"] for r in rt))
        if data.get("has_more"):
            cursor = data["next_cursor"]
        else:
            break
    return "".join(parts)


def clean_parse(text):
    t = text.replace("<br>", "").replace("<empty-block/>", "")
    t = re.sub(r"\\([\\{}\[\]$])", r"\1", t)
    return json.loads(t[t.index("["): t.rindex("]") + 1])


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c]/n)*(cb[c]/n) for c in cats)
    return 1.0 if pe == 1.0 else (po - pe)/(1 - pe)


def fleiss(rows):
    rows = [r for r in rows if len(r) == len(rows[0])]
    N, R = len(rows), len(rows[0])
    cats = sorted({c for r in rows for c in r})
    col = {c: 0 for c in cats}; P_i = []
    for r in rows:
        cnt = Counter(r)
        for c in cats:
            col[c] += cnt[c]
        P_i.append((sum(v*v for v in cnt.values()) - R)/(R*(R-1)))
    Pbar = sum(P_i)/N
    Pe = sum((col[c]/(N*R))**2 for c in cats)
    return (Pbar - Pe)/(1 - Pe)


def majority(votes):
    votes = [v for v in votes if v]
    if not votes:
        return None
    c = Counter(votes).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None
    return c[0][0]


# ---- load established 6-rater panel ----
gold = pd.read_csv("gold_cohort_sample.csv")
gold["llm_topic"] = gold["llm_topic"].astype(str).str.strip().str.lower()
cohort = dict(zip(gold["gold_id"], gold["cohort"].astype(str)))
base = {}
for path in sorted(glob.glob("labels_*_gold.json")):
    m = os.path.basename(path)[len("labels_"):-len("_gold.json")]
    data = json.load(open(path, encoding="utf-8"))
    base[m] = {int(d["id"]): str(d["topic"]).strip().lower() for d in data if d.get("topic")}
base["deepseek"] = {int(k): v for k, v in zip(gold["gold_id"], gold["llm_topic"]) if v and v != "nan"}
IDS = sorted(gold["gold_id"])

# ---- fetch + parse both Qwen ----
qp_raw = clean_parse(fetch(PLUS)); qm_raw = clean_parse(fetch(MAX))
print(f"Qwen-Plus rows={len(qp_raw)}  Qwen-Max rows={len(qm_raw)}")
qp = {int(d["id"]): d for d in qp_raw}; qm = {int(d["id"]): d for d in qm_raw}
qp_t = {i: str(qp[i]["topic"]).strip().lower() for i in qp}
qm_t = {i: str(qm[i]["topic"]).strip().lower() for i in qm}

# ---- established consensus over the 6 base raters ----
base_con = {}
for gid in IDS:
    base_con[gid] = majority([base[m][gid] for m in base if gid in base[m]])


def score_vs_consensus(cand, label):
    print(f"\n=== {label} vs established 6-rater consensus ===")
    for cname, sel in [("2020", ["2020"]), ("2024", ["2024"]),
                       ("recent", ["recent"]), ("PRE (2020+2024)", ["2020", "2024"]),
                       ("OVERALL", ["2020", "2024", "recent"])]:
        ids = [i for i in IDS if cohort[i] in sel and base_con[i] is not None]
        a = [cand[i] for i in ids]; b = [base_con[i] for i in ids]
        acc = sum(x == y for x, y in zip(a, b))/len(ids)
        print(f"  {cname:16s} n={len(ids):3d}  acc={acc:.3f}  kappa={cohen_kappa(a,b):.3f}")


score_vs_consensus(qp_t, "Qwen-Plus")
score_vs_consensus(qm_t, "Qwen-Max")

# ---- Plus vs Max mutual redundancy ----
print("\n=== Qwen-Plus vs Qwen-Max (mutual redundancy) ===")
for cname, sel in [("2020", ["2020"]), ("2024", ["2024"]), ("recent", ["recent"]),
                   ("OVERALL", ["2020", "2024", "recent"])]:
    ids = [i for i in IDS if cohort[i] in sel]
    a = [qp_t[i] for i in ids]; b = [qm_t[i] for i in ids]
    acc = sum(x == y for x, y in zip(a, b))/len(ids)
    print(f"  {cname:8s} n={len(ids):3d}  topic-agreement={acc:.3f}  kappa={cohen_kappa(a,b):.3f}")
disagree = [(i, qp_t[i], qm_t[i]) for i in IDS if qp_t[i] != qm_t[i]]
print(f"  disagreements: {len(disagree)}/120")
for i, p, m in disagree:
    print(f"    id {i:3d} [{cohort[i]:6s}]  Plus={p:10s} Max={m:10s} consensus={base_con[i]}")

# ---- independence: agreement with each panel member ----
print("\n=== agreement with each established rater (lower = more independent) ===")
panel = sorted(base)
print("           " + "  ".join(f"{n[:6]:>6}" for n in panel))
for cand, nm in [(qp_t, "Plus"), (qm_t, "Max")]:
    cells = [f"{sum(cand[i]==base[m][i] for i in IDS)/120:.2f}" for m in panel]
    print(f"  {nm:8s} " + "  ".join(f"{c:>6}" for c in cells))
print(f"  {'Plus~Max':8s} " + f"{sum(qp_t[i]==qm_t[i] for i in IDS)/120:.2f}")

# ---- effect on Fleiss kappa ----
print("\n=== Fleiss kappa under different panel compositions (120 ids) ===")
def fk(extra):
    mats = []
    for gid in IDS:
        row = [base[m][gid] for m in base]
        for e in extra:
            row.append(e[gid])
        mats.append(row)
    return fleiss(mats)
print(f"  base 6                : {fk([]):.3f}")
print(f"  6 + Plus  (7)         : {fk([qp_t]):.3f}")
print(f"  6 + Max   (7)         : {fk([qm_t]):.3f}")
print(f"  6 + both  (8)         : {fk([qp_t, qm_t]):.3f}")

# ---- affect compression for both Qwen ----
print("\n=== affect-range usage (Qwen Plus / Max) ===")
def affrow(d, nm):
    val = [d[i]["valence"] for i in d]; aro = [d[i]["arousal"] for i in d]
    inten = [st.mean([d[i][k] for k in ["arousal","urgency","threat","anger"]]) for i in d]
    return {"rater": nm, "val_mean": round(st.mean(val),2), "val_sd": round(st.pstdev(val),2),
            "val_min": min(val), "val_max": max(val),
            "pct_val+2": round(100*sum(v==2 for v in val)/len(val),1),
            "aro_mean": round(st.mean(aro),2), "aro_sd": round(st.pstdev(aro),2), "aro_max": max(aro),
            "pct_aro5": round(100*sum(a==5 for a in aro)/len(aro),1),
            "intensity_mean": round(st.mean(inten),2), "intensity_sd": round(st.pstdev(inten),2)}
adf = pd.DataFrame([affrow(qp, "qwen-plus"), affrow(qm, "qwen-max")])
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
print(adf.to_string(index=False))

# affect agreement Plus vs Max (exact match rate per dim)
print("\n=== affect exact-match rate Plus vs Max, by dimension ===")
for dim in DIMS:
    match = sum(qp[i][dim] == qm[i][dim] for i in IDS)/120
    mae = st.mean(abs(qp[i][dim]-qm[i][dim]) for i in IDS)
    print(f"  {dim:8s} exact={match:.2f}  MAE={mae:.2f}")

# save candidate label files as NON-glob-matching names (won't auto-join compare)
json.dump([{"id": i, "topic": qp_t[i], **{k: qp[i][k] for k in DIMS}, "confidence": qp[i].get("confidence")} for i in IDS],
          open("cand_qwenplus.json", "w"), ensure_ascii=False)
json.dump([{"id": i, "topic": qm_t[i], **{k: qm[i][k] for k in DIMS}, "confidence": qm[i].get("confidence")} for i in IDS],
          open("cand_qwenmax.json", "w"), ensure_ascii=False)
print("\nwrote cand_qwenplus.json, cand_qwenmax.json (NON-glob names; canonical consensus untouched)")
print("DONE")
