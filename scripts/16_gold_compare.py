"""16_gold_compare.py — consensus gold + per-cohort validation / memorization probe (§2c).

Reads gold_cohort_sample.csv (carries DeepSeek's llm_topic and the keyword rule_topic
by gold_id) plus any rater files labels_<model>_gold.json (array of objects with at
least {id, topic}). Builds a majority-vote consensus gold and reports, per cohort
(2020 / 2024 / recent) and overall:

  * DeepSeek vs consensus  (topic accuracy + Cohen kappa)  -> validity
  * rule     vs consensus  (topic accuracy + Cohen kappa)  -> how weak the rule is
  * Fleiss kappa across all raters (ids where every rater is present)
  * DeepSeek vs consensus-of-OTHER-raters (leave-one-out) -> memorization probe,
    split pre-cutoff (2020+2024) vs post-cutoff (recent)

Defensive about partial coverage: metrics are computed only on ids with enough
votes. Warns when two raters are identical on their shared ids (duplicate-paste guard).

Usage:
    uv run python scripts/16_gold_compare.py
    uv run python scripts/16_gold_compare.py --exclude gemini   # drop a duplicate rater
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

try:
    from narrative_latency import PROC
except ModuleNotFoundError:  # sandbox / standalone
    PROC = Path(".")

TOPICS = ["scam", "political", "health", "other"]


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two aligned categorical label lists."""
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(rows: list[list[str]]) -> float | None:
    """Fleiss' kappa. rows[i] = list of category labels assigned to item i by each rater.
    Assumes a fixed number of raters per item (rows with a different count are dropped)."""
    if not rows:
        return None
    n_raters = Counter(len(r) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == n_raters]
    if not rows or n_raters < 2:
        return None
    N = len(rows)
    cats = sorted({c for r in rows for c in r})
    P_i = []
    col_tot = {c: 0 for c in cats}
    for r in rows:
        cnt = Counter(r)
        for c in cats:
            col_tot[c] += cnt[c]
        P_i.append((sum(v * v for v in cnt.values()) - n_raters) / (n_raters * (n_raters - 1)))
    Pbar = sum(P_i) / N
    Pe = sum((col_tot[c] / (N * n_raters)) ** 2 for c in cats)
    if Pe == 1.0:
        return 1.0
    return (Pbar - Pe) / (1 - Pe)


def majority(votes: list[str]) -> str | None:
    """Strict plurality; returns None on an empty list or a top tie."""
    votes = [v for v in votes if v]
    if not votes:
        return None
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def load_raters(exclude: set[str]) -> dict[str, dict[int, str]]:
    raters: dict[str, dict[int, str]] = {}
    for path in sorted(glob.glob(str(PROC / "labels_*_gold.json"))):
        model = os.path.basename(path)[len("labels_"):-len("_gold.json")]
        if model in exclude:
            print(f"  (excluding rater: {model})")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        raters[model] = {int(d["id"]): str(d["topic"]).strip().lower() for d in data if d.get("topic")}
    return raters


def warn_duplicates(raters: dict[str, dict[int, str]]) -> None:
    for m1, m2 in combinations(sorted(raters), 2):
        shared = set(raters[m1]) & set(raters[m2])
        if len(shared) < 10:
            continue
        same = sum(raters[m1][i] == raters[m2][i] for i in shared)
        if same == len(shared):
            print(f"  ⚠ DUPLICATE WARNING: {m1} and {m2} are identical on all {len(shared)} shared ids "
                  f"— likely a duplicate paste; exclude one with --exclude {m2}")


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Consensus gold + per-cohort validation (§2c).")
    ap.add_argument("--exclude", nargs="*", default=[], help="rater names to drop (e.g. gemini)")
    ap.add_argument("--min-raters", type=int, default=2, help="min votes to form a consensus")
    args = ap.parse_args()

    gold = pd.read_csv(PROC / "gold_cohort_sample.csv")
    gold["llm_topic"] = gold["llm_topic"].astype(str).str.strip().str.lower()
    gold["rule_topic"] = gold["rule_topic"].astype(str).str.strip().str.lower()
    ds = dict(zip(gold["gold_id"], gold["llm_topic"]))       # DeepSeek (pipeline)
    rule = dict(zip(gold["gold_id"], gold["rule_topic"]))
    cohort = dict(zip(gold["gold_id"], gold["cohort"].astype(str)))

    raters = load_raters(set(args.exclude))
    raters["deepseek"] = {int(k): v for k, v in ds.items() if v and v != "nan"}
    print("raters loaded:", ", ".join(f"{m}({len(v)})" for m, v in sorted(raters.items())))
    warn_duplicates(raters)

    # coverage per cohort
    print("\ncoverage per rater x cohort:")
    for m, v in sorted(raters.items()):
        by = Counter(cohort[i] for i in v if i in cohort)
        print(f"  {m:10s} " + "  ".join(f"{c}={by.get(c,0)}" for c in ["2020", "2024", "recent"]))

    # build consensus per gold_id
    recs = []
    for gid in gold["gold_id"]:
        all_votes = [raters[m][gid] for m in raters if gid in raters[m]]
        other_votes = [raters[m][gid] for m in raters if m != "deepseek" and gid in raters[m]]
        recs.append({
            "gold_id": gid,
            "cohort": cohort.get(gid),
            "deepseek": ds.get(gid),
            "rule": rule.get(gid),
            "n_votes": len(all_votes),
            "consensus": majority(all_votes) if len(all_votes) >= args.min_raters else None,
            "others_consensus": majority(other_votes) if len(other_votes) >= args.min_raters else None,
        })
    con = pd.DataFrame(recs)
    con.to_csv(PROC / "gold_cohort_consensus.csv", index=False)

    def report(sub: pd.DataFrame, label: str) -> None:
        v = sub.dropna(subset=["consensus"])
        if not len(v):
            print(f"\n[{label}] no consensus rows yet")
            return
        ds_acc = (v["deepseek"] == v["consensus"]).mean()
        ru_acc = (v["rule"] == v["consensus"]).mean()
        ds_k = cohen_kappa(list(v["deepseek"]), list(v["consensus"]))
        ru_k = cohen_kappa(list(v["rule"]), list(v["consensus"]))
        print(f"\n[{label}]  n={len(v)}")
        print(f"  DeepSeek vs gold : acc={ds_acc:.3f}  kappa={pct(ds_k)}")
        print(f"  rule     vs gold : acc={ru_acc:.3f}  kappa={pct(ru_k)}")
        lo = sub.dropna(subset=["others_consensus"])
        if len(lo):
            print(f"  DeepSeek vs OTHERS (leave-one-out): acc={(lo['deepseek']==lo['others_consensus']).mean():.3f}  n={len(lo)}")

    for c in ["2020", "2024", "recent"]:
        report(con[con["cohort"] == c], c)
    report(con[con["cohort"].isin(["2020", "2024"])], "PRE-cutoff (2020+2024)")
    report(con, "OVERALL")

    # Fleiss across raters on fully-covered ids
    ids_all = [gid for gid in gold["gold_id"] if all(gid in raters[m] for m in raters)]
    rows = [[raters[m][gid] for m in raters] for gid in ids_all]
    print(f"\nFleiss kappa across {len(raters)} raters on {len(ids_all)} fully-covered ids: {pct(fleiss_kappa(rows))}")
    print("wrote:", PROC / "gold_cohort_consensus.csv")


if __name__ == "__main__":
    main()
