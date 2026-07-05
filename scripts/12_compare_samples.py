"""12_compare_samples.py — compare the two pilot cohorts.

Loads the election-aligned analysis sample and the fresh recent control sample
produced by scripts/11_sample500.py, and reports:
  * row counts and cohort x rule_topic composition (counts + within-cohort %);
  * article overlap between the two samples (expected 0 — disjoint windows);
  * training-cutoff status (min/max dates; share after the model cutoff);
  * rule-vs-LLM annotation agreement per cohort, IF an LLM label column exists
    (accuracy + Cohen's kappa); otherwise it says where that plugs in.

If scripts/13_annotate.py has been run, its labeled output
(pilot500_labeled_<mode>.csv, which carries the llm_topic column) is loaded in
preference to the raw sample, so the rule-vs-LLM section fills in automatically.

The logic: the recent sample is entirely POST the model cutoff, so the model
cannot have memorized it; the election sample is entirely PRE cutoff. If
rule-vs-LLM agreement is similar across the two, the annotation is robust to
contamination. If it is markedly higher on the (seen) election sample, that is
a memorization red flag — a sample-level complement to the item-level probes.

Reads:  data/processed/pilot500_labeled_elections.csv  (if present, else)
        data/processed/sample500_elections.csv
        data/processed/pilot500_labeled_recent.csv      (if present, else)
        data/processed/sample500_recent.csv
        (produce the samples first:
             uv run python scripts/11_sample500.py --mode elections
             uv run python scripts/11_sample500.py --mode recent
         then optionally annotate:
             uv run python scripts/13_annotate.py --mode elections
             uv run python scripts/13_annotate.py --mode recent)

Usage:
    uv run python scripts/12_compare_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import PROC, parse_dates_safe

SAMPLES = {
    "elections": PROC / "sample500_elections.csv",
    "recent": PROC / "sample500_recent.csv",
}
LABELED = {
    "elections": PROC / "pilot500_labeled_elections.csv",
    "recent": PROC / "pilot500_labeled_recent.csv",
}

# Training cutoff assumed by the §2c contamination probes. Adjust to match the
# annotating model's documented cutoff.
CUTOFF = pd.Timestamp("2024-07-01", tz="UTC")

# Candidate column names for a model-produced topic label (added in §2).
LLM_LABEL_CANDIDATES = [
    "llm_topic", "llm_label", "model_topic", "deepseek_topic", "gpt_topic",
]


def load(mode: str) -> pd.DataFrame:
    """Load a cohort, preferring the annotated file if scripts/13 has run."""
    labeled, sample = LABELED[mode], SAMPLES[mode]
    path = labeled if labeled.exists() else sample
    if not path.exists():
        raise SystemExit(
            f"Missing {sample}.\n"
            "Produce it first:\n"
            f"    uv run python scripts/11_sample500.py --mode {mode}"
        )
    df = pd.read_csv(path)
    df["article_createdAt"] = parse_dates_safe(df["article_createdAt"])
    return df


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two aligned categorical series (no sklearn dep)."""
    a = a.astype(str).reset_index(drop=True)
    b = b.astype(str).reset_index(drop=True)
    cats = sorted(set(a) | set(b))
    po = (a.values == b.values).mean()
    pe = sum(a.eq(c).mean() * b.eq(c).mean() for c in cats)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def find_llm_col(df: pd.DataFrame) -> str | None:
    for c in LLM_LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def main() -> None:
    ele = load("elections")
    rec = load("recent")
    both = pd.concat([ele, rec], ignore_index=True)

    print("=" * 60)
    print("ROW COUNTS")
    print("=" * 60)
    print(f"elections: {len(ele)}   recent: {len(rec)}   combined: {len(both)}")

    print()
    print("=" * 60)
    print("COHORT x rule_topic")
    print("=" * 60)
    ct = pd.crosstab(both["cohort"], both["rule_topic"])
    print(ct)
    print()
    print("within-cohort %:")
    print((ct.div(ct.sum(axis=1), axis=0) * 100).round(1))

    print()
    print("=" * 60)
    print("OVERLAP (shared text_preview across the two samples)")
    print("=" * 60)
    shared = set(ele["text_preview"]) & set(rec["text_preview"])
    print(f"shared articles: {len(shared)}  (expected 0 — windows are disjoint)")

    print()
    print("=" * 60)
    print(f"TRAINING-CUTOFF STATUS  (cutoff = {CUTOFF:%Y-%m-%d})")
    print("=" * 60)
    for name, df in [("elections", ele), ("recent", rec)]:
        d = df["article_createdAt"]
        post = (d > CUTOFF).mean() * 100
        print(f"{name:<10} {d.min().date()} .. {d.max().date()}   post-cutoff: {post:5.1f}%")

    print()
    print("=" * 60)
    print("RULE vs LLM AGREEMENT")
    print("=" * 60)
    llm_ele, llm_rec = find_llm_col(ele), find_llm_col(rec)
    if not llm_ele and not llm_rec:
        print(f"No LLM label column found yet (looked for {LLM_LABEL_CANDIDATES}).")
        print("Annotate first, then re-run:")
        print("    uv run python scripts/13_annotate.py --mode elections")
        print("    uv run python scripts/13_annotate.py --mode recent")
        print("  - accuracy and Cohen's kappa are reported per cohort;")
        print("  - similar kappa across cohorts => annotation robust to contamination;")
        print("  - kappa notably higher on 'elections' (pre-cutoff) => memorization flag.")
        return
    for name, df, col in [("elections", ele, llm_ele), ("recent", rec, llm_rec)]:
        if not col:
            print(f"{name:<10} no LLM column")
            continue
        d = df.dropna(subset=[col])
        acc = (d["rule_topic"].astype(str) == d[col].astype(str)).mean()
        k = cohen_kappa(d["rule_topic"], d[col])
        print(f"{name:<10} n={len(d):>4}  accuracy={acc:5.3f}  kappa={k:5.3f}")


if __name__ == "__main__":
    main()
