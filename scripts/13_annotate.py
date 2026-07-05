"""13_annotate.py — scaled DeepSeek annotation of a pilot cohort (§2).

Runs the frozen cross-validation annotation prompt (system_prompt.txt — the
canonical §1 instrument) over one of the two pilot samples produced by
scripts/11_sample500.py, at temperature=0 with JSON mode, in batches. Produces
the two-axis affect layer (valence + a composite intensity index) and an
`llm_topic` column, so the output drops straight into
scripts/12_compare_samples.py and the downstream figure / survival model.

Primary rater = DeepSeek (deepseek-chat / V3), the frozen config from the
cross-validation page §3. Set the API key in the environment first:

    export DEEPSEEK_API_KEY=sk-...

Composite intensity = mean(arousal, urgency, threat, anger), per the two-axis
affect decision (cross-validation page §11); valence is kept standalone.

Reads:  data/processed/sample500_<mode>.csv   (from scripts/11_sample500.py)
        system_prompt.txt                     (frozen canonical instrument)

Writes: data/processed/labels_deepseek_<mode>.json  (raw model output, unedited)
        data/processed/pilot500_labeled_<mode>.csv   (sample + parsed labels)

Usage:
    uv run python scripts/13_annotate.py --mode elections
    uv run python scripts/13_annotate.py --mode recent
    uv run python scripts/13_annotate.py --mode elections --limit 50   # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import ROOT, PROC, parse_dates_safe

SYS_PATH = ROOT / "system_prompt.txt"
IN = {
    "elections": PROC / "sample500_elections.csv",
    "recent": PROC / "sample500_recent.csv",
}
RAW_OUT = {
    "elections": PROC / "labels_deepseek_elections.json",
    "recent": PROC / "labels_deepseek_recent.json",
}
CSV_OUT = {
    "elections": PROC / "pilot500_labeled_elections.csv",
    "recent": PROC / "pilot500_labeled_recent.csv",
}

# Frozen DeepSeek config (cross-validation page §3). temperature=0 + JSON mode.
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
BATCH = 25
MAX_RETRIES = 3

TOPICS = {"scam", "political", "health", "other"}
AFFECT = ["valence", "arousal", "urgency", "threat", "anger"]
INTENSITY_PARTS = ["arousal", "urgency", "threat", "anger"]


def get_client():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as e:
        raise SystemExit(
            "The 'openai' package is required.\n"
            "    uv add openai   (or: uv sync --extra llm)"
        ) from e
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit(
            "Set your DeepSeek key first:\n"
            "    export DEEPSEEK_API_KEY=sk-..."
        )
    return OpenAI(base_url=BASE_URL, api_key=key)


def annotate_batch(client, sys_prompt: str, batch: pd.DataFrame) -> list[dict]:
    payload = [{"id": int(r.id), "text": str(r.text_preview)} for r in batch.itertuples()]
    user = (
        "Annotate every message below. Return ONLY a JSON array with one "
        "object per input id, in the same order, each following the schema.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
    )
    obj = json.loads(resp.choices[0].message.content)
    if isinstance(obj, list):
        return obj
    for key in ("results", "data", "annotations", "items"):
        if isinstance(obj.get(key), list):
            return obj[key]
    return [obj]  # single-object fallback


def parse_labels(raw: list[dict]) -> pd.DataFrame:
    lab = pd.DataFrame(raw)
    for c in AFFECT:
        if c in lab.columns:
            lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["intensity"] = lab[INTENSITY_PARTS].mean(axis=1)
    lab = lab.rename(columns={"topic": "llm_topic"})
    return lab


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek annotation of a pilot cohort (§2).")
    ap.add_argument("--mode", choices=["elections", "recent"], default="elections")
    ap.add_argument("--limit", type=int, default=0, help="annotate only the first N rows (smoke test)")
    args = ap.parse_args()

    in_path = IN[args.mode]
    if not in_path.exists():
        raise SystemExit(
            f"Missing {in_path}.\n"
            f"Produce it first:\n    uv run python scripts/11_sample500.py --mode {args.mode}"
        )
    if not SYS_PATH.exists():
        raise SystemExit(
            f"Missing {SYS_PATH}.\n"
            "This is the frozen canonical annotation prompt (cross-validation page §1)."
        )

    sys_prompt = SYS_PATH.read_text(encoding="utf-8")
    rows = pd.read_csv(in_path)
    if "id" not in rows.columns:
        rows = rows.rename(columns={rows.columns[0]: "id"})
    if args.limit:
        rows = rows.iloc[: args.limit].copy()

    client = get_client()

    raw: list[dict] = []
    for i in range(0, len(rows), BATCH):
        batch = rows.iloc[i : i + BATCH]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw += annotate_batch(client, sys_prompt, batch)
                break
            except Exception as e:  # noqa: BLE001 — log + retry any API/parse error
                print(f"  retry batch@{i} attempt {attempt}: {e}")
                if attempt == MAX_RETRIES:
                    raise SystemExit(f"batch@{i} failed after {MAX_RETRIES} attempts")
                time.sleep(2 * attempt)
        print(f"  annotated {min(i + BATCH, len(rows))}/{len(rows)}")

    RAW_OUT[args.mode].write_text(
        json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    lab = parse_labels(raw)
    in_vocab = lab["llm_topic"].isin(TOPICS).mean() if "llm_topic" in lab.columns else 0.0
    merged = rows.merge(lab, on="id", how="left", suffixes=("", "_llm"))
    if "article_createdAt" in merged.columns:
        merged["article_createdAt"] = parse_dates_safe(merged["article_createdAt"])
    merged.to_csv(CSV_OUT[args.mode], index=False, encoding="utf-8")

    print()
    print(f"mode: {args.mode}")
    print(f"annotated: {len(raw)}/{len(rows)}")
    print(f"in-vocab topic rate: {in_vocab:.3f}")
    if "llm_topic" in lab.columns:
        bad = sorted(set(lab.loc[~lab["llm_topic"].isin(TOPICS), "llm_topic"].dropna()))
        if bad:
            print(f"  off-vocab topics: {bad}")
    q = merged.dropna(subset=["llm_topic"]) if "llm_topic" in merged.columns else merged.iloc[:0]
    if len(q):
        acc = (q["rule_topic"].astype(str) == q["llm_topic"].astype(str)).mean()
        print(f"rule vs llm topic agreement: {acc:.3f}  (n={len(q)})")
    print("wrote:", RAW_OUT[args.mode])
    print("wrote:", CSV_OUT[args.mode])


if __name__ == "__main__":
    main()
