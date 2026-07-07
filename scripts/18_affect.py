#!/usr/bin/env python3
"""
scripts/18_affect.py — LLM affect annotation (valence + intensity) layer.

Produces data/processed/cofacts_affect.csv, the last input the survival model
needs for RQ2 (does emotional framing change fact-check latency?). For each
article in the analysis frame it asks an LLM for two axes:
    * valence   — integer -2..+2 (direction; negative/alarming -> positive/reassuring)
    * intensity — composite = mean(arousal, urgency, threat, anger), each 0..4
and writes articleId, valence, the four scales, and the derived intensity.

The transforms (prompt build, JSON parsing, clamping, batching, checkpointing)
are pure and unit-tested in tests/test_affect.py; only default_llm() touches the
network. Runs are RESUMABLE: results append per batch and already-annotated ids
are skipped, so a crash at batch 2,000 keeps the first ~50k rows.

Usage:
    uv run python scripts/18_affect.py                 # annotate the whole frame
    uv run python scripts/18_affect.py --limit 50      # smoke test (first 50)
    uv run python scripts/18_affect.py --batch-size 25 --model gpt-4o-mini

Output feeds 10_survival.py::attach_affect automatically — no further wiring.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import pandas as pd

# Shared repo constants via the scripts/utils.py shim, matching 17_popularity.py.
from utils import ROOT, RAW, PROC

RAW_DEFAULT = RAW / "cofacts"
LATENCY_TOPIC = PROC / "cofacts_latency_topic.csv"
AFFECT_OUT = PROC / "cofacts_affect.csv"

VALENCE_RANGE = (-2, 2)
SCALE_RANGE = (0, 4)
SCALES = ["arousal", "urgency", "threat", "anger"]
COLUMNS = ["articleId", "valence"] + SCALES + ["intensity"]


# --------------------------------------------------------------------------- #
# Pure transforms (unit-tested in tests/test_affect.py)
# --------------------------------------------------------------------------- #
def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def build_prompt(batch):
    """batch: list of {'articleId','text'} -> the user message string."""
    lines = []
    for i, art in enumerate(batch, 1):
        text = (art.get("text") or "").replace("\n", " ").strip()
        if len(text) > 1200:
            text = text[:1200] + " …[truncated]"
        lines.append(f'{i}. id={art["articleId"]} :: {text}')
    joined = "\n".join(lines)
    return (
        "You are an expert annotator of Traditional Chinese LINE rumor "
        "messages. Rate the affect conveyed BY THE MESSAGE TEXT (not its truth):\n"
        "- valence: integer -2..2 (-2 very negative/alarming, 0 neutral, +2 reassuring)\n"
        "- arousal: integer 0..4 (0 calm, 4 highly activating)\n"
        "- urgency: integer 0..4 (0 none, 4 extreme 'act now' pressure)\n"
        "- threat: integer 0..4 (0 none, 4 severe danger/harm framing)\n"
        "- anger: integer 0..4 (0 none, 4 strong outrage/blame)\n"
        "Return ONLY a JSON array, one object per message, no prose, no code fence:\n"
        '{"articleId": "<id>", "valence": int, "arousal": int, '
        '"urgency": int, "threat": int, "anger": int}\n\nMESSAGES:\n' + joined
    )


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def coerce_record(rec):
    """Validate + clamp one raw dict; None if unusable."""
    if not isinstance(rec, dict) or "articleId" not in rec:
        return None
    try:
        out = {"articleId": str(rec["articleId"])}
        out["valence"] = _clamp(int(round(float(rec["valence"]))), *VALENCE_RANGE)
        for s in SCALES:
            out[s] = _clamp(int(round(float(rec[s]))), *SCALE_RANGE)
    except (KeyError, TypeError, ValueError):
        return None
    out["intensity"] = sum(out[s] for s in SCALES) / len(SCALES)
    return out


def extract_json_array(text):
    """Pull a JSON array from an LLM reply, tolerating code fences / prose."""
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def parse_response(text, expected_ids):
    """{articleId: clean_record} restricted to ids we asked about."""
    want = set(expected_ids)
    out = {}
    for rec in extract_json_array(text):
        clean = coerce_record(rec)
        if clean and clean["articleId"] in want:
            out[clean["articleId"]] = clean
    return out


def remaining_ids(all_ids, done_ids):
    """Ids still needing annotation, de-duplicated, original order preserved."""
    done, seen, out = set(done_ids), set(), []
    for a in all_ids:
        if a not in done and a not in seen:
            seen.add(a)
            out.append(a)
    return out


# --------------------------------------------------------------------------- #
# IO wrappers
# --------------------------------------------------------------------------- #
def _raw_path(raw, name):
    for fn in (f"{name}.csv.zip", f"{name}.csv"):
        p = raw / fn
        if p.exists():
            return p
    raise FileNotFoundError(f"neither {name}.csv.zip nor {name}.csv in {raw}")


def load_articles_to_annotate(raw, latency_path=LATENCY_TOPIC):
    """articleId+text for the analysis frame (annotate ~68k, not all 274k)."""
    arts = pd.read_csv(_raw_path(raw, "articles"),
                       usecols=lambda c: c in ("id", "articleId", "text"))
    idcol = "id" if "id" in arts.columns else "articleId"
    arts = arts.rename(columns={idcol: "articleId"})
    arts["articleId"] = arts["articleId"].astype(str)
    arts = arts.dropna(subset=["text"])
    if latency_path.exists():
        d = pd.read_csv(latency_path)
        key = "articleId" if "articleId" in d.columns else (
            "id" if "id" in d.columns else None)
        if key:
            keep = set(d[key].astype(str))
            arts = arts[arts["articleId"].isin(keep)]
            print(f"scoped to {len(keep):,} analysis-frame ids", flush=True)
    else:
        print(f"note: {latency_path.name} not found -> annotating ALL articles",
              flush=True)
    return arts[["articleId", "text"]].to_dict("records")


def load_done_ids(path):
    if not path.exists():
        return set()
    try:
        return set(pd.read_csv(path, usecols=["articleId"])["articleId"].astype(str))
    except Exception:
        return set()


def append_records(path, records):
    if not records:
        return
    df = pd.DataFrame(records, columns=COLUMNS)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def default_llm(model="gpt-4o-mini"):
    """The ONLY network-touching piece. Swap to match the pilot's provider;
    keep the (prompt: str) -> str contract. Reads OPENAI_API_KEY from env."""
    from openai import OpenAI
    client = OpenAI()

    def _call(prompt):
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    return _call


def annotate_batch(llm_call, batch, retries=3, min_frac=0.8):
    """Annotate one batch; retry on error/short parse. Returns [] if it never
    reaches min_frac coverage, so those ids stay queued for a later run."""
    ids = [a["articleId"] for a in batch]
    prompt = build_prompt(batch)
    last = {}
    for attempt in range(1, retries + 1):
        try:
            text = llm_call(prompt)
        except Exception as e:  # noqa: BLE001 - want to retry any provider error
            print(f"  call error {attempt}/{retries}: {e}", flush=True)
            time.sleep(min(2 * attempt, 10))
            continue
        last = parse_response(text, ids)
        if len(last) >= max(1, int(min_frac * len(ids))):
            break
        print(f"  short parse {len(last)}/{len(ids)} attempt {attempt}", flush=True)
        time.sleep(min(1.5 * attempt, 8))
    return [last[i] for i in ids if i in last]


def run(articles, out_path, llm_call, batch_size=25, limit=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path)
    all_ids = [a["articleId"] for a in articles]
    todo_ids = remaining_ids(all_ids, done)
    if limit:
        todo_ids = todo_ids[:limit]
    by_id = {a["articleId"]: a for a in articles}
    todo = [by_id[i] for i in todo_ids]
    n_batches = math.ceil(len(todo) / batch_size) if todo else 0
    print(f"{len(done):,} already annotated; {len(todo):,} to go in "
          f"{n_batches} batches of {batch_size}", flush=True)
    n_ok = 0
    for bi, batch in enumerate(batched(todo, batch_size), 1):
        recs = annotate_batch(llm_call, batch)
        append_records(out_path, recs)
        n_ok += len(recs)
        if bi % 20 == 0 or bi == n_batches:
            print(f"  batch {bi}/{n_batches}  +{len(recs)}  "
                  f"(session total {n_ok:,})", flush=True)
    print(f"done. wrote {n_ok:,} new rows to {out_path}", flush=True)
    return n_ok


def main():
    ap = argparse.ArgumentParser(description="LLM affect annotation layer")
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--out", type=Path, default=AFFECT_OUT)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None,
                    help="annotate only the first N remaining (smoke test)")
    args = ap.parse_args()

    articles = load_articles_to_annotate(args.raw)
    print(f"{len(articles):,} articles in scope", flush=True)
    llm = default_llm(args.model)
    run(articles, args.out, llm, batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
