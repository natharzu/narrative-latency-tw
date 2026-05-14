# Narrative Latency — Taiwan

How long do suspicious LINE messages stay live in Taiwan before Cofacts' community fact-checks them? And has that response speed changed between election cycles?

---

## Headline

> **Between Taiwan's 2020 and 2024 presidential elections, Cofacts' community fact-check response slowed 10× — from a median of 6.7 hours in the 2020 election window to 67.2 hours in the 2024 window.**

- **N** = 5,798 (2020 window) vs 2,191 (2024 window) article-reply pairs
- **Statistical test** Mann–Whitney U (one-sided, 2024 > 2020): **p < 10⁻²⁰⁰**
- **Survivorship-robust** the 10.0× ratio holds when restricted to articles posted ≥6 months before the dataset snapshot (2026-05-10)
- **Caveat** measures *Cofacts community response speed*, not Taiwan-wide rumor-debunking speed (Cofacts users are a self-selected subset of LINE)

![Cofacts latency: 2020 vs 2024 Taiwan presidential elections](viz/election_window_comparison.png)

---

## Why it matters

Most public reporting on disinformation counts *artifacts* (posts, accounts, narratives). This project shifts the unit of analysis from artifact to **temporal pattern** — how long a rumor stays unchallenged is the actual intervention window for fact-checkers and platform trust & safety teams.

Two findings emerge from the same dataset:

1. **Within an election cycle, the Cofacts community mobilizes.** Election-window rumors (90 days around the 2020 or 2024 presidential elections) get fact-checked in a median of **10.7 hours** vs **24.0 hours** for non-election baseline (p ≈ 10⁻¹¹⁷).
2. **Across election cycles, the same community has slowed dramatically.** The 2024 election window is 10× slower than 2020 — so the 2024 slowdown is *despite* election mobilization, not because of its absence.

**Implication for Cofacts and IORG response teams:** civic mobilization alone isn't compensating for structural decline in the editor community. Editor recruitment + AI-assisted triage on the slowest topic cluster (US-skepticism, median 34.3h) are concrete levers.

---

## Method

1. **Source.** Cofacts open dataset (Hugging Face snapshot 2026-05-10, CC BY-SA 4.0) — 283,153 articles + 144,255 replies + 159,055 article-reply links from Taiwan's LINE rumor-reporting bot.
2. **Join.** Each article → its first (chronological) `article_reply` → the matching `replies` row.
3. **Filter.** `status = NORMAL`, `articleType = TEXT`, drop pairs with negative latency or latency > 1 year (13,992 rows dropped, 16.9% drop rate).
4. **Compute.** `latency_hours = reply_createdAt − article_createdAt`. Final dataset: **68,533 article-reply pairs spanning 2016-12 → 2026-05**.
5. **Compare windows.** 2020 election window = 2019-10-13 to 2020-04-10; 2024 election window = 2023-10-15 to 2024-04-12 (each is ±90 days around the election date). Mann–Whitney U test, one-sided.
6. **Cluster tag.** Keyword matching on `text_preview` against IORG's 4-cluster taxonomy (vaccine, US-skepticism, pre-election, CCP information manipulation). 87.7% of articles fell outside the simple keyword taxonomy → reported as "Other" and discussed as a finding in its own right.
7. **Sensitivity.** Re-ran the 2020 vs 2024 comparison restricted to articles posted ≥6 months before the snapshot date — ratio unchanged.

Full pipeline:

    scripts/clean.py        → data/processed/cofacts_latency.csv  (68,533 rows)
    scripts/latency.py      → headline stats + viz/cofacts_latency_distribution.png
                                                + viz/cofacts_latency_by_year.png
    scripts/m4_analysis.py  → election-window stats + cluster tags
                              → data/processed/cofacts_m4.csv
                              → viz/election_window_comparison.png

---

## Data sources

| Source | Role | Records | License |
|---|---|---|---|
| [Cofacts — line-msg-fact-check-tw (HF)](https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw) | Primary timing data (article submissions + community replies) | ~100k articles, ~50k replies, 2016–2026 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Cofacts open data repo](https://github.com/cofacts/opendata) | Alternative access path (CSV releases) | Same as HF mirror | CC BY-SA 4.0 |
| [Cofacts GraphQL API](https://api.cofacts.tw/graphql) | Live query path (for refresh) | Real-time | CC BY-SA 4.0 |
| [IORG — Information Operations Research Group](https://iorg.tw/open) | 4-cluster narrative taxonomy (used for keyword tagging) | 45 hand-curated narratives, 2018–2023 | CC BY-SA 4.0 |
| [Doublethink Lab — Artificial Multiverse report](https://medium.com/doublethinklab/artificial-multiverse-foreign-information-manipulation-and-interference-in-taiwans-2024-national-f3e22ac95fe7) | 2024 election context | Narrative analysis | Open, attribution |

See [`data/raw/SOURCES.md`](data/raw/SOURCES.md) for full retrieval notes and citations.

---

## Charts

- **Election windows compared** — [`viz/election_window_comparison.png`](viz/election_window_comparison.png)
- **Overall latency distribution** — [`viz/cofacts_latency_distribution.png`](viz/cofacts_latency_distribution.png)
- **Median latency by year (2016 → 2026)** — [`viz/cofacts_latency_by_year.png`](viz/cofacts_latency_by_year.png)

![Median latency by year](viz/cofacts_latency_by_year.png)

---

## Limitations

- **Selection bias.** Cofacts users are people who installed a fact-checking chatbot — a self-selected, civically-engaged subset of LINE. This measures *Cofacts community response speed*, not Taiwan-wide rumor debunking.
- **Lower-bound timestamps.** `article.createdAt` is when the rumor reached Cofacts via LINE bot, not when it first appeared anywhere. Real rumor age is at least this large.
- **Cluster tagging is approximate.** 87.7% of articles didn't match any of IORG's 4-cluster keyword set, suggesting IORG's curated taxonomy is too narrow for Cofacts' content space. Reported as "Other" rather than forced into the four buckets.
- **Reply correctness not assessed.** Any normal-status reply counts as a "debunk event" regardless of its factual accuracy.
- **Filter rule pre-registered.** Drop negative-latency and >1-year-latency pairs; restrict to `articleType = TEXT`. 16.9% of joined pairs dropped, reported transparently.

---

## Reproducibility

    # Clone
    git clone https://github.com/natharzu/narrative-latency-tw.git
    cd narrative-latency-tw

    # Install dependencies
    python3 -m pip install --user -r requirements.txt

    # Get the Cofacts snapshot
    # Option A: HuggingFace web UI → download articles.csv.zip, replies.csv.zip,
    #           article_replies.csv.zip from
    #           https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw
    # Option B: huggingface-cli download Cofacts/line-msg-fact-check-tw \
    #           --repo-type=dataset --local-dir=data/raw/cofacts
    mkdir -p data/raw/cofacts
    # place the three .csv.zip files in data/raw/cofacts/

    # Run the pipeline
    python3 scripts/clean.py         # produces data/processed/cofacts_latency.csv
    python3 scripts/latency.py       # headline stats + 2 charts
    python3 scripts/m4_analysis.py   # election-window + cluster stats + 1 chart

**Environment.** Python 3.13, pandas 2.2+, matplotlib 3.10+, scipy 1.11+. See [`requirements.txt`](requirements.txt).

**Snapshot pinning.** All numbers in this README come from Cofacts HF snapshot dated 2026-05-10. Re-running on a newer snapshot will produce different numbers; the 10× ratio is the headline claim for this specific snapshot.

---

## Repository layout

    data/raw/                   IORG narratives + SOURCES.md (Cofacts snapshot is gitignored)
    data/processed/             cofacts_latency.csv, cofacts_m4.csv
    scripts/                    clean.py, latency.py, m4_analysis.py, README.md
    notebooks/                  jupytext-generated mirrors of scripts/
    viz/                        PNG charts (committed)
    report/slides.pdf           5-slide presentation
    PROPOSAL.md                 project brief (mirrors Notion brief)
    requirements.txt

---

## Attribution

Cofacts data:

> This data by Cofacts message reporting chatbot and crowd-sourced fact-checking community is licensed under CC BY-SA 4.0. To provide more info, please visit Cofacts LINE bot https://line.me/ti/p/@cofacts

IORG and Doublethink Lab data used per their respective open-data terms; see [`data/raw/SOURCES.md`](data/raw/SOURCES.md).

---

## Course context

Capstone project — data storytelling, May 2026. Slides: [`report/slides.pdf`](report/slides.pdf). Project brief: [`PROPOSAL.md`](PROPOSAL.md).
