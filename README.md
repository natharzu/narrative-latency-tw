# Narrative Latency

### How fast does Taiwan's community fact-check its own LINE rumors — and how has that speed changed across election cycles?

**Capstone — Practical Data Analysis**
Author: Natalia Harzu

---

## Headline

> **Between Taiwan's 2020 and 2024 presidential elections, Cofacts' community fact-check response slowed 10× — from a median of 6.7 hours in the 2020 window to 67.2 hours in the 2024 window.**

N = 5,825 (2020 window) vs N = 2,200 (2024 window) · Mann–Whitney U, one-sided p < 10⁻²⁰⁰ · Survivorship-robust: the 10.0× ratio holds when restricted to articles posted ≥6 months before the snapshot.

Baseline: overall median 21.2 h across N = 68,533 article-reply pairs, 2016–2026. Snapshot date: 2026-05-10.

---

## 1. Problem framing

In Taiwan's information environment, the *speed* at which a rumor moves through domestic platforms is as consequential as the rumor itself. The window in which a fact-check can intervene closes fast.

Most public reporting treats disinformation as a catalog of *artifacts* (claims, posts, accounts). This project shifts the unit of analysis from artifact to **temporal pattern** — using Cofacts, Taiwan's community-driven LINE rumor-archive, to measure how fast the community itself responds and how that response speed has changed.

The four-stage upstream pipeline (PRC state media → content farms → Want-Want-aligned KOLs → anonymous LINE forwards) is documented qualitatively by IORG and Doublethink Lab. This project does **not** attempt to measure Stage 1 → Stage 4 latency — Cofacts data only contains Stage 4 (domestic LINE) artifacts. We measure response latency *within* that domestic stage, treating the IORG pipeline as the threat context for which our metric is the response signal.

## 2. Research questions

1. **Primary.** What is the distribution of *article → first community fact-check reply* latency across Cofacts' Taiwan LINE rumor archive (HF snapshot 2026-05-10)?
2. **Secondary A.** Are rumors from election-period months (±90 days around 2020 / 2024 national elections) debunked faster or slower than the non-election baseline?
3. **Secondary B.** How has community response latency evolved year-over-year (2018 → 2026), and what does the trend suggest about Cofacts' capacity?

## 3. Data sources

| Source | Role in this project | Records | Access | License |
|---|---|---|---|---|
| **Cofacts** open dataset (LINE rumor archive) | **Primary** — timestamped article submissions + community fact-check replies | ~100k articles, ~50k replies (HF snapshot 2026-05-10) | [HF dataset](https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw) · [opendata repo](https://github.com/cofacts/opendata) · [GraphQL API](https://api.cofacts.tw/graphql) | CC BY-SA 4.0 |
| **IORG** narrative dataset (2018–2023) | Topic-cluster taxonomy reference for tagging Cofacts articles | 45 hand-curated narratives (17 B-series + 28 Dokidoki Alerts) | [iorg.tw/open](https://iorg.tw/open) | CC BY-SA 4.0 |
| **Doublethink Lab** election analyses | Election-window cross-reference (2020, 2024) | Reports + raw indices | [China-Index-raw-data](https://github.com/doublethinklab/China-Index-raw-data) · [Artificial Multiverse report](https://medium.com/doublethinklab/artificial-multiverse-foreign-information-manipulation-and-interference-in-taiwans-2024-national-f3e22ac95fe7) | Open, attribution |

Per-file retrieval notes and provenance: [`data/raw/SOURCES.md`](./data/raw/SOURCES.md).

## 4. Method

1. **Load** three Cofacts tables (`articles`, `replies`, `article_replies`) from the HF snapshot. Filter to `status = NORMAL` and `articleType = TEXT`.
2. **Join** each article to its chronologically first `article_reply`, then to the `replies` row. Compute `latency_hours = reply_createdAt − article_createdAt`.
3. **Drop** pairs with negative latency or latency > 1 year (data errors). Drop rate: 16.9 % on the snapshot.
4. **Tag clusters** via keyword matching using IORG's 4-cluster taxonomy. Collapse residual into "Other"; the "Other" share (87.7 %) is reported as a finding in its own right.
5. **Describe distributions** overall and per cluster: median, IQR, P90, % under 24 h, % under 1 week.
6. **Compare windows.** 2020 + 2024 election windows (±90 days) vs. non-election baseline; Mann–Whitney U, one-sided.
7. **Trend.** Median latency by year, 2018 → 2026.
8. **Sensitivity.** Re-run 2020 vs 2024 comparison restricted to articles posted ≥6 months before the snapshot. The 10× ratio is robust to survivorship.

Pipeline scripts (run in order): [`scripts/01_clean.py`](./scripts/01_clean.py) → [`scripts/02_latency.py`](./scripts/02_latency.py) → [`scripts/03_election_windows.py`](./scripts/03_election_windows.py) → [`scripts/04_final_charts.py`](./scripts/04_final_charts.py). Notebooks are generated via [jupytext](https://jupytext.readthedocs.io/) from the scripts 1–3 above.

## 5. Stakeholder

- **Primary:** Cofacts and IORG response teams — actionable signal for which topic clusters and time windows warrant additional editor capacity and pre-positioned counter-content.
- **Secondary:** Platform trust & safety leads at LINE / Meta; civic-tech organisations operating around Taiwanese election cycles.

Three concrete moves recommended on slide 5 of the deck:
1. **Editor recruitment** focused on the 2024-era throughput gap.
2. **AI-assisted triage** for the slowest topic cluster.
3. **Public monthly latency dashboard** so Cofacts can manage to the metric.

## 6. Deliverables

- `data/raw/` — IORG narrative reference (cluster taxonomy)
- `data/raw/SOURCES.md` — retrieval notes for every source
- `data/processed/cofacts_latency.csv` (N = 68,533) and `cofacts_election_windows.csv`
- `scripts/01_clean.py`, `scripts/02_latency.py`, `scripts/03_election_windows.py`, `scripts/04_final_charts.py`
- `notebooks/01_clean.ipynb`, `02_latency.ipynb`, `03_election_windows.ipynb` (jupytext-generated)
- `viz/cofacts_latency_distribution.png`, `viz/cofacts_latency_by_year.png`, `viz/election_window_comparison.png`
- `report/slides.pdf` — 5-minute deck, 5 slides
- `PROPOSAL.md` — full project brief with risks, milestones, and rubric mapping
- `README.md` — this file

## 7. Rubric mapping

| Criterion | Weight | How this project earns it |
|---|---|---|
| Clarity | 30 | Single headline number (10× slowdown); one side-by-side histogram (2020 vs 2024); no map |
| Insight | 25 | Two findings in one dataset — within-cycle mobilization (election windows faster) **and** across-cycle decay (2020 → 2024, 10× slower). Reframes Taiwan's information-environment story as a temporal-pattern problem rather than an artifact-counting problem. |
| Actionability | 20 | Named stakeholder (Cofacts / IORG) with three specific moves: editor recruitment, AI-assisted triage, public latency dashboard |
| Data rigor | — | Pre-cleaned open sources; reproducible scripts; explicit drop rule; survivorship sensitivity check |
| Story arc | — | Problem → one number → one chart → one recommendation, fits 5 minutes |

## 8. Limitations

- **Selection bias.** Cofacts users are a self-selected subset of LINE users who installed the fact-checking chatbot. The metric measures *Cofacts community response speed*, not Taiwan-wide rumor debunking speed.
- **Lower bound on rumor age.** `article.createdAt` is when the rumor first reached Cofacts, not when it first appeared on LINE.
- **Cluster tagging is approximate.** 87.7 % of articles fall into "Other" under IORG's 4-cluster taxonomy applied via keyword matching. The named-cluster comparisons (pre-election, US-skepticism) carry the bulk of the topic signal.
- **No causal claim about PRC origin.** The analysis is descriptive and temporal. We do not infer that Cofacts-submitted rumors are PRC-originated — only that response speed has changed across election cycles in measurable, statistically significant ways.
- **Reply correctness not adjudicated.** Any normal-status reply is treated as a debunk event regardless of whether the reply itself is factually correct.

## 9. Reproducibility

    git clone https://github.com/natharzu/narrative-latency-tw.git
    cd narrative-latency-tw
    python3 -m pip install --user -r requirements.txt

    # Run the full pipeline
    python3 scripts/01_clean.py
    python3 scripts/02_latency.py
    python3 scripts/03_election_windows.py
    python3 scripts/04_final_charts.py

    # Or open the jupytext-generated notebooks
    jupyter notebook notebooks/02_latency.ipynb
    
    
Snapshot: Hugging Face dataset `Cofacts/line-msg-fact-check-tw` as of 2026-05-10.

## 10. Companion project

[**think-thrice-stickerpack**](https://github.com/natharzu/think-thrice-stickerpack) — a digital-hygiene LINE sticker pack (*Don't Let Them Write Your Ending*) countering the PRC's 疑美論 ("US Abandonment") narrative. Submitted for the Politics of Truth course (Spring 2026) by Group 5. The 10× slowdown documented in this repo is part of the motivation for that pack.

## License

Code and analysis: MIT. Cofacts-derived data: CC BY-SA 4.0 (per upstream license). IORG taxonomy references: CC BY-SA 4.0.


## Updated mechanism (post-submission analysis)

The submitted headline (**10× slower 2024 vs 2020**) is confirmed. Hypothesis testing of four alternative drivers sharpened the mechanism:

| Hypothesis | Test | Result |
|---|---|---|
| Volume surge in 2024 | Article counts per election window | **Rejected** — 2024 had 62% *fewer* submissions (5,798 → 2,191) |
| Right-censoring artifact | Buffer between window end and data max | **Rejected** — 757-day buffer |
| Article complexity shift | text_preview length | **Rejected** — medians 96 vs 82 chars |
| Article type mix shift | articleType / replyType distributions | **Modest** — RUMOR share 52% → 69%, insufficient |

**Volume went down, not up.** The 2024 election cycle saw a documented surge in Taiwan-targeted misinformation activity, yet Cofacts received less than half the rumor submissions of 2020.

**Implication: bilateral platform decline, not queue overload.**
- *Submitters left*: 62% fewer people forwarded rumors to Cofacts during a higher-stakes election cycle.
- *Repliers slowed or left*: the smaller inflow takes 10× longer to verify.

The intervention framing shifts from "more volunteers" to "why did the submission base leave between 2020 and 2024?"
