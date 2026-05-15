# Project Brief — Narrative Latency (Taiwan)

**One-line pitch.** Across Cofacts' Taiwan LINE rumor archive (≈100k messages, 2017–2026), measure how long suspicious messages stay live before the community fact-checks them — and which topic clusters get debunked fastest.

**Stakeholder.** Cofacts and IORG response teams.

**Headline (locked).** *"Between Taiwan's 2020 and 2024 presidential elections, Cofacts' community fact-check response slowed 10× — from a median of 6.7 hours in the 2020 window to 67.2 hours in the 2024 window."* (N = 5,825 vs 2,200; Mann–Whitney one-sided p < 10⁻²⁰⁰; survivorship-robust: 10.0× ratio holds when restricted to articles posted ≥6 months before snapshot.) **Baseline:** overall median 21.2h, N=68,533, 2016–2026.

---

## 1. Problem framing

In Taiwan's information environment, the *speed* at which a foreign-origin narrative crosses from upstream PRC channels into domestic platforms is as consequential as the narrative itself. Fast latency shortens the window for fact-checkers and platform trust & safety teams to intervene.

Most public reporting treats disinformation as a catalog of *artifacts* (claims, posts, accounts). This project shifts the unit of analysis from artifact to **temporal pattern** — a non-trivial reframing that earns Insight points without requiring novel data collection.

### Research questions

1. **Primary.** What is the distribution of article → first fact-check reply latency across Cofacts' Taiwan LINE rumor archive (Hugging Face snapshot 2026-05-10), and how does it vary by topic cluster as defined by IORG's narrative taxonomy?
2. **Secondary A.** Are rumors from election-period months (90 days around 2020 / 2024 national elections) debunked faster or slower than the non-election baseline?
3. **Secondary B.** How has community response latency evolved year-over-year (2018 → 2026), and what does the trend suggest about Cofacts' capacity?

---

## 2. Data plan

| Source | Role | Records | Access | License |
|---|---|---|---|---|
| **Cofacts** open dataset (LINE rumor archive) | Primary — timestamped article submissions + community fact-check replies | ~100k articles, ~50k replies (HF snapshot 2026-05-10) | [HF dataset card](https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw) · [opendata repo](https://github.com/cofacts/opendata) · [GraphQL API](https://api.cofacts.tw/graphql) | CC BY-SA 4.0 |
| **IORG** narrative dataset (2018–2023) | Topic-cluster taxonomy — 4 clusters used to tag Cofacts articles | 45 hand-curated narratives (17 B-series + 28 Dokidoki Alerts) | [iorg.tw/open](https://iorg.tw/open) · [SOURCES.md](data/raw/SOURCES.md) | CC BY-SA 4.0 |
| **Doublethink Lab** election analyses | Optional — election-window flags (2020, 2024) | Reports + raw indices | [China-Index-raw-data](https://github.com/doublethinklab/China-Index-raw-data) · [Artificial Multiverse report](https://medium.com/doublethinklab/artificial-multiverse-foreign-information-manipulation-and-interference-in-taiwans-2024-national-f3e22ac95fe7) | Open, attribution |

### Data schema (target)

    # data/processed/cofacts_latency.csv (one row per article-reply pair)
    articleId            TEXT   PK (Cofacts article ID)
    article_createdAt    TIME   ISO datetime — when the rumor was reported to Cofacts via LINE
    replyId              TEXT   first article_reply (chronological) for this article
    reply_createdAt      TIME   ISO datetime — when the community fact-check reply was authored
    reply_type           ENUM   RUMOR | NOT_RUMOR | OPINIONATED | NOT_ARTICLE
    articleType          ENUM   TEXT (filtered)
    latency_hours        FLOAT  (reply_createdAt - article_createdAt) in hours
    year                 INT    derived from article_createdAt
    topic_cluster        TEXT   IORG taxonomy, keyword-tagged at M4
    election_window      BOOL   within 90 days of 2020 or 2024 national election
    text_preview         TEXT   first 200 chars (PII already hashed by Cofacts)

### Known data risks

- Cofacts users are not a representative LINE sample — they are a self-selected subset who installed the fact-checking chatbot. The metric measures **Cofacts community response speed**, not Taiwan-wide rumor debunking speed.
- `article.createdAt` is when the rumor first reached Cofacts, not when it first appeared on LINE. True rumor age is a lower bound.
- IORG taxonomy is hand-curated and small (4 clusters from 45 narratives). Tagging Cofacts articles via keyword + URL matching is approximate; report sample-level error.
- Cofacts replies may contain factual error; the analysis treats any normal-status reply as a debunk event regardless of correctness.
- Explicit filter rule (pre-registered): drop article-reply pairs with negative latency or latency > 1 year; restrict to `articleType = TEXT`; report drop rate transparently.

---

## 3. Method

1. **Load** three Cofacts tables locally (`articles`, `replies`, `article_replies`) from HF snapshot 2026-05-10. Filter to `status = NORMAL` and `articleType = TEXT`.
2. **Join** each article to its *first* `article_reply` (chronologically), then to the `replies` row. Compute `latency_hours = reply_createdAt − article_createdAt`.
3. **Drop** rows with negative latency or latency > 1 year (data errors). Report drop rate (16.9% on the snapshot).
4. **Tag clusters** using IORG's 4-cluster taxonomy via keyword matching on `text_preview`. Collapse residual into "Other"; report the share that falls into Other (87.7% on the snapshot — finding in its own right).
5. **Describe distributions** overall and per cluster: median, IQR, P90, % under 24h, % under 1 week.
6. **Compare windows**: 2020 + 2024 election windows (±90 days) vs. non-election baseline; Mann–Whitney U, one-sided.
7. **Trend**: median latency by year (2016 → 2026) to track Cofacts community capacity over time.
8. **Sensitivity**: re-run 2020 vs 2024 comparison restricted to articles posted ≥6 months before the snapshot — confirm the ratio is robust to survivorship.

---

## 4. Deliverables

All artifacts live in the GitHub repo so the project reads end-to-end without external links.

- [x] `data/raw/` — 45 IORG narratives (cluster taxonomy reference)
- [x] `data/raw/SOURCES.md` — hyperlinks + retrieval notes for every source
- [x] `scripts/01_clean.py`, `scripts/02_latency.py`, `scripts/03_election_windows.py`
- [x] `data/processed/cofacts_latency.csv` (N=68,533) + `data/processed/cofacts_m4.csv`
- [x] `viz/cofacts_latency_distribution.png`, `viz/cofacts_latency_by_year.png`, `viz/election_window_comparison.png`
- [x] `notebooks/01_clean.ipynb`, `02_latency.ipynb`, `03_election_windows.ipynb` (jupytext-generated)
- [x] `report/slides.pdf` — 5-minute deck, 5 slides
- [x] `PROPOSAL.md` — this file
- [x] `README.md` — final version with locked headline + chart embeds

### 4a. Git hygiene

A commit lands at every milestone exit (≥6 commits expected). Each milestone closes with a descriptive commit message. No single-push finals — the repo must show iteration.

---

## 5. Milestones

**Final deadline: 2026-05-20.** Intermediate submission: 2026-05-13 23:59 CET — submitted on time.

| # | Milestone | Target | Exit criterion |
|---|---|---|---|
| M1 ✅ | Repo created, README committed | 2026-05-11 | `git clone` reproduces scaffolding |
| M2 ✅ | Intermediate submission to Brightspace | 2026-05-13 | Submitted 20:21 Moscow |
| M3 ✅ | Cofacts pipeline producing `cofacts_latency.csv` | 2026-05-14 | Real headline numbers (median 21.2h, N=68,533) |
| M4 ✅ | Election-window stats + cluster tagging | 2026-05-14 | 10× slowdown 2020→2024 locked; p < 10⁻²⁰⁰; survivorship-robust |
| M5 | Final visuals + slide deck | 2026-05-19 | 5-slide deck rehearsed under 5 min |
| M6 | Final submission | 2026-05-20 | Repo link + deck submitted; README final |

---

## 6. Rubric mapping

| Criterion | Weight | How this project earns it |
|---|---|---|
| Clarity | 30 | Single headline number (10× slowdown); one comparison histogram (2020 vs 2024); no map |
| Insight | 25 | Two findings in one dataset: within-cycle mobilization (election windows faster) AND across-cycle decay (2020 → 2024, 10× slower) — a temporal-pattern reframing of Taiwan's information-environment story |
| Actionability | 20 | Named stakeholder (Cofacts / IORG) with three specific moves: editor recruitment, AI-assisted triage for slowest cluster, public monthly latency dashboard |
| Data rigor | — | Pre-cleaned open sources; reproducible scripts; explicit drop-rule; survivorship sensitivity check |
| Story arc | — | Problem → one number → one chart → one recommendation, fits 5 minutes |

---

## 7. Slide outline (5 slides, 5 minutes)

1. **Hook.** Title + giant `10×` + stakeholder name.
2. **Why it matters.** Latency = response window; current public view is artifact-level, not temporal.
3. **Method.** Cofacts article + reply timeline + facts table.
4. **Finding.** Side-by-side histogram — 2020 window (N=5,825, median 6.7h) vs 2024 window (N=2,200, median 67.2h). Annotated with 10× ratio, p < 10⁻²⁰⁰, and survivorship-robust note. Inset shows election windows are faster than baseline overall.
5. **So what.** Three concrete moves for Cofacts / IORG: editor recruitment, AI-assisted triage, public latency dashboard.

**Audience-fit note.** The deck reads standalone (PDF only). Charts exported to `viz/` as PNG so non-coders in the showcase audience can inspect them without running Python.

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cofacts users not representative of all LINE users | High | Frame metric as "Cofacts community response speed"; document selection bias in §2 and on slide 1 caveat line |
| Cluster tagging on Cofacts noisy | Medium (realized) | 87.7% fell into "Other" — reported as a finding; IORG's 4 clusters are too narrow for Cofacts' content space |
| Survivorship inflating 2024 latency | Low (rejected) | Re-ran comparison restricted to articles ≥6 months pre-snapshot; ratio unchanged |
| Stakeholder relevance not obvious to graders | Low | Name Cofacts / IORG on slide 1; include three concrete actions on slide 5 |

---

## 9. Out of scope

- Causal claims about PRC origin. The analysis is descriptive and temporal only.
- Individual-level account or message analysis.
- Cross-language NLP on narrative content — topic clusters come from IORG's existing taxonomy via keyword matching.
- No Streamlit / Marimo app. Output is static (PDF slides + exported PNGs); interactivity adds no value for the response-team stakeholder.

---

## 10. Next actions (M5 → M6)

- [x] Intermediate submission to Brightspace — submitted 2026-05-13 20:21 Moscow
- [x] Cofacts pipeline producing `cofacts_latency.csv` (N=68,533)
- [x] M4 election-window stats + cluster tagging — 10× slowdown locked
- [ ] Generate `notebooks/01_clean.ipynb` + `02_latency.ipynb` + `03_m4.ipynb` via jupytext
- [ ] Final `README.md` with locked headline + chart embeds
- [ ] Build 5-slide deck → `report/slides.pdf` (Keynote → PDF)
- [ ] Self-check against rubric on 2026-05-18
- [ ] Final submission to Brightspace by 2026-05-20
