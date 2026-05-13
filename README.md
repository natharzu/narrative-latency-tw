# Narrative Latency
### Measuring how fast PRC-origin disinformation framings reach Taiwanese platforms

**Capstone — Practical Data Analysis**
Author: Natalia Harzu

---

## 1. Problem

In Taiwan's information environment, the *speed* at which a foreign-origin narrative crosses from upstream PRC channels into domestic platforms (LINE, PTT, Threads) is as consequential as the narrative itself. Fast latency shortens the window for fact-checkers, civic-tech responders, and platform trust & safety teams to intervene.

Most public reporting treats disinformation as a catalog of *artifacts* (claims, posts, accounts). This project shifts the unit of analysis to **temporal patterns** — how long each narrative takes to traverse a four-stage diffusion path, and which topics travel fastest.

## 2. Research question

> **For Taiwan-targeted narratives catalogued between 2021–2023, what is the distribution of latency between Stage 1 (PRC-origin appearance) and Stage 4 (domestic Taiwanese platform spread), and how does it vary by topic cluster?**

Secondary questions:
- Which topic clusters have the shortest median latency?
- Are election-period narratives faster than non-election baseline?
- Where in the Stage 1 → 4 path is the largest delay concentrated?

## 3. Data sources

| Source | Use | Access | License |
|---|---|---|---|
| **IORG** (Information Operations Research Group) — 84-narrative dataset, 2021–2023 | Stage-tagged narratives with timestamps and topic labels | [iorg.tw/open](https://iorg.tw/open) · [methodology](https://iorg.tw/open/rm) | CC BY-SA 4.0 |
| **Doublethink Lab** — election-period analyses (2020, 2024) | Cross-reference of narrative onset and election proximity | [China-Index-raw-data](https://github.com/doublethinklab/China-Index-raw-data) · [Artificial Multiverse report](https://medium.com/doublethinklab/artificial-multiverse-foreign-information-manipulation-and-interference-in-taiwans-2024-national-f3e22ac95fe7) | Open, attribution |
| **Cofacts** open fact-check database | Optional: domestic spread signal, message-level timestamps | [cofacts.tw](https://cofacts.tw) · [GraphQL API](https://api.cofacts.tw/) | CC BY-SA |

Per-file retrieval notes: [`data/raw/SOURCES.md`](./data/raw/SOURCES.md).

All sources are openly licensed and already cleaned at the narrative-record level, so iteration time goes toward analysis and story, not parsing.

## 4. Method

1. **Normalize** the IORG narrative records into a single table: `narrative_id`, `topic_cluster`, `stage_1_date … stage_4_date`, `election_window_flag`.
2. **Compute latencies**: pairwise day-deltas between stages; total Stage 1 → 4 latency.
3. **Cluster topics** using IORG's existing taxonomy; collapse rare categories.
4. **Describe distributions**: median, IQR, and tail behavior per cluster.
5. **Compare windows**: election vs. non-election periods (Mann–Whitney U).
6. **Visualize** the headline finding with one diverging bar chart (median latency per cluster) and one stage-by-stage waterfall (where the lag accumulates).

## 5. Headline (target form)

> *"For Taiwan-targeted narratives 2021–2023, the median time from PRC-origin appearance to domestic LINE/PTT/Threads spread is **X days**, but **Y% of election-period narratives** complete the path in under 48 hours."*

The single number on slide 1; the diverging bar on slide 2; the recommendation on slide 3.

## 6. Stakeholder

- **Primary:** Cofacts and IORG response teams — actionable signal for which topic clusters warrant pre-positioned counter-content.
- **Secondary:** Platform trust & safety leads at LINE / Meta; civic-tech orgs operating during Taiwanese election cycles.

## 7. Deliverables

- `data/` — raw and cleaned narrative tables (with provenance notes)
- `notebooks/01_clean.ipynb` — normalization and stage parsing
- `notebooks/02_latency.ipynb` — latency computation and statistics
- `notebooks/03_viz.ipynb` — final figures
- `report/` — 5-minute slide deck + speaker notes
- `README.md` — this file

## 8. Rubric mapping

| Criterion | Weight | How this project earns it |
|---|---|---|
| Clarity | 30 | Single headline number; one map-free diverging bar |
| Insight | 25 | Shifts unit-of-analysis from artifact to temporal pattern — non-obvious framing |
| Actionability | 20 | Named stakeholder (Cofacts/IORG) with a specific decision: where to pre-position responses |
| Data rigor | — | Pre-cleaned open sources; reproducible notebooks |
| Story arc | — | Problem → one number → one chart → one recommendation, fits 5 minutes |

## 9. Limitations

- IORG's stage tagging reflects observed appearance, not necessarily true origin — latency is a lower bound.
- 84 narratives is small; cluster-level estimates carry wide intervals. Reported with IQR, not point estimates alone.
- Cofacts and platform timestamps are observational; we do not infer causation between PRC-origin and domestic spread, only temporal sequence.

## 10. Reproducibility

```bash

git clone https://github.com/natharzu/narrative-latency-tw.git
cd narrative-latency-tw
pip install -r requirements.txt
jupyter notebook notebooks/01_clean.ipynb

```bash

