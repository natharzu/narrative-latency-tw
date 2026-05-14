# Sources

## Source 1 — Cofacts open dataset (primary timing data)

- **What:** Crowd-sourced fact-checking database of suspicious messages reported via LINE in Taiwan. Provides ISO-timestamped article submissions + community fact-check replies.
- **Where:** https://huggingface.co/datasets/Cofacts/line-msg-fact-check-tw
- **Snapshot used:** 2026-05-10
- **License:** CC BY-SA 4.0
- **Tables used:** `articles`, `replies`, `article_replies`
- **Required attribution (CC BY-SA 4.0):**
  > This data by Cofacts message reporting chatbot and crowd-sourced fact-checking community is licensed under CC BY-SA 4.0. To provide more info, please visit Cofacts LINE bot https://line.me/ti/p/@cofacts

---

## Source 2 — IORG narrative dataset (cluster taxonomy)

- **What:** Hand-curated catalogue of suspected Chinese information manipulation against Taiwan, 2018–2023. Used here purely as a **topic-cluster taxonomy** for tagging Cofacts articles.
- **Where:** https://iorg.tw/open · https://iorg.tw/_en/r/b
- **Retrieved:** 2026-05-13
- **License:** CC BY-SA 4.0
- **Records:** 45 narratives across 4 topic clusters:
  - **CCP information manipulation** (17 narratives, B-series 2018–2020): Kansai Airport, anti-Tsai, post-Brexit democracy, Chen Chu, post-election Taipei mayor, post-election DPP defeats, post-election Han Kuo-yu, etc.
  - **US-skepticism** (11 narratives, Dokidoki Alert DA/11, 2021.8): US troop withdrawal narratives, Afghanistan analogy, US unreliability framings.
  - **Vaccine** (10 narratives, Dokidoki Alert DA/12, 2021.4–2021.8): COVID vaccine procurement disputes, Taiwan-domestic vs. foreign vaccine narratives.
  - **Pre-election** (7 narratives, Dokidoki Alert DA/54, 2021.4–2023.9): 2024 election manipulation narratives, candidate-targeting framings.

### How IORG and Cofacts fit together

- **IORG (Source 2)** = topic-cluster taxonomy. Conceptual scaffold.
- **Cofacts (Source 1)** = timing data. Quantitative engine.
- **Linkage strategy (M4):** Tag a sample of ≥500 Cofacts articles to the 4 IORG clusters via keyword + reference-URL matching. Hand-validate a 50-article subsample to estimate tagging precision. Compute per-cluster latency.

---

## Source 3 (optional) — Doublethink Lab election analyses

- Used only for election-window flag definitions (2020, 2024 Taiwan national elections).
- https://github.com/doublethinklab/China-Index-raw-data
- "Artificial Multiverse" 2024 election FIMI report: https://medium.com/doublethinklab/artificial-multiverse-foreign-information-manipulation-and-interference-in-taiwans-2024-national-f3e22ac95fe7
- License: open, attribution.
