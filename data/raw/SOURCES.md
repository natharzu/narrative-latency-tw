# Data Sources

All narratives in `iorg_narratives_scraped.csv` are from IORG (台灣資訊環境研究中心), published under **CC BY-SA 4.0**. See [iorg.tw/open](https://iorg.tw/open).

## Source breakdown

| Source URL | Cluster | Narratives | Time frame |
|---|---|---|---|
| https://iorg.tw/_en/r/b | ccp_information_manipulation | 17 (B.11–B.73) | 2018.9–2020.8 |
| https://iorg.tw/_en/da/11 | us_skepticism | 11 (DA11.1–DA11.11) | 2021.8 |
| https://iorg.tw/_en/da/12 | vaccine | 10 (DA12.1–DA12.10) | 2021.4–2021.8 |
| https://iorg.tw/_en/da/54 | pre_election | 7 (DA54.1–DA54.7) | 2021.4–2023.9 |

Retrieved 2026-05-13.

## Schema

| Column | Type | Notes |
|---|---|---|
| narrative_id | str | e.g. B.11, DA11.1 |
| case_id | str | e.g. B.1, DA.11 |
| case_name | str | human-readable case label |
| narrative_text | str | IORG-provided summary |
| topic_cluster | str | ccp_information_manipulation / us_skepticism / vaccine / pre_election |
| time_frame_start | str | YYYY.M (monthly) |
| time_frame_end | str | YYYY.M |
| stage_1_date | str | Stage 1 timestamp — TBD M4 |
| stage_2_date | str | Stage 2 timestamp — TBD M4 |
| stage_3_date | str | Stage 3 timestamp — TBD M4 |
| stage_4_date | str | Stage 4 timestamp — TBD M4 |
| election_window | str | e.g. 2020_general — TBD M4 |
| source_url | str | IORG page |
| retrieved_at | str | ISO-8601 datetime |

## Honest limitations

1. **Monthly time frames**, not stage-specific timestamps. The headline latency metric requires per-stage dates, deferred to M4.
2. **Stage 1–4 dates empty.** Per-stage timestamps are not in IORG's consolidated index. Manual transcription from case sub-pages required.
3. **N = 45, not 84.** Earlier planning assumed 84 narratives. IORG does not publish a consolidated CSV; 45 is what is publicly available across the B-series and 3 Dokidoki Alerts. More may be obtainable from alerts 1–10, 13–53 in a follow-up pass.
4. **Election window classification is manual.** Requires per-narrative review against Taiwan electoral calendars.
5. **Narrative text is normalized** to avoid CSV quoting conflicts. Original wording preserved on IORG source URLs.

## Secondary sources (not in CSV)

- **Doublethink Lab** — China Index + Taiwan POWER framework. Used for comparative context.
- **Taiwan Fact-Check Center** — Stage 4 evidence for several B-series narratives.
- **Cofacts** — Potential Stage 4 source in future iterations.

## Versioning

- v1 (2026-05-13): 17 B-series narratives.
- v2 (2026-05-13): +28 Dokidoki Alert narratives → 45 total.
