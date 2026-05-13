"""
Scrape IORG narrative data from the public B-series index + selected Dokidoki Alerts.

Output:
    data/raw/iorg_narratives_scraped.csv
    data/raw/iorg_scrape_log.txt

Source pages (all CC BY-SA 4.0):
    https://iorg.tw/_en/r/b           — B.1-B.7 case studies (consolidated tables)
    https://iorg.tw/_en/da/11         — US-skepticism narratives
    https://iorg.tw/_ua/da/12         — Vaccine narratives
    https://iorg.tw/_ua/da/54         — Pre-election narratives

LIMITATIONS:
    This scraper produces CASE-LEVEL narrative records (one row per narrative,
    with a single time-frame_start / time-frame_end inherited from the parent case).
    Per-stage timestamps (Stage 1 → Stage 4) are NOT in IORG's index tables —
    they live in narrative-specific case study sub-pages, which require either
    deeper scraping or manual augmentation. Treat this CSV as the SKELETON;
    fill stage_1_date … stage_4_date by hand or in a v2 scraper.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

SOURCES = [
    {
        "url": "https://iorg.tw/_en/r/b",
        "label": "B-series consolidated",
        "topic_cluster_default": "ccp_information_manipulation",
    },
    {
        "url": "https://iorg.tw/_en/da/11",
        "label": "US-skepticism (da/11)",
        "topic_cluster_default": "us_skepticism",
    },
    {
        "url": "https://iorg.tw/_ua/da/12",
        "label": "Vaccine narratives (da/12)",
        "topic_cluster_default": "vaccine",
    },
    {
        "url": "https://iorg.tw/_ua/da/54",
        "label": "Pre-election narratives (da/54)",
        "topic_cluster_default": "pre_election",
    },
]

OUT_DIR = Path("data/raw")
OUT_CSV = OUT_DIR / "iorg_narratives_scraped.csv"
OUT_LOG = OUT_DIR / "iorg_scrape_log.txt"

UA = (
    "narrative-latency-tw/0.1 (academic capstone; "
    "https://github.com/natharzu/narrative-latency-tw) "
    "polite-scrape"
)