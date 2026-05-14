"""Scrape IORG narratives v3 - no double underscores anywhere."""
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

SOURCES = [{"url": "https://iorg.tw/_en/r/b", "label": "B-series"}]
OUT_DIR = Path("data/raw")
OUT_CSV = OUT_DIR / "iorg_narratives_scraped.csv"
OUT_LOG = OUT_DIR / "iorg_scrape_log.txt"
UA = "narrative-latency-tw/0.2 (academic capstone)"
PAT = re.compile(r"^\*?\s*([A-Z]+\.\d{2,3})\s*=\s*(.+?)\s*$")