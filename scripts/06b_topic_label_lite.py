"""06b_topic_label_lite.py — attach scam/political/health/other topic labels to
`cofacts_latency.csv` WITHOUT the embedding/clustering stack (scripts 05 + 06).

Why this exists
---------------
The `topic` column produced in ``06_2024_focus.py`` is generated purely by
keyword matching on ``text_preview``; only the *clustering* half of that script
(UMAP + HDBSCAN) needs the sentence embeddings. The full per-article file that
carries the label — ``cofacts_topic_classified.csv`` — is git-ignored as a large
regenerable intermediate, so it is not present in a fresh clone.

This script isolates just the keyword classifier so that a combined
text + latency + topic table can be rebuilt from committed data alone, with no
ML extras installed (``uv sync`` is enough; ``--extra ml`` is not required).

Reads:  data/processed/cofacts_latency.csv        (committed; N ≈ 68.5k)
Writes: data/processed/cofacts_latency_topic.csv  (adds topic + *_hits columns)

Usage:
    uv run python scripts/06b_topic_label_lite.py

Note: the keyword lists below are kept verbatim in sync with
``scripts/06_2024_focus.py``. If you edit one, edit the other (or factor them
into ``narrative_latency``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from narrative_latency import PROC

IN = PROC / "cofacts_latency.csv"
OUT = PROC / "cofacts_latency_topic.csv"

# --- Keyword lists (verbatim from scripts/06_2024_focus.py) ------------------
SCAM = [
    "詐騙", "加賴", "加LINE", "代操", "保證獲利", "翻倍", "穩赚", "內線", "主力", "獲利",
    "投資", "交易", "帳戶", "註冊", "帳號", "密碼", "驗證", "個資", "身分證",
    "兼職", "薪資", "薪水", "0元", "免費", "限時", "中獎", "領取", "點擊", "下單",
    "貸款", "借款", "銀行", "房貸", "信用卡", "卡片", "匯款", "訂單", "實名", "款卡",
    "兼差", "【工作】", "正職", "在家工作", "日領", "月領", "高薪",
]
POLITICAL = [
    "選舉", "投票", "候選", "民進黨", "國民黨", "民眾黨", "時代力量",
    "蔡英文", "賴清德", "韓國瑜", "侯友宜", "柯文哲", "馬英九", "蔣萬安", "郭台銘", "朱立倫",
    "中國", "中共", "共產黨", "國防", "兩岸", "統一", "台獨", "罷免", "公投", "立委", "總統", "縣長", "市長",
    "親中", "抗中", "國安",
]
HEALTH = [
    "疫苗", "接種", "病毒", "肺炎", "新冠", "COVID", "Covid", "covid", "癌症", "致癌",
    "健康", "治療", "醫師", "醫院", "藥物", "副作用", "感染", "食品", "營養", "保健",
]


def hits(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in the text (substring match)."""
    return sum(1 for k in keywords if k in text)


def classify(row) -> str:
    """Assign the topic with the most keyword hits; ties break scam>political>health.

    Falls back to "other" when no keyword matches. Identical logic to
    ``classify()`` in scripts/06_2024_focus.py.
    """
    counts = {
        "scam": row["scam_hits"],
        "political": row["pol_hits"],
        "health": row["health_hits"],
    }
    return "other" if max(counts.values()) == 0 else max(counts, key=counts.get)


def main() -> None:
    print(f"→ Loading {IN.name} …")
    df = pd.read_csv(IN)
    if "text_preview" not in df.columns:
        raise SystemExit(
            "Expected a 'text_preview' column in cofacts_latency.csv; "
            f"found: {list(df.columns)}"
        )
    print(f"  Rows: {len(df):,}")

    text = df["text_preview"].fillna("").astype(str)
    df["scam_hits"] = text.apply(lambda t: hits(t, SCAM))
    df["pol_hits"] = text.apply(lambda t: hits(t, POLITICAL))
    df["health_hits"] = text.apply(lambda t: hits(t, HEALTH))
    df["topic"] = df.apply(classify, axis=1)

    df.to_csv(OUT, index=False)
    print(f"→ Wrote {OUT}  (N={len(df):,})")
    print("\nTopic mix (%):")
    print((df["topic"].value_counts(normalize=True) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
