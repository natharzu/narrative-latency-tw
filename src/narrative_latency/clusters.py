"""IORG-derived keyword taxonomy for tagging Cofacts articles by topic cluster.

The first four clusters (vaccine, us_skepticism, pre_election,
ccp_info_manipulation) are preserved exactly from scripts/03_election_windows.py
(pre_election additionally gains 公投/罷免/大選).

Option 4 (2026-06) expands the taxonomy with nine topic categories discovered by
decomposing the 'Other' bucket via HDBSCAN (see scripts/08). These topics were
stable across both preview- and full-text tagging runs. Keyword lists use
deliberately specific substrings to limit false positives. Tagging is
single-label, first-match in dict order, so 'scam' is evaluated before the topic
categories (a profit/job scam should be tagged by mechanism, not subject), while
the four original IORG clusters stay near the top so their counts barely move.

Degenerate chain-message clusters (repeated-character spam e.g. 吃吃吃,
sticker/download ads with ↓↓↓, phone/underscore noise, and 真的假的 'is this
real?' forwards) are intentionally left in 'Other' -- they are artifacts, not
topics, and keywording them would only add noise.
"""

CLUSTERS = {
    # --- scam / fraud --------------------------------------------------------
    # The slowest tail of 'Other' (investment, job, and part-time scams) and the
    # policy headline. Evaluated FIRST; keywords are specific scam markers so we
    # do not poach legitimate finance/jobs articles (hence no bare 工作/交易/投資).
    "scam": ["詐騙", "詐欺", "飆股", "主力", "帶單", "加賴", "穩賺", "包賺",
             "日領", "兼職", "工作內容", "投資群組"],

    # --- four original IORG clusters (keyword lists unchanged) ---------------
    "vaccine": ["疫苗", "AZ", "BNT", "莫德納", "高端", "BioNTech",
                "vaccine", "vaccination", "Pfizer", "Moderna"],
    "us_skepticism": ["美國", "美军", "美軍", "拜登", "Biden", "Trump", "川普",
                      "阿富汗", "Afghanistan", "美中"],
    "pre_election": ["選舉", "选举", "总统", "總統", "候选", "候選", "投票",
                     "election", "Lai", "賴清德", "蕭美琴", "侯友宜", "柯文哲",
                     "公投", "罷免", "大選"],
    "ccp_info_manipulation": ["中共", "共产党", "共產黨", "解放军", "解放軍",
                              "习近平", "習近平", "Xi Jinping", "PLA", "一国两制",
                              "一國兩制"],

    # --- Option 4 topic categories (discovered from the 'Other' decomposition)
    "health": ["癌", "腫瘤", "口罩", "養生", "偏方", "保健", "中醫", "草藥"],
    "traffic": ["紅燈", "機車", "違規", "罰單", "酒駕", "駕照", "超速"],
    "energy": ["台電", "發電", "電廠", "停電", "核電", "核能", "缺電", "電價",
               "核四"],
    "pension": ["退休", "年金", "公教", "勞保", "勞退"],
    "food_safety": ["萊豬", "瘦肉精", "雞蛋", "進口蛋", "農藥", "食安"],
    "lgbtq": ["同性", "同志", "性戀", "愛滋", "同婚", "婚姻平權"],
    "disaster": ["颱風", "地震", "豪雨", "淹水", "土石流", "海嘯"],
    "international": ["烏克蘭", "俄羅斯", "普丁", "澤倫斯基", "核廢水", "福島",
                     "核食"],
}


def tag(text):
    """Return the first matching cluster name for ``text``, else 'Other'."""
    if not isinstance(text, str):
        return "Other"
    for cluster, kws in CLUSTERS.items():
        if any(kw in text for kw in kws):
            return cluster
    return "Other"
