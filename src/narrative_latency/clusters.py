"""IORG-derived keyword taxonomy for tagging Cofacts articles by topic cluster.

Keyword lists are preserved exactly from scripts/03_election_windows.py.
"""

CLUSTERS = {
    "vaccine": ["疫苗", "AZ", "BNT", "莫德納", "高端", "BioNTech",
                "vaccine", "vaccination", "Pfizer", "Moderna"],
    "us_skepticism": ["美國", "美军", "美軍", "拜登", "Biden", "Trump", "川普",
                      "阿富汗", "Afghanistan", "美中"],
    "pre_election": ["選舉", "选举", "总统", "總統", "候选", "候選", "投票",
                     "election", "Lai", "賴清德", "蕭美琴", "侯友宜", "柯文哲"],
    "ccp_info_manipulation": ["中共", "共产党", "共產黨", "解放军", "解放軍",
                              "习近平", "習近平", "Xi Jinping", "PLA", "一国两制",
                              "一國兩制"],
}


def tag(text):
    """Return the first matching cluster name for ``text``, else 'Other'."""
    if not isinstance(text, str):
        return "Other"
    for cluster, kws in CLUSTERS.items():
        if any(kw in text for kw in kws):
            return cluster
    return "Other"
