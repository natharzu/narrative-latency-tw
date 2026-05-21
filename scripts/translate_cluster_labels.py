"""One-shot: add English category labels to cluster_profiles.csv and cluster_profiles_2024.csv.

Labels are curated (not literal translation) so they read as topic categories
in English-language slides and the Streamlit dashboard.
"""
import pandas as pd

# Global clusters (cluster_id → English category)
LABELS_GLOBAL = {
    -2: "URL-only (no text)",
    5:  "Starbucks & coffee",
    7:  "LGBT & marriage equality",
    10: "Earthquakes",
    13: "Traffic & road safety",
    17: "Taiwan power & blackouts",
    19: "Egg imports (Brazil)",
    20: "Ukraine–Russia war",
    22: "Face masks",
    34: "Eyewear & vision",
    35: "Pension reform",
    43: "Weather & typhoons",
    49: "iPhone / Apple",
    53: "Pork & ractopamine",
    54: "Vaccine misinformation",
    60: "Cancer claims",
    63: "Gargling 'cures'",
    64: "Salmon & sashimi",
    67: "Food & drink myths",
    70: "LINE phishing scams",
    72: "Phone scams",
    73: "Phone & charging tips",
    78: "Free sticker scams",
    81: "Fruit health claims",
    82: "Food & nutrition",
    86: "Family & parenting",
    88: "Junk (food meme spam)",
    89: "COVID-19 containment",
    90: "COVID-19 virus",
    93: "Loan scams",
    95: "Travel subsidy scams",
    101: "Job & salary scams",
    103: "Job platform scams",
    105: "Investment & trading scams",
    107: "Government budget",
    108: "US–China–Taiwan",
    109: "Taiwan–Japan relations",
    110: "Aviation & military jets",
    114: "Elections & referendums",
    115: "Tsai Ing-wen / DPP",
}

# 2024-window clusters (c24 → English category)
LABELS_2024 = {
    0:  "Electricity bills",
    1:  "Traffic enforcement cameras",
    2:  "Senior-citizen scam",
    3:  "Facebook account scam",
    4:  "Pension (military/civil)",
    5:  "Junk (icon/url fragments)",
    6:  "Weight loss & wellness",
    7:  "Gift voucher scam",
    8:  "Mycoplasma outbreak",
    9:  "Sudan-red food contamination",
    10: "Hsiao Bi-khim nationality",
    11: "Lai Ching-te (family)",
    12: "Taiwan–China identity",
    13: "Defense ministry & satellites",
    14: "Lai illegal construction",
    15: "Click-farm scam",
    16: "Junk (slide spam)",
    17: "iPASS transit card",
    18: "Election & voting",
    19: "Lai Ching-te (campaign)",
    20: "DPP / Ko Wen-je",
    21: "Real-name registration scam",
    22: "Loan scam (banks)",
    23: "LINE sticker scam",
    24: "Junk (eating spam)",
    25: "Job & salary scam",
    26: "Stock-market scam",
    27: "Account registration scam",
    28: "Secretary job ads",
    29: "Dash-formatted scam ads",
}

# Patch cluster_profiles.csv
cp = pd.read_csv("data/processed/cluster_profiles.csv")
mapped = cp["cluster_id"].map(LABELS_GLOBAL)
cp["label_en"] = mapped.fillna("(other)")
cp.to_csv("data/processed/cluster_profiles.csv", index=False)
print(f"✓ cluster_profiles.csv: {mapped.notna().sum()}/{len(cp)} clusters mapped (rest = '(other)')")

# Patch cluster_profiles_2024.csv
cp24 = pd.read_csv("data/processed/cluster_profiles_2024.csv")
mapped24 = cp24["c24"].map(LABELS_2024)
cp24["label_en"] = mapped24.fillna("(other)")
cp24.to_csv("data/processed/cluster_profiles_2024.csv", index=False)
print(f"✓ cluster_profiles_2024.csv: {mapped24.notna().sum()}/{len(cp24)} clusters mapped")
