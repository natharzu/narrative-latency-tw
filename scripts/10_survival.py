"""
Phase 5 - Survival analysis of Cofacts article->reply latency.

WHY: scripts/02-03 report medians over REPLIED articles only. Articles never
answered by the snapshot are dropped, which biases the picture (and the bias is
worse in 2024, where more articles go unanswered). Survival analysis keeps them
as right-censored observations.

Rebuilds a censored survival frame from the RAW Cofacts zips (like 01_clean.py),
then fits Kaplan-Meier + Cox PH for TWO event definitions:
    * any         -> first reply of any type
    * substantive -> first RUMOR / NOT_RUMOR (fact-check verdict) reply

Inputs : data/raw/cofacts/{articles,replies,article_replies}.csv.zip
Outputs: data/processed/cofacts_survival.csv
         data/processed/survival_km_medians.csv
         data/processed/survival_cox_hr_<defn>.csv
         data/processed/survival_ph_test_<defn>.csv
         viz/survival_km_2020_vs_2024_<defn>.png
         viz/survival_km_window_vs_baseline_<defn>.png

Run:
    uv run --extra survival python scripts/10_survival.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import median_survival_times

from narrative_latency import (
    RAW, PROC, VIZ, E2020, E2024, SNAPSHOT, in_window, tag, parse_dates_safe,
)
from narrative_latency.survival import build_survival_frame, SUBSTANTIVE_TYPES

COFACTS = RAW / "cofacts"
PROC.mkdir(parents=True, exist_ok=True)
VIZ.mkdir(exist_ok=True)

DEFNS = [
    ("any", "duration_any_h", "event_any"),
    ("substantive", "duration_subst_h", "event_subst"),
]


def load_raw():
    print("Loading raw Cofacts zips...")
    articles = pd.read_csv(COFACTS / "articles.csv.zip")
    replies = pd.read_csv(COFACTS / "replies.csv.zip")
    article_replies = pd.read_csv(COFACTS / "article_replies.csv.zip")

    articles = articles[articles["status"] == "NORMAL"].copy()
    article_replies = article_replies[article_replies["status"] == "NORMAL"].copy()

    articles["createdAt"] = parse_dates_safe(articles["createdAt"])
    article_replies["createdAt"] = parse_dates_safe(article_replies["createdAt"])
    replies["createdAt"] = parse_dates_safe(replies["createdAt"])

    articles = articles[articles["articleType"] == "TEXT"].copy()
    articles = articles.rename(
        columns={"id": "articleId", "createdAt": "article_createdAt"}
    )
    articles = articles[["articleId", "article_createdAt", "text"]]

    article_replies = article_replies.rename(columns={"createdAt": "ar_createdAt"})
    article_replies = article_replies[
        ["articleId", "replyId", "replyType", "ar_createdAt"]
    ]

    replies = replies.rename(
        columns={"id": "replyId", "createdAt": "reply_createdAt"}
    )
    replies = replies[["replyId", "reply_createdAt"]]
    return articles, article_replies, replies


def add_strata(df):
    df = df.copy()
    df["year"] = df["article_createdAt"].dt.year
    df["in_2020_win"] = in_window(df["article_createdAt"], E2020)
    df["in_2024_win"] = in_window(df["article_createdAt"], E2024)
    df["election_window"] = df["in_2020_win"] | df["in_2024_win"]
    df["topic_cluster"] = df["text"].apply(tag)
    return df


def km_median(kmf):
    med = kmf.median_survival_time_
    ci = median_survival_times(kmf.confidence_interval_)
    return med, ci.iloc[0, 0], ci.iloc[0, 1]


def run_km(df, dur, evt, label):
    rows = []
    kmf = KaplanMeierFitter()
    kmf.fit(df[dur], df[evt], label="overall")
    med, lo, hi = km_median(kmf)
    rows.append({"defn": label, "stratum": "overall", "n": len(df),
                 "events": int(df[evt].sum()),
                 "median_h": med, "ci_low_h": lo, "ci_high_h": hi})

    # 2020 window vs 2024 window
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, mask, color in [("2020 window", df["in_2020_win"], "#3b82f6"),
                              ("2024 window", df["in_2024_win"], "#ef4444")]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        k = KaplanMeierFitter().fit(sub[dur], sub[evt], label=name)
        k.plot_survival_function(ax=ax, color=color, ci_show=True)
        med, lo, hi = km_median(k)
        rows.append({"defn": label, "stratum": name, "n": len(sub),
                     "events": int(sub[evt].sum()),
                     "median_h": med, "ci_low_h": lo, "ci_high_h": hi})
    ax.set_title(f"KM survival ({label}): 2020 vs 2024 election windows")
    ax.set_xlabel("Hours since article reported")
    ax.set_ylabel("P(still unanswered)")
    fig.tight_layout()
    out = VIZ / f"survival_km_2020_vs_2024_{label}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")

    # election window vs baseline (whole timeline)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, mask, color in [("election window", df["election_window"], "#10b981"),
                              ("baseline", ~df["election_window"], "#9ca3af")]:
        sub = df[mask]
        k = KaplanMeierFitter().fit(sub[dur], sub[evt], label=name)
        k.plot_survival_function(ax=ax, color=color, ci_show=True)
        med, lo, hi = km_median(k)
        rows.append({"defn": label, "stratum": name, "n": len(sub),
                     "events": int(sub[evt].sum()),
                     "median_h": med, "ci_low_h": lo, "ci_high_h": hi})
    ax.set_title(f"KM survival ({label}): election window vs baseline")
    ax.set_xlabel("Hours since article reported")
    ax.set_ylabel("P(still unanswered)")
    fig.tight_layout()
    out = VIZ / f"survival_km_window_vs_baseline_{label}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")

    # logrank: 2020 vs 2024 windows
    win_df = df[df["election_window"]].copy()
    win_df["which"] = win_df["in_2024_win"].map({True: "2024", False: "2020"})
    if win_df["which"].nunique() == 2:
        lr = multivariate_logrank_test(win_df[dur], win_df["which"], win_df[evt])
        print(f"[{label}] logrank 2020 vs 2024 window: p={lr.p_value:.3e}")
    return rows


def run_cox(df, dur, evt, label):
    model = df[[dur, evt]].copy()
    model["year_c"] = df["year"] - df["year"].mean()
    model["election_window"] = df["election_window"].astype(int)
    if "log_requests" in df.columns:        
        model["log_requests"] = df["log_requests"].values

    dummies = pd.get_dummies(df["topic_cluster"], prefix="topic").astype(float)
    if "topic_Other" in dummies.columns:
        dummies = dummies.drop(columns=["topic_Other"])  # Other = baseline
    model = pd.concat([model, dummies], axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(model, duration_col=dur, event_col=evt)
    print(f"\n===== Cox PH ({label}) =====")
    cph.print_summary()

    out = PROC / f"survival_cox_hr_{label}.csv"
    cph.summary.to_csv(out)
    print(f"Saved {out}")

    try:
        ph = proportional_hazard_test(cph, model, time_transform="rank")
        ph_out = PROC / f"survival_ph_test_{label}.csv"
        ph.summary.to_csv(ph_out)
        print(f"Saved {ph_out} (Schoenfeld PH test)")
    except Exception as e:  # noqa: BLE001
        print(f"[{label}] PH test skipped: {e}")
    return cph

def attach_request_volume(df):
    """Merge reply-request volume (from 17_popularity.py) as a Cox covariate.

    Endogeneity note: request_count accrues only until an article's FIRST reply,
    so it is partly an OUTCOME of fast replies. Read the HR as descriptive; for a
    cleaner causal estimate, swap in an early-window request count (requests in
    the first N hours) built in 17_popularity.py.
    """
    pop_path = PROC / "cofacts_popularity.csv"
    if not pop_path.exists():
        print(f"skip request volume: {pop_path.name} not found "
              "(run scripts/17_popularity.py first)")
        return df
    pop = pd.read_csv(pop_path)[["articleId", "request_count"]]
    df = df.merge(pop, on="articleId", how="left")
    df["request_count"] = df["request_count"].fillna(1).astype(int)
    df["log_requests"] = np.log1p(df["request_count"])
    print(f"attached request volume: median={df['request_count'].median():.0f}, "
          f"max={df['request_count'].max():,}")
    return df

def attach_affect(df):
    """Merge LLM affect scores (valence + composite intensity) as Cox covariates.

    Affect is pre-specified as TWO axes (see the Notion cross-validation page):
        * valence   — direction, kept as-is (-2..+2)
        * intensity — composite = mean(arousal, urgency, threat, anger)

    Reads data/processed/cofacts_affect.csv (articleId + 'valence' and either an
    'intensity' column or the four raw scales). Skips cleanly until the corpus is
    annotated, exactly like attach_request_volume.

    Caveats (Notion 'Smoke tests' + 'Gold-cohort' pages): LLM affect is an
    automatic proxy; arousal vs urgency point opposite ways; affect compression
    (central-tendency bias) attenuates the coefficient; and low-affect URL-only
    rumors are auto-matched fastest, so guard the duplicate/reply-reuse confound.
    """
    aff_path = PROC / "cofacts_affect.csv"
    if not aff_path.exists():
        print(f"skip affect: {aff_path.name} not found "
              "(annotate valence/intensity for the corpus first)")
        return df
    aff = pd.read_csv(aff_path)
    if "intensity" not in aff.columns:
        scales = ["arousal", "urgency", "threat", "anger"]
        have = [c for c in scales if c in aff.columns]
        if not have:
            print(f"skip affect: no 'intensity' and none of {scales} "
                  f"in {aff_path.name}")
            return df
        aff["intensity"] = aff[have].mean(axis=1)
    keep = [c for c in ["articleId", "valence", "intensity"] if c in aff.columns]
    if "valence" not in keep:
        print(f"skip affect: no 'valence' column in {aff_path.name}")
        return df
    df = df.merge(aff[keep], on="articleId", how="left")
    n = df["valence"].notna().sum()
    print(f"attached affect: {n:,}/{len(df):,} rows scored "
          f"(valence median={df['valence'].median():.2f}, "
          f"intensity median={df['intensity'].median():.2f})")
    return df

def main():
    articles, article_replies, replies = load_raw()
    print(f"  NORMAL TEXT articles: {len(articles):,}")

    df = build_survival_frame(
        articles, article_replies, replies,
        snapshot=SNAPSHOT, substantive_types=SUBSTANTIVE_TYPES,
    )
    df = add_strata(df)
    df = attach_request_volume(df) 
    print(f"Survival frame: {len(df):,} articles")
    print(f"  any-reply events:         {df['event_any'].sum():,} "
          f"({df['event_any'].mean():.1%}); censored "
          f"{(1 - df['event_any'].mean()):.1%}")
    print(f"  substantive-reply events: {df['event_subst'].sum():,} "
          f"({df['event_subst'].mean():.1%})")

    keep = ["articleId", "article_createdAt", "year", "in_2020_win",
            "in_2024_win", "election_window", "topic_cluster",
            "duration_any_h", "event_any", "duration_subst_h", "event_subst"]
    df[keep].to_csv(PROC / "cofacts_survival.csv", index=False)
    print(f"Wrote {PROC / 'cofacts_survival.csv'}")

    med_rows = []
    for label, dur, evt in DEFNS:
        med_rows += run_km(df, dur, evt, label)
        run_cox(df, dur, evt, label)

    med = pd.DataFrame(med_rows)
    med.to_csv(PROC / "survival_km_medians.csv", index=False)
    print(f"\nWrote {PROC / 'survival_km_medians.csv'}")
    print(med.to_string(index=False))


if __name__ == "__main__":
    main()
