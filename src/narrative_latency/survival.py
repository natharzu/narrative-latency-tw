"""Survival-analysis data construction for Cofacts reply latency.

scripts/01_clean.py builds cofacts_latency.csv with an INNER join (article ->
its first reply), so every row already has a reply. That dataset cannot express
the central survival quantity: articles never answered by the snapshot, which
are exactly the right-censored observations. This module rebuilds a survival
frame from the raw Cofacts tables so those articles are retained as censored.

Functions here are pure (no IO) and import no heavy deps, so they run in CI and
are unit-tested with synthetic data. scripts/10_survival.py does the file IO
and the lifelines model fits.

Two event definitions (a robustness pair):
  * "any"          -> first reply of any reply_type.
  * "substantive"  -> first reply whose verdict is in SUBSTANTIVE_TYPES
                      (RUMOR / NOT_RUMOR). Opinionated / not-article replies do
                      not count; an article with only those is censored under
                      this definition.
"""
from __future__ import annotations

import pandas as pd

# Cofacts reply verdicts that count as a substantive fact-check.
SUBSTANTIVE_TYPES = ("RUMOR", "NOT_RUMOR")

# Mirrors scripts/01_clean.py: event latencies that are negative or exceed one
# year are treated as data errors and the row is censored instead. Censoring
# windows themselves are never capped.
MAX_LATENCY_HOURS = 24 * 365


def _hours(delta):
    return delta.dt.total_seconds() / 3600.0


def first_reply_times(article_replies, replies, *, substantive_types=SUBSTANTIVE_TYPES):
    """First reply time per article, for both event definitions.

    Ordering follows scripts/01_clean.py: article_replies sorted by their own
    ``ar_createdAt`` (when the reply was attached). The event *time* uses the
    linked reply's ``reply_createdAt`` so durations reconcile with the locked
    ``latency_hours`` metric.

    article_replies : ['articleId', 'replyId', 'replyType', 'ar_createdAt']
    replies         : ['replyId', 'reply_createdAt']

    Returns one row per article that has >=1 linked reply:
        ['articleId', 'any_reply_at', 'subst_reply_at']
    where subst_reply_at is NaT when the article has no substantive reply.
    """
    ar = article_replies.merge(replies, on="replyId", how="inner")
    ar = ar.sort_values("ar_createdAt")

    any_first = (
        ar.groupby("articleId", as_index=False)
        .first()[["articleId", "reply_createdAt"]]
        .rename(columns={"reply_createdAt": "any_reply_at"})
    )

    subst = ar[ar["replyType"].isin(list(substantive_types))]
    subst_first = (
        subst.groupby("articleId", as_index=False)
        .first()[["articleId", "reply_createdAt"]]
        .rename(columns={"reply_createdAt": "subst_reply_at"})
    )

    return any_first.merge(subst_first, on="articleId", how="left")


def build_survival_frame(
    articles,
    article_replies,
    replies,
    *,
    snapshot,
    substantive_types=SUBSTANTIVE_TYPES,
    max_latency_hours=MAX_LATENCY_HOURS,
):
    """One-row-per-article survival frame with right-censoring.

    articles : ['articleId', 'article_createdAt', ...] already filtered to the
        population of interest (e.g. NORMAL + TEXT). Extra columns (e.g. 'text')
        are carried through untouched.
    snapshot : censoring time (tz-aware pd.Timestamp).

    Adds, per article:
        duration_any_h,   event_any   (1 = answered, 0 = censored at snapshot)
        duration_subst_h, event_subst (1 = substantive reply, 0 = censored)
    Rows whose event latency is negative or > max_latency_hours are censored
    (data errors). Articles created after the snapshot are dropped.
    """
    df = articles.copy()
    df = df[df["article_createdAt"].notna()]
    df = df[df["article_createdAt"] <= snapshot].reset_index(drop=True)

    times = first_reply_times(
        article_replies, replies, substantive_types=substantive_types
    )
    df = df.merge(times, on="articleId", how="left")

    censor_h = _hours(snapshot - df["article_createdAt"])

    any_lat = _hours(df["any_reply_at"] - df["article_createdAt"])
    event_any = df["any_reply_at"].notna() & any_lat.between(0, max_latency_hours)
    duration_any = any_lat.where(event_any, censor_h)

    subst_lat = _hours(df["subst_reply_at"] - df["article_createdAt"])
    event_subst = df["subst_reply_at"].notna() & subst_lat.between(0, max_latency_hours)
    duration_subst = subst_lat.where(event_subst, censor_h)

    df = df.assign(
        duration_any_h=duration_any,
        event_any=event_any.astype(int),
        duration_subst_h=duration_subst,
        event_subst=event_subst.astype(int),
    )

    df = df[(df["duration_any_h"] >= 0) & (df["duration_subst_h"] >= 0)]
    return df.reset_index(drop=True)
