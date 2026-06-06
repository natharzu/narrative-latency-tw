"""Roundtrip-safe date/IO helpers for the Cofacts narrative-latency pipeline."""
import pandas as pd


def parse_dates_safe(series, utc=True):
    """Roundtrip-safe datetime parse.

    pandas 2.x strict ISO8601 fails silently on space-separated tz-aware
    strings like '2017-01-11 03:23:00+00:00'. format='mixed' handles them.
    Use this everywhere instead of raw pd.to_datetime.
    """
    return pd.to_datetime(series, format="mixed", utc=utc, errors="coerce")


def reconstruct_article_dates(df):
    """article_createdAt = reply_createdAt - latency_hours (exact arithmetic).

    Both source columns are reliably preserved through every CSV roundtrip.
    Use as fallback when article_createdAt is corrupted.
    """
    reply = parse_dates_safe(df["reply_createdAt"])
    return reply - pd.to_timedelta(df["latency_hours"], unit="h")


def cast_bool_columns(df, columns):
    """Restore object -> bool after CSV roundtrip."""
    for col in columns:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def assign_window(df):
    """Derive 'window' column from in_2020_win / in_2024_win flags."""
    df = df.copy()
    df["window"] = "off"
    df.loc[df["in_2020_win"], "window"] = "2020"
    df.loc[df["in_2024_win"], "window"] = "2024"
    return df
