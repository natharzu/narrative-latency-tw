import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import mannwhitneyu

st.set_page_config(page_title="Narrative Latency — Taiwan", layout="wide")

@st.cache_data
def load():
    df = pd.read_csv("data/processed/cofacts_latency.csv")
    df["article_createdAt"] = pd.to_datetime(
        df["article_createdAt"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["article_createdAt"])
    df["year"] = df["article_createdAt"].dt.year
    return df

df = load()

st.title("Cofacts community response latency, 2016–2026")
st.caption("Cofacts users are a self-selected subset. Metric = community response speed, not Taiwan-wide debunking speed.")

baseline = df["latency_hours"].median()
e2020 = df[df["year"] == 2020]["latency_hours"]
e2024 = df[df["year"] == 2024]["latency_hours"]

c1, c2, c3 = st.columns(3)
c1.metric("Baseline median (all years)", f"{baseline:.1f} h", f"N={len(df):,}")
c2.metric("2020 median", f"{e2020.median():.1f} h", f"N={len(e2020):,}")
c3.metric("2024 median", f"{e2024.median():.1f} h", f"{e2024.median()/e2020.median():.1f}x slower", delta_color="inverse")

if len(e2020) and len(e2024):
    u, p = mannwhitneyu(e2024, e2020, alternative="greater")
    st.write(f"Mann–Whitney one-sided p = {p:.2e}")

tab1, tab2, tab3 = st.tabs(["Distribution", "Year trend", "Clusters"])

with tab1:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    yrs = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
    sub = df[(df["year"] >= yrs[0]) & (df["year"] <= yrs[1])]
    fig = px.histogram(sub, x="latency_hours", log_x=True, nbins=60)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    yearly = df.groupby("year")["latency_hours"].median().reset_index()
    fig = px.line(yearly, x="year", y="latency_hours", markers=True)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if "topic_cluster" in df.columns:
        agg = df.groupby("topic_cluster")["latency_hours"].agg(["median", "count"]).reset_index().sort_values("median")
        fig = px.bar(agg, x="median", y="topic_cluster", orientation="h", hover_data=["count"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("topic_cluster column not present in cofacts_latency.csv — this tab is empty in the current build.")