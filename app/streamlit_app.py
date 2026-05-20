import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu

st.set_page_config(page_title="Narrative Latency — Taiwan", page_icon="⏱️", layout="wide")

st.markdown("""
<style>
.hero-number { font-size: 5.5rem; font-weight: 800; color: #ef4444; line-height: 1; margin-bottom: 0.25rem; }
.hero-sub { font-size: 1.05rem; color: #6b7280; }
.section-divider { margin: 2.5rem 0 1rem 0; border-top: 1px solid #d1d5db; }
</style>
""", unsafe_allow_html=True)

TPL = "plotly_dark"

@st.cache_data
def load():
    df = pd.read_csv("data/processed/cofacts_latency.csv")
    df["article_createdAt"] = pd.to_datetime(df["article_createdAt"], errors="coerce", format="ISO8601", utc=True)
    df = df.dropna(subset=["article_createdAt"])
    df["year"] = df["article_createdAt"].dt.year
    return df

df = load()

# ============ HERO ============
st.markdown("# ⏱️ Narrative Latency — Taiwan")
st.caption("How fast does Cofacts' community fact-check Taiwan's LINE rumors? Interactive view of 68,533 article–reply pairs, 2016–2026.")

# Election windows: ±90 days around Taiwan presidential elections
_e2020 = pd.Timestamp("2020-01-11", tz="UTC")
_e2024 = pd.Timestamp("2024-01-13", tz="UTC")
_win = pd.Timedelta(days=90)
df_2020 = df[(df["article_createdAt"] >= _e2020 - _win) & (df["article_createdAt"] <= _e2020 + _win)]
df_2024 = df[(df["article_createdAt"] >= _e2024 - _win) & (df["article_createdAt"] <= _e2024 + _win)]
e2020 = df_2020["latency_hours"]
e2024 = df_2024["latency_hours"]
ratio = e2024.median() / e2020.median()
u_stat, p_val = mannwhitneyu(e2024, e2020, alternative="greater")

h1, h2, h3 = st.columns([2, 1, 1])
with h1:
    st.markdown(f'<div class="hero-number">{ratio:.0f}×</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-sub"><b>slower in 2024 vs 2020 election window</b><br>'
        f'<span style="color:#9ca3af">{e2020.median():.1f}h median → {e2024.median():.1f}h median</span></div>',
        unsafe_allow_html=True,
    )
with h2:
    st.metric("Articles analyzed", f"{len(df):,}")
    st.metric("Years covered", f"{int(df['year'].min())}–{int(df['year'].max())}")
with h3:
    st.metric("Mann–Whitney p", "< 10⁻²⁰⁰" if p_val < 1e-200 else f"{p_val:.2e}")
    st.metric("Overall median", f"{df['latency_hours'].median():.1f} h")

# Annotated timeline
yearly = df[df["year"] >= 2018].groupby("year").agg(median=("latency_hours", "median"), count=("latency_hours", "size")).reset_index()
fig_tl = go.Figure()
fig_tl.add_trace(go.Scatter(
    x=yearly["year"], y=yearly["median"], mode="lines+markers",
    line=dict(width=3, color="#60a5fa"), marker=dict(size=11, color="#3b82f6"),
    customdata=yearly["count"],
    hovertemplate="<b>%{x}</b><br>Median %{y:.1f}h<br>N=%{customdata:,}<extra></extra>",
))
fig_tl.add_vrect(x0=2019.5, x1=2020.5, fillcolor="#3b82f6", opacity=0.18, line_width=0,
                 annotation_text="2020 election", annotation_position="top left")
fig_tl.add_vrect(x0=2023.5, x1=2024.5, fillcolor="#f59e0b", opacity=0.18, line_width=0,
                 annotation_text="2024 election", annotation_position="top left")
fig_tl.update_layout(template=TPL, title="Median response time by year (Cofacts community)",
                     xaxis_title="Year", yaxis_title="Median latency (hours)", height=360,
                     margin=dict(t=60, b=40))
st.plotly_chart(fig_tl, use_container_width=True)

# ============ SLA ============
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🎯 Set a response-time target")
st.caption("Drag to pick how many hours Cofacts has to respond before a rumor is 'missed'.")

c1, c2 = st.columns([1, 2])
with c1:
    threshold = st.slider("Target SLA (hours)", 1, 168, 24, 1)
    pct_under = 100 * (df["latency_hours"] <= threshold).sum() / len(df)
    pct_over = 100 - pct_under
    missed_n = int(round(pct_over / 100 * len(df)))
    st.markdown(f"### {pct_under:.1f}% met · {pct_over:.1f}% missed")
    st.caption(f"{missed_n:,} of {len(df):,} articles exceeded the {threshold}h window.")

with c2:
    gauge = go.Figure()
    gauge.add_trace(go.Bar(y=["SLA"], x=[pct_under], orientation="h",
                            marker_color="#10b981", text=f"✓ {pct_under:.1f}%",
                            textposition="inside", textfont=dict(size=20, color="white"),
                            hoverinfo="skip"))
    gauge.add_trace(go.Bar(y=["SLA"], x=[pct_over], orientation="h",
                            marker_color="#ef4444", text=f"✗ {pct_over:.1f}%",
                            textposition="inside", textfont=dict(size=20, color="white"),
                            hoverinfo="skip"))
    gauge.update_layout(barmode="stack", template=TPL, height=140, showlegend=False,
                        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=False), margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(gauge, use_container_width=True)

# Cumulative response curve
sorted_lat = df["latency_hours"].sort_values().reset_index(drop=True)
cdf_df = pd.DataFrame({"hours": sorted_lat, "pct": 100 * (sorted_lat.index + 1) / len(sorted_lat)})
cdf_df = cdf_df[cdf_df["hours"] <= 720]
fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(x=cdf_df["hours"], y=cdf_df["pct"], mode="lines", fill="tozeroy",
                              line=dict(color="#3b82f6", width=2),
                              hovertemplate="By %{x:.0f}h: %{y:.1f}%<extra></extra>"))
fig_cdf.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"{threshold}h", annotation_position="top right")
fig_cdf.add_hline(y=pct_under, line_dash="dot", line_color="white", opacity=0.5)
fig_cdf.update_layout(template=TPL, title="Cumulative response curve",
                      xaxis_title="Hours since article posted", yaxis_title="% of rumors responded to",
                      height=360, margin=dict(t=50, b=40))
st.plotly_chart(fig_cdf, use_container_width=True)

# ============ ELECTION CYCLES ============
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🗳️ 2020 vs 2024 election cycles")
st.caption("The headline finding — log-scale x-axis, same binning.")

a, b = st.columns(2)
with a:
    log20 = np.log10(df_2020["latency_hours"].clip(lower=0.01))
    fig20 = px.histogram(x=log20, nbins=40, color_discrete_sequence=["#3b82f6"])
    fig20.update_traces(marker_line_color="white", marker_line_width=0.6)
    fig20.add_vline(x=np.log10(e2020.median()), line_dash="dash", line_color="white",
                    annotation_text=f"median {e2020.median():.1f}h", annotation_position="top right")
    fig20.update_layout(template=TPL, title=f"2020 election window — N={len(e2020):,}", height=340,
                         margin=dict(t=50, b=40), showlegend=False)
    fig20.update_xaxes(title="Latency (hours)", tickvals=[-1,0,1,2,3,4],
                       ticktext=["0.1h","1h","10h","100h","1000h","10000h"])
    st.plotly_chart(fig20, use_container_width=True)
with b:
    log24 = np.log10(df_2024["latency_hours"].clip(lower=0.01))
    fig24 = px.histogram(x=log24, nbins=40, color_discrete_sequence=["#f59e0b"])
    fig24.update_traces(marker_line_color="white", marker_line_width=0.6)
    fig24.add_vline(x=np.log10(e2024.median()), line_dash="dash", line_color="white",
                    annotation_text=f"median {e2024.median():.1f}h", annotation_position="top right")
    fig24.update_layout(template=TPL, title=f"2024 election window — N={len(e2024):,}", height=340,
                         margin=dict(t=50, b=40), showlegend=False)
    fig24.update_xaxes(title="Latency (hours)", tickvals=[-1,0,1,2,3,4],
                       ticktext=["0.1h","1h","10h","100h","1000h","10000h"])
    st.plotly_chart(fig24, use_container_width=True)

st.info(f"📊 2024's median is **{ratio:.1f}×** higher than 2020's, despite both being election years. The slowdown is not driven by election volume — it's a community-capacity story.")

# ============ HEATMAP ============
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 🕐 When do rumors arrive?")
st.caption("Article submission volume by Taipei-local hour × day-of-week.")

d2 = df.copy()
d2["tpe"] = d2["article_createdAt"].dt.tz_convert("Asia/Taipei")
d2["hour"] = d2["tpe"].dt.hour
d2["dow"] = d2["tpe"].dt.day_name()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
heat = d2.groupby(["dow", "hour"]).size().reset_index(name="n").pivot(index="dow", columns="hour", values="n").reindex(dow_order)

fig_heat = px.imshow(heat, color_continuous_scale="Inferno", aspect="auto",
                     labels=dict(x="Hour (Taipei)", y="", color="Articles"))
fig_heat.update_layout(template=TPL, height=340, title="Article submission volume",
                       margin=dict(t=50, b=40))
st.plotly_chart(fig_heat, use_container_width=True)
st.caption("Volume concentrates in weekday evenings (Taipei time) — Cofacts community capacity needs to match that pattern.")

# ============ PERCENTILE DRILL-DOWN ============
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("## 📊 Percentile drill-down by year")
st.caption("Median hides the tail. Switch to P90/P95 to see the slow-response tail by year.")

p_pick = st.select_slider("Percentile", options=[50, 75, 90, 95, 99], value=90,
                           format_func=lambda x: f"P{x}")
pct_yr = df.groupby("year")["latency_hours"].quantile(p_pick / 100).reset_index()
fig_p = px.bar(pct_yr, x="year", y="latency_hours", color="latency_hours",
               color_continuous_scale="Reds", labels={"latency_hours": f"P{p_pick} (hours)"})
fig_p.update_layout(template=TPL, title=f"P{p_pick} response time by year", height=360,
                    margin=dict(t=50, b=40), coloraxis_showscale=False)
st.plotly_chart(fig_p, use_container_width=True)

# ============ Caveats ============
with st.expander("📋 Methodology + caveats"):
    st.markdown("""
- **Source:** Cofacts open dataset, HF snapshot 2026-05-10 (CC BY-SA 4.0). N=68,533 article–reply pairs after filters.
- **Metric:** `latency_hours = first community reply timestamp − article submission timestamp`.
- **Filters:** `articleType = TEXT`, `status = NORMAL`, drop pairs with negative latency or latency > 1 year.
- **Selection bias:** Cofacts users are a self-selected subset of LINE users. The metric measures Cofacts community response speed, not Taiwan-wide rumor debunking speed.
- **Statistical test:** Mann–Whitney U, one-sided, 2024 > 2020.
- **Repo:** [github.com/natharzu/narrative-latency-tw](https://github.com/natharzu/narrative-latency-tw)
""")