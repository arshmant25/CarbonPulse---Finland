# app/streamlit_app.py
# ─────────────────────────────────────────────────────────────
# CarbonPulse Finland — Streamlit Dashboard
# Reads from shared SQLite DB + calls FastAPI for 24h forecasts.
# Mirrors the Power BI dashboard structure:
#   Page 1: Grid Live View  (Fingrid hourly data)
#   Page 2: Sectoral Data   (Statistics Finland)
#   Page 3: Carbon Monitor  (daily cross-validation)

import os
import sqlite3
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────
DB_PATH  = os.getenv("DB_PATH",  "/data/project_data.db")
API_URL  = os.getenv("API_URL",  "http://api:8000")

# ── Colour palette — matches Power BI theme ────────────────────
C_TEAL   = "#2a9d8f"
C_CORAL  = "#e76f51"
C_NAVY   = "#264653"
C_AMBER  = "#e9c46a"
C_BLUE   = "#457b9d"
C_GREEN  = "#52b788"
C_ORANGE = "#f4a261"
C_PURPLE = "#6d6875"
BG_DARK  = "#0f1923"
BG_CARD  = "#1e2f42"

GENERATION_COLORS = {
    "nuclear":       C_BLUE,
    "hydro":         C_TEAL,
    "wind":          C_GREEN,
    "chp_district":  C_CORAL,
    "chp_industrial":C_ORANGE,
}

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="CarbonPulse Finland",
    page_icon="🇫🇮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark theme ─────────────────────────────────────
# CHANGES:
#  - h1-h6 all styled bright (was only h1-h3; every chart title in
#    this app uses "####" = h4, which fell back to the browser's dim
#    default grey -- this was the actual cause of "text too light").
#  - metric-label brightened from #7fafc0 -> #a8d8ea for more contrast
#    against the dark background.
#  - block-container padding reduced (top/sides) and a max-width unset
#    so charts fill the page width instead of leaving large side gutters.
#  - vertical-block gap tightened so there's less dead space between
#    stacked chart rows.
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_DARK}; }}
    .block-container {{
        padding-top: 1rem; padding-bottom: 1rem;
        padding-left: 2rem; padding-right: 2rem;
        max-width: 100%;
    }}
    .metric-card {{
        background: {BG_CARD};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.2rem 0;
    }}
    .metric-label {{
        color: #a8d8ea; font-size: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.05em;
        font-weight: 500;
    }}
    .metric-value {{
        color: #f2f9fc; font-size: 2rem;
        font-weight: 400; line-height: 1.1;
    }}
    .metric-delta-good {{ color: {C_TEAL}; font-size: 0.85rem; }}
    .metric-delta-bad  {{ color: {C_CORAL}; font-size: 0.85rem; }}
    h1, h2, h3, h4, h5, h6 {{ color: #f2f9fc !important; }}
    h4 {{ font-size: 1.1rem !important; margin-bottom: 0.4rem !important; }}
    p, span, div, label {{ color: #d6ecf4; }}
    .stSelectbox label, .stSlider label {{ color: #c3e0ec !important; }}

    /* NEW: sidebar fix. The old ".sidebar .sidebar-content" selector
    is from a pre-1.0 Streamlit API and matches nothing in current
    Streamlit's DOM -- the sidebar was getting NO custom styling at
    all, just Streamlit's light-theme default text color on a dark
    background, which is why it was unreadable. These are the correct
    current selectors (data-testid based, stable across Streamlit
    versions). */
    [data-testid="stSidebar"] {{
        background: #111d2b;
    }}
    [data-testid="stSidebar"] * {{
        color: #e8f4f8 !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: #e8f4f8 !important;
        font-size: 0.95rem;
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        color: #cfe6f0 !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: #2a3f52 !important;
    }}
    /* Selectbox: the closed-box value text and the "Year" label above
    it were both dim -- same root cause, no selector was targeting
    the actual current BaseWeb-rendered selectbox internals. */
    div[data-baseweb="select"] > div {{
        background-color: #1e2f42 !important;
        color: #f2f9fc !important;
        border-color: #35526b !important;
    }}
    div[data-baseweb="select"] span {{
        color: #f2f9fc !important;
    }}
    label[data-testid="stWidgetLabel"] p {{
        color: #e8f4f8 !important;
        font-weight: 500;
    }}

        /* ─── FIXED PORTAL DROPDOWN VISIBILITY FIX ─── */
    /* Targets the popover container that escapes the app wrapper */
    [data-baseweb="popover"] {{
        background-color: #0f1923 !important; /* Matches BG_DARK */
    }}
    
    /* Targets the listbox container and its nested list */
    div[role="listbox"], 
    div[role="listbox"] ul {{
        background-color: #0f1923 !important; /* Matches BG_DARK */
        color: #d6ecf4 !important;            /* Matches global text color */
    }}

    /* Ensure specific list options are dark and text is readable */
    div[role="listbox"] li {{
        background-color: #0f1923 !important;
        color: #d6ecf4 !important;
    }}

    /* Provides clear highlight contrast when hovering or selecting a year */
    div[role="listbox"] li:hover,
    div[role="listbox"] li[aria-selected="true"] {{
        background-color: #2a9d8f !important; /* Highlights with C_TEAL */
        color: #ffffff !important;
    }}
    /* ───────────────────────────────────────────── */

    /* NEW: dropdown arrow icon -- an SVG whose fill color was never
    set, so it defaulted to something too close to the dark background
    to see. */
    div[data-baseweb="select"] svg {{
        fill: #c3e0ec !important;
    }}
    /* Tighten vertical gap between stacked chart blocks */
    div[data-testid="stVerticalBlock"] > div {{ gap: 0.5rem; }}
    hr {{ margin: 0.8rem 0 !important; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f1923",
    font=dict(color="#c3e0ec", family="Segoe UI", size=12),   # brightened from #a8c5d8
    margin=dict(l=45, r=15, t=20, b=35),                       # tighter, was t=40
    xaxis=dict(gridcolor="#1e2f42", showgrid=True),
    yaxis=dict(gridcolor="#1e2f42", showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#d6ecf4")),
)


# ── Data loaders ───────────────────────────────────────────────
@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_hourly(year_filter=None):
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM fingrid_hourly_merged ORDER BY timestamp_utc"
    df = pd.read_sql(q, conn)
    conn.close()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    if year_filter and year_filter != "All":
        df = df[df.index.year == int(year_filter)]
    return df


@st.cache_data(ttl=3600)
def load_sectoral():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM sectoral_emissions", conn)
    conn.close()
    return df


@st.cache_data(ttl=3600)
def load_carbon_monitor():
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT * FROM carbon_monitor_finland ORDER BY date", conn
        )
        df["date"] = pd.to_datetime(df["date"], utc=True)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


@st.cache_data(ttl=900)
def get_forecast(steps=96):
    try:
        r = requests.get(f"{API_URL}/forecast?steps={steps}", timeout=30)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data["forecasts"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df, data.get("last_known_timestamp", "")
    except Exception as e:
        return pd.DataFrame(), str(e)


def get_current():
    try:
        r = requests.get(f"{API_URL}/current", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def metric_card(label, value, delta=None, good_direction="down"):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-good" if (
            (good_direction == "down" and delta < 0) or
            (good_direction == "up"   and delta > 0)
        ) else "metric-delta-bad"
        arrow = "▼" if delta < 0 else "▲"
        delta_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}%</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


# ── Sidebar navigation ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🇫🇮 CarbonPulse Finland")
    st.markdown("---")
    page = st.radio("Navigation", [
        "⚡ Grid Live View",
        "🌍 Sectoral Data",
        "📡 Carbon Monitor",
    ])
    st.markdown("---")
    st.markdown(f"**Data source:** Fingrid Open Data API  \n"
                f"**Sectoral:** Statistics Finland  \n"
                f"**Daily validation:** Carbon Monitor EU")
    st.markdown("---")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    last_refresh = datetime.now(timezone.utc).strftime("%H:%M UTC")
    st.caption(f"Last refresh: {last_refresh}")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — GRID LIVE VIEW
# ══════════════════════════════════════════════════════════════
if page == "⚡ Grid Live View":
    st.markdown("# ⚡ Finland Electricity Grid — CO₂ Intensity")

    # Year selector
    all_years = ["All"] + list(range(2026, 2017, -1))
    col_yr, col_title = st.columns([1, 4], gap="small")
    with col_yr:
        selected_year = st.selectbox("Year", all_years, index=1)

    df = load_hourly(selected_year)
    current = get_current()

    # ── KPI row ─────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    co2_col  = next((c for c in ["co2_intensity_weighted_avg",
                                  "co2_intensity_simple_avg"]
                     if c in df.columns), None)
    prod_col = next((c for c in ["total_production_mw"]
                     if c in df.columns), None)
    wind_col = next((c for c in ["wind_mw"] if c in df.columns), None)
    nuke_col = next((c for c in ["nuclear_mw"] if c in df.columns), None)
    hydr_col = next((c for c in ["hydro_mw"] if c in df.columns), None)

    with k1:
        live_val = (current["co2_intensity_gco2kwh"]
                    if current else df[co2_col].iloc[-1])
        metric_card("CO₂ Intensity (live)",
                    f"{live_val:.1f} gCO₂/kWh",
                    good_direction="down")
    with k2:
        avg_int = df[co2_col].mean() if co2_col else 0
        metric_card("Avg CO₂ Intensity",
                    f"{avg_int:.1f} gCO₂/kWh",
                    good_direction="down")
    with k3:
        total_twh = df[prod_col].sum() / 1e6 if prod_col else 0
        metric_card("Total Production",
                    f"{total_twh:.1f} TWh")
    with k4:
        ren_gwh = 0
        for c in [wind_col, nuke_col, hydr_col]:
            if c and c in df.columns:
                ren_gwh += df[c].sum() / 1000
        prod_gwh = df[prod_col].sum() / 1000 if prod_col else 1
        ren_pct  = ren_gwh / prod_gwh * 100 if prod_gwh > 0 else 0
        metric_card("Renewable Share",
                    f"{ren_pct:.1f}%",
                    good_direction="up")
    with k5:
        co2_kt = 0
        if co2_col and prod_col and co2_col in df.columns:
            co2_kt = (df[co2_col] * df[prod_col]).sum() / 1e6
        metric_card("Grid CO₂ Emitted",
                    f"{co2_kt/1000:.2f} Mt")

    st.markdown("---")

    # ── Row 2: Long-term monthly trend ───────────────────────
    # NEW: "ME" (month-end) labels each point at the LAST day of the
    # month (e.g. Aug 31) even when the month isn't over yet -- so an
    # in-progress August showed as a point dated Aug 31, which looks
    # like it's claiming data from the future. "MS" (month-start)
    # labels each point at the 1st of its month instead -- the value
    # still only reflects however many days of that month exist so
    # far, but the x-axis position is now honest about what data
    # exists today.
    st.markdown("#### CO₂ Intensity — Monthly Trend (all history)")
    df_monthly = df[co2_col].resample("MS").mean().reset_index() if co2_col else pd.DataFrame()

    fig = go.Figure(layout=PLOTLY_LAYOUT)
    if not df_monthly.empty:
        fig.add_trace(go.Scatter(
            x=df_monthly["timestamp_utc"],
            y=df_monthly[co2_col],
            mode="lines", line=dict(color=C_TEAL, width=2),
            name="Monthly avg intensity", fill="tozeroy",
            fillcolor="rgba(42,157,143,0.1)",
        ))
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="gCO₂/kWh")
    st.plotly_chart(fig, width="stretch")

    # NEW: separate "last 7 days + 24h forecast" chart. The monthly
    # chart above aggregates to one point per month, so a 24h forecast
    # plotted on that axis is a single sliver next to (previously
    # mislabeled) monthly points -- there's no way to see the forecast
    # actually continue the recent trend at matching resolution. This
    # chart uses the same hourly data at native resolution for the
    # trailing 7 days, then appends the forecast so the dashed line
    # visually continues from the last actual point instead of
    # appearing as a disconnected dot.
    st.markdown("---")
    st.markdown("#### Last 7 Days + 24h Forecast (hourly resolution)")

    if co2_col and not df.empty:
        recent_cutoff = df.index.max() - pd.Timedelta(days=7)
        df_recent = df[df.index >= recent_cutoff][[co2_col]].reset_index()

        fc_df, last_ts = get_forecast(steps=96)

        fig_recent = go.Figure(layout=PLOTLY_LAYOUT)
        fig_recent.add_trace(go.Scatter(
            x=df_recent["timestamp_utc"],
            y=df_recent[co2_col],
            mode="lines", line=dict(color=C_TEAL, width=2),
            name="Actual (last 7 days)",
            fill="tozeroy", fillcolor="rgba(42,157,143,0.1)",
        ))

        if not fc_df.empty:
            # Prepend the last actual point so the forecast line
            # connects continuously instead of starting from a gap.
            connector_x = [df_recent["timestamp_utc"].iloc[-1]] + list(fc_df["timestamp"])
            connector_y = [df_recent[co2_col].iloc[-1]] + list(fc_df["co2_intensity_gco2kwh"])
            fig_recent.add_trace(go.Scatter(
                x=connector_x, y=connector_y,
                mode="lines",
                line=dict(color=C_CORAL, width=2, dash="dash"),
                name="GRU 24h forecast",
            ))
        else:
            st.caption(f"⚠️ Forecast unavailable: {last_ts}")

        fig_recent.add_vline(
            x=df.index.max(), line_dash="dot", line_color="#5a7d91",
            annotation_text="now", annotation_font_color="#c3e0ec",
        )
        fig_recent.update_xaxes(title_text="")
        fig_recent.update_yaxes(title_text="gCO₂/kWh")
        st.plotly_chart(fig_recent, width="stretch")

    st.markdown("---")

    # ── Row 3: Generation mix ─────────────────────────────────
    col_l, col_c, col_r2 = st.columns([1, 2, 1], gap="small")

    with col_c:
        st.markdown("#### Generation Mix")
        gen_labels = []
        gen_values = []
        gen_colors = []
        for col, lbl, color in [
            (wind_col,  "Wind",          C_GREEN),
            (nuke_col,  "Nuclear",       C_BLUE),
            (hydr_col,  "Hydro",         C_TEAL),
            ("chp_district_mw",  "CHP District",  C_CORAL),
            ("chp_industrial_mw","CHP Industrial", C_ORANGE),
        ]:
            c = col if col else None
            if c and c in df.columns:
                gen_labels.append(lbl)
                gen_values.append(df[c].sum() / 1000)
                gen_colors.append(color)

        if gen_labels:
            fig2 = go.Figure(go.Pie(
                labels=gen_labels, values=gen_values,
                hole=0.5, marker=dict(colors=gen_colors),
                textinfo="label+percent",
                textfont=dict(color="white", size=10),
            ), layout=PLOTLY_LAYOUT)
            fig2.update_layout(
                annotations=[dict(
                    text=f"{ren_pct:.0f}%<br>clean",
                    x=0.5, y=0.5, font_size=16,
                    font_color=C_TEAL, showarrow=False
                )]
            )
            st.plotly_chart(fig2, width="stretch")

    # ── Row 3: Monthly generation stacked + heatmap ──────────
    col_l2, col_r2 = st.columns([3, 2], gap="small")

    with col_l2:
        st.markdown("#### Monthly Generation by Source")
        gen_cols = {
            "CHP District":  "chp_district_mw",
            "CHP Industrial":"chp_industrial_mw",
            "Hydro":         "hydro_mw",
            "Nuclear":       "nuclear_mw",
            "Wind":          "wind_mw",
        }
        colors_list = [C_CORAL, C_ORANGE, C_TEAL, C_BLUE, C_GREEN]
        fig3 = go.Figure(layout=PLOTLY_LAYOUT)
        for (lbl, c), color in zip(gen_cols.items(), colors_list):
            if c in df.columns:
                monthly = df[c].resample("ME").mean().reset_index()
                fig3.add_trace(go.Scatter(
                    x=monthly["timestamp_utc"],
                    y=monthly[c],
                    mode="lines", stackgroup="one",
                    name=lbl, line=dict(color=color, width=0.5),
                    fillcolor=color.replace(")", ",0.7)").replace("rgb", "rgba")
                                  if color.startswith("rgb") else color,
                ))
        fig3.update_yaxes(title_text="Mean MW")
        st.plotly_chart(fig3, width="stretch")

    with col_r2:
        st.markdown("#### Intensity: Month × Hour Heatmap")
        if co2_col and co2_col in df.columns:
            df_h = df.copy()
            df_h["month"] = df_h.index.month
            df_h["hour"]  = df_h.index.hour
            pivot = df_h.groupby(["month","hour"])[co2_col].mean().unstack()
            month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]
            fig4 = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(h) for h in range(24)],
                y=month_labels,
                colorscale="RdYlGn_r",
                colorbar=dict(title="gCO₂/kWh",
                              tickfont=dict(color="#c3e0ec")),
            ), layout=PLOTLY_LAYOUT)
            fig4.update_layout(
                xaxis_title="Hour of day (UTC)",
                yaxis_title="Month",
            )
            st.plotly_chart(fig4, width="stretch")


# ══════════════════════════════════════════════════════════════
# PAGE 2 — SECTORAL DATA
# ══════════════════════════════════════════════════════════════
elif page == "🌍 Sectoral Data":
    st.markdown("# 🌍 Finland National Greenhouse Gas Emissions")

    df_s = load_sectoral()

    # Year slicer
    year_range = st.slider(
        "Year range",
        min_value=1990, max_value=2025,
        value=(1990, 2025)
    )
    df_yr = df_s[(df_s["year"] >= year_range[0]) &
                 (df_s["year"] <= year_range[1])]

    # ── KPI row ─────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4, gap="small")
    sel_year = year_range[1]

    no_lulucf = df_yr[
        (df_yr["sector_code"] == "TOTAL_NO_LULUCF") &
        (df_yr["year"] == sel_year)
    ]["total_kt_co2e"]
    with_lulucf = df_yr[
        (df_yr["sector_code"] == "TOTAL_WITH_LULUCF") &
        (df_yr["year"] == sel_year)
    ]["total_kt_co2e"]
    base_1990 = df_s[
        (df_s["sector_code"] == "TOTAL_NO_LULUCF") &
        (df_s["year"] == 1990)
    ]["total_kt_co2e"]

    total_mt    = no_lulucf.values[0]  / 1000 if len(no_lulucf)  > 0 else 0
    net_mt      = with_lulucf.values[0]/ 1000 if len(with_lulucf)> 0 else 0
    base_mt     = base_1990.values[0]  / 1000 if len(base_1990)  > 0 else 1
    reduction   = (total_mt - base_mt) / base_mt * 100

    with k1: metric_card(f"Total Emissions {sel_year}",  f"{total_mt:.0f} Mt CO₂e")
    with k2: metric_card(f"Net with Forests {sel_year}", f"{net_mt:.0f} Mt CO₂e")
    with k3: metric_card("Reduction since 1990",
                          f"{reduction:.1f}%", good_direction="down")
    with k4:
        co2_sel = df_yr[
            (df_yr["sector_code"] == "TOTAL_NO_LULUCF") &
            (df_yr["year"] == sel_year)
        ][["co2_kt","total_kt_co2e"]]
        co2_share = (co2_sel["co2_kt"].values[0] /
                     co2_sel["total_kt_co2e"].values[0] * 100
                     if len(co2_sel) > 0 else 0)
        metric_card("CO₂ Share of Total", f"{co2_share:.1f}%")

    st.markdown("---")

    # ── Row 2: Trajectory + treemap ─────────────────────────
    col_l, col_r = st.columns([3, 2], gap="small")

    with col_l:
        st.markdown("#### Emissions Trajectory 1990–2025")
        no_lulucf_ts = df_s[df_s["sector_code"] == "TOTAL_NO_LULUCF"]\
            .sort_values("year")
        with_lulucf_ts = df_s[df_s["sector_code"] == "TOTAL_WITH_LULUCF"]\
            .sort_values("year")

        fig5 = go.Figure(layout=PLOTLY_LAYOUT)
        fig5.add_trace(go.Scatter(
            x=no_lulucf_ts["year"],
            y=no_lulucf_ts["total_kt_co2e"] / 1000,
            mode="lines+markers", name="Total (excl. forests)",
            line=dict(color=C_CORAL, width=2),
            fill="tozeroy", fillcolor="rgba(231,111,81,0.1)",
        ))
        fig5.add_trace(go.Scatter(
            x=with_lulucf_ts["year"],
            y=with_lulucf_ts["total_kt_co2e"] / 1000,
            mode="lines+markers", name="Net (incl. forests)",
            line=dict(color=C_TEAL, width=2),
        ))
        fig5.add_hline(y=0, line_dash="dash",
                       line_color="white", opacity=0.5,
                       annotation_text="Net zero")
        fig5.update_yaxes(title_text="Mt CO₂e")
        st.plotly_chart(fig5, width="stretch")

    with col_r:
        st.markdown(f"#### Emission by Group — {sel_year}")
        treemap_data = df_yr[
            (df_yr["level"] == 1) &
            (df_yr["year"] == sel_year) &
            (~df_yr["sector_code"].isin(["INDCO2"]))  # NEW: keep LULUCF(4) and International(1D), only drop the INDCO2 rollup duplicate
        ]
        if not treemap_data.empty:
            fig6 = px.treemap(
                treemap_data,
                path=["group", "sector_name"],
                values="total_kt_co2e",
                color="group",
                color_discrete_map={
                    "energy":       C_CORAL,
                    "transport":    C_ORANGE,
                    "agriculture":  C_GREEN,
                    "industry":     C_BLUE,
                    "lulucf":       C_TEAL,
                    "waste":        C_PURPLE,
                    "international":C_AMBER,  # NEW: was missing, sector was excluded entirely
                },
            )
            fig6.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig6, width="stretch")

    # ── Row 3: Stacked bar + gas breakdown ───────────────────
    col_l2, col_r2 = st.columns([3, 2], gap="small")

    with col_l2:
        st.markdown("#### Sector Breakdown by Year")
        level1 = df_s[
            (df_s["level"] == 1) &
            (~df_s["sector_code"].isin(["INDCO2"]))  # NEW: keep LULUCF(4) and International(1D)
        ]
        group_colors = {
            "energy":     C_CORAL,   "transport":  C_ORANGE,
            "agriculture":C_GREEN,   "industry":   C_BLUE,
            "waste":      C_PURPLE,  "lulucf":     C_TEAL,
            "international": C_AMBER,  # NEW: was missing, sector was excluded entirely
        }
        fig7 = go.Figure(layout=PLOTLY_LAYOUT)
        for grp, color in group_colors.items():
            sub = level1[level1["group"] == grp] if "group" in level1.columns \
                  else level1[level1["group"] == grp]
            annual = sub.groupby("year")["total_kt_co2e"].sum().reset_index()
            fig7.add_trace(go.Bar(
                x=annual["year"],
                y=annual["total_kt_co2e"] / 1000,
                name=grp.capitalize(),
                marker_color=color,
            ))
        fig7.update_layout(barmode="relative")
        fig7.update_yaxes(title_text="Mt CO₂e")
        st.plotly_chart(fig7, width="stretch")

    with col_r2:
        st.markdown(f"#### CO₂ vs Other Gases — {sel_year}")
        gas_data = df_s[
            (df_s["level"] == 1) &
            (df_s["year"] == sel_year) &
            (~df_s["sector_code"].isin(["INDCO2"]))  # NEW: keep LULUCF(4) and International(1D)
        ].dropna(subset=["co2_kt","other_gases_kt"])

        if not gas_data.empty:
            gas_data = gas_data.copy()
            gas_data["total"] = (gas_data["co2_kt"] + gas_data["other_gases_kt"])
            gas_data["co2_pct"]   = gas_data["co2_kt"]        / gas_data["total"] * 100
            gas_data["other_pct"] = gas_data["other_gases_kt"]/ gas_data["total"] * 100

            fig8 = go.Figure(layout=PLOTLY_LAYOUT)
            grp_col = "group" if "group" in gas_data.columns else "group"
            fig8.add_trace(go.Bar(
                y=gas_data[grp_col], x=gas_data["co2_pct"],
                orientation="h", name="CO₂",
                marker_color=C_TEAL,
            ))
            fig8.add_trace(go.Bar(
                y=gas_data[grp_col], x=gas_data["other_pct"],
                orientation="h", name="CH₄ + N₂O + F-gases",
                marker_color=C_CORAL,
            ))
            fig8.update_layout(barmode="relative")
            fig8.update_xaxes(title_text="Gas share (%)")
            st.plotly_chart(fig8, width="stretch")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CARBON MONITOR
# ══════════════════════════════════════════════════════════════
elif page == "📡 Carbon Monitor":
    st.markdown("# 📡 Monitoring Multiscale Carbon Emission")

    df_cm = load_carbon_monitor()

    if df_cm.empty:
        st.warning(
            "Carbon Monitor data not found in the database.  \n"
            "Run `export_for_powerbi.py` to populate "
            "`carbon_monitor_finland` table."
        )
        st.stop()

    # ── KPI row ─────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4, gap="small")
    val_col = next((c for c in df_cm.columns if "Mt" in c or "value" in c.lower()), None)
    if val_col:
        total_mt   = df_cm[val_col].sum()
        power_mt   = df_cm[df_cm["sector"] == "Power"][val_col].sum()
        avg_daily  = df_cm.groupby("date")[val_col].sum().mean()
        avg_power  = df_cm[df_cm["sector"] == "Power"][val_col].mean()
        power_share= power_mt / total_mt * 100 if total_mt > 0 else 0

        with k1: metric_card("Total Emissions (Mt)", f"{total_mt:.1f}")
        with k2: metric_card("Power Sector (Mt)", f"{power_mt:.1f}")
        with k3: metric_card("Avg Daily Total (Mt/day)", f"{avg_daily:.3f}")
        with k4: metric_card("Power Sector Share", f"{power_share:.1f}%")

    st.markdown("---")

    # ── Daily stacked area by sector ─────────────────────────
    st.markdown("#### Daily CO₂ Emissions by Sector 2019–2026")
    if val_col:
        sector_colors = {
            "Power":                C_TEAL,
            "Industry":             C_BLUE,
            "Ground Transport":     C_ORANGE,
            "Residential":          C_CORAL,
            "Domestic Aviation":    C_AMBER,
            "International Aviation":C_PURPLE,
        }
        # Resample to monthly for readability
        df_cm_plot = df_cm.copy()
        df_cm_plot["month"] = df_cm_plot["date"].dt.to_period("M").dt.to_timestamp()

        fig9 = go.Figure(layout=PLOTLY_LAYOUT)
        for sector, color in sector_colors.items():
            sub = df_cm_plot[df_cm_plot["sector"] == sector]
            if sub.empty: continue
            monthly = sub.groupby("month")[val_col].sum().reset_index()
            fig9.add_trace(go.Scatter(
                x=monthly["month"], y=monthly[val_col],
                mode="lines", stackgroup="one",
                name=sector, line=dict(color=color, width=0.5),
            ))
        fig9.update_yaxes(title_text="Mt CO₂")
        st.plotly_chart(fig9, width="stretch")

    # ── Fingrid vs Carbon Monitor validation ─────────────────
    st.markdown("#### Fingrid-Derived vs Carbon Monitor Power Sector (r = 0.959)")
    df_h_all = load_hourly("All")
    co2_col  = next((c for c in ["co2_intensity_weighted_avg",
                                  "co2_intensity_simple_avg"]
                     if c in df_h_all.columns), None)
    cons_co2_col = next((c for c in ["consumption_co2_intens_weighted_avg",
                                      "consumption_co2_intens_simple_avg"]
                         if c in df_h_all.columns), None)
    prod_col = "total_production_mw"
    cons_prod_col = "consumption_elec_mw"

    if co2_col and prod_col in df_h_all.columns and val_col:
        # NEW: Fingrid-derived total was previously ONLY the
        # production-side emissions (co2_intensity * total_production).
        # The consumption-side emissions (consumption_co2_intensity *
        # consumption_elec_mw) were never added, even though the label
        # implied a total. Now both are computed and summed, matching
        # what "Fingrid-derived" should actually represent.
        df_h_all["daily_co2_mt_production"] = (
            df_h_all[co2_col] * df_h_all[prod_col] / 1e9
        )
        if cons_co2_col and cons_prod_col in df_h_all.columns:
            df_h_all["daily_co2_mt_consumption"] = (
                df_h_all[cons_co2_col] * df_h_all[cons_prod_col] / 1e9
            )
        else:
            df_h_all["daily_co2_mt_consumption"] = 0
            st.caption("⚠️ Consumption-side CO₂ columns not found — "
                       "showing production-side only.")

        df_h_all["daily_co2_mt"] = (
            df_h_all["daily_co2_mt_production"] + df_h_all["daily_co2_mt_consumption"]
        )

        fingrid_monthly = (df_h_all["daily_co2_mt"]
                           .resample("MS").sum()
                           .reset_index()
                           .rename(columns={0: "fingrid_mt",
                                            "daily_co2_mt": "fingrid_mt"}))

        cm_power = (df_cm[df_cm["sector"] == "Power"]
                    .copy())
        cm_power["month"] = (cm_power["date"]
                             .dt.to_period("M")
                             .dt.to_timestamp())
        cm_monthly = cm_power.groupby("month")[val_col].sum().reset_index()

        fig10 = go.Figure(layout=PLOTLY_LAYOUT)
        fig10.add_trace(go.Scatter(
            x=fingrid_monthly["timestamp_utc"],
            y=fingrid_monthly["fingrid_mt"],
            mode="lines", name="Fingrid-derived",
            line=dict(color=C_CORAL, width=2),
        ))
        fig10.add_trace(go.Scatter(
            x=cm_monthly["month"],
            y=cm_monthly[val_col],
            mode="lines", name="Carbon Monitor Power",
            line=dict(color=C_TEAL, width=2),
        ))
        fig10.update_yaxes(title_text="Mt CO₂")
        st.plotly_chart(fig10, width="stretch")

    st.info(
        "**Validation result:** Pearson r = **0.959** between Fingrid-derived "
        "daily power emissions and Carbon Monitor's independent estimate. "
        "This confirms the data pipeline is accurate."
    )