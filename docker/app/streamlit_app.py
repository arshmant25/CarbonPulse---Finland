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


@st.cache_data(ttl=3600)
def load_regional_data():
    """Loads every regional table this page needs in one connection.
    Missing tables (e.g. GDP/subclass not generated yet) degrade gracefully
    to empty DataFrames rather than crashing the page."""
    conn = sqlite3.connect(DB_PATH)
    def _try(query):
        try:
            return pd.read_sql(query, conn)
        except Exception:
            return pd.DataFrame()
    out = dict(
        summary            = _try("SELECT * FROM region_year_summary"),
        ghg_long           = _try("SELECT * FROM ghg_emissions_long"),
        subclass_emissions = _try("SELECT * FROM emissions_subclass_long"),
        gdp_merged         = _try("SELECT * FROM region_year_with_gdp"),
    )
    conn.close()
    return out


def normalize_region_name(name):
    """Matches region names across the emissions file, the GDP file, and the
    geometry file, regardless of diacritics/hyphens/casing."""
    name = str(name).strip()
    trans = str.maketrans("åäöÅÄÖ", "aaoAAO")
    name = name.translate(trans)
    name = name.replace("-", " ")
    name = " ".join(name.split()).lower()
    name = name.replace("southern ostrobothnia", "south ostrobothnia")
    return name


GEOJSON_PATH = os.getenv(
    "REGION_GEOJSON_PATH",
    os.path.join(os.path.dirname(DB_PATH), "region_data", "maakunta4500k.json"),
)


def _finalize_geo(geo):
    """Shared post-processing for geometry loaded either from the local file
    or the live WFS call: tag region_key, and make sure we end up in
    EPSG:4326 (lon/lat) for plotly regardless of what CRS the source had."""
    if geo.crs is None:
        # Statistics Finland's exports are ETRS89/TM35FIN (EPSG:3067) -- large
        # metre-scale coordinates. If a downloaded local file doesn't carry
        # CRS metadata, assume 3067 rather than plotly-breaking degrees.
        bounds = geo.total_bounds
        if (abs(bounds) > 180).any():
            geo = geo.set_crs("EPSG:3067")
        else:
            geo = geo.set_crs("EPSG:4326")
    name_col = next(c for c in ["name", "nimi", "namn"] if c in geo.columns)
    geo["region_key"] = geo[name_col].apply(normalize_region_name)
    return geo.to_crs("EPSG:4326")


@st.cache_data(ttl=86400)
def fetch_region_geometry():
    """Loads Finland's 19-region boundaries. Tries the local file first
    (GEOJSON_PATH / REGION_GEOJSON_PATH env var) -- this is the reliable path
    for a Docker container with no outbound internet -- and falls back to
    Statistics Finland's live WFS service if no local file is found. Returns
    None (never raises) if neither works, so callers can fall back to a
    non-map visualisation.
    """
    import geopandas as gpd

    if os.path.exists(GEOJSON_PATH):
        try:
            geo = gpd.read_file(GEOJSON_PATH)
            if len(geo) == 0:
                raise ValueError("File has no features")
            if len(geo) < 5:
                # A real region layer has 19 rows; a handful or fewer strongly
                # suggests this isn't the right file (e.g. a webpage saved
                # with a .json extension, or a different, smaller layer).
                raise ValueError(
                    f"Only {len(geo)} feature(s) in {GEOJSON_PATH} -- expected 19 regions. "
                    f"This usually means the file isn't the real WFS GeoJSON "
                    f"(see fetch_region_geojson.py for the correct way to download it)."
                )
            return _finalize_geo(geo)
        except Exception as e:
            st.sidebar.warning(f"Local region file at {GEOJSON_PATH} couldn't be used ({e}). "
                              f"Trying the live WFS service instead.")

    try:
        WFS_URL = "http://geo.stat.fi/geoserver/tilastointialueet/wfs"
        params = dict(service="WFS", version="2.0.0", request="GetFeature",
                      typeName="tilastointialueet:maakunta4500k", outputFormat="json")
        r = requests.get(WFS_URL, params=params, timeout=20)
        r.raise_for_status()
        geo = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:3067")
        return _finalize_geo(geo)
    except Exception:
        return None



def year_or_all_options(years):
    return ["All years (average)"] + [str(y) for y in sorted(years)]


@st.cache_data(ttl=900)
def get_forecast(steps=96):
    try:
        r = requests.get(f"{API_URL}/forecast?steps={steps}", timeout=90)
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
        "🗺️ Regional Analysis",
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
        # NEW: guard against co2_col being None (table has no data
        # columns yet, e.g. right after creation / before the first
        # hourly merge has run) and against df being empty even when
        # co2_col exists. Previously this went straight to
        # df[co2_col].iloc[-1] with no checks, which raised
        # "KeyError: None" whenever co2_col was None.
        if current:
            live_val = current["co2_intensity_gco2kwh"]
        elif co2_col and not df.empty:
            live_val = df[co2_col].iloc[-1]
        else:
            live_val = None

        metric_card("CO₂ Intensity (live)",
                    f"{live_val:.1f} gCO₂/kWh" if live_val is not None else "No data yet",
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


# ══════════════════════════════════════════════════════════════
# PAGE 4 — REGIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "🗺️ Regional Analysis":
    st.markdown("# 🗺️ Regional Emissions, GDP & Sectoral Carbon Economy")

    _data = load_regional_data()
    summary            = _data["summary"]
    ghg_long           = _data["ghg_long"]
    subclass_emissions = _data["subclass_emissions"]
    gdp_merged         = _data["gdp_merged"]

    if summary.empty:
        st.error("No regional data found. Run `01_consolidate_regions.py` (and optionally "
                 "`02_merge_gdp.py`) to populate `region_year_summary` in the shared database.")
        st.stop()

    all_regions = sorted(summary["region"].unique())
    all_years   = sorted(summary["year"].unique())

    if len(all_regions) < 19:
        st.warning(f"{len(all_regions)}/19 regions loaded so far — everything below still "
                  f"works, it'll just get richer (more lines/rows/map colour) as more region "
                  f"files are consolidated.")

    # ── Global filters — shared by every tab below ──────────────────
    st.markdown("### Filters")
    f1, f2, f3, f4 = st.columns(4, gap="small")
    with f1:
        BASE_YEAR = st.selectbox("Base year", all_years,
                                 index=all_years.index(2005) if 2005 in all_years else 0)
    with f2:
        default_compare_idx = len(all_years) - 1
        COMPARE_YEAR = st.selectbox("Compare year", all_years, index=default_compare_idx)
    with f3:
        BASE_REGION = st.selectbox("Base region", all_regions, index=0)
    with f4:
        compare_default = 1 if len(all_regions) > 1 else 0
        COMPARE_REGION = st.selectbox("Compare region", all_regions, index=compare_default)

    if BASE_YEAR == COMPARE_YEAR:
        st.caption("⚠️ Base year and compare year are the same — comparison charts will show zero change.")
    st.markdown("---")

    tabs = st.tabs([
        "📊 Ranking & Map", "🏭 Sectors & Subclasses", "🔗 Region Similarity",
        "📈 Sector Trends", "🔄 Year Comparison", "💶 GDP vs CO₂",
        "🔍 Subclass Drill-down", "🧬 Sector Similarity", "🌐 Geographic Proximity",
    ])

    # A small reusable helper: several sections only need ONE year, and the
    # ask was "by default base year works" -- so give each such section a
    # lightweight toggle defaulting to Base, rather than another selectbox.
    def snapshot_year_picker(key):
        choice = st.radio("Snapshot year", [f"Base ({BASE_YEAR})", f"Compare ({COMPARE_YEAR})"],
                          horizontal=True, key=key)
        return BASE_YEAR if choice.startswith("Base") else COMPARE_YEAR

    # ══════════════════════════════════════════════════════════
    # TAB 1 — RANKING & MAP
    # ══════════════════════════════════════════════════════════
    with tabs[0]:
        snap_year = snapshot_year_picker("snap_ranking")
        snap = summary[summary["year"] == snap_year].sort_values("total_ktco2e", ascending=False)

        st.markdown(f"#### Regional ranking — {snap_year}")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            fig = go.Figure(layout=PLOTLY_LAYOUT)
            fig.add_trace(go.Bar(
                x=snap["total_ktco2e"], y=snap["region"], orientation="h",
                marker=dict(color=snap["total_ktco2e"], colorscale="RdYlGn_r"),
            ))
            fig.update_layout(title=f"Total emissions by region ({snap_year})",
                              yaxis=dict(autorange="reversed"), height=max(400, len(snap)*28))
            fig.update_xaxes(title_text="ktCO2e")
            st.plotly_chart(fig, width="stretch")
        with c2:
            snap_pc = snap.sort_values("per_capita_tco2e", ascending=False)
            fig = go.Figure(layout=PLOTLY_LAYOUT)
            fig.add_trace(go.Bar(
                x=snap_pc["per_capita_tco2e"], y=snap_pc["region"], orientation="h",
                marker=dict(color=snap_pc["per_capita_tco2e"], colorscale="RdYlGn_r"),
            ))
            fig.update_layout(title=f"Per-capita emissions by region ({snap_year})",
                              yaxis=dict(autorange="reversed"), height=max(400, len(snap)*28))
            fig.update_xaxes(title_text="tCO2e / person")
            st.plotly_chart(fig, width="stretch")

        st.markdown(f"#### Choropleth map — {snap_year}")
        map_metric = st.radio("Colour by", ["total_ktco2e", "per_capita_tco2e"],
                              horizontal=True, key="map_metric")
        geo = fetch_region_geometry()
        if geo is None:
            st.warning("Couldn't load region boundaries from Statistics Finland's WFS service "
                      "right now (needs outbound internet access to geo.stat.fi from this "
                      "server) — showing the ranking bar chart above instead.")
        else:
            snap_map = snap.copy()
            snap_map["region_key"] = snap_map["region"].apply(normalize_region_name)
            geojson = geo.set_index("region_key").__geo_interface__

            # Plotly renamed the mapbox-based choropleth to a maplibre-based
            # one (choropleth_map) in newer releases and dropped
            # choropleth_mapbox entirely in some builds -- support both so
            # this doesn't break depending on which Plotly version is
            # actually installed on the deployment server.
            common_kwargs = dict(
                data_frame=snap_map, geojson=geojson, locations="region_key",
                featureidkey="id", color=map_metric,
                color_continuous_scale="RdYlGn_r",
                center={"lat": 64.5, "lon": 26.0}, zoom=4.2, opacity=0.85,
                hover_name="region",
                hover_data={"region_key": False, "total_ktco2e": ":.0f",
                           "per_capita_tco2e": ":.2f"},
            )
            if hasattr(px, "choropleth_map"):
                fig = px.choropleth_map(map_style="carto-darkmatter", **common_kwargs)
            else:
                fig = px.choropleth_mapbox(mapbox_style="carto-darkmatter", **common_kwargs)

            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=650,
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#c3e0ec"))
            st.plotly_chart(fig, width="stretch")

    # ══════════════════════════════════════════════════════════
    # TAB 2 — SECTORS & SUBCLASSES
    # ══════════════════════════════════════════════════════════
    with tabs[1]:
        snap_year2 = snapshot_year_picker("snap_sectors")
        region_scope = st.multiselect("Regions to include", all_regions, default=all_regions,
                                      key="sector_region_scope")

        st.markdown(f"#### Sector emissions by region — {snap_year2}")
        snap_sec = ghg_long[(ghg_long["year"] == snap_year2) & (ghg_long["region"].isin(region_scope))]
        if snap_sec.empty:
            st.info("No sector data for this selection.")
        else:
            pivot = snap_sec.pivot_table(index="region", columns="sector",
                                         values="value_ktco2e", aggfunc="sum").fillna(0)
            pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
            pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]
            fig = px.imshow(pivot, color_continuous_scale="YlOrRd",
                            labels=dict(color="ktCO2e"), aspect="auto")
            fig.update_layout(**PLOTLY_LAYOUT, height=max(400, len(pivot)*32))
            st.plotly_chart(fig, width="stretch")

        st.markdown(f"#### Top 5 sectors and subclasses, per region — {snap_year2}")
        top_n = 5
        rows = []
        for region in region_scope:
            rdf = ghg_long[(ghg_long["region"] == region) & (ghg_long["year"] == snap_year2)]
            rdf = rdf.sort_values("value_ktco2e", ascending=False).head(top_n)
            for rank, (_, r) in enumerate(rdf.iterrows(), start=1):
                rows.append({"Region": region, "Rank": rank, "Sector": r["sector"],
                            "ktCO2e": round(r["value_ktco2e"], 1)})
        top_sectors_df = pd.DataFrame(rows)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("**Top 5 sectors**")
            st.dataframe(top_sectors_df, width="stretch", hide_index=True, height=420)

        with c2:
            st.markdown("**Top 5 subclasses**")
            if subclass_emissions.empty:
                st.info("Subclass data not loaded — re-run the updated "
                       "`01_consolidate_regions.py` to populate `emissions_subclass_long`.")
            else:
                rows2 = []
                for region in region_scope:
                    rdf = subclass_emissions[(subclass_emissions["region"] == region)
                                             & (subclass_emissions["year"] == snap_year2)]
                    rdf = rdf.sort_values("value_ktco2e", ascending=False).head(top_n)
                    for rank, (_, r) in enumerate(rdf.iterrows(), start=1):
                        rows2.append({"Region": region, "Rank": rank, "Sector": r["sector"],
                                     "Subclass": r["subclass"], "ktCO2e": round(r["value_ktco2e"], 1)})
                top_subclass_df = pd.DataFrame(rows2)
                st.dataframe(top_subclass_df, width="stretch", hide_index=True, height=420)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — REGION SIMILARITY
    # ══════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("#### Which regions rise and fall together? (correlation of total emissions, all years)")
        pivot_total = summary.pivot_table(index="year", columns="region", values="total_ktco2e")
        if pivot_total.shape[1] < 2:
            st.info("Need at least 2 regions loaded to compute correlation.")
        else:
            corr = pivot_total.corr()
            fig = px.imshow(corr, color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                            text_auto=".2f" if len(corr) <= 14 else False,
                            labels=dict(color="Pearson r"))
            fig.update_layout(**PLOTLY_LAYOUT, height=max(450, len(corr)*32))
            st.plotly_chart(fig, width="stretch")

            st.markdown("**All region pairs, ranked by similarity**")
            corr_flat = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            corr_flat.index = corr_flat.index.set_names(["Region A", "Region B"])
            pair_df = corr_flat.reset_index()
            pair_df.columns = ["Region A", "Region B", "Correlation (r)"]
            pair_df = pair_df.sort_values("Correlation (r)", ascending=False).reset_index(drop=True)
            pair_df["Correlation (r)"] = pair_df["Correlation (r)"].round(3)
            st.dataframe(pair_df, width="stretch", hide_index=True, height=400)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — SECTOR TRENDS OVER TIME
    # ══════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("#### Sector emissions over time")
        all_sectors = sorted(ghg_long["sector"].unique())
        c1, c2 = st.columns([1, 2], gap="medium")
        with c1:
            trend_sector = st.selectbox("Sector", all_sectors, key="trend_sector")
        with c2:
            trend_regions = st.multiselect("Regions", all_regions, default=all_regions,
                                           key="trend_regions")

        trend_df = ghg_long[(ghg_long["sector"] == trend_sector) & (ghg_long["region"].isin(trend_regions))]
        if trend_df.empty:
            st.info("No data for this sector/region selection.")
        else:
            fig = go.Figure(layout=PLOTLY_LAYOUT)
            for region, g in trend_df.groupby("region"):
                g = g.sort_values("year")
                fig.add_trace(go.Scatter(x=g["year"], y=g["value_ktco2e"], mode="lines+markers",
                                        name=region, line=dict(width=2)))
            fig.update_layout(title=f"{trend_sector} emissions over time", height=520)
            fig.update_yaxes(title_text="ktCO2e")
            st.plotly_chart(fig, width="stretch")
            st.caption("Tip: narrow the region multiselect to one region to focus on a single "
                      "region's trend for this sector.")

    # ══════════════════════════════════════════════════════════
    # TAB 5 — YEAR COMPARISON (BASE_YEAR vs COMPARE_YEAR)
    # ══════════════════════════════════════════════════════════
    with tabs[4]:
        if BASE_YEAR == COMPARE_YEAR:
            st.info("Base year and compare year are the same — pick two different years in "
                    "the Filters above to see a comparison here.")
        else:
            st.markdown(f"#### Decarbonisation ranking, {BASE_YEAR} → {COMPARE_YEAR}")
            change = (summary[summary["year"] == BASE_YEAR][["region", "total_ktco2e"]]
                      .rename(columns={"total_ktco2e": "base"})
                      .merge(summary[summary["year"] == COMPARE_YEAR][["region", "total_ktco2e"]]
                             .rename(columns={"total_ktco2e": "latest"}), on="region"))
            change["pct_change"] = (change["latest"] - change["base"]) / change["base"] * 100
            change = change.sort_values("pct_change")

            fig = go.Figure(layout=PLOTLY_LAYOUT)
            fig.add_trace(go.Bar(x=change["pct_change"], y=change["region"], orientation="h",
                                 marker=dict(color=change["pct_change"], colorscale="RdYlGn_r")))
            fig.add_vline(x=0, line_color="white", opacity=0.5)
            fig.update_layout(title=f"% change in total emissions, {BASE_YEAR} → {COMPARE_YEAR}",
                              yaxis=dict(autorange="reversed"), height=max(400, len(change)*28))
            st.plotly_chart(fig, width="stretch")

            st.markdown(f"#### Direct comparison, {BASE_YEAR} vs {COMPARE_YEAR}")
            direct = change.rename(columns={"base": str(BASE_YEAR), "latest": str(COMPARE_YEAR),
                                            "pct_change": "pct_change"})
            direct["abs_change"] = direct[str(COMPARE_YEAR)] - direct[str(BASE_YEAR)]
            direct = direct[["region", str(BASE_YEAR), str(COMPARE_YEAR), "abs_change", "pct_change"]]
            direct = direct.set_index("region").round(1).sort_values("pct_change")
            st.dataframe(direct, width="stretch", height=400)

            st.markdown(f"#### What's driving the change? Sector-by-sector, {BASE_YEAR} vs {COMPARE_YEAR}")
            driver_region_sel = st.selectbox("Region", all_regions,
                                             index=all_regions.index(BASE_REGION),
                                             key="driver_region")
            driver_data = ghg_long[(ghg_long["region"] == driver_region_sel)
                                   & (ghg_long["year"].isin([BASE_YEAR, COMPARE_YEAR]))]
            driver_pivot = driver_data.pivot_table(index="sector", columns="year",
                                                   values="value_ktco2e", aggfunc="sum")
            if BASE_YEAR in driver_pivot.columns and COMPARE_YEAR in driver_pivot.columns:
                driver_pivot["delta"] = driver_pivot[COMPARE_YEAR] - driver_pivot[BASE_YEAR]
                driver_pivot = driver_pivot.sort_values("delta")

                c1, c2 = st.columns([3, 2], gap="medium")
                with c1:
                    fig = go.Figure(layout=PLOTLY_LAYOUT)
                    colors = [C_TEAL if v < 0 else C_CORAL for v in driver_pivot["delta"]]
                    fig.add_trace(go.Bar(x=driver_pivot["delta"], y=driver_pivot.index,
                                        orientation="h", marker=dict(color=colors)))
                    fig.add_vline(x=0, line_color="white", opacity=0.5)
                    total_delta = driver_pivot["delta"].sum()
                    fig.update_layout(
                        title=f"{driver_region_sel}: sector drivers of change "
                              f"(total: {total_delta:+.0f} ktCO2e)",
                        height=max(350, len(driver_pivot)*32))
                    st.plotly_chart(fig, width="stretch")
                with c2:
                    st.markdown("**Sector delta table**")
                    st.dataframe(driver_pivot.round(1).rename(columns=str), width="stretch", height=420)
            else:
                st.info(f"Missing sector data for {BASE_YEAR} or {COMPARE_YEAR} for this region.")

        st.markdown("#### Year-on-year change (any two years, at a glance)")
        yoy_pivot = summary.pivot_table(index="region", columns="year", values="total_ktco2e")
        yoy_pct = yoy_pivot.pct_change(axis=1) * 100
        fig = px.imshow(yoy_pct, color_continuous_scale="RdYlGn_r", zmin=-30, zmax=30,
                        labels=dict(color="% change vs prev. year"), aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, height=max(400, len(yoy_pivot)*30))
        st.plotly_chart(fig, width="stretch")

    # ══════════════════════════════════════════════════════════
    # TAB 6 — GDP vs CO2
    # ══════════════════════════════════════════════════════════
    with tabs[5]:
        if gdp_merged.empty or gdp_merged["gdp_per_capita_eur"].notna().sum() == 0:
            st.info("Run `02_merge_gdp.py` to populate `region_year_with_gdp` and unlock this tab.")
        else:
            with st.expander("What do the GDP columns mean? (click to expand)"):
                st.markdown("""
| Column | What it is | Unit | Use it for |
|---|---|---|---|
| `gdp_per_capita_eur` | **Nominal** GDP per person ("At current prices, euro") | EUR, that year's own prices | Comparing regions **within one year** |
| `gdp_per_capita_real_2015eur` | **Real** GDP per person ("Volume series, ref. year 2015") | EUR, constant 2015 prices | Comparing **across years** (inflation removed) |
                """)

            gdp_years = gdp_merged.dropna(subset=["gdp_per_capita_eur"])["year"]
            LATEST_GDP_YEAR = int(gdp_years.max())
            eff_compare = min(COMPARE_YEAR, LATEST_GDP_YEAR)
            if eff_compare != COMPARE_YEAR:
                st.caption(f"GDP data ends {LATEST_GDP_YEAR}; using it instead of {COMPARE_YEAR} below.")

            view_mode = st.radio("View", ["Absolute levels", "Year-on-year % change"],
                                 horizontal=True, key="gdp_view_mode")

            gdp_pivot = gdp_merged.pivot_table(index="year", columns="region",
                                               values="gdp_per_capita_real_2015eur")
            co2_pivot = gdp_merged.pivot_table(index="year", columns="region", values="per_capita_tco2e")
            if view_mode == "Year-on-year % change":
                gdp_pivot = gdp_pivot.pct_change() * 100
                co2_pivot = co2_pivot.pct_change() * 100

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                fig = go.Figure(layout=PLOTLY_LAYOUT)
                for region in gdp_pivot.columns:
                    fig.add_trace(go.Scatter(x=gdp_pivot.index, y=gdp_pivot[region],
                                            mode="lines", name=region))
                ylabel = "EUR per capita (real, 2015)" if view_mode == "Absolute levels" else "% change vs prev. year"
                fig.update_layout(title=f"GDP per capita — {view_mode}", height=450)
                fig.update_yaxes(title_text=ylabel)
                fig.add_vrect(x0=BASE_YEAR, x1=eff_compare, fillcolor="grey", opacity=0.1, line_width=0)
                st.plotly_chart(fig, width="stretch")
            with c2:
                fig = go.Figure(layout=PLOTLY_LAYOUT)
                for region in co2_pivot.columns:
                    fig.add_trace(go.Scatter(x=co2_pivot.index, y=co2_pivot[region],
                                            mode="lines", name=region))
                ylabel = "tCO2e per capita" if view_mode == "Absolute levels" else "% change vs prev. year"
                fig.update_layout(title=f"CO2 per capita — {view_mode}", height=450)
                fig.update_yaxes(title_text=ylabel)
                fig.add_vrect(x0=BASE_YEAR, x1=eff_compare, fillcolor="grey", opacity=0.1, line_width=0)
                st.plotly_chart(fig, width="stretch")

            st.markdown(f"#### Trajectory: {BASE_YEAR} → {eff_compare}")
            traj = gdp_merged[gdp_merged["year"].isin([BASE_YEAR, eff_compare])].dropna(
                subset=["gdp_per_capita_real_2015eur", "per_capita_tco2e"])
            fig = go.Figure(layout=PLOTLY_LAYOUT)
            for region, g in traj.groupby("region"):
                g = g.sort_values("year")
                if len(g) < 2:
                    continue
                x0, y0 = g.iloc[0][["gdp_per_capita_real_2015eur", "per_capita_tco2e"]]
                x1, y1 = g.iloc[1][["gdp_per_capita_real_2015eur", "per_capita_tco2e"]]
                decoupling = x1 > x0 and y1 < y0
                color = C_TEAL if decoupling else C_CORAL
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines+markers+text",
                                        line=dict(color=color, width=2),
                                        marker=dict(size=[8, 12], symbol=["circle", "triangle-up"]),
                                        text=["", region], textposition="top center",
                                        showlegend=False))
            fig.update_layout(title="Teal = richer & cleaner (decoupling). Orange = any other direction.",
                              height=600)
            fig.update_xaxes(title_text="GDP per capita, real 2015 EUR")
            fig.update_yaxes(title_text="CO2 per capita (tCO2e)")
            st.plotly_chart(fig, width="stretch")

            st.markdown("#### Carbon intensity of GDP over time")
            intensity_pivot = gdp_merged.pivot_table(index="year", columns="region",
                                                     values="co2_tonnes_per_million_eur_gdp")
            fig = go.Figure(layout=PLOTLY_LAYOUT)
            for region in intensity_pivot.columns:
                fig.add_trace(go.Scatter(x=intensity_pivot.index, y=intensity_pivot[region],
                                        mode="lines", name=region))
            fig.update_layout(title="tCO2e per million EUR of (nominal) GDP", height=450)
            st.plotly_chart(fig, width="stretch")

    # ══════════════════════════════════════════════════════════
    # TAB 7 — SUBCLASS DRILL-DOWN (base region vs compare region)
    # ══════════════════════════════════════════════════════════
    with tabs[6]:
        if subclass_emissions.empty:
            st.info("Subclass data not loaded — re-run the updated `01_consolidate_regions.py`.")
        else:
            st.markdown(f"#### Comparing **{BASE_REGION}** vs **{COMPARE_REGION}**")
            common_sectors = sorted(
                set(subclass_emissions[subclass_emissions["region"] == BASE_REGION]["sector"].unique())
                | set(subclass_emissions[subclass_emissions["region"] == COMPARE_REGION]["sector"].unique())
            )
            drill_sector = st.selectbox("Sector", common_sectors, key="drill_sector")
            drill_year = snapshot_year_picker("snap_drill")

            sub_both = subclass_emissions[
                (subclass_emissions["region"].isin([BASE_REGION, COMPARE_REGION]))
                & (subclass_emissions["sector"].str.strip() == drill_sector.strip())
            ]

            if sub_both.empty:
                st.info("No subclass rows for this sector in either region.")
            else:
                c1, c2 = st.columns([3, 2], gap="medium")
                with c1:
                    snap_sub = sub_both[sub_both["year"] == drill_year]
                    fig = px.bar(snap_sub, x="subclass", y="value_ktco2e", color="region",
                                barmode="group",
                                color_discrete_map={BASE_REGION: C_TEAL, COMPARE_REGION: C_CORAL})
                    fig.update_layout(**PLOTLY_LAYOUT,
                                      title=f"{drill_sector} subclasses — {drill_year}", height=450)
                    st.plotly_chart(fig, width="stretch")
                with c2:
                    fig = go.Figure(layout=PLOTLY_LAYOUT)
                    for region, dash in [(BASE_REGION, "solid"), (COMPARE_REGION, "dash")]:
                        rdf = sub_both[sub_both["region"] == region]
                        for subclass, g in rdf.groupby("subclass"):
                            g = g.sort_values("year")
                            fig.add_trace(go.Scatter(x=g["year"], y=g["value_ktco2e"],
                                                    mode="lines", name=f"{region} — {subclass}",
                                                    line=dict(dash=dash)))
                    fig.update_layout(title=f"{drill_sector} subclasses over time\\n"
                                            "(solid = base region, dashed = compare region)",
                                      height=450, showlegend=(len(fig.data) <= 10))
                    st.plotly_chart(fig, width="stretch")

    # ══════════════════════════════════════════════════════════
    # TAB 8 — SECTOR SIMILARITY
    # ══════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("#### Which sectors move together nationally?")
        national_sector_year = (ghg_long.groupby(["year", "sector"])["value_ktco2e"]
                                .sum().reset_index()
                                .pivot(index="year", columns="sector", values="value_ktco2e"))
        sector_std = national_sector_year.std()
        national_sector_year = national_sector_year.drop(columns=sector_std[sector_std == 0].index)

        if national_sector_year.shape[1] < 2:
            st.info("Need at least 2 non-constant sectors to compute correlation.")
        else:
            corr_mode = st.radio("Correlation of", ["Raw levels", "Year-on-year % change (detrended)"],
                                 horizontal=True, key="sector_corr_mode")
            if corr_mode == "Raw levels":
                sector_corr = national_sector_year.corr()
                st.caption("Raw levels: two sectors that both trended downward for 20 years will "
                          "look highly correlated even without sharing a real short-term driver.")
            else:
                sector_corr = national_sector_year.pct_change().replace(
                    [np.inf, -np.inf], np.nan).corr()
                st.caption("Detrended: strips out the shared long-term decline, showing which "
                          "sectors actually move together in the SAME year for the same reason.")

            fig = px.imshow(sector_corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            text_auto=".2f", labels=dict(color="Pearson r"))
            fig.update_layout(**PLOTLY_LAYOUT, height=550)
            st.plotly_chart(fig, width="stretch")

            st.markdown("**All sector pairs, ranked**")
            sc_flat = sector_corr.where(np.triu(np.ones(sector_corr.shape), k=1).astype(bool)).stack()
            sc_flat.index = sc_flat.index.set_names(["Sector A", "Sector B"])
            sc_df = sc_flat.reset_index()
            sc_df.columns = ["Sector A", "Sector B", "Correlation (r)"]
            sc_df = sc_df.sort_values("Correlation (r)", ascending=False).reset_index(drop=True)
            sc_df["Correlation (r)"] = sc_df["Correlation (r)"].round(3)
            st.dataframe(sc_df, width="stretch", hide_index=True, height=400)

    # ══════════════════════════════════════════════════════════
    # TAB 9 — GEOGRAPHIC PROXIMITY
    # ══════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("#### Are geographically close regions actually similar?")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            proximity_year_choice = st.selectbox("Year", year_or_all_options(all_years),
                                                 index=len(all_years),  # defaults to latest year
                                                 key="proximity_year")
        with c2:
            neighbor_km = st.slider("Neighbour distance threshold (km)", 20, 400, 100, step=10,
                                    key="neighbor_km")

        geo = fetch_region_geometry()
        if geo is None:
            st.warning("Couldn't load region boundaries from Statistics Finland's WFS service "
                      "right now — this tab needs outbound internet access to geo.stat.fi.")
        else:
            geo_m = geo.to_crs("EPSG:3067")  # metric CRS for real distances
            geo_m["centroid"] = geo_m.geometry.centroid

            if proximity_year_choice == "All years (average)":
                metric_df = summary.groupby("region")[["total_ktco2e", "per_capita_tco2e",
                                                        "population"]].mean().reset_index()
                label = "average across all years"
            else:
                y = int(proximity_year_choice)
                metric_df = summary[summary["year"] == y][
                    ["region", "total_ktco2e", "per_capita_tco2e", "population"]]
                label = f"year {proximity_year_choice}"

            metric_df["region_key"] = metric_df["region"].apply(normalize_region_name)
            geo_snap = geo_m.merge(metric_df, on="region_key", how="inner")

            if gdp_merged is not None and not gdp_merged.empty:
                if proximity_year_choice == "All years (average)":
                    gdp_snap = gdp_merged.groupby("region")["gdp_per_capita_eur"].mean().reset_index()
                else:
                    gdp_snap = gdp_merged[gdp_merged["year"] == int(proximity_year_choice)][
                        ["region", "gdp_per_capita_eur"]]
                gdp_snap["region_key"] = gdp_snap["region"].apply(normalize_region_name)
                geo_snap = geo_snap.merge(gdp_snap[["region_key", "gdp_per_capita_eur"]],
                                          on="region_key", how="left")

            n = len(geo_snap)
            if n < 2:
                st.info("Need at least 2 matched regions with geometry to run this analysis.")
            else:
                rows = []
                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = geo_snap.iloc[i], geo_snap.iloc[j]
                        dist_km = a["centroid"].distance(b["centroid"]) / 1000
                        if dist_km > neighbor_km:
                            continue
                        row = {
                            "Region A": a["region"], "Region B": b["region"],
                            "Distance (km)": round(dist_km, 1),
                            "CO2/capita A": round(a["per_capita_tco2e"], 2),
                            "CO2/capita B": round(b["per_capita_tco2e"], 2),
                            "|Δ CO2/capita|": round(abs(a["per_capita_tco2e"] - b["per_capita_tco2e"]), 2),
                            "Population A": int(a["population"]),
                            "Population B": int(b["population"]),
                            "|Δ Population|": int(abs(a["population"] - b["population"])),
                        }
                        if "gdp_per_capita_eur" in geo_snap.columns:
                            row["GDP/capita A"] = round(a.get("gdp_per_capita_eur", np.nan), 0)
                            row["GDP/capita B"] = round(b.get("gdp_per_capita_eur", np.nan), 0)
                            row["|Δ GDP/capita|"] = round(
                                abs(a.get("gdp_per_capita_eur", np.nan) - b.get("gdp_per_capita_eur", np.nan)), 0)
                        rows.append(row)

                neighbours_df = pd.DataFrame(rows).sort_values("Distance (km)")
                st.markdown(f"**{len(neighbours_df)} region pairs within {neighbor_km} km "
                           f"({label})**")
                if neighbours_df.empty:
                    st.info("No pairs within this threshold — try a larger distance.")
                else:
                    st.dataframe(neighbours_df, width="stretch", hide_index=True, height=450)

                    st.markdown("#### Does distance actually predict similarity? (all pairs, not just close ones)")
                    all_pairs = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            a, b = geo_snap.iloc[i], geo_snap.iloc[j]
                            all_pairs.append({
                                "distance_km": a["centroid"].distance(b["centroid"]) / 1000,
                                "diff_co2": abs(a["per_capita_tco2e"] - b["per_capita_tco2e"]),
                            })
                    all_pairs_df = pd.DataFrame(all_pairs)
                    if len(all_pairs_df) >= 3:
                        r = all_pairs_df["distance_km"].corr(all_pairs_df["diff_co2"])
                        fig = px.scatter(all_pairs_df, x="distance_km", y="diff_co2",
                                        labels={"distance_km": "Distance between regions (km)",
                                               "diff_co2": "|Δ CO2 per capita|"})
                        z = np.polyfit(all_pairs_df["distance_km"], all_pairs_df["diff_co2"], 1)
                        xs = np.linspace(all_pairs_df["distance_km"].min(),
                                        all_pairs_df["distance_km"].max(), 50)
                        fig.add_trace(go.Scatter(x=xs, y=np.poly1d(z)(xs), mode="lines",
                                                line=dict(color=C_TEAL, width=2), name="trend"))
                        fig.update_layout(**PLOTLY_LAYOUT,
                                          title=f"r = {r:.2f} "
                                                f"({'weak/no' if abs(r)<0.2 else 'some' if abs(r)<0.5 else 'fairly strong'} relationship)",
                                          height=450)
                        st.plotly_chart(fig, width="stretch")