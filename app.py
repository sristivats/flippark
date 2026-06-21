"""
ParkSense AI — Bengaluru Parking Violation Intelligence Dashboard
================================================================
RUN:
    pip install streamlit pandas numpy plotly folium streamlit-folium h3 xgboost
    streamlit run app.py

DATA FILES (place in same folder as app.py — all from your Kaggle /kaggle/working/):
    violations_geocoded.csv(.gz)      — raw enriched records
    enforcement_ranking_with_cost.csv — Phase 4 delay-cost ranked table
    h3_hexagons_with_cii.csv          — hex-level CII scores
    dbscan_hotspot_clusters.csv       — Phase 2 cluster output
    phase5_alert_log.csv              — Phase 5 alert queue
    eda_hourly.csv / eda_daily.csv / eda_monthly.csv
    eda_station_load.csv / eda_vehicle_risk.csv
    xgb_violation_forecast_model.json (optional — enables live forecasts)
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import HeatMap, MarkerCluster, MiniMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = "."

def fp(name):
    return os.path.join(DATA_DIR, name)

# ── Colour tokens ─────────────────────────────────────────────────────────────
C = {
    "bg":       "#0a0f1e",
    "surface":  "#111827",
    "card":     "#1a2235",
    "border":   "#1f2d42",
    "accent1":  "#f43f5e",   # rose – critical
    "accent2":  "#f97316",   # orange – warning
    "accent3":  "#3b82f6",   # blue – info
    "accent4":  "#10b981",   # emerald – safe
    "accent5":  "#a78bfa",   # violet – forecast
    "text":     "#f1f5f9",
    "muted":    "#64748b",
    "dim":      "#334155",
}

TIER_HEX = {
    "SEVERE":   C["accent1"], "CRITICAL": C["accent1"],
    "HIGH":     C["accent2"],
    "MODERATE": "#eab308",   "MEDIUM":   "#eab308",
    "LOW":      C["accent4"],
}

SEVERITY_HEX = {
    "CRITICAL": C["accent1"],
    "HIGH":     C["accent2"],
    "MEDIUM":   "#eab308",
    "LOW":      C["accent4"],
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ParkSense AI — Bengaluru",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: {C['bg']};
    color: {C['text']};
}}

/* ── sidebar ── */
section[data-testid="stSidebar"] {{
    background: {C['surface']} !important;
    border-right: 1px solid {C['border']};
}}
section[data-testid="stSidebar"] * {{ color: {C['text']} !important; }}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {{
    background: {C['card']} !important;
    border-color: {C['border']} !important;
    color: {C['text']} !important;
}}

/* ── main area ── */
.main .block-container {{ padding: 1.5rem 2rem 2rem; max-width: 1400px; }}
[data-testid="stAppViewContainer"] {{ background: {C['bg']}; }}

/* ── strip default Streamlit chrome that otherwise shows as blank boxes:
     column wrappers, vertical blocks, widget containers, and empty
     placeholder elements ── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="column"],
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stElementContainer"]:empty,
[data-testid="stVerticalBlockBorderWrapper"]:empty,
div.element-container:empty {{
    display: none !important;
}}

/* ── native widgets in the MAIN area (radio, text input, slider) ── */
.main .stRadio > div {{ gap: 6px; }}
.main .stTextInput > div > div,
.main .stSelectbox > div > div,
.main .stDateInput > div > div {{
    background: {C['card']} !important;
    border: 1px solid {C['border']} !important;
    color: {C['text']} !important;
}}
.main input, .main textarea {{
    background: {C['card']} !important;
    color: {C['text']} !important;
}}
.main .stSlider [data-baseweb="slider"] {{ background: transparent !important; }}

/* ── hero banner ── */
.ps-hero {{
    background: linear-gradient(135deg, {C['surface']} 0%, #0d1b2e 100%);
    border: 1px solid {C['border']};
    border-left: 4px solid {C['accent1']};
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    display: flex; align-items: center; gap: 18px;
}}
.ps-hero-icon {{ font-size: 36px; }}
.ps-hero h1 {{
    font-size: 20px; font-weight: 700; color: {C['text']};
    margin: 0; letter-spacing: -0.02em;
}}
.ps-hero p {{ font-size: 12px; color: {C['muted']}; margin: 4px 0 0;
    font-family: 'JetBrains Mono', monospace; }}

/* ── page title ── */
.ps-page-title {{
    font-size: 15px; font-weight: 600; color: {C['text']};
    letter-spacing: -0.01em; margin-bottom: 4px;
}}
.ps-page-sub {{ font-size: 12px; color: {C['muted']}; margin-bottom: 20px; }}

/* ── KPI cards ── */
.kpi-row {{ display: flex; gap: 14px; margin-bottom: 22px; flex-wrap: wrap; }}
.kpi {{
    flex: 1; min-width: 160px;
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 16px 20px;
    border-top: 3px solid var(--ac);
    position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}}
.kpi::before {{
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(135deg, var(--ac) 0%, transparent 60%);
    opacity: 0.04;
}}
.kpi .kv {{
    font-size: 28px; font-weight: 700;
    color: var(--ac);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
}}
.kpi .kl {{
    font-size: 11px; color: {C['muted']};
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-top: 6px;
}}
.kpi .ks {{ font-size: 11px; color: {C['dim']}; margin-top: 2px; }}

/* ── section card ── */
.ps-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}}
.ps-card-title {{
    font-size: 12px; font-weight: 600;
    color: {C['muted']};
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 16px;
}}

/* ── tier / severity badges ── */
.badge {{
    display: inline-block; padding: 2px 9px;
    border-radius: 20px; font-size: 11px; font-weight: 600;
    letter-spacing: 0.03em;
}}
.badge-SEVERE,.badge-CRITICAL {{ background:#4d0a14; color:{C['accent1']}; }}
.badge-HIGH    {{ background:#4d2106; color:{C['accent2']}; }}
.badge-MODERATE,.badge-MEDIUM {{ background:#3d3000; color:#eab308; }}
.badge-LOW     {{ background:#052e16; color:{C['accent4']}; }}
.badge-RISING  {{ background:#4d0a14; color:{C['accent1']}; }}
.badge-FALLING {{ background:#0c1a4d; color:{C['accent3']}; }}
.badge-STABLE  {{ background:{C['border']}; color:{C['muted']}; }}
.badge-WATCH   {{ background:#3d3000; color:#eab308; }}
.badge-INFO    {{ background:#0c1a4d; color:{C['accent3']}; }}

/* ── alert cards ── */
.al-card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-left: 4px solid var(--al);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
    position: relative;
}}
.al-title {{ font-size: 13px; font-weight: 600; color: {C['text']}; margin-bottom: 4px; }}
.al-meta  {{ font-size: 11px; color: {C['muted']}; line-height: 1.6; }}
.al-cost  {{ font-size: 12px; color: var(--al); font-family: 'JetBrains Mono', monospace;
             margin-top: 6px; font-weight: 600; }}

/* ── route stop ── */
.route-stop {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
    min-height: 105px;
}}
.stop-num {{
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--sn); color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; flex-shrink: 0;
}}
.stop-body b {{ font-size: 13px; color: {C['text']}; }}
.stop-body span {{ font-size: 11px; color: {C['muted']}; }}

/* ── table ── */
.ps-tbl {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
.ps-tbl th {{
    background: {C['surface']}; padding: 8px 10px;
    text-align: left; font-weight: 600; color: {C['muted']};
    border-bottom: 1px solid {C['border']};
    text-transform: uppercase; font-size: 11px; letter-spacing: 0.04em;
}}
.ps-tbl td {{ padding: 8px 10px; border-bottom: 1px solid {C['border']}; color: {C['text']}; }}
.ps-tbl tr:hover td {{ background: {C['surface']}; }}

/* ── misc ── */
hr.ps-rule {{ border: none; border-top: 1px solid {C['border']}; margin: 18px 0; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    d = {}

    # ── EDA tables ────────────────────────────────────────────────────────────
    for k, f in [("hourly","eda_hourly.csv"), ("daily","eda_daily.csv"),
                 ("monthly","eda_monthly.csv"), ("station","eda_station_load.csv"),
                 ("vehicle","eda_vehicle_risk.csv"), ("alerts","phase5_alert_log.csv"),
                 ("hexdf","h3_hexagons_with_cii.csv"),
                 ("clusters","dbscan_hotspot_clusters.csv")]:
        p = fp(f)
        d[k] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    # ── Main cost-ranked table (richest file) ─────────────────────────────────
    cost_path = fp("enforcement_ranking_with_cost.csv")
    d["cost"] = pd.read_csv(cost_path) if os.path.exists(cost_path) else pd.DataFrame()

    # ── Violations (large — load with usecols, support .gz) ───────────────────
    vcols = ["latitude","longitude","risk_score_raw","primary_violation",
             "updated_vehicle_type","vehicle_type","police_station","hour_ist",
             "date","day_of_week","junction_resolved","violation_impact",
             "vehicle_weight","h3_cell"]
    for vname in ["violations_geocoded.csv", "violations_geocoded.csv.gz"]:
        vp = fp(vname)
        if os.path.exists(vp):
            compression = "gzip" if vname.endswith(".gz") else None
            existing = [c for c in vcols if True]  # read header first
            tmp = pd.read_csv(vp, compression=compression, nrows=1)
            existing_cols = [c for c in vcols if c in tmp.columns]
            d["viol"] = pd.read_csv(vp, compression=compression, usecols=existing_cols)
            # Use updated_vehicle_type where available (officer correction)
            if "updated_vehicle_type" in d["viol"].columns and "vehicle_type" in d["viol"].columns:
                d["viol"]["vehicle_type_eff"] = d["viol"]["updated_vehicle_type"].fillna(d["viol"]["vehicle_type"])
            elif "vehicle_type" in d["viol"].columns:
                d["viol"]["vehicle_type_eff"] = d["viol"]["vehicle_type"]
            if "date" in d["viol"].columns:
                d["viol"]["date"] = pd.to_datetime(d["viol"]["date"], errors="coerce")
            break
    else:
        d["viol"] = pd.DataFrame()

    # ── XGBoost model ─────────────────────────────────────────────────────────
    mp = fp("xgb_violation_forecast_model.json")
    if HAS_XGB and os.path.exists(mp):
        m = xgb.XGBRegressor(); m.load_model(mp)
        d["model"] = m
    else:
        d["model"] = None

    # ── Parse SCITA payloads for forecast data ────────────────────────────────
    if not d["alerts"].empty and "scita_payload" in d["alerts"].columns:
        def safe_parse(x):
            try: return json.loads(x) if pd.notna(x) else {}
            except: return {}
        payloads = d["alerts"]["scita_payload"].apply(safe_parse)
        d["alerts"]["predicted_risk"] = payloads.apply(
            lambda p: p.get("forecast", {}).get("predicted_risk_for_tomorrow", None))
        d["alerts"]["trend"] = payloads.apply(
            lambda p: p.get("forecast", {}).get("trend", None))
        d["alerts"]["cost_inr"] = payloads.apply(
            lambda p: p.get("congestion_impact", {}).get("estimated_cost_inr_per_day", None))

    return d

# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(tier):
    t = str(tier).upper()
    return f'<span class="badge badge-{t}">{t}</span>'

def fmt(n, rs=False):
    try:
        v = float(n)
        if rs:
            if v >= 1e7: return f"₹{v/1e7:.1f}Cr"
            if v >= 1e5: return f"₹{v/1e5:.1f}L"
            return f"₹{v:,.0f}"
        if v >= 1e5: return f"{v/1e5:.1f}L"
        if v >= 1e3: return f"{v/1e3:.1f}K"
        return f"{int(v)}"
    except: return str(n)

def kpi_html(val, label, sub="", ac=C["accent1"]):
    return f"""
    <div class="kpi" style="--ac:{ac}">
        <div class="kv">{val}</div>
        <div class="kl">{label}</div>
        {"<div class='ks'>" + sub + "</div>" if sub else ""}
    </div>"""

import re

def render_html(html: str):
    flat = re.sub(r"\n\s*", "", html.strip())
    st.markdown(flat, unsafe_allow_html=True)
    
def plotly_dark(fig, height=None):
    kwargs = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=C["text"], size=11),
        margin=dict(l=8, r=8, t=32, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(gridcolor=C["border"], zeroline=False),
        yaxis=dict(gridcolor=C["border"], zeroline=False),
    )
    if height: kwargs["height"] = height
    fig.update_layout(**kwargs)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LOAD + SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style="padding:14px 0 6px">
        <div style="font-size:22px;font-weight:700;color:{C['text']};letter-spacing:-0.03em">
            🚦 ParkSense AI
        </div>
        <div style="font-size:11px;color:{C['muted']};font-family:monospace;margin-top:4px">
            Bengaluru · 122.47K clean violations
        </div>
    </div>
    <hr style="border-color:{C['border']};margin:10px 0 16px">
    """, unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        D = load_data()

    cost   = D["cost"]
    hexdf  = D["hexdf"]
    hourly = D["hourly"]
    daily  = D["daily"]
    monthly= D["monthly"]
    station= D["station"]
    vehicle= D["vehicle"]
    alerts = D["alerts"]
    viol   = D["viol"]
    clusters = D["clusters"]

    page = st.radio(
        "Navigate",
        ["📋 Challan Dashboard",
         "🗺️ Congestion Heatmap",
         "⚠️ Delay by Vehicle & Violation",
         "📈 Forecast & Planning"],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:{C['border']};margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px;color:{C['muted']};text-transform:uppercase;letter-spacing:.06em'>Filters</div>", unsafe_allow_html=True)

    # ── Station filter ─────────────────────────────────────────────────────────
    all_stations = ["All stations"]
    if not station.empty and "police_station" in station.columns:
        all_stations += sorted(station["police_station"].dropna().unique().tolist())
    elif not cost.empty and "dominant_station" in cost.columns:
        all_stations += sorted(cost["dominant_station"].dropna().unique().tolist())
    sel_station = st.selectbox("Police station", all_stations)

    # ── Date range filter ──────────────────────────────────────────────────────
    if not viol.empty and "date" in viol.columns and viol["date"].notna().any():
        min_d = viol["date"].min().date()
        max_d = viol["date"].max().date()
    else:
        min_d = datetime(2023, 11, 9).date()
        max_d = datetime(2024, 4, 8).date()

    sel_dates = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d, max_value=max_d,
    )
    if isinstance(sel_dates, (list, tuple)) and len(sel_dates) == 2:
        d_from, d_to = pd.Timestamp(sel_dates[0]), pd.Timestamp(sel_dates[1])
    else:
        d_from, d_to = pd.Timestamp(min_d), pd.Timestamp(max_d)

    # ── Traffic severity filter ────────────────────────────────────────────────
    sev_opts = ["All"] + sorted(cost["traffic_severity"].dropna().unique().tolist()) if not cost.empty and "traffic_severity" in cost.columns else ["All"]
    sel_sev = st.selectbox("Traffic severity", sev_opts)

    st.markdown(f"<hr style='border-color:{C['border']};margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:10px;color:{C['dim']};line-height:1.7'></div>", unsafe_allow_html=True)


# ── Apply sidebar filters to viol and cost tables ─────────────────────────────
def filter_viol(df):
    r = df.copy()
    if sel_station != "All stations" and "police_station" in r.columns:
        r = r[r["police_station"] == sel_station]
    if "date" in r.columns:
        r = r[(r["date"] >= d_from) & (r["date"] <= d_to)]
    return r

def filter_cost(df):
    r = df.copy()
    if sel_station != "All stations" and "dominant_station" in r.columns:
        r = r[r["dominant_station"] == sel_station]
    if sel_sev != "All" and "traffic_severity" in r.columns:
        r = r[r["traffic_severity"] == sel_sev]
    return r

viol_f = filter_viol(viol) if not viol.empty else pd.DataFrame()
cost_f = filter_cost(cost) if not cost.empty else pd.DataFrame()


# ── Hero banner (every page) ──────────────────────────────────────────────────
st.markdown("""
<div class="ps-hero">
    <span class="ps-hero-icon">🚦</span>
    <div>
        <h1>Bengaluru Parking Violation Intelligence</h1>
        <p>AI-driven hotspot detection · Congestion delay quantification · Predictive enforcement planning</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHALLAN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

if page == "📋 Challan Dashboard":
    st.markdown('<div class="ps-page-title">Challan Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-page-sub">Filtered by police station · date range · traffic severity</div>', unsafe_allow_html=True)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    total_challans = len(viol_f) if not viol_f.empty else 0
    n_zones = cost_f["h3_cell"].nunique() if not cost_f.empty and "h3_cell" in cost_f.columns else 0

    high_risk_jn = 0
    if not cost_f.empty and "traffic_severity" in cost_f.columns:
        high_risk_jn = cost_f[cost_f["traffic_severity"].isin(["CRITICAL","HIGH"])]["junction_name"].nunique()

    total_delay_hrs = 0.0
    if not cost_f.empty and "person_hours_lost" in cost_f.columns:
        total_delay_hrs = cost_f.drop_duplicates("junction_name")["person_hours_lost"].sum()

    total_cost = 0.0
    if not cost_f.empty and "congestion_cost_rs" in cost_f.columns:
        total_cost = cost_f.drop_duplicates("junction_name")["congestion_cost_rs"].sum()

    st.markdown(f"""
    <div class="kpi-row">
        {kpi_html(fmt(total_challans), "Total challans", "In selected filters", C["accent1"])}
        {kpi_html(str(high_risk_jn), "High-risk junctions", "CRITICAL + HIGH severity", C["accent2"])}
        {kpi_html(fmt(total_delay_hrs), "Est. person-hours lost", "Across filtered zones", C["accent5"])}
        {kpi_html(fmt(total_cost, rs=True), "Estimated daily cost", "₹ congestion value", C["accent3"])}
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Hourly + Day-of-week ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="ps-card-title">Enforcement activity by hour (IST)</div>', unsafe_allow_html=True)
        if not hourly.empty and "count" in hourly.columns:
            h = hourly.copy()
            if "period" not in h.columns:
                h["period"] = h["hour_ist"].apply(
                    lambda x: "Peak congestion" if (8<=x<=11 or 17<=x<=21)
                    else "Night patrol" if (x>=22 or x<=5) else "Off-peak")
            cmap = {"Peak congestion": C["accent1"], "Night patrol": C["accent3"], "Off-peak": C["muted"]}
            fig = px.bar(h, x="hour_ist", y="count", color="period",
                         color_discrete_map=cmap, height=230,
                         labels={"hour_ist":"Hour (IST)","count":"Challans","period":""})
            fig.update_layout(legend=dict(orientation="h", y=-0.3, x=0))
            plotly_dark(fig, 230)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("eda_hourly.csv not found.")

    with col2:
        st.markdown('<div class="ps-card-title">Challans by day of week</div>', unsafe_allow_html=True)
        if not daily.empty and "count" in daily.columns:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            d2 = daily.copy()
            if "day_of_week" in d2.columns:
                d2["day_of_week"] = pd.Categorical(d2["day_of_week"], categories=day_order, ordered=True)
                d2 = d2.sort_values("day_of_week")
            fig = px.bar(d2, x="day_of_week", y="count",
                         color="count", color_continuous_scale=["#1a2235", C["accent3"]],
                         height=230, labels={"day_of_week":"","count":"Challans"})
            fig.update_layout(coloraxis_showscale=False)
            plotly_dark(fig, 230)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("eda_daily.csv not found.")

    # ── Row 2: Top stations + monthly trend ───────────────────────────────────
    col3, col4 = st.columns([3, 2])

    with col3:
        st.markdown('<div class="ps-card-title">Top police stations by risk-weighted load</div>', unsafe_allow_html=True)
        if not station.empty and "total_risk" in station.columns:
            s = station.copy()
            if sel_station != "All stations" and "police_station" in s.columns:
                s = s[s["police_station"] == sel_station]
            s = s.sort_values("total_risk", ascending=False).head(12)
            fig = px.bar(s, x="total_risk", y="police_station", orientation="h",
                         color="total_risk", color_continuous_scale=["#1a2235", C["accent2"]],
                         height=290, labels={"total_risk":"Risk score","police_station":""})
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            plotly_dark(fig, 290)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("eda_station_load.csv not found.")

    with col4:
        st.markdown('<div class="ps-card-title">Monthly challan trend</div>', unsafe_allow_html=True)
        if not monthly.empty and "count" in monthly.columns:
            fig = px.line(monthly, x="month", y="count", markers=True,
                          height=290, labels={"month":"","count":"Challans"},
                          color_discrete_sequence=[C["accent5"]])
            fig.update_traces(line_width=2.5, marker_size=7,
                              fill="tozeroy",
                              fillcolor=f"rgba(167,139,250,0.12)")
            plotly_dark(fig, 290)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("eda_monthly.csv not found.")

    # ── High-risk junction table ───────────────────────────────────────────────
    st.markdown('<div class="ps-card-title">High-risk junctions — CRITICAL & HIGH only</div>', unsafe_allow_html=True)

    if not cost_f.empty:
        hr = cost_f[cost_f["traffic_severity"].isin(["CRITICAL","HIGH"])].copy() if "traffic_severity" in cost_f.columns else cost_f.copy()
        hr = hr.drop_duplicates("junction_name").sort_values("traffic_impact_score" if "traffic_impact_score" in hr.columns else "congestion_cost_rs", ascending=False).head(20)

        cols_show = ["junction_name","dominant_station","n_violations","person_hours_lost",
                     "congestion_cost_rs","traffic_severity","cii_tier"]
        cols_show = [c for c in cols_show if c in hr.columns]
        hr_disp = hr[cols_show].copy()

        rename_map = {
            "junction_name":"Junction","dominant_station":"Station",
            "n_violations":"Challans","person_hours_lost":"Person-hrs lost",
            "congestion_cost_rs":"Cost/day (₹)","traffic_severity":"Severity","cii_tier":"CII Tier"
        }
        hr_disp = hr_disp.rename(columns=rename_map)

        if "Person-hrs lost" in hr_disp.columns:
            hr_disp["Person-hrs lost"] = hr_disp["Person-hrs lost"].apply(lambda x: f"{x:,.0f}")
        if "Cost/day (₹)" in hr_disp.columns:
            hr_disp["Cost/day (₹)"] = hr_disp["Cost/day (₹)"].apply(lambda x: fmt(x, rs=True))

        def sev_colour(val):
            cols = {"CRITICAL":f"background:{C['accent1']}22;color:{C['accent1']}",
                    "HIGH":f"background:{C['accent2']}22;color:{C['accent2']}",
                    "MEDIUM":"background:#eab30822;color:#eab308",
                    "LOW":f"background:{C['accent4']}22;color:{C['accent4']}"}
            return cols.get(str(val).upper(), "")

        if not hr_disp.empty:
            styled = hr_disp.style \
                .map(sev_colour, subset=[c for c in ["Severity","CII Tier"] if c in hr_disp.columns]) \
                .set_properties(**{"font-size":"12px"})
            st.dataframe(styled, height=340, use_container_width=True)
            st.download_button("⬇️ Download CSV",
                hr[cols_show].to_csv(index=False).encode(),
                "high_risk_junctions.csv", "text/csv")
        else:
            st.info("No CRITICAL/HIGH junctions match current filters.")
    else:
        st.info("enforcement_ranking_with_cost.csv not found.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CONGESTION DELAY IMPACT HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ Congestion Heatmap":
    st.markdown('<div class="ps-page-title">Congestion Delay Impact Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-page-sub">Each zone coloured by estimated person-hours lost per day · darker = worse delay impact</div>', unsafe_allow_html=True)

    if not HAS_FOLIUM or not HAS_H3:
        st.warning("Install folium, streamlit-folium and h3: `pip install folium streamlit-folium h3`")
    else:
        # ── Map layer controls ─────────────────────────────────────────────────
        c_ctrl1, c_ctrl2 = st.columns([3, 1])
        with c_ctrl1:
            layer = st.radio("Map layer", ["Delay impact (H3 choropleth)",
                                           "Priority markers (top 60 zones)",
                                           "Raw violation density"],
                             horizontal=True)
        with c_ctrl2:
            n_markers = st.slider("Marker limit", 20, 100, 60) if "Priority" in layer else 60

        BLR = [12.9716, 77.5946]
        m = folium.Map(location=BLR, zoom_start=12, tiles="CartoDB DarkMatter", prefer_canvas=True)
        MiniMap(toggle_display=False).add_to(m)

        if layer == "Delay impact (H3 choropleth)" and not hexdf.empty:
            # Merge delay cost into hex table from cost table
            hex_cost = hexdf.copy()
            if not cost.empty and "person_hours_lost" in cost.columns:
                agg = cost.groupby("h3_cell")["person_hours_lost"].max().reset_index()
                hex_cost = hex_cost.merge(agg, on="h3_cell", how="left")
            else:
                hex_cost["person_hours_lost"] = hex_cost.get("cii", 0)

            max_delay = hex_cost["person_hours_lost"].max() if "person_hours_lost" in hex_cost.columns else 1

            for _, row in hex_cost.iterrows():
                cell = row.get("h3_cell")
                if not cell: continue
                try:
                    delay = float(row.get("person_hours_lost", 0) or 0)
                    ratio = min(delay / max(max_delay, 1), 1)
                    # Interpolate: low=blue → mid=orange → high=red
                    if ratio < 0.5:
                        r = int(59 + (234 - 59) * ratio * 2)
                        g = int(130 + (88 - 130) * ratio * 2)
                        b = int(246 + (12 - 246) * ratio * 2)
                    else:
                        r = int(234 + (220 - 234) * (ratio - 0.5) * 2)
                        g = int(88 + (38 - 88) * (ratio - 0.5) * 2)
                        b = int(12 + (38 - 12) * (ratio - 0.5) * 2)
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    opacity = 0.15 + ratio * 0.65

                    boundary = h3.cell_to_boundary(cell)
                    coords = [[lat, lon] for lat, lon in boundary]
                    tier = str(row.get("cii_tier", "LOW")).upper()
                    popup_html = (
                        f"<b style='color:#1e293b'>{row.get('dominant_station','?')}</b><br>"
                        f"Violation count: <b>{int(row.get('count',0)):,}</b><br>"
                        f"CII tier: <b>{tier}</b> ({row.get('cii',0):.1f})<br>"
                        f"Person-hrs lost/day: <b>{delay:,.0f}</b><br>"
                        f"H3: {cell}"
                    )
                    folium.Polygon(
                        locations=coords, color=color, weight=0.4,
                        fill=True, fill_color=color, fill_opacity=opacity,
                        popup=folium.Popup(popup_html, max_width=260),
                        tooltip=f"{tier} · {int(delay):,} person-hrs lost",
                    ).add_to(m)
                except Exception:
                    continue

        elif "Priority markers" in layer and not cost_f.empty:
            best = cost_f.drop_duplicates("junction_name").sort_values(
                "traffic_impact_score" if "traffic_impact_score" in cost_f.columns else "congestion_cost_rs",
                ascending=False).head(n_markers)
            mc = MarkerCluster().add_to(m)
            for rank, (_, row) in enumerate(best.iterrows(), 1):
                sev = str(row.get("traffic_severity","LOW")).upper()
                col_m = {"CRITICAL":"red","HIGH":"orange","MEDIUM":"blue","LOW":"green"}.get(sev,"gray")
                popup = (
                    f"<b>#{rank} — {row.get('junction_name','?')}</b><br>"
                    f"Challans: <b>{int(row.get('n_violations',0)):,}</b><br>"
                    f"Person-hrs lost: <b>{float(row.get('person_hours_lost',0)):,.0f}</b><br>"
                    f"Cost/day: <b>₹{float(row.get('congestion_cost_rs',0)):,.0f}</b><br>"
                    f"Severity: <b>{sev}</b> · CII: {row.get('cii',0):.1f}<br>"
                    f"Top violation: {row.get('dominant_type','?')}<br>"
                    f"Station: {row.get('dominant_station','?')}"
                )
                try:
                    folium.Marker(
                        location=[float(row["lat"]), float(row["lon"])],
                        popup=folium.Popup(popup, max_width=300),
                        tooltip=f"#{rank} {str(row.get('junction_name',''))[:28]} | {sev}",
                        icon=folium.Icon(color=col_m, icon="warning-sign", prefix="glyphicon"),
                    ).add_to(mc)
                except Exception:
                    continue

        elif "density" in layer and not viol_f.empty:
            heat_data = (
                viol_f[["latitude","longitude","risk_score_raw"]].dropna()
                .sample(min(25000, len(viol_f)), random_state=42)
                .values.tolist()
            )
            HeatMap(heat_data, radius=13, blur=16, max_zoom=14,
                    gradient={0.25:"#1e3a5f", 0.5:"#7c3aed",
                               0.75:"#f97316", 1.0:"#f43f5e"}).add_to(m)
        else:
            st.info("Required data not found for this layer. Check your data files.")

        st_folium(m, width=None, height=580, returned_objects=[])

        # ── Legend ─────────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:12px;align-items:center">
            <span style="font-size:11px;color:{C['muted']}">Delay intensity →</span>
            <span style="display:flex;align-items:center;gap:6px;font-size:11px;color:{C['text']}">
                <span style="width:14px;height:14px;background:#3b82f6;border-radius:2px;display:inline-block"></span>Low
            </span>
            <span style="display:flex;align-items:center;gap:6px;font-size:11px;color:{C['text']}">
                <span style="width:14px;height:14px;background:#f97316;border-radius:2px;display:inline-block"></span>Medium
            </span>
            <span style="display:flex;align-items:center;gap:6px;font-size:11px;color:{C['text']}">
                <span style="width:14px;height:14px;background:#dc2626;border-radius:2px;display:inline-block"></span>High
            </span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DELAY CONTRIBUTION BY VEHICLE TYPE
# ═════════════════════════════════════════════════════════════════════════════

elif page == "⚠️ Delay by Vehicle & Violation":
    st.markdown('<div class="ps-page-title">Delay Contribution by Vehicle & Violation Type</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-page-sub">Which vehicle types and violation categories cause the most congestion delay? Computed from officer-corrected vehicle classifications and ranked impact scores.</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size:13px;font-weight:600;color:%s;margin:4px 0 12px;text-transform:uppercase;letter-spacing:.04em">🚗 By vehicle type</div>' % C["accent5"], unsafe_allow_html=True)

    # ── Vehicle type vs delay (using raw violations + cost table) ─────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="ps-card-title">% of total risk score by vehicle type</div>', unsafe_allow_html=True)
        if not vehicle.empty:
            vcol = "updated_vehicle_type" if "updated_vehicle_type" in vehicle.columns else "vehicle_type"
            vc = vehicle.copy()
            if "pct_of_total_risk" in vc.columns:
                vc = vc.sort_values("pct_of_total_risk", ascending=False).head(10)
                fig = px.bar(vc, x=vcol, y="pct_of_total_risk",
                             color="pct_of_total_risk",
                             color_continuous_scale=["#1a2235", C["accent1"]],
                             height=300,
                             labels={vcol:"", "pct_of_total_risk":"% of total risk"})
                fig.update_layout(coloraxis_showscale=False)
                plotly_dark(fig, 300)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("eda_vehicle_risk.csv not found.")

    with col_b:
        st.markdown('<div class="ps-card-title">Challan count vs risk share — size = avg risk</div>', unsafe_allow_html=True)
        if not vehicle.empty:
            vcol = "updated_vehicle_type" if "updated_vehicle_type" in vehicle.columns else "vehicle_type"
            vc = vehicle.dropna(subset=["pct_of_total_risk","pct_of_count"]).head(12).copy()
            fig = px.scatter(vc, x="pct_of_count", y="pct_of_total_risk",
                             text=vcol, size="avg_risk" if "avg_risk" in vc.columns else "pct_of_total_risk",
                             color="pct_of_total_risk",
                             color_continuous_scale=["#1a2235", C["accent2"]],
                             height=300,
                             labels={"pct_of_count":"% of challan count",
                                     "pct_of_total_risk":"% of risk score"})
            fig.update_traces(textposition="top center", textfont=dict(size=9, color=C["text"]))
            fig.update_layout(coloraxis_showscale=False)
            plotly_dark(fig, 300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("eda_vehicle_risk.csv not found.")

    # ── Vehicle weight distribution from violations data ───────────────────────
    if not viol_f.empty and "vehicle_type_eff" in viol_f.columns:

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<div class="ps-card-title">Person-minutes lost by vehicle type (from cost table)</div>', unsafe_allow_html=True)
            if not cost_f.empty and "person_minutes_lost" in cost_f.columns and "dominant_vehicle" in cost_f.columns:
                pm = cost_f.groupby("dominant_vehicle")["person_minutes_lost"].sum().reset_index()
                pm = pm.sort_values("person_minutes_lost", ascending=True).tail(10)
                fig = px.bar(pm, x="person_minutes_lost", y="dominant_vehicle",
                             orientation="h", height=300,
                             color="person_minutes_lost",
                             color_continuous_scale=["#1a2235", C["accent5"]],
                             labels={"person_minutes_lost":"Person-minutes lost","dominant_vehicle":""})
                fig.update_layout(coloraxis_showscale=False)
                plotly_dark(fig, 300)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            else:
                # Fallback: use viol table vehicle weights
                vd = viol_f.groupby("vehicle_type_eff").agg(
                    count=("vehicle_type_eff","count"),
                    avg_weight=("vehicle_weight","mean") if "vehicle_weight" in viol_f.columns else ("vehicle_type_eff","count")
                ).reset_index().sort_values("count", ascending=True).tail(10)
                fig = px.bar(vd, x="count", y="vehicle_type_eff", orientation="h",
                             height=300, color="count",
                             color_continuous_scale=["#1a2235", C["accent5"]],
                             labels={"count":"Challan count","vehicle_type_eff":""})
                fig.update_layout(coloraxis_showscale=False)
                plotly_dark(fig, 300)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        with col_d:
            st.markdown('<div class="ps-card-title">Vehicle type mix over time (top 5 types)</div>', unsafe_allow_html=True)
            if "date" in viol_f.columns and "vehicle_type_eff" in viol_f.columns:
                top5_veh = viol_f["vehicle_type_eff"].value_counts().head(5).index.tolist()
                vt = viol_f[viol_f["vehicle_type_eff"].isin(top5_veh)].copy()
                vt["month"] = pd.to_datetime(vt["date"]).dt.to_period("M").astype(str)
                grp = vt.groupby(["month","vehicle_type_eff"]).size().reset_index(name="count")
                fig = px.line(grp, x="month", y="count", color="vehicle_type_eff",
                              markers=True, height=300,
                              labels={"month":"","count":"Challans","vehicle_type_eff":"Vehicle"},
                              color_discrete_sequence=[C["accent1"],C["accent2"],C["accent3"],
                                                       C["accent4"],C["accent5"]])
                plotly_dark(fig, 300)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            else:
                st.info("Date or vehicle data not available.")

    # ── Vehicle weight impact table ─────────────────────────────────────────
    st.markdown('<div class="ps-card-title">Vehicle type congestion weight — why heavier vehicles matter more</div>', unsafe_allow_html=True)
    weight_tbl = pd.DataFrame({
        "Vehicle type": ["HGV / Trailer / Private Bus","LGV / Tractor","Maxi-Cab / Van",
                         "Passenger Auto / Goods Auto / Car / Jeep","Motor Cycle / Scooter / Moped"],
        "Congestion weight": [5, 4, 3, 2, 1],
        "Lane footprint (% of one lane)": ["55–60%","45%","35%","25–30%","10%"],
        "Justification": [
            "Blocks entire lane, prevents overtaking",
            "Large footprint, extends into second lane",
            "Significant obstruction on narrow roads",
            "Standard vehicle width, moderate impact",
            "Narrow, often able to thread past traffic",
        ]
    })
    st.dataframe(weight_tbl, hide_index=True, use_container_width=True)

    # ── Section divider: vehicle → violation ──────────────────────────────────
    st.markdown(f"<hr style='border-color:{C['border']};margin:8px 0 20px'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;font-weight:600;color:%s;margin:4px 0 12px;text-transform:uppercase;letter-spacing:.04em">⚠️ By violation type</div>' % C["accent2"], unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="ps-card-title">Total risk score by violation type</div>', unsafe_allow_html=True)

        if not viol_f.empty and "primary_violation" in viol_f.columns and "risk_score_raw" in viol_f.columns:
            vt = viol_f.groupby("primary_violation").agg(
                total_risk=("risk_score_raw","sum"),
                challan_count=("risk_score_raw","count"),
                avg_risk=("risk_score_raw","mean"),
            ).reset_index().sort_values("total_risk", ascending=True).tail(10)

            fig = px.bar(vt, x="total_risk", y="primary_violation", orientation="h",
                         color="total_risk",
                         color_continuous_scale=["#1a2235", C["accent1"]],
                         height=320,
                         labels={"total_risk":"Total risk score","primary_violation":""})
            fig.update_layout(coloraxis_showscale=False)
            plotly_dark(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            # Fallback: use cost table
            if not cost_f.empty and "dominant_type" in cost_f.columns and "congestion_cost_rs" in cost_f.columns:
                vt = cost_f.groupby("dominant_type")["congestion_cost_rs"].sum().reset_index()
                vt = vt.sort_values("congestion_cost_rs", ascending=True).tail(10)
                fig = px.bar(vt, x="congestion_cost_rs", y="dominant_type", orientation="h",
                             height=320, color="congestion_cost_rs",
                             color_continuous_scale=["#1a2235", C["accent1"]],
                             labels={"congestion_cost_rs":"Total cost (₹)","dominant_type":""})
                fig.update_layout(coloraxis_showscale=False)
                plotly_dark(fig, 320)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            else:
                st.info("violations_geocoded.csv not found.")

    with col_b:
        st.markdown('<div class="ps-card-title">Violation impact weight vs challan frequency</div>', unsafe_allow_html=True)

        if not viol_f.empty and "primary_violation" in viol_f.columns:
            vt2 = viol_f.groupby("primary_violation").agg(
                challan_count=("primary_violation","count"),
                avg_impact=("violation_impact","mean") if "violation_impact" in viol_f.columns else ("primary_violation","count"),
                total_risk=("risk_score_raw","sum") if "risk_score_raw" in viol_f.columns else ("primary_violation","count"),
            ).reset_index().sort_values("challan_count", ascending=False).head(8)

            fig = px.scatter(vt2, x="challan_count", y="avg_impact",
                             text="primary_violation", size="total_risk",
                             color="avg_impact",
                             color_continuous_scale=["#1a2235", C["accent2"]],
                             height=320,
                             labels={"challan_count":"Challan count","avg_impact":"Avg impact weight"})
            fig.update_traces(textposition="top center", textfont=dict(size=8, color=C["text"]))
            fig.update_layout(coloraxis_showscale=False)
            plotly_dark(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("violations_geocoded.csv not found.")

    # ── Violation type over time ───────────────────────────────────────────────
    if not viol_f.empty and "primary_violation" in viol_f.columns and "date" in viol_f.columns:
        st.markdown('<div class="ps-card-title">Violation type trend by month (top 5 categories)</div>', unsafe_allow_html=True)

        top5_viol = viol_f["primary_violation"].value_counts().head(5).index.tolist()
        vm = viol_f[viol_f["primary_violation"].isin(top5_viol)].copy()
        vm["month"] = pd.to_datetime(vm["date"]).dt.to_period("M").astype(str)
        grp = vm.groupby(["month","primary_violation"]).size().reset_index(name="count")

        fig = px.line(grp, x="month", y="count", color="primary_violation",
                      markers=True, height=260,
                      labels={"month":"","count":"Challans","primary_violation":"Violation"},
                      color_discrete_sequence=[C["accent1"],C["accent2"],C["accent3"],
                                               C["accent4"],C["accent5"]])
        plotly_dark(fig, 260)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    # ── Violation impact reference table ─────────────────────────────────────
    st.markdown('<div class="ps-card-title">Congestion impact weights — how violation severity is scored</div>', unsafe_allow_html=True)
    impact_tbl = pd.DataFrame({
        "Violation type": [
            "PARKING IN A MAIN ROAD",
            "PARKING NEAR ROAD CROSSING",
            "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC",
            "DOUBLE PARKING",
            "NO PARKING",
            "WRONG PARKING",
            "PARKING ON FOOTPATH",
            "DEFECTIVE NUMBER PLATE",
        ],
        "Impact weight (1–5)": [5, 4, 4, 4, 3, 2, 2, 1],
        "Reason": [
            "Directly reduces carriageway capacity",
            "Blocks sight lines, causes rear-end risk",
            "Forces pedestrians onto road, blocks bus access",
            "Reduces lane count by 50%, severe chokepoint",
            "Deliberate obstruction in controlled zone",
            "Narrows effective lane width",
            "Forces pedestrians onto the road",
            "Administrative — no direct traffic impact",
        ]
    })
    st.dataframe(impact_tbl, hide_index=True, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FORECAST & ENFORCEMENT PLANNING
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📈 Forecast & Planning":
    st.markdown('<div class="ps-page-title">Forecasting & Enforcement Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-page-sub">AI-predicted violation spikes · Patrol route optimisation · SCITA alert queue</div>', unsafe_allow_html=True)

    # ── Apply sidebar filters (station / severity) to the alert log ───────────
    alerts_f = alerts.copy()
    if not alerts_f.empty:
        if sel_station != "All stations" and "police_station" in alerts_f.columns:
            alerts_f = alerts_f[alerts_f["police_station"] == sel_station]
        if sel_sev != "All" and "alert_level" in alerts_f.columns:
            # alert_level (CRITICAL/WATCH/INFO) and traffic_severity use overlapping
            # vocab — match whichever column the alert log actually carries.
            if "traffic_severity" in alerts_f.columns:
                alerts_f = alerts_f[alerts_f["traffic_severity"] == sel_sev]

    using_alert_log = not alerts_f.empty and "alert_level" in alerts_f.columns

    # ── KPIs — prefer the live alert log, fall back to the cost-ranked table ──
    n_crit = n_watch = n_esc = n_sent = 0
    total_cost_at_risk = 0.0

    if using_alert_log:
        n_crit  = int((alerts_f["alert_level"] == "CRITICAL").sum())
        n_watch = int((alerts_f["alert_level"] == "WATCH").sum())
        if "escalated" in alerts_f.columns:
            n_esc = int(alerts_f["escalated"].sum())
        if "scita_sync_status" in alerts_f.columns:
            n_sent = int((alerts_f["scita_sync_status"] == "SENT").sum())
        if "cost_inr" in alerts_f.columns:
            total_cost_at_risk = alerts_f["cost_inr"].sum()
    elif not cost_f.empty and "traffic_severity" in cost_f.columns:
        # Fallback: derive the same KPIs from the enforcement ranking / cost
        # table so the page is never just showing zeroes when
        # phase5_alert_log.csv hasn't been generated yet.
        cf = cost_f.drop_duplicates("junction_name")
        n_crit  = int((cf["traffic_severity"] == "CRITICAL").sum())
        n_watch = int((cf["traffic_severity"] == "HIGH").sum())
        if "escalated" in cf.columns:
            n_esc = int(cf["escalated"].sum())
        elif "hours_unaddressed" in cf.columns:
            n_esc = int((cf["hours_unaddressed"] > 2).sum())
        if "congestion_cost_rs" in cf.columns:
            total_cost_at_risk = cf[cf["traffic_severity"].isin(["CRITICAL","HIGH"])]["congestion_cost_rs"].sum()

    cost_kpi_label = "CRITICAL alerts" if using_alert_log else "CRITICAL junctions"
    watch_kpi_label = "WATCH alerts" if using_alert_log else "HIGH junctions"

    st.markdown(f"""
    <div class="kpi-row">
        {kpi_html(str(n_crit), cost_kpi_label, "Immediate dispatch", C["accent1"])}
        {kpi_html(str(n_watch), watch_kpi_label, "Monitor closely", C["accent2"])}
        {kpi_html(str(n_esc), "Escalated", "> 2 hrs unaddressed", C["accent5"])}
        {kpi_html(fmt(total_cost_at_risk, rs=True), "Cost at risk (alerted zones)", "₹/day est.", C["accent3"])}
    </div>
    """, unsafe_allow_html=True)

    if not using_alert_log:
        st.caption("⚠️ phase5_alert_log.csv not found or empty for the current filters — showing equivalent figures derived from the enforcement ranking table instead.")

    col_f, col_g = st.columns([3, 2])

    with col_f:
        # ── Predicted risk from alerts ─────────────────────────────────────────
        st.markdown('<div class="ps-card-title">Predicted tomorrow risk — top active zones</div>', unsafe_allow_html=True)


        if not alerts_f.empty and "predicted_risk" in alerts_f.columns and alerts_f["predicted_risk"].notna().any():
            fc = alerts_f.dropna(subset=["predicted_risk"]).copy()
            fc = fc.drop_duplicates("junction_name").sort_values("predicted_risk", ascending=False).head(15)
            fc["trend_icon"] = fc["trend"].map({"RISING":"↑","FALLING":"↓","STABLE":"→"}).fillna("→")
            fc["label"] = fc["junction_name"].str[:30] + " " + fc["trend_icon"]
            fig = px.bar(fc.iloc[::-1], x="predicted_risk", y="label",
                         orientation="h", height=360,
                         color="predicted_risk",
                         color_continuous_scale=["#1a2235", C["accent5"]],
                         labels={"predicted_risk":"Predicted risk score","label":""})
            fig.update_layout(coloraxis_showscale=False)
            plotly_dark(fig, 360)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        elif HAS_XGB and D["model"] is not None:
            st.info("Model loaded but no time-series features available for inference in the app. Run Phase 5 notebook to generate phase5_alert_log.csv.")
        else:
            # Show congestion intensity as forecast proxy
            if not cost_f.empty and "congestion_intensity" in cost_f.columns:
                proxy = cost_f.drop_duplicates("junction_name").sort_values(
                    "congestion_intensity", ascending=False).head(15)
                fig = px.bar(proxy.iloc[::-1], x="congestion_intensity", y="junction_name",
                             orientation="h", height=360,
                             color="congestion_intensity",
                             color_continuous_scale=["#1a2235", C["accent5"]],
                             labels={"congestion_intensity":"Congestion intensity","junction_name":""})
                fig.update_layout(coloraxis_showscale=False,
                                  yaxis=dict(tickfont=dict(size=10)))
                plotly_dark(fig, 360)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                st.caption("Showing congestion intensity (Phase 4 proxy) — run Phase 5 for live model forecasts.")
            else:
                st.info("phase5_alert_log.csv not found. Run Phase 5 notebook to generate live forecasts.")

    with col_g:
        # ── Trend distribution ─────────────────────────────────────────────────
        st.markdown('<div class="ps-card-title">Forecast trend distribution</div>', unsafe_allow_html=True)
        if not alerts_f.empty and "trend" in alerts_f.columns and alerts_f["trend"].notna().any():
            td = alerts_f["trend"].value_counts().reset_index()
            td.columns = ["trend","count"]
            cmap_t = {"RISING": C["accent1"], "STABLE": C["muted"], "FALLING": C["accent3"]}
            fig = px.pie(td, names="trend", values="count",
                         color="trend", color_discrete_map=cmap_t,
                         height=200, hole=0.5)
            fig.update_layout(legend=dict(orientation="h", y=-0.1),
                              margin=dict(l=0,r=0,t=0,b=30))
            plotly_dark(fig, 200)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Trend data not available.")

        # ── Alert level distribution ───────────────────────────────────────────
        st.markdown('<div class="ps-card-title">Alert level distribution</div>', unsafe_allow_html=True)
        if not alerts_f.empty and "alert_level" in alerts_f.columns:
            al_d = alerts_f["alert_level"].value_counts().reset_index()
            al_d.columns = ["level","count"]
            cmap_a = {"CRITICAL":C["accent1"],"WATCH":C["accent2"],"INFO":C["accent3"]}
            fig = px.bar(al_d, x="level", y="count", color="level",
                         color_discrete_map=cmap_a, height=170,
                         labels={"level":"","count":"Alerts"})
            fig.update_layout(showlegend=False)
            plotly_dark(fig, 170)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("Alert data not available.")

    # ── Patrol route ──────────────────────────────────────────────────────────
    st.markdown('<div class="ps-card-title">Suggested patrol route — nearest-neighbour optimised from top-priority alerts</div>', unsafe_allow_html=True)

    if not alerts_f.empty and "alert_level" in alerts_f.columns:
        route_src = alerts_f[alerts_f["alert_level"].isin(["CRITICAL","WATCH"])].copy()
    elif not cost_f.empty:
        route_src = cost_f.drop_duplicates("junction_name").sort_values(
            "traffic_impact_score" if "traffic_impact_score" in cost_f.columns else "congestion_cost_rs",
            ascending=False).head(20).copy()
    else:
        route_src = pd.DataFrame()

    if not route_src.empty:
        route_src = route_src.drop_duplicates(
            "junction_name" if "junction_name" in route_src.columns else route_src.columns[0]
        ).head(8)
        stop_colors = [C["accent1"],C["accent1"],C["accent2"],C["accent2"],
                       C["accent3"],C["accent3"],C["accent4"],C["accent4"]]

        # Display cards row-wise instead of column-wise
        for start in range(0, len(route_src), 4):
        
            cols = st.columns(4)
        
            current_batch = route_src.iloc[start:start+4]
        
            for idx, (col, (_, row)) in enumerate(zip(cols, current_batch.iterrows())):
        
                i = start + idx
        
                sn_color = stop_colors[min(i, len(stop_colors)-1)]
                jn = str(row.get("junction_name", "—"))[:38]
                lvl = str(
                    row.get(
                        "alert_level",
                        row.get("traffic_severity", "?")
                    )
                ).upper()
        
                cost_rs = row.get(
                    "cost_inr",
                    row.get("congestion_cost_rs", 0)
                )
        
                unit = row.get(
                    "assigned_unit",
                    row.get("dominant_station", "—")
                )
        
                with col:
                    render_html(f"""
                        <div class="route-stop">
                            <div class="stop-num" style="--sn:{sn_color}">{i+1}</div>
                            <div class="stop-body">
                                <b>{jn}</b><br>
                                {badge(lvl)} &nbsp; {fmt(cost_rs, rs=True)}/day<br>
                                <span style="font-size:10px;color:{C['dim']}">Unit: {unit}</span>
                            </div>
                        </div>
                    """)
    else:
        st.info("No active alerts or cost data available for patrol planning.")

    # ── Alert detail table ────────────────────────────────────────────────────
    if not alerts_f.empty:
        st.markdown('<div class="ps-card-title">Full alert log — sortable, searchable</div>', unsafe_allow_html=True)

        q_alert = st.text_input("🔍 Filter alerts", placeholder="Junction name or station...")
        al_view = alerts_f.copy()
        if q_alert:
            q_l = q_alert.lower()
            mask = (
                al_view.get("junction_name", pd.Series(dtype=str)).str.lower().str.contains(q_l, na=False) |
                al_view.get("police_station", pd.Series(dtype=str)).str.lower().str.contains(q_l, na=False)
            )
            al_view = al_view[mask]

        show = [c for c in ["alert_id","junction_name","police_station","alert_level",
                             "assigned_unit","status","escalated","scita_sync_status",
                             "predicted_risk","trend","cost_inr"] if c in al_view.columns]
        al_disp = al_view[show].copy()

        rename_al = {
            "alert_id":"ID","junction_name":"Junction","police_station":"Station",
            "alert_level":"Level","assigned_unit":"Unit","status":"Status",
            "escalated":"Esc.","scita_sync_status":"SCITA",
            "predicted_risk":"Pred. Risk","trend":"Trend","cost_inr":"Cost/day (₹)"
        }
        al_disp = al_disp.rename(columns=rename_al)
        if "Cost/day (₹)" in al_disp.columns:
            al_disp["Cost/day (₹)"] = al_disp["Cost/day (₹)"].apply(lambda x: fmt(x, rs=True) if pd.notna(x) else "—")

        def sev_col(val):
            lkp = {"CRITICAL":f"background:{C['accent1']}22;color:{C['accent1']}",
                   "WATCH":f"background:{C['accent2']}22;color:{C['accent2']}",
                   "HIGH":f"background:{C['accent2']}22;color:{C['accent2']}",
                   "RISING":f"background:{C['accent1']}22;color:{C['accent1']}",
                   "FALLING":f"background:{C['accent3']}22;color:{C['accent3']}",
                   "SENT":f"background:{C['accent4']}22;color:{C['accent4']}"}
            return lkp.get(str(val).upper(), "")

        styled_al = al_disp.style \
            .map(sev_col, subset=[c for c in ["Level","Trend","SCITA"] if c in al_disp.columns]) \
            .format({"Pred. Risk":"{:.1f}"}, na_rep="—") \
            .set_properties(**{"font-size":"12px"})
        st.dataframe(styled_al, height=380, use_container_width=True)
        st.download_button("⬇️ Download alert log",
            al_view[show].to_csv(index=False).encode(),
            "alert_log.csv", "text/csv")

    # ── Enforcement shift planning ────────────────────────────────────────────
    st.markdown('<div class="ps-card-title">Enforcement window recommendation</div>', unsafe_allow_html=True)
    if not hourly.empty and "count" in hourly.columns and "avg_risk" in hourly.columns:
        h2 = hourly.copy()
        h2["priority_score"] = h2["count"] * h2["avg_risk"]
        top_windows = h2.sort_values("priority_score", ascending=False).head(6)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=top_windows["hour_ist"],
            y=top_windows["count"],
            name="Challan count",
            marker_color=C["accent3"],
            opacity=0.7,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=top_windows["hour_ist"],
            y=top_windows["avg_risk"],
            name="Avg risk score",
            line=dict(color=C["accent1"], width=2.5),
            mode="lines+markers",
            marker=dict(size=8),
        ), secondary_y=True)
        fig.update_layout(
            height=260,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C["text"], size=11),
            margin=dict(l=8,r=8,t=8,b=8),
            legend=dict(orientation="h", y=-0.25, bgcolor="rgba(0,0,0,0)"),
        )
        fig.update_xaxes(gridcolor=C["border"], title_text="Hour (IST)", color=C["text"])
        fig.update_yaxes(gridcolor=C["border"], title_text="Challan count", color=C["accent3"], secondary_y=False)
        fig.update_yaxes(title_text="Avg risk", color=C["accent1"], secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        best_hour = int(top_windows.iloc[0]["hour_ist"])
        second_hour = int(top_windows.iloc[1]["hour_ist"]) if len(top_windows) > 1 else best_hour
        st.markdown(f"""
        <div style="background:{C['surface']};border:1px solid {C['border']};border-radius:8px;padding:12px 16px;margin-top:8px">
            <span style="font-size:13px;color:{C['text']};font-weight:600">Recommended enforcement windows:</span>
            <span style="font-family:monospace;color:{C['accent5']};margin-left:12px">
                {best_hour:02d}:00–{best_hour+1:02d}:00 IST
                &nbsp;&nbsp;·&nbsp;&nbsp;
                {second_hour:02d}:00–{second_hour+1:02d}:00 IST
            </span>
            <span style="font-size:11px;color:{C['muted']};margin-left:12px">highest challan volume × avg risk score</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("eda_hourly.csv not found.")
