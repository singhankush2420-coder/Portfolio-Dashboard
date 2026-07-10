# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:40:06 2026

@author: Ankush
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False
from scipy.stats import skew, kurtosis, norm, gaussian_kde
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas as rl_canvas
try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False
    def st_autorefresh(*args, **kwargs):
        pass  ## graceful no-op if package missing

## ── Suppress matplotlib tight_layout warnings globally ─────────────────────
import warnings as _warnings
_warnings.filterwarnings("ignore", message="Tight layout not applied")
_warnings.filterwarnings("ignore", message=".*tight_layout.*")

## ─────────────────────────────────────────────────────────────────────────────
##  PAGE CONFIG
## ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PortfolioIQ: Risk, Performance & Attribution",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


## ── Welcome modal — shown once per session ─────────────────────────────────
if "welcome_shown" not in st.session_state:
    st.session_state["welcome_shown"] = False

if not st.session_state["welcome_shown"]:
    _modal_html = """
<style>
.piq-ov{display:flex;align-items:center;justify-content:center;padding:40px 20px;min-height:80vh;}
.piq-md{background:#0D1117;border:0.5px solid #30363D;border-radius:12px;max-width:560px;width:100%;overflow:hidden;}
.piq-tp{padding:32px 36px 24px;}
.piq-br{display:flex;align-items:center;gap:8px;margin-bottom:24px;}
.piq-dot{width:8px;height:8px;border-radius:50%;background:#C8A951;display:inline-block;}
.piq-bn{font-size:12px;font-weight:600;color:#8B949E;letter-spacing:0.5px;}
.piq-qm{font-size:52px;color:#C8A951;line-height:0.5;display:block;margin-bottom:10px;font-family:Georgia,serif;}
.piq-qt{font-family:Georgia,serif;font-size:15px;color:#E6EDF3;line-height:1.75;font-style:italic;}
.piq-sc{font-size:11px;color:#8B949E;margin-top:10px;}
.piq-dv{width:32px;height:1px;background:#30363D;margin:20px 0;}
.piq-bd{font-size:13px;color:#8B949E;line-height:1.75;}
.piq-bd b{color:#E6EDF3;font-weight:600;}
.piq-bt{padding:16px 36px 28px;}
.piq-ds{padding:12px 14px;background:#161B22;border-radius:8px;margin-bottom:20px;border:0.5px solid #30363D;}
.piq-ds p{font-size:10.5px;color:#6E7681;line-height:1.6;text-transform:uppercase;letter-spacing:0.3px;}
</style>
<div class="piq-ov"><div class="piq-md">
<div class="piq-tp">
<div class="piq-br"><span class="piq-dot"></span><span class="piq-bn">PORTFOLIOIQ — RISK, PERFORMANCE &amp; ATTRIBUTION</span></div>
<span class="piq-qm">"</span>
<p class="piq-qt">Under heaven all can see beauty as beauty only because there is ugliness. All can know good as good only because there is evil.</p>
<p class="piq-sc">— Tao Te Ching, Chapter 2 &nbsp;·&nbsp; Lao Tzu</p>
<div class="piq-dv"></div>
<p class="piq-bd">A <b>profit</b> means little without understanding the losses that were possible.<br>A <b>return</b> means little without understanding the risks that created it.<br><br>PortfolioIQ was built to uncover the story behind both.</p>
</div>
<div class="piq-bt">
<div class="piq-ds"><p>THIS DASHBOARD IS FOR EDUCATIONAL PURPOSE ONLY. THIS DASHBOARD DOES NOT PROVIDE ANY INVESTMENT ADVICE.</p></div>
</div>
</div></div>
"""
    st.markdown(_modal_html, unsafe_allow_html=True)

    ## Single working button — styled with golden border via CSS targeting,
    ## positioned to visually sit inside the modal overlay using negative margin
    st.markdown("""
<style>
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: 10000;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    pointer-events: none;
    margin-top: 0 !important;
}
div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"]) > div {
    pointer-events: auto;
}
button[kind="secondary"] {
    border: 1px solid #C8A951 !important;
    border-radius: 8px !important;
    background: transparent !important;
    color: #C8A951 !important;
    font-weight: 600 !important;
    position: relative;
    top: -68px;
}
button[kind="secondary"]:hover {
    background: rgba(200,169,81,0.08) !important;
    color: #C8A951 !important;
    border: 1px solid #C8A951 !important;
}
button[kind="secondary"] p { color: #C8A951 !important; }
</style>
""", unsafe_allow_html=True)

    _bc = st.columns([1, 1.4, 1])
    with _bc[1]:
        if st.button("Begin Analysis", key="welcome_ok", width='stretch'):
            st.session_state["welcome_shown"] = True
            st.rerun()
    st.stop()
## ── Global CSS injection ──────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Global app background ── */
.stApp {
    background: linear-gradient(180deg, #081018 0%, #0B1622 100%);
    color: #F3F6FA;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Remove blank gap at top ── */
/* Remove excess padding but keep header for sidebar toggle */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}

/* ── Hide Streamlit chrome but keep header for sidebar toggle ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Sidebar toggle always visible ── */
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: #C8A951 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1723 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    min-width: 240px !important;
}
[data-testid="stSidebar"] * { color: #F3F6FA; }
[data-testid="stSidebar"] label {
    color: #9FB3C8 !important;
    font-size: 12px !important;
}
[data-testid="stSidebar"] .stCheckbox label p {
    color: #C9D1D9 !important;
    font-size: 13px !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    font-weight: 400 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #9FB3C8;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    font-weight: 600;
    margin-bottom: 4px;
}

/* ── Top bar ── */
.top-bar {
    background: linear-gradient(90deg, #0D1723 0%, #0B1622 100%);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 10px 0 12px 0;
    margin-bottom: 12px;
    margin-top: 0px;
}
.top-bar-title {
    font-size: 24px;
    font-weight: 800;
    color: #F3F6FA;
    letter-spacing: -0.3px;
    line-height: 1.2;
}
.top-bar-sub {
    font-size: 12px;
    color: #8EA1B4;
    margin-top: 3px;
}
.gold-text { color: #C8A951; }

/* ── Section header ── */
.section-header {
    background: linear-gradient(90deg, #1E3A5F 0%, #0D1723 100%);
    border-left: 4px solid #C8A951;
    border-radius: 0 10px 10px 0;
    padding: 10px 16px;
    margin: 16px 0 12px 0;
}
.section-header h3 {
    color: #F3F6FA !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    margin: 0 !important;
    letter-spacing: 0.3px;
}
.section-header p {
    color: #8EA1B4 !important;
    font-size: 11px !important;
    margin: 2px 0 0 0 !important;
}

/* ── KPI metric cards ── */
.kpi-card {
    background: linear-gradient(145deg, #0F1C2B, #13263A);
    padding: 16px 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 18px rgba(0,0,0,0.3);
    text-align: center;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border: 1px solid rgba(200,169,81,0.3);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.green::before  { background: #00E676; }
.kpi-card.red::before    { background: #FF5252; }
.kpi-card.amber::before  { background: #C8A951; }
.kpi-card.blue::before   { background: #42A5F5; }
.kpi-label {
    font-size: 11px;
    color: #8EA1B4;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}
.kpi-value        { font-size: 22px; font-weight: 800; color: #F3F6FA; line-height: 1.1; }
.kpi-value.green  { color: #00E676; }
.kpi-value.red    { color: #FF5252; }
.kpi-value.amber  { color: #C8A951; }
.kpi-sub          { font-size: 11px; color: #8EA1B4; margin-top: 4px; }

/* ── Streamlit metric overrides ── */
[data-testid="stMetricValue"] {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #F3F6FA !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    color: #8EA1B4 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #0F1C2B, #13263A) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
    transition: all 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(200,169,81,0.3) !important;
    transform: translateY(-2px);
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid rgba(200,169,81,0.3);
    gap: 4px;
    padding: 4px 4px 0 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: #122235;
    border-radius: 10px 10px 0 0;
    color: #8EA1B4;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 18px;
    border: 1px solid rgba(255,255,255,0.05);
    border-bottom: none;
    transition: all 0.2s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #F3F6FA !important;
    background: #1A3050 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #1E3A5F, #1A2E4A) !important;
    color: #C8A951 !important;
    border-color: rgba(200,169,81,0.3) !important;
}
[data-testid="stTabPanel"] {
    background: linear-gradient(180deg, #0D1723 0%, #081018 100%);
    border: 1px solid rgba(255,255,255,0.04);
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 16px 20px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05) !important;
}

/* ── Form ── */
[data-testid="stForm"] {
    background: linear-gradient(145deg, #0F1C2B, #13263A) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #C8A951, #A8893A) !important;
    color: #081018 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 8px 20px !important;
    font-size: 13px !important;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #D4B660, #C8A951) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(200,169,81,0.3) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #1E3A5F, #162C4A) !important;
    color: #C8A951 !important;
    border: 1px solid rgba(200,169,81,0.4) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    width: 100%;
    transition: all 0.2s ease;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #C8A951, #A8893A) !important;
    color: #081018 !important;
    transform: translateY(-1px);
}

/* ── Info/warning/error boxes ── */
[data-testid="stInfo"]    { background: #0D1723; border-color: #42A5F5; border-radius: 10px; }
[data-testid="stWarning"] { background: #0D1723; border-color: #C8A951; border-radius: 10px; }
[data-testid="stError"]   { background: #0D1723; border-color: #FF5252; border-radius: 10px; }
[data-testid="stSuccess"] { background: #0D1723; border-color: #00E676; border-radius: 10px; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: linear-gradient(145deg, #0F1C2B, #13263A) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
}

/* ── Info box custom ── */
.info-box {
    background: linear-gradient(145deg, #0F1C2B, #13263A);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 4px solid #C8A951;
    border-radius: 0 10px 10px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #C9D1D9;
    margin: 8px 0;
}

/* ── Dividers ── */
hr { border-color: rgba(255,255,255,0.05) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #081018; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 8px; }
::-webkit-scrollbar-thumb:hover { background: #C8A951; }

/* ── Footer ── */
.dash-footer {
    margin-top: 32px;
    padding: 14px 20px;
    background: #0D1723;
    border-top: 1px solid rgba(255,255,255,0.05);
    border-radius: 0 0 12px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

</style>
""", unsafe_allow_html=True)

## ── Top bar rendered AFTER sidebar (so _display_name is available) ───────────
_top_left, _top_right = st.columns([4, 1])

## ─────────────────────────────────────────────────────────────────────────────
##  SIDEBAR
## ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding:4px 0 12px 0;border-bottom:1px solid #21262D;margin-bottom:8px'>
  <div style='font-size:15px;font-weight:700;color:#E6EDF3;letter-spacing:-0.2px'>
    &#9632; <span style='color:#C8A951'>Settings</span>
  </div>
  <div style='font-size:11px;color:#8B949E;margin-top:2px'>Configure analysis parameters</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("**Report Personalisation**")
_salutation  = st.sidebar.selectbox(
    "Salutation", ["Mr.", "Ms.", "Mrs.", "Dr.", "Prof."], index=0
)
_client_name = st.sidebar.text_input(
    "Your Name", placeholder="e.g. User Name", max_chars=50
)
_display_name = f"{_salutation} {_client_name.strip()}" if _client_name.strip() else ""

## ── Top bar — rendered here so _display_name is already known ────────────────
_greeting_html = (
    f' &nbsp;|&nbsp; <span style="color:#F3F6FA;font-weight:600">'
    f'Welcome, <span style="color:#C8A951">{_display_name}</span></span>'
    if _display_name else ""
)
with _top_left:
    st.markdown(f"""
<div class="top-bar">
  <div class="top-bar-title">
    <span style="color:#C8A951">&#9632;</span>
    PortfolioIQ: Risk, Performance & Attribution{_greeting_html}
  </div>
  <div class="top-bar-sub">
    "Understanding downside risk to build stronger portfolio returns"
  </div>
</div>
<div style="margin-bottom:8px"></div>
""", unsafe_allow_html=True)
_pdf_placeholder = st.empty()


st.sidebar.markdown("---")
st.sidebar.markdown("**Portfolio**")
num_stocks = st.sidebar.number_input(
    "Number of stocks in Portfolio", min_value=1, max_value=50, value=4
)
alpha        = st.sidebar.slider("Confidence Level", 0.90, 0.99, value=0.95)
horizon_days = st.sidebar.slider("VaR Horizon (Days)", 1, 30, value=10)

st.sidebar.markdown("---")
st.sidebar.markdown("**VaR Methods**")
use_hist = st.sidebar.checkbox("Historical Method", value=True)
use_para = st.sidebar.checkbox("Parametric Method")
use_mc   = st.sidebar.checkbox("Monte Carlo Simulation")

mc_sims = 0
if use_mc:
    mc_sims = st.sidebar.slider("Monte Carlo Simulations", 1000, 10000, value=5000)

if not (use_hist or use_para or use_mc):
    st.sidebar.warning("⚠️ Select at least one VaR method to see risk analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Beta Settings**")
beta_horizon    = st.sidebar.selectbox("Beta Lookback Period", ["6M", "1Y", "5Y"], index=1)
beta_period_map = {"6M": 126, "1Y": 252, "5Y": 1260}

st.sidebar.markdown("---")
st.sidebar.markdown("**Benchmark**")

## Benchmark options — NSE and BSE
## Per SEBI/NSE methodology: Nifty 100 = Large Cap (top 100), Midcap 150 = Mid Cap, etc.
BENCHMARK_OPTIONS = {
    "Auto-detect (Recommended)":      None,
    "Nifty 100 (NSE Large Cap)":      ("^CNX100",           "Nifty 100"),
    "Nifty 50 (NSE Top 50)":          ("^NSEI",             "Nifty 50"),
    "Nifty Midcap 150 (NSE Mid)":     ("NIFTYMIDCAP150.NS", "Nifty Midcap 150"),
    "Nifty Smallcap 250 (NSE Small)": ("NIFTYSMLCAP250.NS", "Nifty Smallcap 250"),
    "Nifty 500 (NSE Broad)":          ("^CRSLDX",          "Nifty 500"),
    "Sensex (BSE Large Cap)":         ("^BSESN",            "Sensex"),
    "BSE Midcap":                     ("BSE-MIDCAP.BO",  "BSE Midcap"),
    "BSE Smallcap":                   ("BSE-SMLCAP.BO",  "BSE Smallcap"),
    "BSE 500":                        ("BSE-500.BO",     "BSE 500"),
}
selected_benchmark_label = st.sidebar.selectbox(
    "Benchmark Index",
    list(BENCHMARK_OPTIONS.keys()),
    index=0,
    help="Auto-detect picks the best benchmark based on your stocks exchange and market cap"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Auto Refresh**")
auto_refresh    = st.sidebar.checkbox("Enable Auto Refresh", value=True)
refresh_minutes = st.sidebar.selectbox("Refresh Interval (Minutes)", [15, 20, 30], index=1)
ttl_seconds     = refresh_minutes * 60
refresh_ms      = refresh_minutes * 60 * 1000

if auto_refresh:
    st_autorefresh(interval=refresh_ms, key="data_refresh")

## ─────────────────────────────────────────────────────────────────────────────
# %%  HOLDINGS TABLE
## ─────────────────────────────────────────────────────────────────────────────
## ── Holdings form — collapsible after first submission ──────────────────────

## ── Fallback cap classification — used when Yahoo Finance returns no marketCap ──
_NIFTY50_TICKERS = {
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","KOTAKBANK",
    "HINDUNILVR","SBIN","BHARTIARTL","ITC","AXISBANK","LT",
    "BAJFINANCE","ASIANPAINT","MARUTI","TITAN","SUNPHARMA",
    "ULTRACEMCO","NESTLEIND","WIPRO","HCLTECH","TECHM","TATASTEEL",
    "JSWSTEEL","POWERGRID","NTPC","ONGC","COALINDIA","ADANIENT",
    "ADANIPORTS","DRREDDY","CIPLA","DIVISLAB","APOLLOHOSP",
    "BAJAJFINSV","BAJAJ-AUTO","EICHERMOT","HEROMOTOCO","MM",
    "TATAMOTORS","TATACONSUM","BRITANNIA","BPCL","GRASIM",
    "HINDALCO","INDUSINDBK","SBILIFE","HDFCLIFE","ICICIPRULI",
    "VEDL","SHRIRAMFIN","BEL","TRENT","JSWENERGY","NHPC",
}
_KNOWN_MIDCAP_TICKERS = {
    "MPHASIS","PERSISTENT","COFORGE","LTIM","LTTS","NAUKRI",
    "AUBANK","BALKRISIND","BANDHANBNK","BATAINDIA","BERGEPAINT",
    "BIOCON","CROMPTON","CUMMINSIND","DALBHARAT","DEEPAKNTR",
    "DIXON","FEDERALBNK","GODREJCP","GODREJPROP","HDFCAMC",
    "IDFCFIRSTB","INDIAMART","IGL","JKCEMENT","KANSAINER",
    "LICHSGFIN","MARICO","METROPOLIS","MUTHOOTFIN","NMDC",
    "OFSS","PAGEIND","PIIND","POLYCAB","RADICO","RAMCOCEM",
    "SAIL","SRF","SUPREMEIND","TORNTPHARM","TORNTPOWER",
    "VOLTAS","ZOMATO","PAYTM","NYKAA","IRCTC","IRFC","RVNL",
}

## ── ETF Framework — Tiered Classification ────────────────────────────────────
## Tier 1: Sector ETFs  → direct sector index benchmark
## Tier 2: Broad Market → benchmark itself (near-zero active contribution)
## Tier 3: Thematic     → Nifty 500 proxy with methodology note
## Tier 4: Unknown      → excluded from BHB, user notified
##
## benchmark_weight is NOT hardcoded here — it is fetched dynamically
## from NIFTY_SECTOR_WEIGHTS at runtime so it stays current
## ─────────────────────────────────────────────────────────────────────────────

## Tier 1 — Sector ETFs: keyword → (sector_label, attribution_benchmark_ticker)
ETF_SECTOR_MAP = {
    ## Banking / Financial
    "bank":           ("Financial Services",  "^NSEBANK"),
    "banking":        ("Financial Services",  "^NSEBANK"),
    "nsebank":        ("Financial Services",  "^NSEBANK"),
    "financial":      ("Financial Services",  "^NSEBANK"),
    ## Technology / IT
    "it ":            ("Technology",          "^CNXIT"),
    " it":            ("Technology",          "^CNXIT"),
    "nifty it":       ("Technology",          "^CNXIT"),
    "infotech":       ("Technology",          "^CNXIT"),
    "information tech":("Technology",         "^CNXIT"),
    "tech etf":       ("Technology",          "^CNXIT"),
    ## Pharma / Healthcare
    "pharma":         ("Healthcare",          "^CNXPHARMA"),
    "healthcare":     ("Healthcare",          "^CNXPHARMA"),
    "health care":    ("Healthcare",          "^CNXPHARMA"),
    ## Auto / Consumer Cyclical
    "auto":           ("Consumer Cyclical",   "^CNXAUTO"),
    "automobile":     ("Consumer Cyclical",   "^CNXAUTO"),
    ## FMCG / Consumer Defensive
    "fmcg":           ("Consumer Defensive",  "^CNXFMCG"),
    ## Energy
    "energy":         ("Energy",              "^CNXENERGY"),
    "oil":            ("Energy",              "^CNXENERGY"),
    ## Metal / Mining
    "metal":          ("Basic Materials",     "^CNXMETAL"),
    "mining":         ("Basic Materials",     "^CNXMETAL"),
    ## Realty
    "realt":          ("Real Estate",         "^CNXREALTY"),
    ## Infrastructure
    "infra":          ("Industrials",         "^CNXINFRA"),
    "infrastructure": ("Industrials",         "^CNXINFRA"),
}

## Tier 2 — Broad Market ETFs: keyword list (active contribution ≈ zero)
ETF_BROAD_MARKET_KEYWORDS = [
    ## Nifty 50 variants — SPECIFIC only, not generic 'nifty'
    "nifty 50 ", "nifty50", "nifty bees", "niftybees",
    "n50", "nif50",
    ## Nifty 100 / Next 50 / Nifty 500
    "nifty 100", "nifty100", "nifty next 50", "niftynext",
    "nifty 500", "nifty500",
    ## Sensex / BSE broad
    "sensex", "bse 500", "bse500",
    ## Generic Nifty ETF names — "HDFC Nifty ETF", "Kotak Nifty ETF"
    ## NOTE: "nifty etf" must NOT match "nifty it etf" or "nifty auto etf"
    ## So we keep it specific
    "nifty etf",
    ## Catches "SBI ETF Nifty 50", "Axis ETF Nifty 50" etc where order differs
    "etf nifty 50",
]

## Tier 3 — Thematic ETFs: keyword → (theme_label, proxy_benchmark)
## Proxy = Nifty 500 (broadest available on Yahoo Finance)
## Label shown to user: "Nifty 500 (Proxy)" — transparent about limitation
ETF_THEMATIC_MAP = {
    "defence":     ("Defence & Aerospace",   "^CRSLDX"),
    "defense":     ("Defence & Aerospace",   "^CRSLDX"),
    "psu":         ("PSU Theme",             "^CNXPSE"),
    "public sector":("PSU Theme",            "^CNXPSE"),
    "manufactur":  ("Manufacturing",         "^CRSLDX"),
    "mobility":    ("Mobility / EV",         "NIFTYMIDCAP150.NS"),
    "ev ":         ("EV / Clean Energy",     "NIFTYMIDCAP150.NS"),
    "electric":    ("EV / Clean Energy",     "NIFTYMIDCAP150.NS"),
    "midcap":      ("Mid Cap",               "NIFTYMIDCAP150.NS"),
    "mid cap":     ("Mid Cap",               "NIFTYMIDCAP150.NS"),
    "smallcap":    ("Small Cap",             "^CRSLDX"),
    "small cap":   ("Small Cap",             "^CRSLDX"),
    ## PSU / CPSE
    "cpse":        ("PSU Theme",             "^CNXPSE"),
    "bharat 22":   ("PSU Theme",             "^CNXPSE"),
    ## Infra
    "infrabees":   ("Industrials",           "^CNXINFRA"),
    ## Factor / Smart Beta ETFs
    "low vol":     ("Factor ETF",            "^CRSLDX"),
    "low volat":   ("Factor ETF",            "^CRSLDX"),
    "momentum":    ("Factor ETF",            "^CRSLDX"),
    "quality":     ("Factor ETF",            "^CRSLDX"),
    ## Consumption / India themes
    "consumption": ("Consumption Theme",     "^CRSLDX"),
    ## REIT / InvIT
    "reit":        ("Real Estate",           "^CNXREALTY"),
    "invit":       ("Real Estate",           "^CNXREALTY"),
}

## Tier 4 — Non-equity ETFs: these are EXCLUDED from BHB
## No equity sector benchmark makes sense for them
ETF_EXCLUDE_KEYWORDS = [
    "gold", "silver", "commodity", "liquid", "overnight",
    "money market", "bond", "gilt", "debt", "bharat bond",
    "nasdaq", "s&p 500", "s&p500", "hangseng", "hang seng",
    "global", "international", "world", "us equity",
    "nyse", "fang", "us tech", "china", "taiwan",
    "japan", "europe", "emerging market", "mirae asset nyse",
]

## Manual overrides — known ETFs where name-matching might fail
ETF_DEFINITIONS = {
    ## ── Defence ETFs (Tier 3 Thematic) ──────────────────────────────────────
    "GROWWDEFNC.NS": {
        "sector": "Defence & Aerospace", "industry": "ETF — Nifty India Defence Index",
        "is_etf": True, "tracks": "Nifty India Defence Index",
        "etf_tier": 3, "attribution_bm": "^CRSLDX",
        "attribution_bm_label": "Nifty 500 (Proxy)",
    },
    "MODEFENCE.NS": {
        "sector": "Defence & Aerospace", "industry": "ETF — Nifty India Defence Index",
        "is_etf": True, "tracks": "Nifty India Defence Index",
        "etf_tier": 3, "attribution_bm": "^CRSLDX",
        "attribution_bm_label": "Nifty 500 (Proxy)",
    },
    ## ── Nifty 50 ETFs (Tier 2 Broad Market) ─────────────────────────────────
    "NIFTYBEES.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    ## HDFCNIFETF.NS is delisted on Yahoo Finance — keeping for legacy support
    ## Users should use HDFCNIFTY.NS instead
    "HDFCNIFETF.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index (Delisted on Yahoo)",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "HDFCNIFTY.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "HDFCNIF50.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "ICICIB22.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "UTINIFTETF.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "SBIETFNIF50.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    ## ── Gold ETFs (Tier 4 Non-Equity) ────────────────────────────────────────
    "GOLDBEES.NS": {
        "sector": "Non-Equity ETF", "industry": "ETF — Gold (Physical)",
        "is_etf": True, "tracks": "Physical Gold",
        "etf_tier": 4, "attribution_bm": None,
        "attribution_bm_label": "Excluded — non-equity ETF",
    },
    "HDFCMFGETF.NS": {
        "sector": "Non-Equity ETF", "industry": "ETF — Gold (Physical)",
        "is_etf": True, "tracks": "Physical Gold",
        "etf_tier": 4, "attribution_bm": None,
        "attribution_bm_label": "Excluded — non-equity ETF",
    },
    "NIPPONBEES.NS": {
        "sector": "Non-Equity ETF", "industry": "ETF — Gold (Physical)",
        "is_etf": True, "tracks": "Physical Gold",
        "etf_tier": 4, "attribution_bm": None,
        "attribution_bm_label": "Excluded — non-equity ETF",
    },
    ## ── Bank ETFs (Tier 1 Sector) ─────────────────────────────────────────────
    "BANKBEES.NS": {
        "sector": "Financial Services", "industry": "ETF — Nifty Bank Index",
        "is_etf": True, "tracks": "Nifty Bank",
        "etf_tier": 1, "attribution_bm": "^NSEBANK",
        "attribution_bm_label": "^NSEBANK",
    },
    "HDFCNIFBAN.NS": {
        "sector": "Financial Services", "industry": "ETF — Nifty Bank Index",
        "is_etf": True, "tracks": "Nifty Bank",
        "etf_tier": 1, "attribution_bm": "^NSEBANK",
        "attribution_bm_label": "^NSEBANK",
    },
}

def classify_etf(ticker, info_data):
    """
    Auto-classify any ETF into one of four tiers using its Yahoo Finance name.
    Returns a dict compatible with ETF_DEFINITIONS format.
    Falls back gracefully at every tier.
    """
    name = (
        (info_data.get("longName",  "") or "") + " " +
        (info_data.get("shortName", "") or "")
    ).lower().strip()

    ## Check manual override first
    if ticker in ETF_DEFINITIONS:
        return ETF_DEFINITIONS[ticker]

    ## Tier 4 — Non-equity: exclude from BHB entirely (check first — gold/liquid must not go to sector)
    for kw in ETF_EXCLUDE_KEYWORDS:
        if kw in name:
            return {
                "sector":       "Non-Equity ETF",
                "industry":     f"ETF — {info_data.get('longName', ticker)}",
                "is_etf":       True,
                "tracks":       info_data.get("longName", "Unknown Index"),
                "etf_tier":     4,
                "attribution_bm": None,
                "attribution_bm_label": "Excluded — non-equity ETF",
            }

    ## Tier 1 — Sector ETF (check BEFORE broad market)
    ## "Nifty IT ETF" must go to Tier 1, not Tier 2 via generic "nifty" match
    for kw, (sector, bm_ticker) in ETF_SECTOR_MAP.items():
        if kw in name:
            return {
                "sector":       sector,
                "industry":     f"ETF — {info_data.get('longName', ticker)}",
                "is_etf":       True,
                "tracks":       info_data.get("longName", "Sector Index"),
                "etf_tier":     1,
                "attribution_bm": bm_ticker,
                "attribution_bm_label": bm_ticker,
            }

    ## Tier 2 — Broad market: active contribution ≈ zero
    for kw in ETF_BROAD_MARKET_KEYWORDS:
        if kw in name:
            return {
                "sector":       "Broad Market ETF",
                "industry":     f"ETF — {info_data.get('longName', ticker)}",
                "is_etf":       True,
                "tracks":       info_data.get("longName", "Broad Market Index"),
                "etf_tier":     2,
                "attribution_bm": None,
                "attribution_bm_label": "Primary Benchmark (tracks index)",
            }

    ## Tier 3 — Thematic ETF: use Nifty 500 proxy
    for kw, (theme, bm_ticker) in ETF_THEMATIC_MAP.items():
        if kw in name:
            return {
                "sector":       theme,
                "industry":     f"ETF — {info_data.get('longName', ticker)}",
                "is_etf":       True,
                "tracks":       info_data.get("longName", "Thematic Index"),
                "etf_tier":     3,
                "attribution_bm": bm_ticker,
                "attribution_bm_label": f"{bm_ticker} (Proxy)",
            }

    ## Tier 4 fallback — unrecognised ETF → exclude from BHB
    return {
        "sector":       "Unknown ETF",
        "industry":     f"ETF — {info_data.get('longName', ticker)}",
        "is_etf":       True,
        "tracks":       info_data.get("longName", "Unknown"),
        "etf_tier":     4,
        "attribution_bm": None,
        "attribution_bm_label": "Not mapped — excluded from BHB",
    }

## Manual sector overrides for stocks Yahoo Finance cannot classify correctly
## e.g. newly listed stocks, demerged entities, small caps
MANUAL_SECTOR_MAP = {  ## Priority 3 fallback when Yahoo returns Unknown
    "TMCV.NS":"Consumer Cyclical","TMPV.NS":"Consumer Cyclical",
    "TITAN.NS":"Consumer Durables","CRISIL.NS":"Financial Services",
    "DABUR.NS":"Consumer Defensive","ASIANPAINT.NS":"Basic Materials",
    "TECHM.NS":"Technology","WIPRO.NS":"Technology","INFY.NS":"Technology",
    "TCS.NS":"Technology","HCLTECH.NS":"Technology","LTIM.NS":"Technology",
    "PERSISTENT.NS":"Technology","COFORGE.NS":"Technology","MPHASIS.NS":"Technology",
    "BAJFINANCE.NS":"Financial Services","BAJAJFINSV.NS":"Financial Services",
    "HDFCBANK.NS":"Financial Services","ICICIBANK.NS":"Financial Services",
    "KOTAKBANK.NS":"Financial Services","AXISBANK.NS":"Financial Services",
    "SBIN.NS":"Financial Services","SBILIFE.NS":"Financial Services",
    "HDFCLIFE.NS":"Financial Services","SHRIRAMFIN.NS":"Financial Services",
    "RELIANCE.NS":"Energy","ONGC.NS":"Energy","BPCL.NS":"Energy",
    "IOC.NS":"Energy","GAIL.NS":"Energy","NTPC.NS":"Utilities",
    "POWERGRID.NS":"Utilities","NHPC.NS":"Utilities","TATAPOWER.NS":"Utilities",
    "SUNPHARMA.NS":"Healthcare","DRREDDY.NS":"Healthcare","CIPLA.NS":"Healthcare",
    "DIVISLAB.NS":"Healthcare","APOLLOHOSP.NS":"Healthcare",
    "MARUTI.NS":"Consumer Cyclical","TATAMOTORS.NS":"Consumer Cyclical",
    "MM.NS":"Consumer Cyclical","BAJAJ-AUTO.NS":"Consumer Cyclical",
    "HEROMOTOCO.NS":"Consumer Cyclical","EICHERMOT.NS":"Consumer Cyclical",
    "ITC.NS":"Consumer Defensive","HINDUNILVR.NS":"Consumer Defensive",
    "NESTLEIND.NS":"Consumer Defensive","BRITANNIA.NS":"Consumer Defensive",
    "MARICO.NS":"Consumer Defensive","TATACONSUM.NS":"Consumer Defensive",
    "LT.NS":"Industrials","BEL.NS":"Industrials","SIEMENS.NS":"Industrials",
    "ABB.NS":"Industrials","HAVELLS.NS":"Industrials",
    "BHARTIARTL.NS":"Communication Services",
    "TATASTEEL.NS":"Basic Materials","JSWSTEEL.NS":"Basic Materials",
    "HINDALCO.NS":"Basic Materials","ULTRACEMCO.NS":"Basic Materials",
    "SAIL.NS":"Basic Materials","NMDC.NS":"Basic Materials",
    "GODREJPROP.NS":"Real Estate","DLF.NS":"Real Estate",
    "VOLTAS.NS":"Consumer Durables",
    ## Stocks confirmed returning N/A on Streamlit Cloud
    "BHARATFORG.NS":"Industrials",
    "OFSS.NS":"Technology",
    "AMBUJACEM.NS":"Basic Materials",
    "PFIZER.NS":"Healthcare",
    ## Additional common NSE stocks
    "ADANIENT.NS":"Industrials","ADANIPORTS.NS":"Industrials",
    "ADANIGREEN.NS":"Utilities","ADANIPOWER.NS":"Utilities",
    "PIIND.NS":"Basic Materials","DEEPAKNTR.NS":"Basic Materials",
    "ASTRAL.NS":"Industrials","POLYCAB.NS":"Industrials",
    "DMART.NS":"Consumer Defensive","TRENT.NS":"Consumer Cyclical",
    "NYKAA.NS":"Consumer Cyclical","ZOMATO.NS":"Consumer Services",
    "PAYTM.NS":"Financial Services","IRCTC.NS":"Industrials",
    "IRFC.NS":"Financial Services","RVNL.NS":"Industrials",
    "PFC.NS":"Financial Services","RECLTD.NS":"Financial Services",
    "HUDCO.NS":"Financial Services","IREDA.NS":"Financial Services",
    "CANBK.NS":"Financial Services","PNB.NS":"Financial Services",
    "BANKBARODA.NS":"Financial Services","FEDERALBNK.NS":"Financial Services",
    "IDFCFIRSTB.NS":"Financial Services","AUBANK.NS":"Financial Services",
    "INDUSINDBK.NS":"Financial Services",
    "TORNTPHARM.NS":"Healthcare","BIOCON.NS":"Healthcare",
    "GLENMARK.NS":"Healthcare","ALKEM.NS":"Healthcare",
    "JSWENERGY.NS":"Utilities","TORNTPOWER.NS":"Utilities",
    "DALBHARAT.NS":"Basic Materials","SHREECEM.NS":"Basic Materials",
    "RAMCOCEM.NS":"Basic Materials","JKCEMENT.NS":"Basic Materials",
    "IGL.NS":"Energy","MGL.NS":"Energy","GUJGASLTD.NS":"Energy",
    "VBL.NS":"Consumer Defensive","RADICO.NS":"Consumer Defensive",
    "JUBLFOOD.NS":"Consumer Services","DEVYANI.NS":"Consumer Services",
    "WESTLIFE.NS":"Consumer Services","SAPIENT.NS":"Technology",
    "KPITTECH.NS":"Technology","TATAELXSI.NS":"Technology",
    "KAYNES.NS":"Industrials","DIXON.NS":"Industrials",
    "APLAPOLLO.NS":"Industrials","GRINDWELL.NS":"Industrials",
}

## ─────────────────────────────────────────────────────────────────────────────
# %%  DATA FETCHERS
## ─────────────────────────────────────────────────────────────────────────────

## ─────────────────────────────────────────────────────────────────────────────
##  QUOTE OF THE DAY
## ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_QUOTES = [
    ("The ability to observe without evaluating is the highest form of intelligence.", "J. Krishnamurti"),
    ("You are not a drop in the ocean. You are the entire ocean in a drop.", "Rumi"),
    ("Muddy water is best cleared by leaving it alone.", "Alan Watts"),
    ("Nature does not hurry, yet everything is accomplished.", "Lao Tzu"),
    ("The wound is the place where the light enters you.", "Rumi"),
    ("Yesterday I was clever, so I wanted to change the world. Today I am wise, so I am changing myself.", "Rumi"),
    ("The quieter you become, the more you are able to hear.", "Rumi"),
    ("Until you make the unconscious conscious, it will direct your life and you will call it fate.", "Carl Jung"),
    ("The privilege of a lifetime is to become who you truly are.", "Carl Jung"),
    ("To the mind that is still, the whole universe surrenders.", "Lao Tzu"),
    ("What you seek is seeking you.", "Rumi"),
    ("There is a crack in everything. That is how the light gets in.", "Leonard Cohen"),
    ("The present moment always will have been.", "Alan Watts"),
    ("We are shaped by our thoughts; we become what we think.", "Buddha"),
    ("Do not go where the path may lead, go instead where there is no path and leave a trail.", "Ralph Waldo Emerson"),
]

## ── AMFI Cap Classification loader ──────────────────────────────────────────
## Source: amfi_cap_classification.csv (AMFI Jun 2026 list)
## Per SEBI Circular Oct 6, 2017: Top 100=Large Cap, 101-250=Mid Cap, 251+=Small Cap
## Update every January and July from: amfiindia.com/research-information/other-data
@st.cache_data(ttl=86400*180)  ## refresh every 6 months
def load_amfi_cap_map():
    try:
        import os
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "amfi_cap_classification.csv")
        _df = pd.read_csv(_p, dtype=str).fillna("")
        _n  = {str(r["NSE"]).strip(): r["Category"]
               for _, r in _df.iterrows() if str(r["NSE"]).strip()}
        _b  = {str(r["BSE"]).strip(): r["Category"]
               for _, r in _df.iterrows() if str(r["BSE"]).strip()}
        return _n, _b
    except Exception:
        return {}, {}

_AMFI_NSE_MAP, _AMFI_BSE_MAP = load_amfi_cap_map()


def enrich_info(raw_ticker):
    """
    Auto-detect exchange, resolve missing suffix, classify cap category.
    Returns dict with keys: ticker, exchange, cap_category, market_cap, display_name
    """
    raw = str(raw_ticker).strip().upper()

    ## ── Step 1: determine exchange suffix ────────────────────────────────────
    if raw.endswith(".NS"):
        candidates = [raw]
        exchange   = "NSE"
    elif raw.endswith(".BO"):
        candidates = [raw]
        exchange   = "BSE"
    else:
        ## No suffix — try NSE first, then BSE
        candidates = [raw + ".NS", raw + ".BO"]
        exchange   = None

    ## ── Step 2: try each candidate until one returns data ────────────────────
    resolved_ticker = raw  ## fallback
    info_data       = {}
    for cand in candidates:
        try:
            info_data = yf.Ticker(cand).info
            ## Check if we got real data back
            if info_data.get("regularMarketPrice") or info_data.get("currentPrice") or info_data.get("marketCap"):
                resolved_ticker = cand
                exchange = "NSE" if cand.endswith(".NS") else "BSE"
                break
        except Exception:
            continue

    ## ── Step 3: classify market cap ──────────────────────────────────────────
    market_cap = info_data.get("marketCap", 0) or 0

    ## ── Cap classification — AMFI official list (primary source) ──────────
    ## Per SEBI Circular Oct 6, 2017: Top 100=Large, 101-250=Mid, 251+=Small
    ## Source: amfi_cap_classification.csv (AMFI Jun 2026)
    ## Update: every January and July from amfiindia.com
    _base = raw.replace(".NS","").replace(".BO","")
    _exch = "NSE" if resolved_ticker.endswith(".NS") else "BSE"
    ## Priority 1: AMFI official classification (most accurate)
    if _exch == "NSE" and _base in _AMFI_NSE_MAP:
        cap_cat = _AMFI_NSE_MAP[_base]
    elif _exch == "BSE" and _base in _AMFI_BSE_MAP:
        cap_cat = _AMFI_BSE_MAP[_base]
    elif _base in _AMFI_NSE_MAP:
        cap_cat = _AMFI_NSE_MAP[_base]
    elif _base in _AMFI_BSE_MAP:
        cap_cat = _AMFI_BSE_MAP[_base]
    ## Priority 2: Yahoo marketCap fallback (new IPOs not yet in AMFI list)
    elif market_cap >= 290_000_000_000:
        cap_cat = "Large Cap"
    elif market_cap >= 85_000_000_000:
        cap_cat = "Mid Cap"
    elif market_cap > 0:
        cap_cat = "Small Cap"
    else:
        cap_cat = "Unknown"  ## New IPO not yet classified by AMFI

    ## ── Step 4: detect ETF — check manual definitions first, then quoteType ──
    ## Manual definitions take priority — handles cases where Yahoo quoteType is missing
    if resolved_ticker in ETF_DEFINITIONS or raw in ETF_DEFINITIONS:
        _etf_def_key = resolved_ticker if resolved_ticker in ETF_DEFINITIONS else raw
        etf_info     = ETF_DEFINITIONS[_etf_def_key]
        ## Check if this is a delisted ticker (industry contains "Delisted")
        _is_delisted = "Delisted" in etf_info.get("industry", "")
        return {
            "ticker":       resolved_ticker,
            "exchange":     exchange or "NSE",
            "cap_category": "ETF",
            "market_cap":   market_cap,
            "display_name": info_data.get("shortName", resolved_ticker),
            "is_etf":       True,
            "etf_info":     etf_info,
            "sector":       etf_info.get("sector", "ETF"),
            "delisted":     _is_delisted,
        }

    quote_type = info_data.get("quoteType", "")
    if quote_type == "ETF":
        etf_info = classify_etf(resolved_ticker, info_data)
        return {
            "ticker":       resolved_ticker,
            "exchange":     exchange or "Unknown",
            "cap_category": "ETF",
            "market_cap":   market_cap,
            "display_name": info_data.get("shortName", resolved_ticker),
            "is_etf":       True,
            "etf_info":     etf_info,
            "sector":       etf_info["sector"],
        }

    ## Get sector from Yahoo Finance info (already fetched above)
    _sector_yf = info_data.get("sector", None) or None

    ## If sector is missing from .info, try quoteSummary API directly
    ## This is more reliable for NSE stocks from cloud server IPs
    if not _sector_yf or _sector_yf == "Unknown":
        try:
            import requests as _req2
            _url2 = f"https://query2.finance.yahoo.com/v11/finance/quoteSummary/{resolved_ticker}"
            _r2   = _req2.get(_url2, params={"modules":"assetProfile"},
                              headers={"User-Agent":"Mozilla/5.0"}, timeout=5)
            if _r2.status_code == 200:
                _prof = _r2.json().get("quoteSummary",{}).get("result",[{}])
                if _prof:
                    _s = _prof[0].get("assetProfile",{}).get("sector","")
                    if _s and _s != "Unknown":
                        _sector_yf = _s
        except Exception:
            pass

    return {
        "ticker":       resolved_ticker,
        "exchange":     exchange or "Unknown",
        "cap_category": cap_cat,
        "market_cap":   market_cap,
        "display_name": info_data.get("shortName", resolved_ticker),
        "is_etf":       False,
        "etf_info":     None,
        "sector":       _sector_yf,
    }
import datetime as _dt_hdf
_default_date = _dt_hdf.date.today() - _dt_hdf.timedelta(days=365)
if "holdings_df" not in st.session_state:
    st.session_state["holdings_df"] = pd.DataFrame({
        "Ticker":    [""] * num_stocks,
        "Buy Date":  [_default_date] * num_stocks,
        "Buy Price": [0.0] * num_stocks,
        "Quantity":  [0] * num_stocks
    })

_cur_df  = st.session_state["holdings_df"]
_cur_len = len(_cur_df)

if _cur_len < num_stocks:
    ## Growing — add empty rows to reach num_stocks
    _extra = pd.DataFrame({
        "Ticker":    [""] * (num_stocks - _cur_len),
        "Buy Date":  [_default_date] * (num_stocks - _cur_len),
        "Buy Price": [0.0] * (num_stocks - _cur_len),
        "Quantity":  [0] * (num_stocks - _cur_len)
    })
    st.session_state["holdings_df"] = pd.concat([_cur_df, _extra], ignore_index=True)

elif _cur_len > num_stocks:
    ## Shrinking — only remove EMPTY rows, never delete rows with real ticker data
    _filled_mask = _cur_df["Ticker"].astype(str).str.strip().ne("")
    _filled_count = _filled_mask.sum()
    if _filled_count <= num_stocks:
        ## Keep all filled rows + enough empty rows to reach num_stocks
        _filled_rows = _cur_df[_filled_mask]
        _empty_rows  = _cur_df[~_filled_mask].head(max(0, num_stocks - _filled_count))
        st.session_state["holdings_df"] = pd.concat(
            [_filled_rows, _empty_rows], ignore_index=True
        )
    ## else: more filled rows than num_stocks requested — keep all filled rows as-is
    ## (don't delete user's actual data just because they lowered the count)
if "show_holdings_form" not in st.session_state:
    st.session_state["show_holdings_form"] = True

if st.session_state.get("portfolio_submitted", False) and not st.session_state["show_holdings_form"]:
    hdf_preview = st.session_state["holdings_df"]
    tickers_str = ", ".join(hdf_preview["Ticker"].tolist())
    col_info, col_btn = st.columns([5, 1])
    with col_info:
     bar_html = (
            '<div style="padding:10px 16px;background:linear-gradient(145deg,#0F1C2B,#13263A);'
            'border:1px solid rgba(255,255,255,0.06);border-left:4px solid #C8A951;'
            'border-radius:0 10px 10px 0;font-size:13px;color:#C9D1D9;margin-bottom:8px">'
            + (f'<span style="color:#C8A951;font-weight:700">{_display_name}</span> &nbsp;|&nbsp; ' if _display_name else '')
            + '<strong style="color:#C8A951">&#9632; Portfolio loaded:</strong>'
            + f' &nbsp; {tickers_str} &nbsp;|&nbsp; {len(hdf_preview)} positions'
            + ' &nbsp;|&nbsp; <span style="color:#8EA1B4">Click Edit to modify</span>'
            + '</div>'
        )
    with col_btn:
        if st.button("✏️ Edit Holdings", key="edit_holdings_btn"):
            st.session_state["show_holdings_form"] = True
            st.rerun()
    holdings_df = st.session_state["holdings_df"].copy()
    submitted   = False

else:
    st.markdown(
        '<div class="section-header"><h3>&#9998; Enter Holdings Details</h3>'
        '<p>Add your portfolio positions — ticker, buy date, buy price, quantity</p></div>',
        unsafe_allow_html=True
    )


    ## ── Company name search — Approach B Quick-Add ─────────────────────────
    def _search_nse_ticker(query):
        try:
            import requests as _rq
            _r = _rq.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q":query,"quotesCount":10,"newsCount":0,
                        "enableFuzzyQuery":True,"region":"IN"},
                headers={"User-Agent":"Mozilla/5.0"}, timeout=5
            )
            return [
                {"name": q.get("shortname", q.get("longname","")),
                 "ticker": q["symbol"]}
                for q in _r.json().get("quotes",[])
                if q.get("symbol","").endswith((".NS",".BO"))
                and q.get("quoteType") in ("EQUITY","ETF","MUTUALFUND")
            ]
        except Exception:
            return []

    _TICKER_CHANGES = {
        "HDFCNIFETF.NS":     ("HDFCNIFTY.NS",  "HDFC AMC renamed this ETF in 2023"),
        "HDFC.NS":           ("HDFCBANK.NS",   "HDFC Ltd merged into HDFC Bank in 2023"),
        "ZEEL.NS":           ("ZEEENT.NS",     "Zee Entertainment ticker changed"),
        "MINDTREE.NS":       ("LTIM.NS",       "Mindtree merged into LTIMindtree in 2023"),
        "LTECH.NS":          ("LTIM.NS",       "L&T Infotech merged into LTIMindtree"),
        "TATAMOTORS-DVR.NS": ("TATAMOTORS.NS", "DVR shares merged with ordinary in 2021"),
    }

    for _k in ["qa_ticker","qa_name","qa_warning"]:
        if _k not in st.session_state:
            st.session_state[_k] = ""

    st.markdown(
        "<p style='font-size:12px;font-weight:600;color:#C8A951;margin:6px 0 3px 0'>"
        "Search &amp; Add — find company, fill details, add to portfolio</p>",
        unsafe_allow_html=True
    )
    _csearch = st.text_input(
        "Search", placeholder="e.g. Reliance, HDFC Bank, Gold ETF...",
        label_visibility="collapsed", key="company_search"
    )
    if _csearch and len(_csearch.strip()) >= 3:
        with st.spinner("Searching..."):
            _results = _search_nse_ticker(_csearch.strip())
        if _results:
            _sel = st.selectbox(
                "Select",
                ["select company"] + [f"{r['name']}  ({r['ticker']})" for r in _results],
                label_visibility="collapsed", key="sel_company"
            )
            if _sel and _sel != "select company":
                _raw_t = _sel.split("(")[-1].rstrip(")")
                _raw_n = _sel.split("(")[0].strip()
                if _raw_t in _TICKER_CHANGES:
                    _new_t, _rsn = _TICKER_CHANGES[_raw_t]
                    st.session_state["qa_ticker"]  = _new_t
                    st.session_state["qa_name"]    = _raw_n
                    st.session_state["qa_warning"] = f"Ticker changed: {_raw_t} -> {_new_t} | {_rsn}"
                else:
                    st.session_state["qa_ticker"]  = _raw_t
                    st.session_state["qa_name"]    = _raw_n
                    st.session_state["qa_warning"] = ""
        else:
            st.caption("No results. Try a shorter name or type the ticker directly.")

    if st.session_state.get("qa_ticker"):
        _qt = st.session_state["qa_ticker"]
        _qn = st.session_state["qa_name"]
        if st.session_state.get("qa_warning"):
            st.warning(st.session_state["qa_warning"])
        st.markdown(
            f"<div style='padding:8px 12px;background:#161B22;"
            f"border:0.5px solid #30363D;border-radius:8px;margin:4px 0'>"
            f"<b style='color:#F3F6FA'>{_qn}</b>"
            f"<code style='color:#58A6FF;margin-left:8px'>{_qt}</code></div>",
            unsafe_allow_html=True
        )
        _c1,_c2,_c3,_c4 = st.columns([2,2,2,1])
        with _c1:
            _qprice = st.number_input("Buy Price Rs", min_value=0.01,
                                      value=1.0, format="%.2f", key="qa_price")
        with _c2:
            _qqty   = st.number_input("Quantity", min_value=1,
                                      value=1, step=1, key="qa_qty")
        with _c3:
            import datetime as _dt
            _qdate  = st.date_input(
                "Buy Date",
                value=_dt.date.today(),
                min_value=_dt.date(1990, 1, 1),
                max_value=_dt.date.today(),
                key="qa_date"
            )
        with _c4:
            st.markdown("<div style='margin-top:22px'></div>", unsafe_allow_html=True)
            _qadd   = st.button("Add", key="qa_add_btn", width='stretch')
        if _qadd:
            if _qprice <= 0:
                st.error("Enter a valid buy price.")
            else:
                import pandas as pd
                ## Store _qdate as datetime.date directly — NOT as string
                ## String causes type mismatch crash in data_editor on rerun
                _new_row = pd.DataFrame([{
                    "Ticker": _qt, "Buy Date": _qdate,
                    "Buy Price": float(_qprice), "Quantity": int(_qqty)
                }])
                _df = st.session_state.get("holdings_df",
                    pd.DataFrame(columns=["Ticker","Buy Date","Buy Price","Quantity"]))
                _empty = _df["Ticker"].astype(str).str.strip().eq("")
                if _empty.any():
                    _idx = _empty.idxmax()
                    _df.at[_idx,"Ticker"]    = _qt
                    _df.at[_idx,"Buy Date"]  = _qdate
                    _df.at[_idx,"Buy Price"] = float(_qprice)
                    _df.at[_idx,"Quantity"]  = int(_qqty)
                    st.session_state["holdings_df"] = _df
                else:
                    st.session_state["holdings_df"] = pd.concat([_df,_new_row],ignore_index=True)
                for _k in ["qa_ticker","qa_name","qa_warning"]:
                    st.session_state[_k] = ""
                st.success(f"{_qt} added to holdings table.")
                st.rerun()

    st.markdown("---")



    with st.form("holdings_form"):
        import datetime as _dt_col
        ## Ensure Buy Date column is always datetime.date — never string
        ## Prevents _check_type_compatibilities crash after Add button rerun
        _df_safe = st.session_state["holdings_df"].copy()
        _df_safe["Buy Date"] = pd.to_datetime(
            _df_safe["Buy Date"], errors="coerce"
        ).dt.date.where(
            pd.to_datetime(_df_safe["Buy Date"], errors="coerce").notna(),
            other=_dt_col.date.today() - _dt_col.timedelta(days=365)
        )
        holdings_df = st.data_editor(
            _df_safe,
            num_rows="fixed",
            hide_index=True,
        )
        submitted = st.form_submit_button("✅ Start Porfolio Analysis")

if submitted:
    st.session_state["holdings_df"]        = holdings_df.copy()
    st.session_state["portfolio_submitted"] = True
    st.session_state["show_holdings_form"]  = False
    st.rerun()  ## re-render so form collapses immediately
elif not st.session_state.get("portfolio_submitted", False):
    st.info("Fill the table and click **Update Portfolio** to run analysis.")
    st.stop()
else:
    holdings_df = st.session_state["holdings_df"].copy()

st.session_state["holdings_df"] = holdings_df.copy()

holdings_df["Ticker"]    = holdings_df["Ticker"].astype(str).str.upper().str.strip()
holdings_df["Buy Date"]  = pd.to_datetime(holdings_df["Buy Date"],  errors="coerce")
holdings_df["Buy Price"] = pd.to_numeric(holdings_df["Buy Price"],  errors="coerce")
holdings_df["Quantity"]  = pd.to_numeric(holdings_df["Quantity"],   errors="coerce")

if holdings_df["Ticker"].eq("").any():
    st.warning("⚠️ Please fill all ticker symbols.")
    st.stop()
if holdings_df["Buy Date"].isna().any():
    st.warning("⚠️ Please enter valid Buy Dates.")
    st.stop()
if holdings_df["Buy Price"].isna().any() or (holdings_df["Buy Price"] <= 0).any():
    st.warning("⚠️ Buy Price must be greater than 0.")
    st.stop()
if holdings_df["Quantity"].isna().any() or (holdings_df["Quantity"] <= 0).any():
    st.warning("⚠️ Quantity must be greater than 0.")
    st.stop()

tickers              = holdings_df["Ticker"].tolist()
portfolio_start_date = holdings_df["Buy Date"].min()
portfolio_end_date   = pd.Timestamp.today()

## ── Auto-detect exchange, resolve suffix, classify cap category ───────────
with st.spinner("Detecting exchange and market cap for each holding..."):
    enriched_info = {}
    resolved_tickers = []
    for raw_t in tickers:
        info_e = enrich_info(raw_t)
        enriched_info[raw_t] = info_e
        resolved_tickers.append(info_e["ticker"])

    ## Update tickers list with resolved (suffixed) tickers
    holdings_df["Ticker"]    = resolved_tickers
    holdings_df["Exchange"]  = [enriched_info[t]["exchange"]     for t in tickers]
    holdings_df["Cap"]       = [enriched_info[t]["cap_category"] for t in tickers]
    tickers                  = holdings_df["Ticker"].tolist()

    ## Warn user about any delisted/invalid tickers
    _delisted_warn = [
        t for t in tickers
        if enriched_info.get(t, {}).get("delisted", False)
    ]
    if _delisted_warn:
        _dl_str = ", ".join([t.replace(".NS","").replace(".BO","") for t in _delisted_warn])
        st.warning(
            f"⚠️ **Ticker(s) not found on Yahoo Finance: {_dl_str}**\n\n"
            f"These tickers may be delisted or renamed. "
            f"Price data will be unavailable — please check the correct ticker symbol. "
            f"Example: HDFCNIFETF.NS → try HDFCNIFTY.NS instead."
        )

## ─────────────────────────────────────────────────────────────────────────────
# %%  CONSTANTS — INSTITUTIONAL PARAMETERS
## ─────────────────────────────────────────────────────────────────────────────
## RBI Repo Rate — update when RBI changes monetary policy
## Last changed: June 2025 to 6.00%
RISK_FREE_ANNUAL = 0.0525  ## RBI Repo Rate 5.25% — unchanged, MPC Jun 5 2026
risk_free_daily  = (1 + RISK_FREE_ANNUAL) ** (1 / 252) - 1

## ══════════════════════════════════════════════════════════════════════════════
## BENCHMARK SECTOR WEIGHTS — sourced from NSE India factsheets + Dhan.co
## Last updated: May 2026
## Keys use Yahoo Finance sector names — yf.Ticker().info returns these names
## Update monthly by visiting:
##   Nifty 50: https://www.nseindia.com (factsheet PDF)
##   Others:   https://dhan.co/indices/<index>-sector-weightage/
## ══════════════════════════════════════════════════════════════════════════════

## ── Nifty 50 — Source: NSE India Factsheet May 2026 (user-provided) ──────────
_NIFTY50_WEIGHTS = {  ## Nifty 50 NSE factsheet Jun 2026
    "Financial Services":     0.3700,
    "Energy":                 0.0979,
    "Industrials":            0.0812,
    "Technology":             0.0741,
    "Basic Materials":        0.0683,
    "Consumer Cyclical":      0.0674,
    "Consumer Defensive":     0.0581,
    "Communication Services": 0.0515,
    "Healthcare":             0.0490,
    "Consumer Durables":      0.0275,
    "Consumer Services":      0.0275,
    "Utilities":              0.0273,
}

_NIFTY_SMALLCAP250_WEIGHTS = {  ## Nifty Smallcap 250 NSE factsheet Jun 2026
    "Financial Services":     0.2231,
    "Industrials":            0.2095,
    "Healthcare":             0.1401,
    "Basic Materials":        0.0993,
    "Consumer Cyclical":      0.0924,
    "Consumer Services":      0.0443,
    "Technology":             0.0385,
    "Consumer Durables":      0.0323,
    "Consumer Defensive":     0.0317,
    "Communication Services": 0.0280,
    "Energy":                 0.0251,
    "Utilities":              0.0200,
    "Real Estate":            0.0156,
}

_NIFTY100_WEIGHTS = {  ## Nifty 100 NSE factsheet Jun 2026
    "Financial Services":     0.3410,
    "Industrials":            0.0988,
    "Energy":                 0.0920,
    "Basic Materials":        0.0758,
    "Consumer Cyclical":      0.0707,
    "Consumer Defensive":     0.0632,
    "Technology":             0.0630,
    "Healthcare":             0.0520,
    "Communication Services": 0.0421,
    "Utilities":              0.0415,
    "Consumer Services":      0.0324,
    "Consumer Durables":      0.0225,
    "Real Estate":            0.0050,
}

## ── Nifty Midcap 150 — Source: Dhan.co, May 2026 ────────────────────────────
_NIFTY_MIDCAP150_WEIGHTS = {  ## Nifty Midcap 150 NSE factsheet Jun 2026
    "Financial Services":     0.2864,
    "Industrials":            0.1639,
    "Healthcare":             0.1016,
    "Basic Materials":        0.0791,
    "Consumer Cyclical":      0.0764,
    "Consumer Services":      0.0528,
    "Technology":             0.0445,
    "Consumer Durables":      0.0426,
    "Consumer Defensive":     0.0414,
    "Communication Services": 0.0305,
    "Utilities":              0.0292,
    "Real Estate":            0.0266,
    "Energy":                 0.0249,
}

## ── Nifty 500 — Source: Dhan.co, May 2026 ────────────────────────────────────
_NIFTY500_WEIGHTS = {  ## Nifty 500 NSE factsheet Jun 2026
    "Financial Services":     0.3167,
    "Industrials":            0.1244,
    "Basic Materials":        0.0791,
    "Consumer Cyclical":      0.0742,
    "Healthcare":             0.0719,
    "Energy":                 0.0708,
    "Technology":             0.0565,
    "Consumer Defensive":     0.0553,
    "Communication Services": 0.0382,
    "Consumer Services":      0.0380,
    "Utilities":              0.0366,
    "Consumer Durables":      0.0278,
    "Real Estate":            0.0106,
}

## ── Sensex — BSE/Asia Index factsheet Jun 2026 (total 99.99%) ────────────────
_SENSEX_WEIGHTS = {
    "Financial Services":     0.4051,  ## 40.51%
    "Consumer Cyclical":      0.1164,  ## 11.64%
    "Industrials":            0.0977,  ## 9.77%  (Industrials+Services)
    "Energy":                 0.0965,  ## 9.65%
    "Technology":             0.0836,  ## 8.36%
    "Communication Services": 0.0622,  ## 6.22%
    "Consumer Defensive":     0.0514,  ## 5.14%
    "Utilities":              0.0331,  ## 3.31%
    "Basic Materials":        0.0317,  ## 3.17%
    "Healthcare":             0.0222,  ## 2.22%
}

## ── BSE Midcap — BSE/Asia Index factsheet Jun 2026 (total 100.00%) ───────────
_BSE_MIDCAP_WEIGHTS = {
    "Financial Services":     0.2492,  ## 24.92%  (Financial Services+Diversified)
    "Consumer Cyclical":      0.2028,  ## 20.28%
    "Industrials":            0.1856,  ## 18.56%  (Industrials+Services)
    "Healthcare":             0.1068,  ## 10.68%
    "Basic Materials":        0.0994,  ## 9.94%
    "Technology":             0.0657,  ## 6.57%
    "Energy":                 0.0364,  ## 3.64%
    "Consumer Defensive":     0.0257,  ## 2.57%
    "Utilities":              0.0178,  ## 1.78%
    "Communication Services": 0.0106,  ## 1.06%
}

## ── BSE 500 — BSE/Asia Index factsheet Jun 2026 (total 99.99%) ───────────────
_BSE500_WEIGHTS = {
    "Financial Services":     0.3114,  ## 31.14%  (Financial Services+Diversified)
    "Consumer Cyclical":      0.1532,  ## 15.32%
    "Industrials":            0.1264,  ## 12.64%  (Industrials+Services)
    "Basic Materials":        0.0772,  ## 7.72%
    "Healthcare":             0.0723,  ## 7.23%
    "Energy":                 0.0720,  ## 7.20%
    "Technology":             0.0575,  ## 5.75%
    "Consumer Defensive":     0.0566,  ## 5.66%
    "Communication Services": 0.0372,  ## 3.72%
    "Utilities":              0.0361,  ## 3.61%
}

## ── BSE Smallcap — BSE/Asia Index factsheet Jun 2026 (total 100.00%) ─────────
_BSE_SMALLCAP_WEIGHTS = {
    "Industrials":            0.2632,  ## 26.32%  (Industrials+Services)
    "Consumer Cyclical":      0.2107,  ## 21.07%
    "Financial Services":     0.1750,  ## 17.50%  (Financial Services+Diversified)
    "Healthcare":             0.1203,  ## 12.03%
    "Basic Materials":        0.0929,  ## 9.29%
    "Consumer Defensive":     0.0532,  ## 5.32%
    "Technology":             0.0334,  ## 3.34%
    "Utilities":              0.0238,  ## 2.38%
    "Communication Services": 0.0159,  ## 1.59%
    "Energy":                 0.0116,  ## 1.16%
}

## ── Master lookup — indexed by benchmark ticker ──────────────────────────────
BENCHMARK_SECTOR_WEIGHTS = {
    "^CNX100":     {"name": "Nifty 100",       "updated": "Jun 2026", "weights": _NIFTY100_WEIGHTS},
    "^NSEI":       {"name": "Nifty 50",        "updated": "May 2026", "weights": _NIFTY50_WEIGHTS},
    "^CRSLDX":      {"name": "Nifty 500",  "updated": "Jun 2026", "weights": _NIFTY500_WEIGHTS},
    "NIFTYSMLCAP250.NS": {"name": "Nifty Smallcap 250","updated": "Jun 2026", "weights": _NIFTY_SMALLCAP250_WEIGHTS},
    "NIFTYMIDCAP150.NS":     {"name": "Nifty Midcap 150",       "updated": "May 2026", "weights": _NIFTY_MIDCAP150_WEIGHTS},
    "^BSESN":      {"name": "Sensex",          "updated": "Jun 2026", "weights": _SENSEX_WEIGHTS},
    "BSE-MIDCAP.BO":    {"name": "BSE Midcap",    "updated": "Jun 2026", "weights": _BSE_MIDCAP_WEIGHTS},
    "BSE-500.BO":       {"name": "BSE 500",       "updated": "Jun 2026", "weights": _BSE500_WEIGHTS},
    "BSE-SMLCAP.BO":    {"name": "BSE Smallcap",  "updated": "Jun 2026", "weights": _BSE_SMALLCAP_WEIGHTS},
}

## ── Backwards-compatible alias — always points to selected benchmark ──────────
## This gets overwritten after BENCHMARK_TICKER is determined at runtime
NIFTY_SECTOR_WEIGHTS = _NIFTY50_WEIGHTS  ## default until runtime

## ══════════════════════════════════════════════════════════════════════════════
## SECTOR INDEX TICKERS — one set per exchange
## Yahoo Finance sector names (US-style) mapped to exchange-specific sector indices
## e.g. BHARTIARTL.NS → Yahoo sector "Communication Services" → ^CNXSERVICE (NSE)
##      BHARTIARTL.BO → Yahoo sector "Communication Services" → BSE-TECK.BO (BSE)
## ══════════════════════════════════════════════════════════════════════════════

## ── NSE sector indices (used when BENCHMARK_TICKER starts with ^NSEI / ^NSE / ^CNX / NIFTYMIDCAP150.NS)
NSE_SECTOR_INDEX_TICKERS = {
    ## ^CNXTelecom does NOT exist on Yahoo Finance — use ^CNXSERVICE
    "Communication Services": "^CNXSERVICE",
    "Technology":             "^CNXIT",
    "Financial Services":     "^NSEBANK",
    "Energy":                 "^CNXENERGY",
    "Consumer Cyclical":      "^CNXAUTO",    ## Auto proxy for Consumer Cyclical
    "Consumer Durables":      "^CNXFMCG",    ## FMCG proxy — no dedicated Durables index on Yahoo
    "Consumer Services":      "^CNXSERVICE", ## Services index as proxy
    "Healthcare":             "^CNXPHARMA",
    "Industrials":            "^CNXINFRA",
    "Consumer Defensive":     "^CNXFMCG",
    "Basic Materials":        "^CNXMETAL",
    "Utilities":              "^CNXENERGY",  ## Energy/Power proxy — no dedicated Power index on Yahoo
    "Real Estate":            "^CNXREALTY",
    "Capital Goods":          "^CNXINFRA",
    "Services":               "^CNXSERVICE",
}

## ── BSE sector indices (used when BENCHMARK_TICKER starts with ^BSE / ^BSESN / BSE-MIDCAP.BO)
## Yahoo Finance BSE sector index tickers — verified working tickers
BSE_SECTOR_INDEX_TICKERS = {
    "Financial Services":     "BSE-BANK.BO",    ## BSE Bankex
    "Technology":             "BSE-IT.BO",      ## BSE IT
    "Healthcare":             "BSE-HC.BO",      ## BSE Healthcare
    "Consumer Cyclical":      "BSE-AUTO.BO",    ## BSE Auto
    "Consumer Durables":      "BSE-CD.BO",      ## BSE Consumer Durables
    "Consumer Defensive":     "BSE-FMCG.BO",   ## BSE FMCG
    "Consumer Services":      "BSE-CG.BO",      ## BSE CG proxy — no dedicated BSE Services index
    "Basic Materials":        "BSE-METAL.BO",   ## BSE Metal
    "Industrials":            "BSE-CG.BO",      ## BSE Capital Goods
    "Energy":                 "BSE-OILGAS.BO",  ## BSE Oil & Gas
    "Real Estate":            "BSE-REALTY.BO",  ## BSE Realty
    "Utilities":              "BSE-POWER.BO",   ## BSE Power
    "Communication Services": "BSE-TECK.BO",   ## BSE Teck (closest proxy)
    "Capital Goods":          "BSE-CG.BO",      ## BSE Capital Goods
    "Services":               "BSE-TECK.BO",    ## BSE Teck proxy
}

## ── Backwards-compatible alias — set dynamically at runtime ──────────────────
## Overwritten after BENCHMARK_TICKER is known (see below)
SECTOR_INDEX_TICKERS = NSE_SECTOR_INDEX_TICKERS  ## default

## ETF definitions — ETFs do not have a single sector classification
## BHB selection effect is not applicable to ETFs
## They represent a sector ALLOCATION decision, not a stock SELECTION decision
## ── ETF Framework — Tiered Classification ────────────────────────────────────
## Tier 1: Sector ETFs  → direct sector index benchmark
## Tier 2: Broad Market → benchmark itself (near-zero active contribution)
## Tier 3: Thematic     → Nifty 500 proxy with methodology note
## Tier 4: Unknown      → excluded from BHB, user notified
##
## benchmark_weight is NOT hardcoded here — it is fetched dynamically
## from NIFTY_SECTOR_WEIGHTS at runtime so it stays current
## ─────────────────────────────────────────────────────────────────────────────

## Tier 1 — Sector ETFs: keyword → (sector_label, attribution_benchmark_ticker)
ETF_SECTOR_MAP = {
    ## Banking / Financial
    "bank":           ("Financial Services",  "^NSEBANK"),
    "banking":        ("Financial Services",  "^NSEBANK"),
    "nsebank":        ("Financial Services",  "^NSEBANK"),
    "financial":      ("Financial Services",  "^NSEBANK"),
    ## Technology / IT
    "it ":            ("Technology",          "^CNXIT"),
    " it":            ("Technology",          "^CNXIT"),
    "nifty it":       ("Technology",          "^CNXIT"),
    "infotech":       ("Technology",          "^CNXIT"),
    "information tech":("Technology",         "^CNXIT"),
    "tech etf":       ("Technology",          "^CNXIT"),
    ## Pharma / Healthcare
    "pharma":         ("Healthcare",          "^CNXPHARMA"),
    "healthcare":     ("Healthcare",          "^CNXPHARMA"),
    "health care":    ("Healthcare",          "^CNXPHARMA"),
    ## Auto / Consumer Cyclical
    "auto":           ("Consumer Cyclical",   "^CNXAUTO"),
    "automobile":     ("Consumer Cyclical",   "^CNXAUTO"),
    ## FMCG / Consumer Defensive
    "fmcg":           ("Consumer Defensive",  "^CNXFMCG"),
    ## Energy
    "energy":         ("Energy",              "^CNXENERGY"),
    "oil":            ("Energy",              "^CNXENERGY"),
    ## Metal / Mining
    "metal":          ("Basic Materials",     "^CNXMETAL"),
    "mining":         ("Basic Materials",     "^CNXMETAL"),
    ## Realty
    "realt":          ("Real Estate",         "^CNXREALTY"),
    ## Infrastructure
    "infra":          ("Industrials",         "^CNXINFRA"),
    "infrastructure": ("Industrials",         "^CNXINFRA"),
}

## Tier 2 — Broad Market ETFs: keyword list (active contribution ≈ zero)
ETF_BROAD_MARKET_KEYWORDS = [
    ## Nifty 50 variants — SPECIFIC only, not generic 'nifty'
    "nifty 50 ", "nifty50", "nifty bees", "niftybees",
    "n50", "nif50",
    ## Nifty 100 / Next 50 / Nifty 500
    "nifty 100", "nifty100", "nifty next 50", "niftynext",
    "nifty 500", "nifty500",
    ## Sensex / BSE broad
    "sensex", "bse 500", "bse500",
    ## Generic Nifty ETF names — "HDFC Nifty ETF", "Kotak Nifty ETF"
    ## NOTE: "nifty etf" must NOT match "nifty it etf" or "nifty auto etf"
    ## So we keep it specific
    "nifty etf",
    ## Catches "SBI ETF Nifty 50", "Axis ETF Nifty 50" etc where order differs
    "etf nifty 50",
]

## Tier 3 — Thematic ETFs: keyword → (theme_label, proxy_benchmark)
## Proxy = Nifty 500 (broadest available on Yahoo Finance)
## Label shown to user: "Nifty 500 (Proxy)" — transparent about limitation
ETF_THEMATIC_MAP = {
    "defence":     ("Defence & Aerospace",   "^CRSLDX"),
    "defense":     ("Defence & Aerospace",   "^CRSLDX"),
    "psu":         ("PSU Theme",             "^CNXPSE"),
    "public sector":("PSU Theme",            "^CNXPSE"),
    "manufactur":  ("Manufacturing",         "^CRSLDX"),
    "mobility":    ("Mobility / EV",         "NIFTYMIDCAP150.NS"),
    "ev ":         ("EV / Clean Energy",     "NIFTYMIDCAP150.NS"),
    "electric":    ("EV / Clean Energy",     "NIFTYMIDCAP150.NS"),
    "midcap":      ("Mid Cap",               "NIFTYMIDCAP150.NS"),
    "mid cap":     ("Mid Cap",               "NIFTYMIDCAP150.NS"),
    "smallcap":    ("Small Cap",             "^CRSLDX"),
    "small cap":   ("Small Cap",             "^CRSLDX"),
    ## PSU / CPSE
    "cpse":        ("PSU Theme",             "^CNXPSE"),
    "bharat 22":   ("PSU Theme",             "^CNXPSE"),
    ## Infra
    "infrabees":   ("Industrials",           "^CNXINFRA"),
    ## Factor / Smart Beta ETFs
    "low vol":     ("Factor ETF",            "^CRSLDX"),
    "low volat":   ("Factor ETF",            "^CRSLDX"),
    "momentum":    ("Factor ETF",            "^CRSLDX"),
    "quality":     ("Factor ETF",            "^CRSLDX"),
    ## Consumption / India themes
    "consumption": ("Consumption Theme",     "^CRSLDX"),
    ## REIT / InvIT
    "reit":        ("Real Estate",           "^CNXREALTY"),
    "invit":       ("Real Estate",           "^CNXREALTY"),
}

## Tier 4 — Non-equity ETFs: these are EXCLUDED from BHB
## No equity sector benchmark makes sense for them
ETF_EXCLUDE_KEYWORDS = [
    "gold", "silver", "commodity", "liquid", "overnight",
    "money market", "bond", "gilt", "debt", "bharat bond",
    "nasdaq", "s&p 500", "s&p500", "hangseng", "hang seng",
    "global", "international", "world", "us equity",
    "nyse", "fang", "us tech", "china", "taiwan",
    "japan", "europe", "emerging market", "mirae asset nyse",
]

## Manual overrides — known ETFs where name-matching might fail
ETF_DEFINITIONS = {
    ## ── Defence ETFs (Tier 3 Thematic) ──────────────────────────────────────
    "GROWWDEFNC.NS": {
        "sector": "Defence & Aerospace", "industry": "ETF — Nifty India Defence Index",
        "is_etf": True, "tracks": "Nifty India Defence Index",
        "etf_tier": 3, "attribution_bm": "^CRSLDX",
        "attribution_bm_label": "Nifty 500 (Proxy)",
    },
    "MODEFENCE.NS": {
        "sector": "Defence & Aerospace", "industry": "ETF — Nifty India Defence Index",
        "is_etf": True, "tracks": "Nifty India Defence Index",
        "etf_tier": 3, "attribution_bm": "^CRSLDX",
        "attribution_bm_label": "Nifty 500 (Proxy)",
    },
    ## ── Nifty 50 ETFs (Tier 2 Broad Market) ─────────────────────────────────
    "NIFTYBEES.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    ## HDFCNIFETF.NS is delisted on Yahoo Finance — keeping for legacy support
    ## Users should use HDFCNIFTY.NS instead
    "HDFCNIFETF.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index (Delisted on Yahoo)",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "HDFCNIFTY.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "HDFCNIF50.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "ICICIB22.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "UTINIFTETF.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    "SBIETFNIF50.NS": {
        "sector": "Broad Market ETF", "industry": "ETF — Nifty 50 Index",
        "is_etf": True, "tracks": "Nifty 50",
        "etf_tier": 2, "attribution_bm": None,
        "attribution_bm_label": "Primary Benchmark (tracks Nifty 50)",
    },
    ## ── Gold ETFs (Tier 4 Non-Equity) ────────────────────────────────────────
    "GOLDBEES.NS": {
        "sector": "Non-Equity ETF", "industry": "ETF — Gold (Physical)",
        "is_etf": True, "tracks": "Physical Gold",
        "etf_tier": 4, "attribution_bm": None,
        "attribution_bm_label": "Excluded — non-equity ETF",
    },
    "HDFCMFGETF.NS": {
        "sector": "Non-Equity ETF", "industry": "ETF — Gold (Physical)",
        "is_etf": True, "tracks": "Physical Gold",
        "etf_tier": 4, "attribution_bm": None,
        "attribution_bm_label": "Excluded — non-equity ETF",
    },
    "NIPPONBEES.NS": {
        "sector": "Non-Equity ETF", "industry": "ETF — Gold (Physical)",
        "is_etf": True, "tracks": "Physical Gold",
        "etf_tier": 4, "attribution_bm": None,
        "attribution_bm_label": "Excluded — non-equity ETF",
    },
    ## ── Bank ETFs (Tier 1 Sector) ─────────────────────────────────────────────
    "BANKBEES.NS": {
        "sector": "Financial Services", "industry": "ETF — Nifty Bank Index",
        "is_etf": True, "tracks": "Nifty Bank",
        "etf_tier": 1, "attribution_bm": "^NSEBANK",
        "attribution_bm_label": "^NSEBANK",
    },
    "HDFCNIFBAN.NS": {
        "sector": "Financial Services", "industry": "ETF — Nifty Bank Index",
        "is_etf": True, "tracks": "Nifty Bank",
        "etf_tier": 1, "attribution_bm": "^NSEBANK",
        "attribution_bm_label": "^NSEBANK",
    },
}

def classify_etf(ticker, info_data):
    """
    Auto-classify any ETF into one of four tiers using its Yahoo Finance name.
    Returns a dict compatible with ETF_DEFINITIONS format.
    Falls back gracefully at every tier.
    """
    name = (
        (info_data.get("longName",  "") or "") + " " +
        (info_data.get("shortName", "") or "")
    ).lower().strip()

    ## Check manual override first
    if ticker in ETF_DEFINITIONS:
        return ETF_DEFINITIONS[ticker]

    ## Tier 4 — Non-equity: exclude from BHB entirely (check first — gold/liquid must not go to sector)
    for kw in ETF_EXCLUDE_KEYWORDS:
        if kw in name:
            return {
                "sector":       "Non-Equity ETF",
                "industry":     f"ETF — {info_data.get('longName', ticker)}",
                "is_etf":       True,
                "tracks":       info_data.get("longName", "Unknown Index"),
                "etf_tier":     4,
                "attribution_bm": None,
                "attribution_bm_label": "Excluded — non-equity ETF",
            }

    ## Tier 1 — Sector ETF (check BEFORE broad market)
    ## "Nifty IT ETF" must go to Tier 1, not Tier 2 via generic "nifty" match
    for kw, (sector, bm_ticker) in ETF_SECTOR_MAP.items():
        if kw in name:
            return {
                "sector":       sector,
                "industry":     f"ETF — {info_data.get('longName', ticker)}",
                "is_etf":       True,
                "tracks":       info_data.get("longName", "Sector Index"),
                "etf_tier":     1,
                "attribution_bm": bm_ticker,
                "attribution_bm_label": bm_ticker,
            }

    ## Tier 2 — Broad market: active contribution ≈ zero
    for kw in ETF_BROAD_MARKET_KEYWORDS:
        if kw in name:
            return {
                "sector":       "Broad Market ETF",
                "industry":     f"ETF — {info_data.get('longName', ticker)}",
                "is_etf":       True,
                "tracks":       info_data.get("longName", "Broad Market Index"),
                "etf_tier":     2,
                "attribution_bm": None,
                "attribution_bm_label": "Primary Benchmark (tracks index)",
            }

    ## Tier 3 — Thematic ETF: use Nifty 500 proxy
    for kw, (theme, bm_ticker) in ETF_THEMATIC_MAP.items():
        if kw in name:
            return {
                "sector":       theme,
                "industry":     f"ETF — {info_data.get('longName', ticker)}",
                "is_etf":       True,
                "tracks":       info_data.get("longName", "Thematic Index"),
                "etf_tier":     3,
                "attribution_bm": bm_ticker,
                "attribution_bm_label": f"{bm_ticker} (Proxy)",
            }

    ## Tier 4 fallback — unrecognised ETF → exclude from BHB
    return {
        "sector":       "Unknown ETF",
        "industry":     f"ETF — {info_data.get('longName', ticker)}",
        "is_etf":       True,
        "tracks":       info_data.get("longName", "Unknown"),
        "etf_tier":     4,
        "attribution_bm": None,
        "attribution_bm_label": "Not mapped — excluded from BHB",
    }

## Manual sector overrides for stocks Yahoo Finance cannot classify correctly
## e.g. newly listed stocks, demerged entities, small caps
MANUAL_SECTOR_MAP = {  ## Priority 3 fallback when Yahoo returns Unknown
    "TMCV.NS":"Consumer Cyclical","TMPV.NS":"Consumer Cyclical",
    "TITAN.NS":"Consumer Durables","CRISIL.NS":"Financial Services",
    "DABUR.NS":"Consumer Defensive","ASIANPAINT.NS":"Basic Materials",
    "TECHM.NS":"Technology","WIPRO.NS":"Technology","INFY.NS":"Technology",
    "TCS.NS":"Technology","HCLTECH.NS":"Technology","LTIM.NS":"Technology",
    "PERSISTENT.NS":"Technology","COFORGE.NS":"Technology","MPHASIS.NS":"Technology",
    "BAJFINANCE.NS":"Financial Services","BAJAJFINSV.NS":"Financial Services",
    "HDFCBANK.NS":"Financial Services","ICICIBANK.NS":"Financial Services",
    "KOTAKBANK.NS":"Financial Services","AXISBANK.NS":"Financial Services",
    "SBIN.NS":"Financial Services","SBILIFE.NS":"Financial Services",
    "HDFCLIFE.NS":"Financial Services","SHRIRAMFIN.NS":"Financial Services",
    "RELIANCE.NS":"Energy","ONGC.NS":"Energy","BPCL.NS":"Energy",
    "IOC.NS":"Energy","GAIL.NS":"Energy","NTPC.NS":"Utilities",
    "POWERGRID.NS":"Utilities","NHPC.NS":"Utilities","TATAPOWER.NS":"Utilities",
    "SUNPHARMA.NS":"Healthcare","DRREDDY.NS":"Healthcare","CIPLA.NS":"Healthcare",
    "DIVISLAB.NS":"Healthcare","APOLLOHOSP.NS":"Healthcare",
    "MARUTI.NS":"Consumer Cyclical","TATAMOTORS.NS":"Consumer Cyclical",
    "MM.NS":"Consumer Cyclical","BAJAJ-AUTO.NS":"Consumer Cyclical",
    "HEROMOTOCO.NS":"Consumer Cyclical","EICHERMOT.NS":"Consumer Cyclical",
    "ITC.NS":"Consumer Defensive","HINDUNILVR.NS":"Consumer Defensive",
    "NESTLEIND.NS":"Consumer Defensive","BRITANNIA.NS":"Consumer Defensive",
    "MARICO.NS":"Consumer Defensive","TATACONSUM.NS":"Consumer Defensive",
    "LT.NS":"Industrials","BEL.NS":"Industrials","SIEMENS.NS":"Industrials",
    "ABB.NS":"Industrials","HAVELLS.NS":"Industrials",
    "BHARTIARTL.NS":"Communication Services",
    "TATASTEEL.NS":"Basic Materials","JSWSTEEL.NS":"Basic Materials",
    "HINDALCO.NS":"Basic Materials","ULTRACEMCO.NS":"Basic Materials",
    "SAIL.NS":"Basic Materials","NMDC.NS":"Basic Materials",
    "GODREJPROP.NS":"Real Estate","DLF.NS":"Real Estate",
    "VOLTAS.NS":"Consumer Durables",
    ## Stocks confirmed returning N/A on Streamlit Cloud
    "BHARATFORG.NS":"Industrials",
    "OFSS.NS":"Technology",
    "AMBUJACEM.NS":"Basic Materials",
    "PFIZER.NS":"Healthcare",
    ## Additional common NSE stocks
    "ADANIENT.NS":"Industrials","ADANIPORTS.NS":"Industrials",
    "ADANIGREEN.NS":"Utilities","ADANIPOWER.NS":"Utilities",
    "PIIND.NS":"Basic Materials","DEEPAKNTR.NS":"Basic Materials",
    "ASTRAL.NS":"Industrials","POLYCAB.NS":"Industrials",
    "DMART.NS":"Consumer Defensive","TRENT.NS":"Consumer Cyclical",
    "NYKAA.NS":"Consumer Cyclical","ZOMATO.NS":"Consumer Services",
    "PAYTM.NS":"Financial Services","IRCTC.NS":"Industrials",
    "IRFC.NS":"Financial Services","RVNL.NS":"Industrials",
    "PFC.NS":"Financial Services","RECLTD.NS":"Financial Services",
    "HUDCO.NS":"Financial Services","IREDA.NS":"Financial Services",
    "CANBK.NS":"Financial Services","PNB.NS":"Financial Services",
    "BANKBARODA.NS":"Financial Services","FEDERALBNK.NS":"Financial Services",
    "IDFCFIRSTB.NS":"Financial Services","AUBANK.NS":"Financial Services",
    "INDUSINDBK.NS":"Financial Services",
    "TORNTPHARM.NS":"Healthcare","BIOCON.NS":"Healthcare",
    "GLENMARK.NS":"Healthcare","ALKEM.NS":"Healthcare",
    "JSWENERGY.NS":"Utilities","TORNTPOWER.NS":"Utilities",
    "DALBHARAT.NS":"Basic Materials","SHREECEM.NS":"Basic Materials",
    "RAMCOCEM.NS":"Basic Materials","JKCEMENT.NS":"Basic Materials",
    "IGL.NS":"Energy","MGL.NS":"Energy","GUJGASLTD.NS":"Energy",
    "VBL.NS":"Consumer Defensive","RADICO.NS":"Consumer Defensive",
    "JUBLFOOD.NS":"Consumer Services","DEVYANI.NS":"Consumer Services",
    "WESTLIFE.NS":"Consumer Services","SAPIENT.NS":"Technology",
    "KPITTECH.NS":"Technology","TATAELXSI.NS":"Technology",
    "KAYNES.NS":"Industrials","DIXON.NS":"Industrials",
    "APLAPOLLO.NS":"Industrials","GRINDWELL.NS":"Industrials",
}

## ─────────────────────────────────────────────────────────────────────────────
# %%  DATA FETCHERS
## ─────────────────────────────────────────────────────────────────────────────

## ─────────────────────────────────────────────────────────────────────────────
##  QUOTE OF THE DAY
## ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_QUOTES = [
    ("The ability to observe without evaluating is the highest form of intelligence.", "J. Krishnamurti"),
    ("You are not a drop in the ocean. You are the entire ocean in a drop.", "Rumi"),
    ("Muddy water is best cleared by leaving it alone.", "Alan Watts"),
    ("Nature does not hurry, yet everything is accomplished.", "Lao Tzu"),
    ("The wound is the place where the light enters you.", "Rumi"),
    ("Yesterday I was clever, so I wanted to change the world. Today I am wise, so I am changing myself.", "Rumi"),
    ("The quieter you become, the more you are able to hear.", "Rumi"),
    ("Until you make the unconscious conscious, it will direct your life and you will call it fate.", "Carl Jung"),
    ("The privilege of a lifetime is to become who you truly are.", "Carl Jung"),
    ("To the mind that is still, the whole universe surrenders.", "Lao Tzu"),
    ("What you seek is seeking you.", "Rumi"),
    ("There is a crack in everything. That is how the light gets in.", "Leonard Cohen"),
    ("The present moment always will have been.", "Alan Watts"),
    ("We are shaped by our thoughts; we become what we think.", "Buddha"),
    ("Do not go where the path may lead, go instead where there is no path and leave a trail.", "Ralph Waldo Emerson"),
]

@st.cache_data(ttl=86400)
def get_quote_of_day():
    try:
        import requests as _req
        resp = _req.get("https://zenquotes.io/api/today", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and "q" in data[0] and "a" in data[0]:
                q = data[0]["q"].strip(); a = data[0]["a"].strip()
                if len(q) > 10 and a.lower() not in ("unknown",""):
                    return q, a
    except Exception:
        pass
    import datetime
    seed = int(datetime.date.today().strftime("%Y%m%d")) % len(_FALLBACK_QUOTES)
    return _FALLBACK_QUOTES[seed]

@st.cache_data(ttl=ttl_seconds)
def fetch_price_history(tickers, start, end):
    ## Try with auto_adjust=True first (yfinance >= 0.2.x)
    try:
        data = yf.download(tickers, start=start, end=end,
                           progress=False, auto_adjust=True)
    except Exception:
        data = pd.DataFrame()

    if data.empty:
        ## Fallback: try without auto_adjust
        try:
            data = yf.download(tickers, start=start, end=end,
                               progress=False, auto_adjust=False)
        except Exception:
            return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    ## Handle both MultiIndex (multiple tickers) and flat (single ticker)
    if isinstance(data.columns, pd.MultiIndex):
        ## Newer yfinance: auto_adjust=True returns "Close" not "Adj Close"
        level0 = data.columns.get_level_values(0).unique().tolist()
        col = "Close" if "Close" in level0 else (
              "Adj Close" if "Adj Close" in level0 else level0[0])
        prices = data[col]
    else:
        col = "Close" if "Close" in data.columns else (
              "Adj Close" if "Adj Close" in data.columns else data.columns[0])
        prices = data[[col]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices.dropna(how="all")


@st.cache_data(ttl=ttl_seconds)
def fetch_latest_prices(tickers):
    ## Try auto_adjust=True first (newer yfinance)
    try:
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
    except Exception:
        data = pd.DataFrame()

    if data.empty:
        try:
            data = yf.download(tickers, period="5d", progress=False, auto_adjust=False)
        except Exception:
            return pd.Series(dtype=float)

    if data.empty:
        return pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0).unique().tolist()
        col    = "Close" if "Close" in level0 else (
                 "Adj Close" if "Adj Close" in level0 else level0[0])
        latest = data[col].ffill().iloc[-1]
    else:
        col    = "Close" if "Close" in data.columns else (
                 "Adj Close" if "Adj Close" in data.columns else data.columns[0])
        latest = data[[col]].ffill().iloc[-1]
        if hasattr(latest, 'squeeze'):
            latest = latest.squeeze()
    return latest


@st.cache_data(ttl=ttl_seconds)
def fetch_benchmark(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        data = pd.DataFrame()
    if data.empty:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        except Exception:
            return pd.Series(dtype=float)
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0).unique().tolist()
        col    = "Close" if "Close" in level0 else (
                 "Adj Close" if "Adj Close" in level0 else level0[0])
        series = data[col].squeeze()
    else:
        col    = "Close" if "Close" in data.columns else (
                 "Adj Close" if "Adj Close" in data.columns else data.columns[0])
        series = data[col]
    return series.dropna()



@st.cache_data(ttl=ttl_seconds)
def fetch_sector_info(ticker):
    """
    Fetch sector and industry for a ticker.
    Uses multiple methods since yf.Ticker().info returns incomplete
    data for some NSE stocks from cloud server IPs.
    """
    ## Method 1: yf.Ticker().info (standard)
    try:
        info   = yf.Ticker(ticker).info
        sector = info.get("sector",   "") or ""
        indust = info.get("industry", "") or ""
        if sector and sector != "Unknown":
            return {"sector": sector, "industry": indust}
    except Exception:
        pass

    ## Method 2: yf.Ticker().fast_info then get_info (yfinance 1.5+)
    try:
        tk  = yf.Ticker(ticker)
        if hasattr(tk, 'get_info'):
            info2  = tk.get_info()
            sector = info2.get("sector",   "") or ""
            indust = info2.get("industry", "") or ""
            if sector and sector != "Unknown":
                return {"sector": sector, "industry": indust}
    except Exception:
        pass

    ## Method 3: Direct Yahoo Finance quoteSummary API
    ## More reliable than yf.Ticker().info for NSE stocks from cloud IPs
    try:
        import requests as _req
        _url = f"https://query2.finance.yahoo.com/v11/finance/quoteSummary/{ticker}"
        _params = {"modules": "assetProfile"}
        _headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        _r = _req.get(_url, params=_params, headers=_headers, timeout=6)
        if _r.status_code == 200:
            _data   = _r.json()
            _profile= _data.get("quoteSummary",{}).get("result",[{}])[0].get("assetProfile",{})
            sector  = _profile.get("sector",   "") or ""
            indust  = _profile.get("industry", "") or ""
            if sector and sector != "Unknown":
                return {"sector": sector, "industry": indust}
    except Exception:
        pass

    ## All methods failed
    return {"sector": "Unknown", "industry": "Unknown"}


@st.cache_data(ttl=ttl_seconds)
def detect_portfolio_benchmark(enriched_rows):
    """
    Look at the exchange and cap category of each holding (weighted by value)
    and suggest the most appropriate benchmark index ticker.
    """
    if not enriched_rows:
        return "^NSEI", "Nifty 50", "NSE"

    ## Count by exchange (weight-agnostic for benchmark selection)
    exchanges = [r["exchange"] for r in enriched_rows]
    bse_count = exchanges.count("BSE")
    nse_count = exchanges.count("NSE")
    dominant_exchange = "BSE" if bse_count > nse_count else "NSE"

    ## Count cap categories — EXCLUDE ETFs from this count
    ## ETFs are not themselves large/mid/small cap stocks
    ## Benchmark should be based on the equity stock portion only
    caps  = [r["cap_category"] for r in enriched_rows
             if r["cap_category"] not in ("ETF", "Non-Equity ETF", "Broad Market ETF",
                                           "Unknown", "Unclassified")]
    large = caps.count("Large Cap")
    mid   = caps.count("Mid Cap")
    small = caps.count("Small Cap")
    total = max(len(caps), 1)  ## use equity-only count

    ## Determine dominant cap type from stock portion
    ## Per SEBI/NSE methodology: Large=top 100, Mid=101-250, Small=251+
    if large / total >= 0.6:
        dominant_cap = "Large Cap"
    elif mid / total >= 0.6:
        dominant_cap = "Mid Cap"
    elif small / total >= 0.6:
        dominant_cap = "Small Cap"
    elif (large + mid) / total >= 0.7:
        dominant_cap = "Large Mid Mixed"
    else:
        dominant_cap = "Mixed"

    ## Pick benchmark based on exchange + cap
    ## NSE benchmarks match SEBI cap definitions exactly per NSE methodology doc
    if dominant_exchange == "BSE":
        if dominant_cap == "Large Cap":
            return "^BSESN",        "Sensex (BSE 30)",        "BSE"
        elif dominant_cap == "Mid Cap":
            return "BSE-MIDCAP.BO", "BSE Midcap",             "BSE"
        elif dominant_cap == "Small Cap":
            return "BSE-SMLCAP.BO", "BSE Smallcap",            "BSE"
        else:
            return "BSE-500.BO",    "BSE 500",                 "BSE"
    else:  ## NSE
        if dominant_cap == "Large Cap":
            return "^CNX100",           "Nifty 100",           "NSE"
        elif dominant_cap == "Mid Cap":
            return "NIFTYMIDCAP150.NS", "Nifty Midcap 150",    "NSE"
        elif dominant_cap == "Small Cap":
            return "NIFTYSMLCAP250.NS", "Nifty Smallcap 250",  "NSE"
        elif dominant_cap == "Large Mid Mixed":
            return "^CRSLDX",           "Nifty 500",           "NSE"
        else:
            return "^CRSLDX",           "Nifty 500",           "NSE"


@st.cache_data(ttl=ttl_seconds)
def fetch_top5_in_sector(sector_name, exchange="NSE", exclude_tickers=None):
    """
    Fetch top 5 stocks by market cap in a given Yahoo Finance sector.
    Uses a curated list of Nifty 500 stocks per sector — refreshed on each
    auto-refresh cycle (same TTL as rest of dashboard).
    exchange: "NSE" (.NS suffix) or "BSE" (.BO suffix)
    exclude_tickers: list of tickers already in portfolio (shown differently)
    """
    ## ── Curated sector → Nifty 500 top stocks map (NSE tickers) ─────────────
    ## Source: NSE India, May 2026 — top stocks by market cap per sector
    SECTOR_TOP_STOCKS = {
        "Communication Services": [
            "BHARTIARTL.NS","IDEA.NS","INDUSTOWER.NS","TATACOMM.NS","RAILTEL.NS",
            "MTNL.NS","STLTECH.NS","HFCL.NS",
        ],
        "Technology": [
            "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS",
            "LTIM.NS","MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","OFSS.NS",
        ],
        "Financial Services": [
            "HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS","AXISBANK.NS",
            "BAJFINANCE.NS","BAJAJFINSV.NS","HDFCAMC.NS","SBILIFE.NS","HDFCLIFE.NS",
        ],
        "Energy": [
            "RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","HINDPETRO.NS",
            "GAIL.NS","OIL.NS","MRPL.NS",
        ],
        "Consumer Cyclical": [
            "MARUTI.NS","M&M.NS","TATAMOTORS.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS",
            "EICHERMOT.NS","BOSCHLTD.NS","TVSMOTOR.NS",
        ],
        "Healthcare": [
            "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS","APOLLOHOSP.NS",
            "LUPIN.NS","BIOCON.NS","TORNTPHARM.NS","AUROPHARMA.NS",
        ],
        "Industrials": [
            "LT.NS","ADANIPORTS.NS","SIEMENS.NS","ABB.NS","HAVELLS.NS",
            "CUMMINSIND.NS","THERMAX.NS","BEL.NS","HAL.NS","BHEL.NS",
        ],
        "Consumer Defensive": [
            "HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
            "MARICO.NS","GODREJCP.NS","COLPAL.NS","EMAMILTD.NS",
        ],
        "Basic Materials": [
            "TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","COALINDIA.NS",
            "NMDC.NS","SAIL.NS","NATIONALUM.NS","HINDCOPPER.NS",
        ],
        "Utilities": [
            "POWERGRID.NS","NTPC.NS","ADANIGREEN.NS","TATAPOWER.NS","TORNTPOWER.NS",
            "CESC.NS","NHPC.NS","SJVN.NS",
        ],
        "Real Estate": [
            "DLF.NS","GODREJPROP.NS","OBEROIRLTY.NS","PRESTIGE.NS","BRIGADE.NS",
            "PHOENIXLTD.NS","SOBHA.NS","MAHLIFE.NS",
        ],
        "Capital Goods": [
            "HAL.NS","BEL.NS","BHEL.NS","COCHINSHIP.NS","MAZAGON.NS",
            "GRINDWELL.NS","ELGIEQUIP.NS","KENNAMETAL.NS",
        ],
    }

    exclude = set(exclude_tickers or [])
    suffix  = ".NS" if exchange == "NSE" else ".BO"

    ## Get candidate tickers for this sector
    candidates = SECTOR_TOP_STOCKS.get(sector_name, [])
    if not candidates:
        return pd.DataFrame()

    ## Adjust suffix if needed
    adjusted = []
    for t in candidates:
        base = t.replace(".NS","").replace(".BO","")
        adjusted.append(base + suffix)

    ## Fetch market cap and current price for each
    rows = []
    for ticker in adjusted[:8]:   ## check top 8, return best 5
        try:
            ## Method 1: fast_info (lighter, more reliable than .info)
            tk       = yf.Ticker(ticker)
            fi       = tk.fast_info
            price    = getattr(fi, 'last_price',    None) or getattr(fi, 'regular_market_price', None) or 0
            prev     = getattr(fi, 'previous_close', price) or price
            mkt_cap  = getattr(fi, 'market_cap',    None) or 0
            chg_1d   = ((price - prev) / prev * 100) if prev else 0
            ret_52w  = 0.0

            ## Get name from .info only if fast_info gave us a valid price
            name = ticker.replace(".NS","").replace(".BO","")
            if price and price > 0:
                try:
                    name = tk.info.get("shortName", name) or name
                except Exception:
                    pass

            ## Method 2: fallback to .info if fast_info gave nothing
            if not price or price == 0:
                try:
                    info    = tk.info
                    mkt_cap = info.get("marketCap", 0) or 0
                    price   = info.get("currentPrice") or info.get("regularMarketPrice", 0) or 0
                    prev    = info.get("previousClose", price) or price
                    chg_1d  = ((price - prev) / prev * 100) if prev else 0
                    name    = info.get("shortName", name) or name
                    ret_52w = (info.get("52WeekChange") or 0.0) * 100
                except Exception:
                    pass

            if price and price > 0:
                rows.append({
                    "Ticker":       ticker,
                    "Name":         name,
                    "Price":        price,
                    "1D Change %":  chg_1d,
                    "52W Return %": ret_52w,
                    "Market Cap":   mkt_cap,
                    "In Portfolio": ticker in exclude or
                                    ticker.replace(suffix,"") + ".NS" in exclude or
                                    ticker.replace(suffix,"") + ".BO" in exclude,
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Market Cap", ascending=False).head(5).reset_index(drop=True)
    df["Rank"] = range(1, len(df)+1)
    return df


## ── Sector news proxy tickers — 2 representative stocks per Yahoo Finance sector ──
## Used to fetch sector-level context news alongside user's actual holdings
## Proxies are major liquid NSE stocks that best represent each sector's narrative
SECTOR_NEWS_PROXIES = {
    "Financial Services":     ["HDFCBANK.NS",   "ICICIBANK.NS"],
    "Technology":             ["TCS.NS",         "INFY.NS"],
    "Energy":                 ["RELIANCE.NS",    "ONGC.NS"],
    "Consumer Cyclical":      ["MARUTI.NS",      "TATAMOTORS.NS"],
    "Consumer Defensive":     ["HINDUNILVR.NS",  "ITC.NS"],
    "Healthcare":             ["SUNPHARMA.NS",   "DRREDDY.NS"],
    "Basic Materials":        ["TATASTEEL.NS",   "HINDALCO.NS"],
    "Industrials":            ["LT.NS",          "BEL.NS"],
    "Communication Services": ["BHARTIARTL.NS",  "ZEEENT.NS"],
    "Utilities":              ["NTPC.NS",        "POWERGRID.NS"],
    "Real Estate":            ["DLF.NS",         "GODREJPROP.NS"],
    "Consumer Durables":      ["TITAN.NS",       "VOLTAS.NS"],
    "Consumer Services":      ["DMART.NS",       "JUBLFOOD.NS"],
    "Capital Goods":          ["LT.NS",          "SIEMENS.NS"],
    "Services":               ["LT.NS",          "HCLTECH.NS"],
}

@st.cache_data(ttl=ttl_seconds)
def fetch_ticker_news(ticker, news_type="stock"):
    """
    Fetch news for a stock ticker.
    Primary:  Google News RSS (confirmed working on Streamlit Cloud)
    Fallback: yf.Search() for when Google News blocks
    Returns list of dicts: title, source, published, link, query_type
    """
    import datetime as _dt, urllib.request, urllib.parse
    import xml.etree.ElementTree as ET

    _clean  = ticker.replace(".NS","").replace(".BO","")
    results = []

    ## ── Method 1: Google News RSS (primary — best quality, India-specific) ───
    try:
        _query   = urllib.parse.quote(f"{_clean} NSE stock India")
        _url     = (f"https://news.google.com/rss/search?"
                    f"q={_query}&hl=en-IN&gl=IN&ceid=IN:en")
        _req     = urllib.request.Request(_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
            "Accept":     "application/rss+xml, application/xml, text/xml, */*",
        })
        with urllib.request.urlopen(_req, timeout=6) as _resp:
            _root = ET.fromstring(_resp.read())
        for _item in _root.findall(".//item")[:6]:
            _title = _item.findtext("title", "")
            _link  = _item.findtext("link",  "")
            _src   = _item.find("{https://news.google.com/rss}source")
            _source= _src.text if _src is not None else "Google News"
            _pub   = _item.findtext("pubDate", "")
            try:
                from email.utils import parsedate_to_datetime as _pdt
                _pub_str = _pdt(_pub).strftime("%d %b %Y %H:%M")
                _pub_ts  = int(_pdt(_pub).timestamp())
            except Exception:
                _pub_str = _pub[:16] if _pub else ""
                _pub_ts  = 0
            if _title:
                results.append({
                    "title":      _title,
                    "source":     _source,
                    "published":  _pub_str,
                    "pub_ts":     _pub_ts,
                    "link":       _link,
                    "query_type": news_type,
                    "ticker":     ticker,
                })
    except Exception:
        pass

    ## ── Method 2: yf.Search fallback (if Google News fails) ─────────────────
    if not results:
        try:
            _s   = yf.Search(_clean, news_count=6, max_results=0)
            _raw = _s.news or []
            for item in _raw[:6]:
                _title = item.get("title","") or item.get("headline","")
                _link  = (item.get("link","")
                          or item.get("url","")
                          or item.get("canonicalUrl",{}).get("url",""))
                _source = item.get("publisher","") or "Yahoo Finance"
                _pub_ts = item.get("providerPublishTime", 0) or 0
                try:
                    _pub_str = (_dt.datetime.fromtimestamp(int(_pub_ts))
                                .strftime("%d %b %Y %H:%M") if _pub_ts else "")
                except Exception:
                    _pub_str = ""
                if _title:
                    results.append({
                        "title":      _title,
                        "source":     _source,
                        "published":  _pub_str,
                        "pub_ts":     int(_pub_ts) if _pub_ts else 0,
                        "link":       _link,
                        "query_type": news_type,
                        "ticker":     ticker,
                    })
        except Exception:
            pass

    return results


@st.cache_data(ttl=ttl_seconds)
def fetch_sector_return(sector_ticker, from_date, to_date):
    try:
        data = yf.download(
            sector_ticker, start=from_date, end=to_date,
            progress=False, auto_adjust=True
        )
        if data.empty:
            return np.nan
        ## Handle both MultiIndex and flat — and both "Close" / "Adj Close"
        if isinstance(data.columns, pd.MultiIndex):
            level0 = data.columns.get_level_values(0).unique().tolist()
            col    = "Close" if "Close" in level0 else (
                     "Adj Close" if "Adj Close" in level0 else level0[0])
            prices = data[col].squeeze()
        else:
            col    = "Close" if "Close" in data.columns else (
                     "Adj Close" if "Adj Close" in data.columns else data.columns[0])
            prices = data[col]
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()
        prices    = prices.dropna()
        available = prices.index[prices.index >= pd.Timestamp(from_date)]
        if len(available) < 2:
            return np.nan
        return float(prices.iloc[-1] / prices[available[0]]) - 1
    except Exception:
        return np.nan


## ─────────────────────────────────────────────────────────────────────────────
# %%  HELPER FUNCTIONS
## ─────────────────────────────────────────────────────────────────────────────
def compute_returns(prices):
    return prices.pct_change().dropna()

def portfolio_series(returns, weights):
    return np.dot(returns.values, weights)

def historical_var(returns, alpha=0.95):
    return -np.percentile(returns, (1 - alpha) * 100)

def historical_es(returns, alpha=0.95):
    cutoff = np.percentile(returns, (1 - alpha) * 100)
    return -returns[returns <= cutoff].mean()

def xirr(cashflows, dates, guess=0.1):
    from scipy.optimize import brentq
    dates     = pd.to_datetime(dates)
    cashflows = np.array(cashflows, dtype=float)
    if not (np.any(cashflows > 0) and np.any(cashflows < 0)):
        return np.nan
    t0   = dates.min()
    days = np.array((dates - t0).days, dtype=float)
    def npv(rate):
        if rate <= -0.999999:
            return np.inf
        return np.sum(cashflows / (1 + rate) ** (days / 365.25))
    try:
        return brentq(npv, -0.99, 50, xtol=1e-8, maxiter=2000)
    except Exception:
        return np.nan

def compute_twr_portfolio_returns(prices, holdings_df):
    all_dates          = prices.index
    daily_port_returns = []
    for i in range(1, len(all_dates)):
        current_date = all_dates[i]
        prev_date    = all_dates[i - 1]
        ## Use prev_date not current_date — stock enters TWR the day AFTER purchase
        ## This matches real-world TWR sub-period chaining correctly
        active = holdings_df[
            pd.to_datetime(holdings_df["Buy Date"]) <= prev_date
        ].copy()
        if active.empty:
            continue
        available = [t for t in active["Ticker"].tolist() if t in prices.columns]
        if not available:
            continue
        try:
            prev_px    = prices.loc[prev_date,    available]
            current_px = prices.loc[current_date, available]
        except KeyError:
            continue
        ## Per-ticker filter: drop individual bad tickers, not the whole day
        valid      = prev_px.notna() & current_px.notna() & (prev_px > 0) & (current_px > 0)
        prev_px    = prev_px[valid]
        current_px = current_px[valid]
        if prev_px.empty:
            continue
        qty_map    = active.set_index("Ticker")["Quantity"]
        valid_list = prev_px.index.tolist()
        day_qty    = pd.Series({t: float(qty_map[t])
                                for t in valid_list if t in qty_map.index})
        if day_qty.empty:
            continue
        mkt_val        = prev_px * day_qty
        total_val      = mkt_val.sum()
        if total_val == 0:
            continue
        day_weights    = mkt_val / total_val
        stock_rets     = (current_px / prev_px) - 1
        port_daily_ret = (stock_rets * day_weights).sum()
        daily_port_returns.append((current_date, float(port_daily_ret)))
    if not daily_port_returns:
        return pd.Series(dtype=float)
    dates, rets = zip(*daily_port_returns)
    return pd.Series(list(rets), index=pd.DatetimeIndex(dates)).dropna()

def apply_dark_theme(ax, fig):
    ax.set_facecolor("#0E1117")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(axis="both", labelsize=9, colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")

## ─────────────────────────────────────────────────────────────────────────────
# %%  PLOT FUNCTIONS
## ─────────────────────────────────────────────────────────────────────────────
def plot_kde(port_returns):
    data     = port_returns.dropna().values
    kde_fn   = gaussian_kde(data, bw_method=0.4)
    x        = np.linspace(data.min() * 1.6, data.max() * 1.6, 600)
    y_kde    = kde_fn(x)
    mu       = data.mean()
    sigma    = data.std()
    med      = np.median(data)
    y_norm   = norm.pdf(x, mu, sigma)
    skew_val = skew(data)
    kurt_val = kurtosis(data, fisher=False)

    fig, ax  = plt.subplots(figsize=(12, 5))

    ## ── Loss / gain zone shading ─────────────────────────────────────────────
    ax.axvspan(x.min(), 0, alpha=0.04, color="#FF1744")
    ax.axvspan(0, x.max(), alpha=0.04, color="#2979FF")

    ## ── Fill under KDE ──────────────────────────────────────────────────────
    ax.fill_between(x, y_kde, alpha=0.22, color="#2979FF", label="_nolegend_")
    ax.plot(x, y_kde, linewidth=2.4, color="#2979FF",
            label="Empirical distribution (KDE)")

    ## ── Normal distribution overlay ─────────────────────────────────────────
    ax.plot(x, y_norm, linewidth=1.8, color="#FF9100", linestyle="--",
            alpha=0.85, label="Normal distribution (same μ, σ)")

    ## ── Mean line ───────────────────────────────────────────────────────────
    ax.axvline(mu, color="#00E676", linewidth=1.8, linestyle="-.",
               alpha=0.9, label=f"Mean ({mu:.4f})")
    ax.text(mu + sigma * 0.08, ax.get_ylim()[1] * 0.88,
            f"Mean\n{mu:.4f}", color="#00E676", fontsize=8,
            ha="left", va="top", fontweight="bold")

    ## ── Median line ─────────────────────────────────────────────────────────
    ax.axvline(med, color="#69F0AE", linewidth=1.5, linestyle=":",
               alpha=0.9, label=f"Median ({med:.4f})")
    ax.text(med - sigma * 0.08, ax.get_ylim()[1] * 0.72,
            f"Median\n{med:.4f}", color="#69F0AE", fontsize=8,
            ha="right", va="top", fontweight="bold")

    ## ── Zero line ───────────────────────────────────────────────────────────
    ax.axvline(0, color="white", linewidth=0.9, linestyle="-", alpha=0.3)

    ## ── Percentile rug marks ────────────────────────────────────────────────
    pct_labels = {5: "p5", 25: "p25", 50: "p50", 75: "p75", 95: "p95"}
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(data, p)
        col = "#9E9E9E"
        ax.axvline(val, ymin=0, ymax=0.04, color=col,
                   linewidth=1.3, alpha=0.7)
        if p in pct_labels:
            ax.text(val, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06,
                    pct_labels[p], color="#9E9E9E", fontsize=7,
                    ha="center", va="top")

    ## ── Kurtosis annotation — dynamic based on actual value ─────────────────
    if kurt_val > 4:
        kurt_color  = "#FF9100"
        kurt_face   = "#4A1900"
        kurt_edge   = "#FF9100"
        kurt_label  = (f"Kurtosis {kurt_val:.2f} — heavy fat tails\n"
                       f"{kurt_val/3:.1f}x more extreme moves vs normal")
    elif kurt_val > 3:
        kurt_color  = "#FFB74D"
        kurt_face   = "#3E2800"
        kurt_edge   = "#FFB74D"
        kurt_label  = (f"Kurtosis {kurt_val:.2f} — mild fat tails\n"
                       f"{kurt_val/3:.1f}x more extreme moves vs normal")
    else:
        kurt_color  = "#69F0AE"
        kurt_face   = "#003020"
        kurt_edge   = "#69F0AE"
        kurt_label  = f"Kurtosis {kurt_val:.2f} — near normal (no fat tails)"

    ax.text(0.98, 0.97, kurt_label, transform=ax.transAxes,
            fontsize=8.5, color=kurt_color, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=kurt_face, edgecolor=kurt_edge, alpha=0.9))

    ## ── Skewness annotation — dynamic based on actual value ─────────────────
    if skew_val < -0.5:
        skew_color = "#FF5252"
        skew_face  = "#3B0000"
        skew_edge  = "#FF5252"
        skew_label = (f"Skewness {skew_val:.2f} — strong left skew\n"
                      "Large losses occur more often than large gains")
    elif skew_val < -0.1:
        skew_color = "#FF8A80"
        skew_face  = "#2D0000"
        skew_edge  = "#FF8A80"
        skew_label = (f"Skewness {skew_val:.2f} — slight left skew\n"
                      "Minor downside tail bias present")
    elif skew_val > 0.5:
        skew_color = "#69F0AE"
        skew_face  = "#003020"
        skew_edge  = "#69F0AE"
        skew_label = (f"Skewness {skew_val:.2f} — strong right skew\n"
                      "Large gains occur more often than large losses")
    elif skew_val > 0.1:
        skew_color = "#B9F6CA"
        skew_face  = "#001A0A"
        skew_edge  = "#B9F6CA"
        skew_label = (f"Skewness {skew_val:.2f} — slight right skew\n"
                      "Minor upside tail bias — favourable")
    else:
        skew_color = "#90CAF9"
        skew_face  = "#001830"
        skew_edge  = "#90CAF9"
        skew_label = (f"Skewness {skew_val:.2f} — approximately symmetric\n"
                      "Gains and losses roughly balanced")

    ax.text(0.98, 0.76, skew_label, transform=ax.transAxes,
            fontsize=8.5, color=skew_color, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=skew_face, edgecolor=skew_edge, alpha=0.9))

    ## ── Summary metric boxes below the plot ─────────────────────────────────
    ax.set_title("Portfolio Daily Return Distribution — Empirical vs Normal",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Daily Return", fontsize=10)
    ax.set_ylabel("Density",      fontsize=10)
    ax.tick_params(axis="both",   labelsize=9)

    legend = ax.legend(fontsize=8.5, loc="upper left",
                       framealpha=0.85, ncol=1)
    legend.get_frame().set_facecolor("#0E1117")
    for text in legend.get_texts():
        text.set_color("white")

    apply_dark_theme(ax, fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    ## ── Four metric cards below chart ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean daily return",  f"{mu:.4f}",
              help="Average daily return of the portfolio")
    c2.metric("Daily volatility σ", f"{sigma:.4f}",
              f"{sigma * np.sqrt(252):.2%} annualised",
              help="Standard deviation of daily returns")

    kurt_delta = (f"Fat tails — {kurt_val/3:.1f}× normal"
                  if kurt_val > 3 else "Near normal distribution")
    c3.metric("Kurtosis (Pearson)", f"{kurt_val:.4f}",
              kurt_delta,
              delta_color="inverse" if kurt_val > 3 else "normal",
              help=">3 = fat tails. Extreme moves occur more often than normal.")

    skew_delta = ("Left tail — losses heavier" if skew_val < -0.1
                  else "Right tail — gains heavier" if skew_val > 0.1
                  else "Approximately symmetric")
    c4.metric("Skewness",           f"{skew_val:.4f}",
              skew_delta,
              delta_color="inverse" if skew_val < -0.1 else "normal",
              help="Negative = left tail risk. Positive = right tail advantage.")

def plot_var_distribution(horizon_returns, var_value, es_value,
                          method_name, horizon_days, mu_h=None, sigma_h=None):
    """
    Improved VaR histogram — color-coded risk zones, styled markers,
    metric cards below chart. No green color anywhere.
    """
    fig, ax = plt.subplots(figsize=(9, 4.2))

    var_thr = -var_value   ## negative — loss side
    es_thr  = -es_value    ## further negative

    if "Parametric" in method_name and mu_h is not None and sigma_h is not None:
        ## ── Parametric: smooth KDE curve with colored fills ──────────────────
        x = np.linspace(mu_h - 4.5 * sigma_h, mu_h + 4.5 * sigma_h, 800)
        y = norm.pdf(x, mu_h, sigma_h)

        ## Color zones — no green
        ax.fill_between(x, y,
                        where=(x <= es_thr),
                        color="#E2534A", alpha=0.85,
                        label=f"Extreme loss (< ES)")
        ax.fill_between(x, y,
                        where=((x > es_thr) & (x <= var_thr)),
                        color="#F09595", alpha=0.75,
                        label=f"VaR tail")
        ax.fill_between(x, y,
                        where=(x > var_thr),
                        color="#378ADD", alpha=0.25,
                        label="Normal range")
        ax.plot(x, y, color="#378ADD", linewidth=2.0)

        from matplotlib.patches import Patch as _PP
        _lp = ax.legend(handles=[
            _PP(color="#E2534A", alpha=0.85, label="Extreme loss (< ES)"),
            _PP(color="#F09595", alpha=0.75, label="VaR tail"),
            _PP(color="#378ADD", alpha=0.45, label="Normal range"),
        ], fontsize=9, loc="upper right",
           framealpha=0.92, edgecolor="#C8A951")
        for _t in _lp.get_texts(): _t.set_color("#F3F6FA")
        _lp.get_frame().set_facecolor("#0D1723")

    else:
        ## ── Historical / Monte Carlo: color-coded histogram bars ─────────────
        returns = np.array(horizon_returns)
        counts, bin_edges = np.histogram(returns, bins=40, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        for i, (cnt, center) in enumerate(zip(counts, bin_centers)):
            if center <= es_thr:
                col = "#E2534A"         ## extreme loss — dark red
            elif center <= var_thr:
                col = "#F09595"         ## VaR tail — light red
            else:
                col = "#378ADD"         ## normal range — blue
            ax.bar(center, cnt,
                   width=(bin_edges[1]-bin_edges[0])*0.92,
                   color=col, alpha=0.85 if center <= es_thr else
                               0.75 if center <= var_thr else 0.35,
                   edgecolor="none")

        ## Legend proxy patches — no green
        from matplotlib.patches import Patch
        _leg = ax.legend(handles=[
            Patch(color="#E2534A", alpha=0.85, label="Extreme loss (< ES)"),
            Patch(color="#F09595", alpha=0.75, label="VaR tail"),
            Patch(color="#378ADD", alpha=0.45, label="Normal range"),
        ], fontsize=9, loc="upper left",
           framealpha=0.92, edgecolor="#C8A951")
        for txt in _leg.get_texts():
            txt.set_color("#F3F6FA")
        _leg.get_frame().set_facecolor("#0D1723")

    ## ── VaR vertical line ────────────────────────────────────────────────────
    ax.axvline(var_thr, color="#E24B4A", linewidth=1.8,
               linestyle="--", alpha=0.95)
    y_top = ax.get_ylim()[1]
    ax.text(var_thr - 0.0002, y_top * 0.97,
            f"VaR\n{var_thr:.2%}",
            color="#E24B4A", fontsize=8, ha="right", va="top",
            fontweight="bold")

    ## ── ES vertical line ─────────────────────────────────────────────────────
    ax.axvline(es_thr, color="#BA7517", linewidth=1.6,
               linestyle=":", alpha=0.95)
    ax.text(es_thr - 0.0002, y_top * 0.78,
            f"ES\n{es_thr:.2%}",
            color="#BA7517", fontsize=8, ha="right", va="top",
            fontweight="bold")

    ## ── Zero line ────────────────────────────────────────────────────────────
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.3)

    ## ── Styling ──────────────────────────────────────────────────────────────
    ax.set_title(f"Return Distribution — {method_name}",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(f"{horizon_days}-Day Portfolio Returns", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    if "Parametric" in method_name:
        legend = ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85,
                           edgecolor="#21262D")
        legend.get_frame().set_facecolor("#0E1117")
        for txt in legend.get_texts():
            txt.set_color("white")

    apply_dark_theme(ax, fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)



def plot_drawdown(port_returns_series):
    cumulative   = (1 + port_returns_series).cumprod()
    rolling_max  = cumulative.cummax()
    drawdown     = (cumulative - rolling_max) / rolling_max
    max_dd       = float(drawdown.min())
    max_dd_date  = drawdown.idxmin()

    ## Current drawdown — last value in the series (updates end of each day)
    current_dd   = float(drawdown.iloc[-1])
    current_date = drawdown.index[-1]

    ## Find ALL recovery events (every time drawdown returns to 0 after a trough)
    ## Only mark the recovery from the MAX drawdown specifically
    post_max       = drawdown[drawdown.index > max_dd_date]
    recovery_dates = post_max[post_max >= -0.001]
    recovery_date  = recovery_dates.index[0] if not recovery_dates.empty else None

    ## Is portfolio currently in drawdown?
    currently_in_dd = current_dd < -0.001

    fig, ax = plt.subplots(figsize=(12, 5))

    ## ── Drawdown fill ───────────────────────────────────────────────────────
    ax.fill_between(drawdown.index, drawdown.values * 100, 0,
                    alpha=0.5, color="#FF1744", label="Drawdown")
    ax.plot(drawdown.index, drawdown.values * 100,
            linewidth=1.5, color="#FF1744")

    ## ── Max drawdown vertical line ──────────────────────────────────────────
    ax.axvline(max_dd_date, color="#C8A951", linestyle="--",
               linewidth=1.5, label=f"Max DD: {max_dd:.2%}")

    ## ── Max DD annotation — top right of chart to avoid covering the trough ─
    ax.annotate(
        f"Max DD\n{max_dd:.2%}\n{max_dd_date.strftime('%b %Y')}",
        xy=(max_dd_date, max_dd * 100),
        xytext=(0.72, 0.18), textcoords="axes fraction",
        fontsize=8.5, color="#C8A951", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1200",
                  edgecolor="#C8A951", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="#C8A951", lw=1.2)
    )

    ## ── Recovery from max drawdown ──────────────────────────────────────────
    if recovery_date is not None:
        duration_days = (recovery_date - max_dd_date).days
        ax.axvline(recovery_date, color="#00C853", linestyle=":",
                   linewidth=1.5, label=f"Recovered max DD: {duration_days} days")
        ax.annotate(
            f"Recovered\n{recovery_date.strftime('%b %Y')}",
            xy=(recovery_date, 0),
            xytext=(0.45, 0.85), textcoords="axes fraction",
            fontsize=8, color="#00C853",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#001A08",
                      edgecolor="#00C853", alpha=0.85),
            arrowprops=dict(arrowstyle="->", color="#00C853", lw=1.2)
        )

    ## ── Current status marker at the right edge ─────────────────────────────
    ## Dynamic x-offset — push label LEFT when current date is near right edge
    _date_range  = (drawdown.index[-1] - drawdown.index[0]).days
    _date_pos    = (current_date - drawdown.index[0]).days / max(_date_range, 1)
    _offset_x    = -80 if _date_pos > 0.85 else -55

    if currently_in_dd:
        ## Red dot at current drawdown level
        ax.scatter([current_date], [current_dd * 100],
                   color="#FF1744", s=60, zorder=5)
        ax.annotate(
            f"Today\n{current_dd:.2%}",
            xy=(current_date, current_dd * 100),
            xytext=(_offset_x, 10), textcoords="offset points",
            fontsize=8.5, color="#FF6D6D", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A0000",
                      edgecolor="#FF1744", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#FF1744", lw=1.2)
        )
    else:
        ## Green dot — fully recovered
        ax.scatter([current_date], [current_dd * 100],
                   color="#00C853", s=60, zorder=5)
        ax.annotate(
            f"Today\n{current_dd:.2%}\nRecovered",
            xy=(current_date, current_dd * 100),
            xytext=(_offset_x, -20), textcoords="offset points",
            fontsize=8.5, color="#00C853", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#001A08",
                      edgecolor="#00C853", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#00C853", lw=1.2)
        )

    ## ── Zero line ───────────────────────────────────────────────────────────
    ax.axhline(0, color="white", linestyle="-", linewidth=0.8, alpha=0.4)

    ax.set_title("Portfolio Drawdown — Peak-to-Trough Decline",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date",         fontsize=10)
    ax.set_ylabel("Drawdown (%)", fontsize=10)

    ## ── Legend — moved to upper right so it never covers the trough ─────────
    legend = ax.legend(fontsize=8.5, loc="upper right",
                       framealpha=0.85, ncol=1)
    legend.get_frame().set_facecolor("#0E1117")
    legend.get_frame().set_edgecolor("#21262D")
    for text in legend.get_texts():
        text.set_color("white")

    apply_dark_theme(ax, fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    ## ── Metric cards below chart ─────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Max Drawdown",    f"{max_dd:.2%}",
              help="Largest peak-to-trough decline over the full period")
    d2.metric("Max DD Date",     max_dd_date.strftime("%d %b %Y"),
              help="Date when maximum drawdown was reached")
    d3.metric("Current Drawdown", f"{current_dd:.2%}",
              delta="In drawdown" if currently_in_dd else "Recovered",
              delta_color="inverse" if currently_in_dd else "normal",
              help=f"Drawdown as of {current_date.strftime('%d %b %Y')} — updates end of each trading day")
    if recovery_date is not None:
        duration_days = (recovery_date - max_dd_date).days
        d4.metric("Recovery (Max DD)", recovery_date.strftime("%b %Y"),
                  f"{duration_days} days to recover",
                  help="When the portfolio recovered from its maximum drawdown")
    else:
        d4.metric("Recovery (Max DD)", "Not yet",
                  f"Still {abs(max_dd):.2%} below peak",
                  delta_color="inverse",
                  help="Portfolio has not yet recovered from maximum drawdown")

    return max_dd, recovery_date

## ─────────────────────────────────────────────────────────────────────────────
# %%  INSTITUTIONAL PDF REPORT — WITH CHARTS
## ─────────────────────────────────────────────────────────────────────────────
def create_institutional_pdf(
    holdings_df, risk_df, portfolio_stats, attr_df,
    total_invested, total_current, total_pnl,
    portfolio_xirr, max_dd, recovery_date,
    RISK_FREE_ANNUAL,
    port_returns=None,
    benchmark_data=None,
    sector_df=None,
    stress_scenarios=None,
    portfolio_beta=None,
    total_current_val=None,
    benchmark_name="Nifty 50",
    benchmark_ticker="^NSEI",
    portfolio_exchange="NSE",
    client_name="",
    quote_text="",
    quote_author="",
):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.patches as _mpatches
    import warnings
    warnings.filterwarnings("ignore", message="Tight layout not applied")

    ## ── PDF palette ─────────────────────────────────────────────────────────
    DARK_NAVY    = rl_colors.HexColor("#0A1628")
    MID_NAVY     = rl_colors.HexColor("#112244")
    STEEL        = rl_colors.HexColor("#1E3A5F")
    STEEL_LIGHT  = rl_colors.HexColor("#EAF0F8")
    ACCENT       = rl_colors.HexColor("#C8A951")
    ACCENT_LIGHT = rl_colors.HexColor("#F5EDD4")
    PDF_GREEN    = rl_colors.HexColor("#1A6B3C")
    GREEN_LIGHT  = rl_colors.HexColor("#E6F4EC")
    PDF_RED      = rl_colors.HexColor("#8B1A1A")
    RED_LIGHT    = rl_colors.HexColor("#FBE9E9")
    AMBER        = rl_colors.HexColor("#B8690A")
    AMBER_LIGHT  = rl_colors.HexColor("#FEF3E2")
    WHITE        = rl_colors.white
    BLACK        = rl_colors.HexColor("#0D0D0D")
    GREY_700     = rl_colors.HexColor("#4A4A4A")
    GREY_500     = rl_colors.HexColor("#777777")
    GREY_300     = rl_colors.HexColor("#BBBBBB")
    GREY_100     = rl_colors.HexColor("#F4F4F4")
    BORDER       = rl_colors.HexColor("#D0D8E4")
    ROW_ALT      = rl_colors.HexColor("#F7F9FC")

    C_NAVY="#0A1628"; C_STEEL="#1E3A5F"; C_GOLD="#C8A951"
    C_GREEN="#1A6B3C"; C_RED="#8B1A1A"; C_AMBER="#B8690A"
    C_TEAL="#00897B"; C_BG="#F7F9FC"

    PAGE_W, PAGE_H = A4
    L_MARGIN=1.8*cm; R_MARGIN=1.8*cm; T_MARGIN=1.8*cm; B_MARGIN=2.0*cm
    CW = PAGE_W - L_MARGIN - R_MARGIN

    ## ── Styles ──────────────────────────────────────────────────────────────
    def _S(name, **kw): return ParagraphStyle(name, **kw)
    TITLE_S   = _S("TT", fontSize=22,fontName="Helvetica-Bold",textColor=WHITE,   alignment=TA_CENTER,leading=28)
    SUBTITLE_S= _S("ST", fontSize=10,fontName="Helvetica",     textColor=GREY_300,alignment=TA_CENTER,leading=14)
    H_SECTION = _S("HS", fontSize=10,fontName="Helvetica-Bold",textColor=WHITE,   leading=14)
    H_SUB     = _S("HB", fontSize=9, fontName="Helvetica-Bold",textColor=STEEL,   leading=13,spaceAfter=3,spaceBefore=5)
    BODY_S    = _S("BS", fontSize=8.5,fontName="Helvetica",    textColor=GREY_700,leading=13,spaceAfter=3)
    CAPTION_S = _S("CS", fontSize=7, fontName="Helvetica",     textColor=GREY_500,leading=10,spaceAfter=2)
    DISC_S    = _S("DS", fontSize=6.5,fontName="Helvetica",    textColor=GREY_500,leading=9, spaceAfter=1)
    DISC_LIGHT= _S("DL", fontSize=6.5,fontName="Helvetica",    textColor=GREY_300,leading=9)
    META_S    = _S("MS", fontSize=8, fontName="Helvetica",     textColor=GREY_700,alignment=TA_CENTER,leading=11)
    META_BOLD = _S("MB", fontSize=8, fontName="Helvetica-Bold",textColor=BLACK,   alignment=TA_CENTER,leading=11)
    KPI_LBL   = _S("KL", fontSize=7, fontName="Helvetica",     textColor=GREY_500,alignment=TA_CENTER,leading=9)
    KPI_VAL   = _S("KV", fontSize=12,fontName="Helvetica-Bold",textColor=DARK_NAVY,alignment=TA_CENTER,leading=14)
    KPI_VAL_G = _S("KG", fontSize=12,fontName="Helvetica-Bold",textColor=PDF_GREEN,alignment=TA_CENTER,leading=14)
    KPI_VAL_R = _S("KR", fontSize=12,fontName="Helvetica-Bold",textColor=PDF_RED,  alignment=TA_CENTER,leading=14)

    def _section_block(title, subtitle=""):
        bar = Table([[Paragraph(title, H_SECTION)]], colWidths=[CW])
        bar.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),STEEL),("LEFTPADDING",(0,0),(-1,-1),10),
            ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LINEBEFORE",(0,0),(0,-1),3,ACCENT),
        ]))
        items = [bar]
        if subtitle:
            items += [Spacer(1,1*mm), Paragraph(subtitle, CAPTION_S)]
        items.append(Spacer(1,2.5*mm))
        return items

    def _base_ts():
        return TableStyle([
            ("BACKGROUND",(0,0),(-1,0),STEEL),("TEXTCOLOR",(0,0),(-1,0),WHITE),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),
            ("TOPPADDING",(0,0),(-1,-1),4.5),("BOTTOMPADDING",(0,0),(-1,-1),4.5),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("BOX",(0,0),(-1,-1),0.5,BORDER),
            ("INNERGRID",(0,0),(-1,-1),0.3,BORDER),("ALIGN",(0,0),(0,-1),"LEFT"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,ROW_ALT]),
        ])

    def _info_box(text, style="info"):
        cm_map={"info":(STEEL_LIGHT,STEEL),"success":(GREEN_LIGHT,PDF_GREEN),
                "warning":(AMBER_LIGHT,AMBER),"risk":(RED_LIGHT,PDF_RED)}
        bg,border=cm_map.get(style,(STEEL_LIGHT,STEEL))
        t=Table([[Paragraph(text,BODY_S)]],colWidths=[CW])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.8,border),
            ("LINEBEFORE",(0,0),(0,-1),3,border),("LEFTPADDING",(0,0),(-1,-1),10),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ]))
        return t

    def _cap_box(text):
        t=Table([[Paragraph(text,CAPTION_S)]],colWidths=[CW])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),GREY_100),("LEFTPADDING",(0,0),(-1,-1),8),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LINEABOVE",(0,0),(-1,-1),0.5,BORDER),
        ]))
        return t

    def _fig2img(fig, w_cm, h_cm, dpi=130):
        buf = io.BytesIO()
        fig.savefig(buf,format="png",dpi=dpi,bbox_inches="tight",facecolor=fig.get_facecolor())
        buf.seek(0); plt.close(fig)
        return Image(buf,width=w_cm*cm,height=h_cm*cm)

    def _styled(fig, ax):
        fig.patch.set_facecolor(C_BG); ax.set_facecolor(C_BG)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC"); ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(colors="#555555",labelsize=8)
        ax.xaxis.label.set_color("#555555"); ax.yaxis.label.set_color("#555555")
        ax.title.set_color(C_NAVY)
        ax.yaxis.grid(True,alpha=0.4,color="#CCCCCC",linestyle="--"); ax.set_axisbelow(True)

    ## ── Numbered canvas ─────────────────────────────────────────────────────
    class _NC(rl_canvas.Canvas):
        def __init__(self,*a,**k): rl_canvas.Canvas.__init__(self,*a,**k); self._sv=[]
        def showPage(self): self._sv.append(dict(self.__dict__)); self._startPage()
        def save(self):
            total=len(self._sv)
            for s in self._sv:
                self.__dict__.update(s); self._chrome(total); rl_canvas.Canvas.showPage(self)
            rl_canvas.Canvas.save(self)
        def _chrome(self,total):
            pg=self._pageNumber
            if pg==1: return
            self.setFillColor(DARK_NAVY); self.rect(0,0,PAGE_W,1.2*cm,fill=1,stroke=0)
            self.setFillColor(GREY_300); self.setFont("Helvetica",6.5)
            self.drawString(L_MARGIN,0.44*cm,"PORTFOLIO ANALYSIS REPORT")
            self.setFillColor(ACCENT); self.setFont("Helvetica-Bold",6.5)
            self.drawCentredString(PAGE_W/2,0.44*cm,"STRICTLY CONFIDENTIAL — NOT FOR DISTRIBUTION")
            self.setFillColor(GREY_300); self.setFont("Helvetica",6.5)
            self.drawRightString(PAGE_W-R_MARGIN,0.44*cm,f"Page {pg} of {total}")
            self.setStrokeColor(ACCENT); self.setLineWidth(1.5)
            self.line(L_MARGIN,PAGE_H-0.9*cm,PAGE_W-R_MARGIN,PAGE_H-0.9*cm)
            self.setFillColor(GREY_500); self.setFont("Helvetica",6)
            gd=pd.Timestamp.now().strftime("%d %b %Y %H:%M")
            self.drawRightString(PAGE_W-R_MARGIN,PAGE_H-0.65*cm,f"Generated: {gd}")

    ## ══════════════════════════════════════════════════════════════════════════
    ##  CHART BUILDERS — using live portfolio data
    ## ══════════════════════════════════════════════════════════════════════════

    def _chart_composition():
        wts   = (holdings_df["Current Value"]/holdings_df["Current Value"].sum()*100).tolist()
        ticks = holdings_df["Ticker"].str.replace(".NS","").tolist()
        pal   = [C_STEEL,C_GOLD,C_RED,C_GREEN,C_TEAL,"#7B1FA2","#0288D1","#F57F17","#4CAF50","#E64A19"][:len(ticks)]
        nifty_approx = {"Communication Services":3.8,"Consumer Cyclical":8.9,"Utilities":2.2,
                        "Industrials":4.3,"Technology":13.2,"Financial Services":32.9,
                        "Healthcare":4.8,"Energy":11.9,"Basic Materials":3.2,"Consumer Defensive":3.9}
        nifty_w = []
        if sector_df is not None and not sector_df.empty and "Sector" in sector_df.columns:
            for t in holdings_df["Ticker"].tolist():
                row_s = sector_df[sector_df["Ticker"]==t]
                if not row_s.empty:
                    nifty_w.append(nifty_approx.get(row_s.iloc[0]["Sector"],0.0))
                else:
                    nifty_w.append(0.0)
        else:
            nifty_w = [0.0]*len(ticks)

        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4)); fig.patch.set_facecolor(C_BG)
        wedges,_,ats=ax1.pie(wts,autopct="%1.1f%%",colors=pal,explode=[0.04]*len(ticks),
                              startangle=140,pctdistance=0.78,wedgeprops=dict(linewidth=1.5,edgecolor="white"))
        for at in ats: at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
        ax1.set_title("Portfolio Weight Distribution",fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax1.legend(wedges,[f"{t} ({w:.1f}%)" for t,w in zip(ticks,wts)],
                   loc="upper center", bbox_to_anchor=(0.5,-0.05),
                   ncol=min(3,len(ticks)), fontsize=8, frameon=False,
                   labelcolor="white")
        ax1.set_facecolor(C_BG)
        x=np.arange(len(ticks)); w=0.36
        b1=ax2.bar(x-w/2,wts,width=w,color=C_STEEL,label="Portfolio",alpha=0.9,edgecolor="white")
        b2=ax2.bar(x+w/2,nifty_w,width=w,color=C_GOLD,label=BENCHMARK_NAME,alpha=0.9,edgecolor="white")
        for bar in b1:
            h_=bar.get_height()
            ax2.text(bar.get_x()+bar.get_width()/2,h_+0.5,f"{h_:.1f}%",ha="center",va="bottom",fontsize=7,color=C_STEEL,fontweight="bold")
        ax2.set_xticks(x); ax2.set_xticklabels(ticks,fontsize=8,rotation=15,ha="right")
        ax2.set_ylabel("Weight (%)",fontsize=9)
        ax2.set_title(f"Portfolio vs {benchmark_name} Weights",fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax2.legend(fontsize=8,frameon=False); _styled(fig,ax2); ax2.set_facecolor(C_BG)
        plt.tight_layout(pad=1.5); return fig

    def _chart_pnl():
        ticks=[h.replace(".NS","") for h in holdings_df["Ticker"].tolist()]
        pnls=holdings_df["P&L"].tolist(); rets=holdings_df["Return %"].tolist()
        c_p=[C_GREEN if p>=0 else C_RED for p in pnls]; c_r=[C_GREEN if r>=0 else C_RED for r in rets]
        _pnl_h = max(3.8, len(ticks) * 0.5)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,_pnl_h)); fig.patch.set_facecolor(C_BG)
        bars1=ax1.barh(ticks,pnls,color=c_p,edgecolor="white",height=0.55)
        ## Extend x-axis to prevent labels overflowing into ticker names
        _mx_p = max(abs(p) for p in pnls) if pnls else 1
        ax1.set_xlim(-_mx_p*1.35, _mx_p*1.35)
        for bar,val in zip(bars1,pnls):
            _lbl_x = bar.get_width() + _mx_p*0.03 if val>=0 else bar.get_width() - _mx_p*0.03
            ax1.text(_lbl_x, bar.get_y()+bar.get_height()/2,
                     f"Rs.{val:+,.0f}",
                     ha="left" if val>=0 else "right", va="center",
                     fontsize=7.5, fontweight="bold",
                     color=C_GREEN if val>=0 else C_RED)
        ax1.axvline(0,color="#AAAAAA",lw=1,linestyle="--")
        ax1.set_xlabel("P&L (Rs.)",fontsize=9); ax1.set_title("Absolute P&L per Holding",fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        _styled(fig,ax1); ax1.set_facecolor(C_BG); ax1.xaxis.grid(True,alpha=0.4,color="#CCCCCC"); ax1.yaxis.grid(False)
        bars2=ax2.barh(ticks,rets,color=c_r,edgecolor="white",height=0.55)
        _mx_r = max(abs(r) for r in rets) if rets else 1
        ax2.set_xlim(-_mx_r*1.35, _mx_r*1.35)
        for bar,val in zip(bars2,rets):
            _lbl_x = bar.get_width() + _mx_r*0.03 if val>=0 else bar.get_width() - _mx_r*0.03
            ax2.text(_lbl_x, bar.get_y()+bar.get_height()/2,
                     f"{val:+.1f}%",
                     ha="left" if val>=0 else "right", va="center",
                     fontsize=7.5, fontweight="bold",
                     color=C_GREEN if val>=0 else C_RED)
        ax2.axvline(0,color="#AAAAAA",lw=1,linestyle="--")
        ax2.set_xlabel("Return (%)",fontsize=9); ax2.set_title("Return % per Holding",fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        _styled(fig,ax2); ax2.set_facecolor(C_BG); ax2.xaxis.grid(True,alpha=0.4,color="#CCCCCC"); ax2.yaxis.grid(False)
        plt.tight_layout(pad=1.5); return fig

    def _chart_cumulative():
        fig,ax=plt.subplots(figsize=(10,4)); fig.patch.set_facecolor(C_BG)
        if port_returns is not None and len(port_returns)>2:
            if benchmark_data is not None and len(benchmark_data)>2:
                bm_r=benchmark_data.pct_change().dropna()
                al=pd.concat([port_returns,bm_r],axis=1).dropna(); al.columns=["P","B"]
                ## ── CRITICAL: rebase BOTH series to 100 at the SAME start date ──
                ## Without this, cumprod() starts at (1+first_return)*100 not 100
                ## making the comparison between portfolio and benchmark wrong
                pc=(1+al["P"]).cumprod(); bc=(1+al["B"]).cumprod()
                pc=(pc/pc.iloc[0])*100;   bc=(bc/bc.iloc[0])*100
                port_total_pct = pc.iloc[-1]-100
                bench_total_pct= bc.iloc[-1]-100
                ax.plot(pc.index,pc.values,color=C_STEEL,lw=2.2,
                        label=f"Portfolio (TWR) {port_total_pct:+.2f}%",zorder=3)
                ax.plot(bc.index,bc.values,color=C_GOLD, lw=2, linestyle="--",
                        label=f"{benchmark_name} {bench_total_pct:+.2f}%",zorder=3)
                ax.fill_between(pc.index,pc.values,bc.values,
                                where=(pc.values>=bc.values),alpha=0.15,color=C_GREEN)
                ax.fill_between(pc.index,pc.values,bc.values,
                                where=(pc.values<bc.values), alpha=0.15,color=C_RED)
            else:
                pc=(1+port_returns).cumprod()
                pc=(pc/pc.iloc[0])*100
                ax.plot(pc.index,pc.values,color=C_STEEL,lw=2.2,
                        label=f"Portfolio (TWR) {pc.iloc[-1]-100:+.2f}%",zorder=3)
        else:
            tr=(total_current-total_invested)/total_invested*100
            ax.text(0.5,0.5,f"Total Return: {tr:+.2f}%",ha="center",va="center",fontsize=14,
                    fontweight="bold",transform=ax.transAxes,color=C_GREEN if tr>=0 else C_RED)
        ax.axhline(100,color="#AAAAAA",lw=0.8,linestyle=":",alpha=0.7)
        ax.set_xlabel("Date",fontsize=9); ax.set_ylabel("Indexed Return (Base=100)",fontsize=9)
        ax.set_title(f"Portfolio vs {benchmark_name} — Cumulative Performance (Base 100)",
                     fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax.legend(fontsize=8.5,frameon=False); _styled(fig,ax)
        plt.tight_layout(pad=1.2); return fig

    def _chart_drawdown():
        fig,ax=plt.subplots(figsize=(10,3.8)); fig.patch.set_facecolor(C_BG)
        if port_returns is not None and len(port_returns)>2:
            cum=((1+port_returns).cumprod()); rm=cum.cummax(); dd=(cum-rm)/rm*100
            current_dd = float(dd.iloc[-1])
            max_dd_val = float(dd.min())
            idx_max    = dd.idxmin()
            current_dt = dd.index[-1]
            _date_range= max((dd.index[-1]-dd.index[0]).days,1)
            ax.fill_between(dd.index,dd.values,0,alpha=0.5,color=C_RED,label="Drawdown")
            ax.plot(dd.index,dd.values,color=C_RED,lw=1.2)
            ax.axvline(idx_max,color=C_AMBER,lw=1.5,linestyle="--",alpha=0.8,
                       label=f"Max DD: {max_dd_val:.2f}%")
            _xoff = 10 if (idx_max-dd.index[0]).days/_date_range < 0.75 else -75
            ax.annotate(f"Max DD\n{max_dd_val:.2f}%\n{idx_max.strftime('%b %Y')}",
                        xy=(idx_max,max_dd_val),xytext=(_xoff,-8),
                        textcoords="offset points",fontsize=7.5,color=C_AMBER,fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",facecolor="#1A1200",edgecolor=C_AMBER,alpha=0.85),
                        arrowprops=dict(arrowstyle="->",color=C_AMBER,lw=1))
            _in_dd   = current_dd < -0.1
            _dcol    = "#FF1744" if _in_dd else "#00C853"
            ax.scatter([current_dt],[current_dd],color=_dcol,s=50,zorder=5)
            _cxoff   = -75 if (current_dt-dd.index[0]).days/_date_range > 0.85 else -55
            _dlabel  = f"Today\n{current_dd:.2f}%" + ("" if _in_dd else "\nRecovered")
            ax.annotate(_dlabel,xy=(current_dt,current_dd),xytext=(_cxoff,10),
                        textcoords="offset points",fontsize=7.5,color=_dcol,fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="#1A0000" if _in_dd else "#001A08",
                                  edgecolor=_dcol,alpha=0.85),
                        arrowprops=dict(arrowstyle="->",color=_dcol,lw=1))
        else:
            ax.text(0.5,0.5,f"Max Drawdown: {max_dd:.2%}",ha="center",va="center",
                    fontsize=14,fontweight="bold",transform=ax.transAxes,color=C_RED)
        ax.axhline(0,color="#AAAAAA",lw=0.8)
        ax.set_xlabel("Date",fontsize=9); ax.set_ylabel("Drawdown (%)",fontsize=9)
        ax.set_title("Portfolio Drawdown — Peak-to-Trough Decline",
                     fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax.legend(fontsize=7.5,frameon=False)
        _styled(fig,ax); plt.tight_layout(pad=1.2); return fig

    def _chart_kde():
        fig,ax=plt.subplots(figsize=(10,3.8)); fig.patch.set_facecolor(C_BG)
        if port_returns is not None and len(port_returns)>10:
            data=port_returns.dropna().values
            kde_fn=gaussian_kde(data,bw_method=0.4)
            x=np.linspace(data.min()*1.5,data.max()*1.5,500)
            y_kde=kde_fn(x); mu_d,s_d=data.mean(),data.std()
            y_norm=norm.pdf(x,mu_d,s_d)
            var_lev=np.percentile(data,(1-portfolio_stats.get("alpha",0.95))*100)
            ax.fill_between(x,y_kde,alpha=0.35,color=C_STEEL)
            ax.plot(x,y_kde,color=C_STEEL,lw=2,label="Empirical KDE")
            ax.plot(x,y_norm,color=C_GOLD,lw=1.5,linestyle="--",label="Normal Dist.",alpha=0.8)
            x_tail=x[x<=var_lev]
            ax.fill_between(x_tail,kde_fn(x_tail),alpha=0.75,color=C_RED,
                            label=f"VaR Tail ({(1-portfolio_stats.get('alpha',0.95)):.0%})")
            ax.axvline(mu_d,color=C_STEEL,lw=1.5,linestyle="--",alpha=0.8,label=f"Mean: {mu_d:.4f}")
            ax.axvline(np.median(data),color=C_TEAL,lw=1.5,linestyle=":",alpha=0.8,label=f"Median: {np.median(data):.4f}")
            ax.axvline(0,color="#AAAAAA",lw=1,alpha=0.6,label="Zero")
        else:
            ax.text(0.5,0.5,"Insufficient data",ha="center",va="center",transform=ax.transAxes,fontsize=11,color=C_AMBER)
        ax.set_xlabel("Daily Return",fontsize=9); ax.set_ylabel("Density",fontsize=9)
        ax.set_title("Daily Return Distribution — Empirical vs Normal",fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax.legend(fontsize=7.5,frameon=False,ncol=3); _styled(fig,ax)
        plt.tight_layout(pad=1.2); return fig

    def _chart_waterfall():
        ## Sort by contribution descending — positive first, negative last
        ticks=[h.replace(".NS","") for h in holdings_df["Ticker"].tolist()]
        wts=(holdings_df["Current Value"]/holdings_df["Current Value"].sum()).values
        rets=(holdings_df["Return %"]/100).values
        contribs=[float(w*r*100) for w,r in zip(wts,rets)]
        paired = sorted(zip(ticks, contribs), key=lambda x: x[1], reverse=True)
        ticks_s    = [p[0] for p in paired]
        contribs_s = [p[1] for p in paired]
        net    = sum(contribs_s)
        labels = ticks_s + ["Net Result"]
        vals   = contribs_s + [net]
        ## Standalone bars — each from zero, no accumulation
        bar_colors = [C_GREEN if v>=0 else C_RED for v in contribs_s] + [C_STEEL]
        _wf_h = max(4.0, len(labels) * 0.45)
        fig,ax=plt.subplots(figsize=(10,_wf_h)); fig.patch.set_facecolor(C_BG)
        bars=ax.bar(range(len(labels)), vals, color=bar_colors,
                    edgecolor="white", lw=1, width=0.6, zorder=3)
        y_max = max(abs(v) for v in vals) * 0.12
        for bar, val in zip(bars, vals):
            y_pos = val + y_max if val >= 0 else val - y_max
            ax.text(bar.get_x()+bar.get_width()/2, y_pos,
                    f"{val:+.2f}%", ha="center", va="bottom" if val>=0 else "top",
                    fontsize=7.5, fontweight="bold",
                    color=C_GREEN if val>=0 else C_RED)
        ax.axhline(0,color="#AAAAAA",lw=1,linestyle="-",alpha=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Return Contribution (%)",fontsize=9)
        ax.set_title("Return Contribution — Weight x Stock Return",
                     fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax.legend(handles=[_mpatches.Patch(facecolor=C_GREEN,label="Positive"),
                            _mpatches.Patch(facecolor=C_RED,label="Negative"),
                            _mpatches.Patch(facecolor=C_STEEL,label="Net result")],
                  fontsize=8,frameon=False)
        _styled(fig,ax); plt.tight_layout(pad=1.2); return fig

    def _chart_bhb():
        if sector_df is None or sector_df.empty: return None
        ticks=[t.replace(".NS","") for t in sector_df["Ticker"].tolist()]
        ## Multiply by 100 to convert decimals to percentage points for display
        ## sector_df stores effects as decimals (e.g. -0.0529) not % (e.g. -5.29)
        alloc  = [v*100 for v in sector_df["Allocation Effect"].fillna(0).tolist()]
        sel    = [v*100 for v in sector_df["Selection Effect"].fillna(0).tolist()]
        inter  = [v*100 for v in sector_df["Interaction Effect"].fillna(0).tolist()]
        total_e= [v*100 for v in sector_df["Total Active Return"].fillna(0).tolist()]
        ## Dynamic height — scales with number of stocks
        _n_stocks_bhb = len(ticks)   ## ticks defined above, x not yet
        _bhb_h = max(4.2, _n_stocks_bhb * 0.5)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,_bhb_h)); fig.patch.set_facecolor(C_BG)
        x=np.arange(len(ticks)); w=0.22
        ax1.bar(x-w,alloc,width=w,color=C_STEEL,label="Allocation",alpha=0.9)
        ax1.bar(x,sel,  width=w,color=C_GOLD, label="Selection",  alpha=0.9)
        ax1.bar(x+w,inter,width=w,color=C_TEAL,label="Interaction",alpha=0.9)
        ax1.axhline(0,color="#AAAAAA",lw=1)
        ax1.set_xticks(x); ax1.set_xticklabels(ticks,fontsize=8,rotation=15,ha="right")
        ax1.set_ylabel("Effect (%)",fontsize=9)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.1f}%"))
        ax1.set_title("BHB Effects by Stock",fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax1.legend(fontsize=7.5,frameon=False); _styled(fig,ax1); ax1.set_facecolor(C_BG)
        c_t=[C_GREEN if t>0 else C_RED for t in total_e]
        bars=ax2.bar(ticks,total_e,color=c_t,edgecolor="white",width=0.55)
        mx_t=max(abs(t) for t in total_e)*0.12 if total_e else 0.01
        for bar,val in zip(bars,total_e):
            ax2.text(bar.get_x()+bar.get_width()/2,val+(mx_t if val>=0 else -mx_t),
                     f"{val:+.2f}%",ha="center",va="bottom" if val>=0 else "top",
                     fontsize=8,fontweight="bold",color=C_GREEN if val>=0 else C_RED)
        ax2.axhline(0,color="#AAAAAA",lw=1)
        ax2.set_xticks(range(len(ticks))); ax2.set_xticklabels(ticks,fontsize=8,rotation=15,ha="right")
        ax2.set_title("Total Active Return per Stock",fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax2.set_ylabel("Total Active Return (%)",fontsize=9)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.1f}%"))
        ## Add 20% padding above highest bar so labels don't clip
        _y2_max = max(abs(v) for v in total_e) if total_e else 1
        _y2_cur = ax2.get_ylim()
        ax2.set_ylim(_y2_cur[0], max(_y2_cur[1], _y2_max * 1.25))
        _styled(fig,ax2); ax2.set_facecolor(C_BG)
        plt.tight_layout(pad=1.5); return fig

    def _chart_sector_vs_stock():
        if sector_df is None or sector_df.empty: return None
        ticks=[t.replace(".NS","") for t in sector_df["Ticker"].tolist()]
        stock_r=sector_df["Stock Return"].tolist()
        sector_r=sector_df["Sector Return"].tolist()
        is_etf_=sector_df["Is ETF"].tolist() if "Is ETF" in sector_df.columns else [False]*len(ticks)
        _sec_h = max(4.0, len(ticks) * 0.5)
        fig,ax=plt.subplots(figsize=(10,_sec_h)); fig.patch.set_facecolor(C_BG)
        x=np.arange(len(ticks)); w=0.5
        c_s=[C_GREEN if r>=0 else C_RED for r in stock_r]
        ax.bar(x,[r*100 for r in stock_r],width=w*1.1,color=c_s,edgecolor="white",alpha=0.9,label="Stock Return",zorder=3)
        for i,(sr,etf) in enumerate(zip(sector_r,is_etf_)):
            if not etf and not (isinstance(sr,float) and np.isnan(sr)):
                ax.scatter(x[i],sr*100,color=C_STEEL,s=70,zorder=5,marker="D")
                ## Smart label positioning — alternate above/below to avoid bar overlap
                _sr_pct = sr*100
                _bar_h  = stock_r[i]*100
                ## Put label above if diamond is above bar top, else below
                if _sr_pct > _bar_h:
                    _va, _yo = "bottom", 1.5
                else:
                    _va, _yo = "top", -1.5
                ax.text(x[i]+0.15, _sr_pct+_yo, f"{_sr_pct:.1f}%",
                        ha="left", va=_va, fontsize=7, color=C_STEEL, fontweight="bold")
            elif etf:
                ax.text(x[i],1,"ETF\nN/A",ha="center",va="bottom",fontsize=7,color="#999999")
        if benchmark_data is not None and len(benchmark_data)>2:
            bm_tot=float(benchmark_data.iloc[-1]/benchmark_data.iloc[0]-1)*100
            ax.axhline(bm_tot,color=C_GOLD,lw=1.8,linestyle="--",alpha=0.9,label=f"{benchmark_name}: {bm_tot:+.1f}%")
        ax.axhline(0,color="#AAAAAA",lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(ticks,fontsize=9)
        ax.set_ylabel("Return (%)",fontsize=9)
        ax.set_title("Stock Return vs Sector Index Return — Alpha vs Beta",fontsize=10,fontweight="bold",color=C_NAVY,pad=8)
        ax.legend(handles=[_mpatches.Patch(color=C_GREEN,label="Stock Return (+)"),
                            _mpatches.Patch(color=C_RED,label="Stock Return (-)"),
                            plt.scatter([],[],color=C_STEEL,s=60,marker="D",label="Sector Return")],
                  fontsize=8,frameon=False)
        _styled(fig,ax); plt.tight_layout(pad=1.2); return fig

    def _chart_var():
        ## Build ONLY the panels the user selected — reads from risk_df
        if port_returns is None or len(port_returns) < 20:
            return None
        if risk_df is None or risk_df.empty:
            return None

        data    = port_returns.dropna().values
        alpha_v = portfolio_stats.get("alpha", 0.95)
        mu_p    = data.mean()
        s_p     = data.std()

        methods_in_report = risk_df["Method"].tolist()
        has_hist  = any("Historical"  in m for m in methods_in_report)
        has_para  = any("Parametric"  in m for m in methods_in_report)
        has_mc    = any("Monte Carlo" in m for m in methods_in_report)

        ## Build panels
        panels = []
        if has_hist:
            var_threshold_h = np.percentile(data, (1-alpha_v)*100)
            tail_losses_h   = data[data <= var_threshold_h]
            es_threshold_h  = float(tail_losses_h.mean()) if len(tail_losses_h) > 0 else var_threshold_h
            panels.append(("Historical", "hist", var_threshold_h, es_threshold_h, data))

        if has_para:
            var_threshold_p = norm.ppf(1-alpha_v, mu_p, s_p)
            es_threshold_p  = mu_p - s_p * (norm.pdf(norm.ppf(1-alpha_v)) / (1-alpha_v))
            panels.append(("Parametric (Normal)", "para", var_threshold_p, es_threshold_p, data))

        sim_r_mc = None
        if has_mc:
            np.random.seed(42)
            T_mc     = portfolio_stats.get("horizon", 1)
            sim_d    = np.random.normal(mu_p, s_p, (5000, T_mc))
            sim_r_mc = (1 + sim_d).prod(axis=1) - 1
            var_threshold_mc = float(np.percentile(sim_r_mc, (1-alpha_v)*100))
            tail_mc          = sim_r_mc[sim_r_mc <= var_threshold_mc]
            es_threshold_mc  = float(tail_mc.mean()) if len(tail_mc) > 0 else var_threshold_mc
            panels.append(("Monte Carlo", "mc", var_threshold_mc, es_threshold_mc, sim_r_mc))

        if not panels:
            return None

        n_panels  = len(panels)
        fig_w     = 6.5 * n_panels
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.5), squeeze=False)
        fig.patch.set_facecolor(C_BG)

        ## PDF color palette
        _RED_DARK  = "#E2534A"    ## extreme loss zone
        _RED_LIGHT = "#F09595"    ## VaR tail zone
        _BLUE      = "#378ADD"    ## normal range
        _AMBER_L   = "#BA7517"    ## ES line color

        for col_idx, (panel_name, panel_type, var_thr, es_thr, ret_data) in enumerate(panels):
            ax = axes[0][col_idx]

            if panel_type == "para":
                ## ── Parametric: smooth KDE curve with colored fills ──────────
                x_r = np.linspace(mu_p - 4.5*s_p, mu_p + 4.5*s_p, 600)
                y_r = norm.pdf(x_r, mu_p, s_p)

                ax.fill_between(x_r, y_r, where=(x_r <= es_thr),
                                color=_RED_DARK, alpha=0.85)
                ax.fill_between(x_r, y_r,
                                where=((x_r > es_thr) & (x_r <= var_thr)),
                                color=_RED_LIGHT, alpha=0.75)
                ax.fill_between(x_r, y_r, where=(x_r > var_thr),
                                color=_BLUE, alpha=0.25)
                ax.plot(x_r, y_r, color=_BLUE, lw=2.0)

            else:
                ## ── Historical / MC: color-coded histogram bars ───────────────
                counts, bin_edges = np.histogram(ret_data, bins=38, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_w = (bin_edges[1] - bin_edges[0]) * 0.92

                for cnt, center in zip(counts, bin_centers):
                    if center <= es_thr:
                        col, alp = _RED_DARK, 0.9
                    elif center <= var_thr:
                        col, alp = _RED_LIGHT, 0.8
                    else:
                        col, alp = _BLUE, 0.35
                    ax.bar(center, cnt, width=bin_w,
                           color=col, alpha=alp, edgecolor="none")

            ## ── VaR line ─────────────────────────────────────────────────────
            ax.axvline(var_thr, color=_RED_DARK, lw=1.8, linestyle="--")
            y_top = ax.get_ylim()[1]
            ax.text(var_thr - s_p*0.15, y_top * 0.97,
                    f"VaR {alpha_v:.0%}\n{var_thr:.3f}",
                    color=_RED_DARK, fontsize=7.5, ha="right", va="top",
                    fontweight="bold")

            ## ── ES line ──────────────────────────────────────────────────────
            ax.axvline(es_thr, color=_AMBER_L, lw=1.6, linestyle=":")
            ax.text(es_thr - s_p*0.15, y_top * 0.78,
                    f"ES\n{es_thr:.3f}",
                    color=_AMBER_L, fontsize=7.5, ha="right", va="top",
                    fontweight="bold")

            ## ── Zero line ────────────────────────────────────────────────────
            ax.axvline(0, color="gray", lw=0.8, alpha=0.4)

            ## ── Legend proxy ─────────────────────────────────────────────────
            from matplotlib.patches import Patch as _Patch
            ax.legend(handles=[
                _Patch(color=_RED_DARK,  alpha=0.85, label="Extreme loss (< ES)"),
                _Patch(color=_RED_LIGHT, alpha=0.75, label="VaR tail"),
                _Patch(color=_BLUE,      alpha=0.35, label="Normal range"),
            ], fontsize=7, frameon=True, loc="upper left",
               framealpha=0.7, edgecolor="#D0D8E4")
            for txt in ax.get_legend().get_texts():
                txt.set_color("white")

            ## ── Method label ─────────────────────────────────────────────────
            ax.set_xlabel("Daily Return", fontsize=9, color="#0A1628")
            ax.set_ylabel("Density",      fontsize=9, color="#0A1628")
            ax.set_title(f"Method {col_idx+1} — {panel_name}",
                         fontsize=10, fontweight="bold",
                         color="#0A1628", pad=8)
            ## Rotate x-axis labels to prevent crushing/overlapping
            ax.tick_params(axis='x', rotation=30, labelsize=7.5)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.2%}")
            )
            _styled(fig, ax)
            ax.set_facecolor(C_BG)

        plt.tight_layout(pad=1.5)
        return fig

    ## ══════════════════════════════════════════════════════════════════════════
    ##  ASSEMBLE PDF
    ## ══════════════════════════════════════════════════════════════════════════
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=A4,
        leftMargin=L_MARGIN,rightMargin=R_MARGIN,topMargin=T_MARGIN,bottomMargin=B_MARGIN,
        title="Portfolio Risk Analytics Report",author="Portfolio Analysis Report")
    elements=[]; total_ret=(total_current-total_invested)/total_invested*100
    xirr_str=f"{portfolio_xirr:.2%}" if (portfolio_xirr is not None and not np.isnan(float(portfolio_xirr))) else "N/A"
    rec_str=recovery_date.strftime("%b %Y") if recovery_date else "Not yet"
    gen_date=pd.Timestamp.now().strftime("%d %B %Y, %H:%M IST")

    ## PAGE 1 — PREMIUM COVER
    ## ── Helper to build a single snapshot card ──────────────────────────────
    _CARD_W = (CW - 3*0.15*cm) / 4   ## exact card width so 4 cards + 3 gaps = CW
    def _card(label, value, val_style=None):
        vs = val_style or KPI_VAL
        t  = Table([[Paragraph(label, KPI_LBL)],
                    [Paragraph(str(value), vs)]],
                   colWidths=[_CARD_W])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,-1), DARK_NAVY),
            ("LINEBELOW",   (0,0), (-1,0),  1.5, ACCENT),
            ("BOX",         (0,0), (-1,-1), 0.5, BORDER),
            ("TOPPADDING",  (0,0), (-1,-1), 10),
            ("BOTTOMPADDING",(0,0),(-1,-1), 10),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("ALIGN",       (0,0), (-1,-1), "LEFT"),
        ]))
        return t

    ## ── Compute all cover values safely ──────────────────────────────────────
    _bm_cover_ret = "N/A"
    _excess_ret   = "N/A"
    _best_stock   = "N/A"
    _worst_stock  = "N/A"
    _var_cover    = "N/A"
    try:
        if benchmark_data is not None:
            _bm_raw = benchmark_data
            ## Ensure 1D Series
            if hasattr(_bm_raw, 'squeeze'):
                _bm_raw = _bm_raw.squeeze()
            if hasattr(_bm_raw, 'iloc') and len(_bm_raw) > 1:
                _bm_clean = _bm_raw.dropna()
                if len(_bm_clean) > 1:
                    _bm_s = float(_bm_clean.iloc[0])
                    _bm_e = float(_bm_clean.iloc[-1])
                    if _bm_s > 0:
                        _bm_pct       = (_bm_e / _bm_s - 1) * 100
                        _bm_cover_ret = f"{_bm_pct:+.2f}%"
                        _excess_ret   = f"{total_ret - _bm_pct:+.2f}%"
    except Exception:
        pass
    try:
        if attr_df is not None and not attr_df.empty and "Return Contribution" in attr_df.columns:
            _rc = attr_df["Return Contribution"].dropna()
            if len(_rc) > 0:
                _best_idx   = _rc.idxmax()
                _worst_idx  = _rc.idxmin()
                _bt = str(attr_df.loc[_best_idx,"Ticker"]).replace(".NS","").replace(".BO","")
                _wt = str(attr_df.loc[_worst_idx,"Ticker"]).replace(".NS","").replace(".BO","")
                _bc = float(attr_df.loc[_best_idx,"Return Contribution"])
                _wc = float(attr_df.loc[_worst_idx,"Return Contribution"])
                _best_stock  = f"{_bt} ({_bc*100:+.2f}%)"
                _worst_stock = f"{_wt} ({_wc*100:+.2f}%)"
    except Exception:
        pass
    try:
        if risk_df is not None and not risk_df.empty:
            _var_val = risk_df.iloc[0]["VaR %"]
            _var_cover = f"{float(_var_val):.2f}%" if _var_val is not None else "N/A"
    except Exception:
        pass
    ## ── Cover header — fills exactly one page ───────────────────────────────
    _QUOTE_S = ParagraphStyle("QS", fontSize=10, fontName="Helvetica-Oblique",
                               textColor=rl_colors.HexColor("#C8A951"),
                               alignment=TA_CENTER, leading=15)
    _QAUTH_S = ParagraphStyle("QA", fontSize=9, fontName="Helvetica",
                               textColor=rl_colors.HexColor("#8B949E"),
                               alignment=TA_CENTER, leading=13)
    _DEDIC_S = ParagraphStyle("DD", fontSize=10, fontName="Helvetica",
                               textColor=rl_colors.HexColor("#8B949E"),
                               alignment=TA_CENTER, leading=14)
    _NAME_S  = ParagraphStyle("NM", fontSize=14, fontName="Helvetica-Bold",
                               textColor=WHITE, alignment=TA_CENTER, leading=18)

    _cover_rows = [
        [Paragraph("PORTFOLIO ANALYSIS REPORT", TITLE_S)],
        [Spacer(1, 0.3*cm)],
        [Paragraph("Performance, Risk &amp; Attribution Report", SUBTITLE_S)],
        [Spacer(1, 0.15*cm)],
        [Paragraph(
            f"Reporting Period: {gen_date.split(',')[0]}  |  "
            f"Exchange: {portfolio_exchange}  |  "
            f"Benchmark: {benchmark_name}",
            SUBTITLE_S)],
        [Spacer(1, 0.3*cm)],
        [HRFlowable(width=CW*0.5, thickness=0.8, color=ACCENT, hAlign="CENTER")],
    ]

    if client_name and client_name.strip():
        _cover_rows += [
            [Spacer(1, 0.3*cm)],
            [Paragraph("Prepared exclusively for", _DEDIC_S)],
            [Spacer(1, 0.1*cm)],
            [Paragraph(client_name.strip(), _NAME_S)],
        ]

    if quote_text and quote_text.strip():
        _safe_q = quote_text.strip().replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        _safe_a = quote_author.strip().replace("&","&amp;") if quote_author else ""
        _cover_rows += [
            [Spacer(1, 0.3*cm)],
            [HRFlowable(width=CW*0.35, thickness=0.5,
                        color=rl_colors.HexColor("#2A3F5F"), hAlign="CENTER")],
            [Spacer(1, 0.2*cm)],
            [Paragraph(f"&#8220;{_safe_q}&#8221;", _QUOTE_S)],
        ]
        if _safe_a:
            _cover_rows += [
                [Spacer(1, 0.1*cm)],
                [Paragraph(f"&#8212; {_safe_a}", _QAUTH_S)],
            ]

    _cover_rows.append([Spacer(1, 0.3*cm)])

    ## Assign fixed heights: content rows get more, spacer rows get less
    ## This prevents uniform distribution making gaps look enormous
    _USABLE_H = PAGE_H - T_MARGIN - B_MARGIN - 1.2*cm - 0.5*cm

    ## Calculate per-row heights based on content type
    _row_heights = []
    _content_h   = 0
    _spacer_h    = 0.4*cm   ## fixed height for Spacer rows
    _hr_h        = 0.3*cm   ## fixed height for HRFlowable rows
    _content_rows_count = 0

    for _rv in _cover_rows:
        import reportlab.platypus as _plat
        _cell = _rv[0]
        if isinstance(_cell, _plat.Spacer):
            _row_heights.append(_spacer_h)
            _spacer_h_total = _spacer_h
        elif isinstance(_cell, _plat.HRFlowable):
            _row_heights.append(_hr_h)
        else:
            _row_heights.append(None)   ## will be set to content_h
            _content_rows_count += 1

    ## Distribute remaining height equally among content rows
    _fixed_total = sum(h for h in _row_heights if h is not None)
    _remaining   = max(_USABLE_H - _fixed_total, _content_rows_count * 1.2*cm)
    _content_h   = _remaining / max(_content_rows_count, 1)
    _row_heights = [(_content_h if h is None else h) for h in _row_heights]

    cover = Table(_cover_rows, colWidths=[CW], rowHeights=_row_heights)
    cover.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), DARK_NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING",   (0,0), (-1,-1), 20),
        ("RIGHTPADDING",  (0,0), (-1,-1), 20),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    elements.append(cover)
    elements.append(HRFlowable(width=CW, thickness=2, color=ACCENT, spaceAfter=0.25*cm))
    elements.append(PageBreak())

    ## ── Meta row ──────────────────────────────────────────────────────────────
    meta = Table([[
        Paragraph("Generated",      META_S),
        Paragraph("Risk-Free Rate", META_S),
        Paragraph("Benchmark",      META_S),
        Paragraph("Confidence",     META_S),
        Paragraph("Exchange",       META_S),
    ],[
        Paragraph(gen_date,                               META_BOLD),
        Paragraph(f"{RISK_FREE_ANNUAL:.2%} p.a. (RBI Repo)", META_BOLD),
        Paragraph(f"{benchmark_name} ({benchmark_ticker})",  META_BOLD),
        Paragraph(f"{portfolio_stats.get('alpha',0.95):.0%}", META_BOLD),
        Paragraph(portfolio_exchange,                     META_BOLD),
    ]], colWidths=[CW/5]*5)
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  GREY_100),
        ("BACKGROUND",    (0,1), (-1,1),  WHITE),
        ("BOX",           (0,0), (-1,-1), 0.5, BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.3, BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
    ]))
    elements.append(meta)
    elements.append(Spacer(1, 0.25*cm))

    ## ── Snapshot cards — Row 1: 4 financial metrics ───────────────────────────
    _r1c1 = _card("PORTFOLIO VALUE",   f"Rs.{total_current:,.0f}",
                  KPI_VAL_G if total_current >= total_invested else KPI_VAL_R)
    _r1c2 = _card("TOTAL P&L",         f"Rs.{total_pnl:+,.0f}",
                  KPI_VAL_G if total_pnl >= 0 else KPI_VAL_R)
    _r1c3 = _card("PORTFOLIO RETURN",  f"{total_ret:+.2f}%",
                  KPI_VAL_G if total_ret >= 0 else KPI_VAL_R)
    _r1c4 = _card("BENCHMARK RETURN",  _bm_cover_ret, KPI_VAL)

    _row1 = Table([[_r1c1, Spacer(0.15*cm,1), _r1c2,
                    Spacer(0.15*cm,1), _r1c3, Spacer(0.15*cm,1), _r1c4]],
                  colWidths=[_CARD_W, 0.15*cm]*3 + [_CARD_W])
    _row1.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
    elements.append(_row1)
    elements.append(Spacer(1, 0.15*cm))

    ## ── Snapshot cards — Row 2: 4 more metrics ────────────────────────────────
    try:
        _exc_style = (KPI_VAL_G if float(_excess_ret.replace('%','').replace('+','')) >= 0
                      else KPI_VAL_R) if _excess_ret not in ("N/A","") else KPI_VAL
    except Exception:
        _exc_style = KPI_VAL
    _r2c1 = _card("EXCESS RETURN", _excess_ret, _exc_style)
    _r2c2 = _card("XIRR",            xirr_str,
                  KPI_VAL_G if (not np.isnan(portfolio_xirr) and portfolio_xirr>0) else KPI_VAL_R)
    _r2c3 = _card("SHARPE RATIO",    f"{portfolio_stats.get('sharpe',0):.4f}",
                  KPI_VAL_G if portfolio_stats.get('sharpe',0)>0 else KPI_VAL_R)
    _r2c4 = _card("MAX DRAWDOWN",    f"{max_dd:.2%}", KPI_VAL_R)

    _row2 = Table([[_r2c1, Spacer(0.15*cm,1), _r2c2,
                    Spacer(0.15*cm,1), _r2c3, Spacer(0.15*cm,1), _r2c4]],
                  colWidths=[_CARD_W, 0.15*cm]*3 + [_CARD_W])
    _row2.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
    elements.append(_row2)
    elements.append(Spacer(1, 0.15*cm))

    ## ── Snapshot cards — Row 3: Best/Worst + VaR + Beta ─────────────────────
    _r3c1 = _card("BEST PERFORMER",  _best_stock,  KPI_VAL_G)
    _r3c2 = _card("WORST PERFORMER", _worst_stock, KPI_VAL_R)
    _r3c3 = _card("VaR (1-Day)",     _var_cover,   KPI_VAL_R)
    try:
        ## Use portfolio_beta parameter first (passed directly from session_state)
        ## Fall back to portfolio_stats dict (also from session_state)
        _beta_raw = portfolio_beta if (portfolio_beta is not None and portfolio_beta == portfolio_beta) else portfolio_stats.get('beta', None)
        _beta_val = float(_beta_raw) if _beta_raw is not None else float('nan')
        _beta_str = f"{_beta_val:.4f}" if _beta_val == _beta_val else "N/A"
    except Exception:
        _beta_str = "N/A"


    _r3c4 = _card("BETA", _beta_str, KPI_VAL)
    _row3 = Table([[_r3c1, Spacer(0.15*cm,1), _r3c2,
                    Spacer(0.15*cm,1), _r3c3, Spacer(0.15*cm,1), _r3c4]],
                  colWidths=[_CARD_W, 0.15*cm]*3 + [_CARD_W])
    _row3.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
    elements.append(_row3)
    elements.append(Spacer(1, 0.2*cm))
    ## ── Executive summary bullets ─────────────────────────────────────────────
    elements.append(HRFlowable(width=CW, thickness=0.5, color=BORDER, spaceAfter=0.1*cm))
    elements.append(Paragraph("EXECUTIVE SUMMARY", H_SUB))
    elements.append(Spacer(1, 0.1*cm))

    sharpe_v = float(portfolio_stats.get("sharpe", float("nan")) or float("nan"))
    beta_v   = float(portfolio_stats.get("beta",   float("nan")) or float("nan"))
    best_r = worst_r = max_wt = None
    try:
        if attr_df is not None and not attr_df.empty:
            if "Return Contribution" in attr_df.columns:
                _rc2 = attr_df["Return Contribution"].dropna()
                if len(_rc2) > 0:
                    best_r  = attr_df.loc[_rc2.idxmax()]
                    worst_r = attr_df.loc[_rc2.idxmin()]
            if "Weight" in attr_df.columns:
                _wt2 = attr_df["Weight"].dropna()
                if len(_wt2) > 0:
                    max_wt = attr_df.loc[_wt2.idxmax()]
    except Exception:
        best_r = worst_r = max_wt = None
    exec_b = []

    ## ── Bullet 1: P&L vs invested ────────────────────────────────────────────
    if total_pnl >= 0:
        exec_b.append(
            f"Portfolio is <b>up Rs.{total_pnl:,.0f} ({total_ret:+.2f}%)</b> "
            f"on invested capital of Rs.{total_invested:,.0f}."
        )
    else:
        exec_b.append(
            f"Portfolio is <b>down Rs.{abs(total_pnl):,.0f} ({total_ret:.2f}%)</b> "
            f"on invested capital of Rs.{total_invested:,.0f}."
        )

    ## ── Bullet 2: Benchmark comparison ───────────────────────────────────────
    try:
        if benchmark_data is not None and len(benchmark_data) > 1:
            _bm_clean = benchmark_data.dropna()
            _bm_ret_exec = (_bm_clean.iloc[-1] / _bm_clean.iloc[0] - 1) * 100
            _exc_exec    = total_ret - _bm_ret_exec
            _vs_bm = "outperformed" if _exc_exec >= 0 else "underperformed"
            exec_b.append(
                f"Portfolio <b>{_vs_bm} {benchmark_name}</b> by "
                f"<b>{abs(_exc_exec):.2f}%</b> "
                f"(portfolio {total_ret:+.2f}% vs benchmark {_bm_ret_exec:+.2f}%)."
            )
    except Exception:
        pass

    ## ── Bullet 3: Sharpe ratio with meaningful interpretation ────────────────
    if not np.isnan(sharpe_v):
        if sharpe_v < 0:
            _sharpe_interp = (
                f"Negative — portfolio returned less than the {RISK_FREE_ANNUAL:.2%} "
                f"risk-free rate on a risk-adjusted basis. Cash/FD would have been more efficient."
            )
        elif sharpe_v < 0.5:
            _sharpe_interp = (
                f"Below institutional threshold of 0.5. Returns are positive but "
                f"insufficient relative to the risk taken."
            )
        elif sharpe_v < 1.0:
            _sharpe_interp = "Acceptable risk-adjusted return. Room for improvement."
        else:
            _sharpe_interp = "Strong — generating good returns per unit of risk taken."
        exec_b.append(
            f"Sharpe Ratio <b>{sharpe_v:.4f}</b> — {_sharpe_interp}"
        )

    ## ── Bullet 4: Beta and drawdown ──────────────────────────────────────────
    if not np.isnan(beta_v):
        _beta_desc = "more" if beta_v > 1 else "less"
        _dd_status = (
            f"Recovered {rec_str}." if recovery_date
            else f"Still in drawdown — not yet recovered."
        )
        exec_b.append(
            f"Beta <b>{beta_v:.4f}</b> — portfolio is {_beta_desc} volatile than "
            f"{benchmark_name}. Max drawdown: <b>{max_dd:.2%}</b>. {_dd_status}"
        )

    ## ── Bullet 5: Best and worst contributors with stock return ──────────────
    if best_r is not None and worst_r is not None:
        try:
            _best_ticker  = str(best_r["Ticker"]).replace(".NS","").replace(".BO","")
            _worst_ticker = str(worst_r["Ticker"]).replace(".NS","").replace(".BO","")
            _best_rc      = float(best_r["Return Contribution"]) * 100
            _worst_rc     = float(worst_r["Return Contribution"]) * 100
            _best_sr      = float(best_r.get("Stock Return", 0)) * 100
            _worst_sr     = float(worst_r.get("Stock Return", 0)) * 100
            exec_b.append(
                f"<b>{_best_ticker}</b> best contributor "
                f"(+{_best_rc:.2f}% to portfolio, stock returned {_best_sr:+.2f}%). "
                f"<b>{_worst_ticker}</b> largest drag "
                f"({_worst_rc:.2f}% to portfolio, stock returned {_worst_sr:+.2f}%)."
            )
        except Exception:
            exec_b.append(
                f"<b>{best_r['Ticker']}</b> best contributor. "
                f"<b>{worst_r['Ticker']}</b> largest drag."
            )

    ## ── Bullet 6: Concentration alert (only if concentrated) ─────────────────
    if max_wt is not None:
        _max_w = float(max_wt["Weight"])
        _max_t = str(max_wt["Ticker"]).replace(".NS","").replace(".BO","")
        if _max_w > 0.50:
            exec_b.append(
                f"<b>High Concentration Alert:</b> {_max_t} = {_max_w:.1%} of portfolio. "
                f"A single stock dominates — amplifies both gains and losses significantly. "
                f"Institutional guidelines typically cap single-stock exposure at 10-20%."
            )
        elif _max_w > 0.30:
            exec_b.append(
                f"<b>Moderate Concentration:</b> {_max_t} = {_max_w:.1%} of portfolio. "
                f"Above typical institutional thresholds — monitor closely."
            )

    ## ── Bullet 7: XIRR vs risk-free rate ─────────────────────────────────────
    try:
        if portfolio_xirr is not None and not np.isnan(float(portfolio_xirr)):
            _xirr_pct = float(portfolio_xirr) * 100
            _vs_rf    = "above" if _xirr_pct > RISK_FREE_ANNUAL * 100 else "below"
            exec_b.append(
                f"XIRR <b>{_xirr_pct:.2f}%</b> p.a. — annualised return accounting for "
                f"actual investment timing. This is <b>{_vs_rf} the risk-free rate</b> "
                f"of {RISK_FREE_ANNUAL:.2%} (RBI Repo)."
            )
    except Exception:
        pass
    exec_t=Table([[Paragraph("<b>EXECUTIVE SUMMARY</b>",H_SUB)],
                  [Paragraph("<br/><br/>".join(f"&#8226; &nbsp;{b}" for b in exec_b),BODY_S)]],colWidths=[CW])
    exec_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),ACCENT_LIGHT),("BACKGROUND",(0,1),(-1,-1),WHITE),
        ("BOX",(0,0),(-1,-1),0.8,ACCENT),("LINEBEFORE",(0,0),(0,-1),3,ACCENT),
        ("LEFTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    elements.append(exec_t); elements.append(Spacer(1,0.2*cm))
    conf_t=Table([[Paragraph("STRICTLY CONFIDENTIAL — Informational purposes only. Not investment advice. Past performance does not guarantee future results. Data: Yahoo Finance.",DISC_LIGHT)]],colWidths=[CW])
    conf_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),DARK_NAVY),("LEFTPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    elements.append(conf_t)

    ## PAGE 2 — HOLDINGS + COMPOSITION
    elements.append(PageBreak())
    elements+=_section_block("SECTION 1 — HOLDINGS SUMMARY & PORTFOLIO COMPOSITION",f"Positions as of {gen_date} | Data: Yahoo Finance")
    h_hdr=["Ticker","Buy Date","Buy Px\n(Rs.)","Qty","Curr Px\n(Rs.)","Invested\n(Rs.)","Curr Val\n(Rs.)","P&L\n(Rs.)","Return %"]
    h_rows=[h_hdr]
    for _,row in holdings_df.iterrows():
        h_rows.append([str(row["Ticker"]),str(row["Buy Date"].date()) if pd.notna(row["Buy Date"]) else "N/A",
            f"{row['Buy Price']:,.2f}",f"{row['Quantity']:.0f}",f"{row['Current Price']:,.2f}",
            f"{row['Invested Value']:,.2f}",f"{row['Current Value']:,.2f}",f"{row['P&L']:+,.2f}",f"{row['Return %']:+.2f}%"])
    h_rows.append(["TOTAL","","","","",f"{total_invested:,.2f}",f"{total_current:,.2f}",f"{total_pnl:+,.2f}",f"{total_ret:+.2f}%"])
    ht=Table(h_rows,colWidths=[2.6*cm,2.1*cm,1.9*cm,0.8*cm,1.8*cm,2.0*cm,2.0*cm,2.0*cm,2.2*cm],repeatRows=1)
    ts=_base_ts()
    for col in range(2,9): ts.add("ALIGN",(col,0),(col,-1),"RIGHT")
    for i,row in enumerate(holdings_df.itertuples(),start=1):
        pnl=row._asdict().get("P&L",0); c=PDF_GREEN if pnl>=0 else PDF_RED; bg=GREEN_LIGHT if pnl>=0 else RED_LIGHT
        ts.add("TEXTCOLOR",(7,i),(7,i),c); ts.add("TEXTCOLOR",(8,i),(8,i),c); ts.add("BACKGROUND",(7,i),(8,i),bg)
    tr=len(h_rows)-1
    ts.add("BACKGROUND",(0,tr),(-1,tr),DARK_NAVY); ts.add("TEXTCOLOR",(0,tr),(-1,tr),WHITE); ts.add("FONTNAME",(0,tr),(-1,tr),"Helvetica-Bold")
    ht.setStyle(ts); elements.append(ht); elements.append(Spacer(1,0.15*cm))
    elements.append(_fig2img(_chart_composition(),17.4,6.5))
    elements.append(_cap_box(f"Left: Portfolio weight distribution. Right: Portfolio weights vs {benchmark_name} approximate benchchmark sector weights. Large gaps between bars indicate significant active bets vs the index."))
    elements.append(Spacer(1,0.25*cm))
    elements.append(_fig2img(_chart_pnl(),17.4,5.5))
    elements.append(_cap_box("Left: Absolute P&L per holding (Rs.). Right: Return % per holding. Wide dispersion between best and worst performers is characteristic of a concentrated portfolio."))

    ## PAGE 3 — PERFORMANCE + BENCHMARK
    elements.append(PageBreak())
    elements+=_section_block("SECTION 2 — PERFORMANCE METRICS",f"Rf = {RISK_FREE_ANNUAL:.2%} p.a. (RBI Repo) | Beta lookback: {portfolio_stats.get('beta_horizon','1Y')} | Benchmark: Nifty 50")
    def fv(k,f=".4f"):
        v=portfolio_stats.get(k,float("nan")); return f"{v:{f}}" if not np.isnan(v) else "N/A"
    pm_hdr=["Metric","Value","Signal","What this means for your portfolio"]
    def _pm_p(txt):
        return Paragraph(txt, ParagraphStyle("PA",fontSize=7.5,fontName="Helvetica",
                          textColor=GREY_700,leading=11,wordWrap="LTR"))
    _bv=portfolio_stats.get("beta",float("nan"))
    _sv=portfolio_stats.get("sharpe",float("nan"))
    _tv=portfolio_stats.get("sortino",float("nan"))
    _trv=portfolio_stats.get("treynor",float("nan"))
    _cv=portfolio_stats.get("cagr",float("nan"))
    _vv=portfolio_stats.get("ann_vol",float("nan"))
    _skv=portfolio_stats.get("skewness",float("nan"))
    _kv=portfolio_stats.get("kurtosis",float("nan"))
    pm_rows=[pm_hdr,
        ["Beta",fv("beta"),
         "DEFENSIVE" if (not np.isnan(_bv) and _bv<1) else "AGGRESSIVE",
         _pm_p(f"Portfolio moves {abs(1-_bv)*100:.1f}% {'less' if not np.isnan(_bv) and _bv<1 else 'more'} than {benchmark_name}. When Nifty falls 1%, this portfolio typically moves {_bv:.2f}. Defensive positioning provides partial protection on down days." if not np.isnan(_bv) else "N/A")],
        ["Sharpe",fv("sharpe"),
         "GOOD" if (not np.isnan(_sv) and _sv>1) else ("BELOW HURDLE" if (not np.isnan(_sv) and _sv<0) else "MODERATE"),
         _pm_p(("Portfolio is generating returns below the 6.00% RBI Repo risk-free hurdle on a risk-adjusted basis. Holding a risk-free instrument would have been more efficient." if not np.isnan(_sv) and _sv<0 else "Portfolio is generating positive risk-adjusted returns above the RBI Repo rate hurdle.") if not np.isnan(_sv) else "N/A")],
        ["Sortino",fv("sortino"),
         "GOOD" if (not np.isnan(_tv) and _tv>1) else ("BELOW HURDLE" if (not np.isnan(_tv) and _tv<0) else "MODERATE"),
         _pm_p(("Downside risk is disproportionately large compared to upside. The portfolio suffers larger-than-expected losses on bad days, worse than the Sharpe ratio alone suggests." if not np.isnan(_tv) and not np.isnan(_sv) and _tv<_sv else "Downside volatility is being managed well relative to overall risk.") if not np.isnan(_tv) else "N/A")],
        ["Treynor",fv("treynor"),
         "NEGATIVE" if (not np.isnan(_trv) and _trv<0) else "POSITIVE",
         _pm_p(("The market-linked portion of portfolio risk is not being rewarded. Market exposure is generating negative excess return above the risk-free rate." if not np.isnan(_trv) and _trv<0 else "Market risk is generating positive excess return above the risk-free rate.") if not np.isnan(_trv) else "N/A")],
        ["CAGR",fv("cagr",".2%"),
         "NEGATIVE" if (not np.isnan(_cv) and _cv<0) else "POSITIVE",
         _pm_p((f"Annualised return is negative over the measurement period. The total return of +2.23% is more meaningful given the short holding period — annualisation amplifies short-term figures significantly." if not np.isnan(_cv) and _cv<0 else f"Portfolio is growing at {fv('cagr','.2%')} per annum.") if not np.isnan(_cv) else "N/A")],
        ["Ann. Volatility",fv("ann_vol",".2%"),"MODERATE",
         _pm_p(f"Portfolio value swings by approximately {fv('ann_vol','.2%')} per year. High volatility is driven by concentration in the largest holding. Compare to Nifty 50 typical range of 13-15%." if not np.isnan(_vv) else "N/A")],
        ["Skewness",fv("skewness"),
         "LEFT TAIL" if (not np.isnan(_skv) and _skv<0) else "RIGHT TAIL",
         _pm_p(("Return distribution leans slightly left — large negative days occur marginally more often than large positive days. A mild signal that losses can occasionally be outsized." if not np.isnan(_skv) and _skv<0 else "Return distribution leans slightly right — large positive days are marginally more frequent. Favourable characteristic.") if not np.isnan(_skv) else "N/A")],
        ["Kurtosis",fv("kurtosis"),
         "FAT TAILS" if (not np.isnan(_kv) and _kv>3) else "NORMAL",
         _pm_p(f"Extreme daily moves occur {(_kv/3):.1f}x more often than a normal bell curve predicts. Surprise events — both sharp drops and rallies — hit harder than standard models assume. This is why Historical VaR is more reliable than Parametric VaR for this portfolio." if not np.isnan(_kv) else "N/A")],
    ]
    pm_t=Table(pm_rows,colWidths=[2.3*cm,1.9*cm,2.6*cm,10.4*cm],repeatRows=1)
    pm_ts=_base_ts()
    pm_ts.add("ALIGN",(1,0),(1,-1),"RIGHT")
    pm_ts.add("ALIGN",(2,0),(2,-1),"CENTER")
    good_s={"DEFENSIVE","GOOD","POSITIVE","RIGHT TAIL","NORMAL"}
    bad_s={"AGGRESSIVE","BELOW HURDLE","NEGATIVE","LEFT TAIL","FAT TAILS"}
    for i in range(1,len(pm_rows)):
        sig=pm_rows[i][2]
        c=PDF_GREEN if sig in good_s else (PDF_RED if sig in bad_s else AMBER)
        pm_ts.add("TEXTCOLOR",(2,i),(2,i),c)
        pm_ts.add("FONTNAME",(2,i),(2,i),"Helvetica-Bold")
    pm_t.setStyle(pm_ts); elements.append(pm_t); elements.append(Spacer(1,0.2*cm))
    elements.append(_fig2img(_chart_cumulative(),17.4,5.5))
    elements.append(_cap_box(f"Cumulative indexed performance (Base 100). Green = portfolio outperforming {benchmark_name}. Red = underperforming."))
    elements.append(Spacer(1,0.1*cm))

    ## ── Period Returns, Capture Ratios, Correlation — fresh page after chart ──
    elements.append(PageBreak())
    elements.append(Paragraph("Period Returns vs Benchmark", H_SUB))
    elements.append(Spacer(1,0.15*cm))
    try:
        if port_returns is not None and benchmark_data is not None and len(port_returns) > 20:
            _pr   = (1 + port_returns).cumprod()
            _br   = (benchmark_data / benchmark_data.iloc[0] * 100)
            _pr   = (_pr / _pr.iloc[0]) * 100

            def _period_ret(series, days):
                if len(series) >= days:
                    return (series.iloc[-1] / series.iloc[-days] - 1) * 100
                return float('nan')

            _periods = [
                ("1 Month",      21),
                ("3 Months",     63),
                ("6 Months",    126),
                ("Since Inception", len(_pr)),
            ]
            _pr_hdr  = ["Period", "Portfolio", benchmark_name, "Excess Return"]
            _pr_rows = [_pr_hdr]
            for _pname, _pdays in _periods:
                _port_r = _period_ret(_pr, _pdays)
                _bm_r   = _period_ret(_br, _pdays)
                if not (pd.isna(_port_r) or pd.isna(_bm_r)):
                    _exc = _port_r - _bm_r
                    _pr_rows.append([
                        _pname,
                        f"{_port_r:+.2f}%",
                        f"{_bm_r:+.2f}%",
                        f"{_exc:+.2f}%",
                    ])

            if len(_pr_rows) > 1:
                _prt = Table(_pr_rows, colWidths=[CW*0.28, CW*0.24, CW*0.24, CW*0.24])
                _pts = _base_ts()
                for _ci in [1,2,3]: _pts.add("ALIGN",(_ci,0),(_ci,-1),"CENTER")
                ## Color excess return cells
                for _ri in range(1, len(_pr_rows)):
                    try:
                        _exc_v = float(_pr_rows[_ri][3].replace('%','').replace('+',''))
                        _ec = PDF_GREEN if _exc_v >= 0 else PDF_RED
                        _pts.add("TEXTCOLOR",(3,_ri),(3,_ri), _ec)
                        _pts.add("FONTNAME",(3,_ri),(3,_ri),"Helvetica-Bold")
                        ## Color portfolio return
                        _pv = float(_pr_rows[_ri][1].replace('%','').replace('+',''))
                        _pts.add("TEXTCOLOR",(1,_ri),(1,_ri), PDF_GREEN if _pv>=0 else PDF_RED)
                    except Exception:
                        pass
                _prt.setStyle(_pts)
                elements.append(_prt)
                elements.append(Spacer(1,0.15*cm))
    except Exception as _pe:
        elements.append(_cap_box(f"Period returns unavailable: {_pe}"))

    ## ── Capture Ratios ───────────────────────────────────────────────────────
    try:
        if port_returns is not None and benchmark_data is not None:
            _port_r  = port_returns.dropna()
            _bm_pct  = benchmark_data.pct_change().dropna()
            _common  = _port_r.index.intersection(_bm_pct.index)
            _p2      = _port_r.loc[_common]
            _b2      = _bm_pct.loc[_common]
            _up_m    = _b2 > 0
            _dn_m    = _b2 < 0
            def _cap(p, b, m):
                if m.sum() < 2: return float('nan')
                return ((1+p[m]).prod()**(252/m.sum())-1) / ((1+b[m]).prod()**(252/m.sum())-1) * 100
            _uc = _cap(_p2, _b2, _up_m)
            _dc = _cap(_p2, _b2, _dn_m)

            elements.append(Paragraph("Capture Ratios", H_SUB))
            elements.append(Spacer(1,0.1*cm))
            _cap_hdr = ["Metric","Value","What it means"]
            _cap_rows = [_cap_hdr]
            if not pd.isna(_uc):
                _uc_sig = "Good" if _uc >= 100 else "Partial upside capture"
                _cap_rows.append(["Up Capture Ratio", f"{_uc:.1f}%",
                    f"Portfolio captures {_uc:.1f}% of {benchmark_name} gains. "
                    f"{'Above 100% = outperforms in rallies.' if _uc>=100 else 'Below 100% = lags in rallies.'}"])
            if not pd.isna(_dc):
                _cap_rows.append(["Down Capture Ratio", f"{_dc:.1f}%",
                    f"Portfolio falls {_dc:.1f}% as much as {benchmark_name} in downturns. "
                    f"{'Below 100% = defensive.' if _dc<=100 else 'Above 100% = amplifies losses.'}"])
            if len(_cap_rows) > 1:
                _crt = Table(_cap_rows, colWidths=[CW*0.25, CW*0.15, CW*0.60])
                _crs = _base_ts()
                _crs.add("ALIGN",(1,0),(1,-1),"CENTER")
                for _ri in range(1, len(_cap_rows)):
                    try:
                        _v = float(_cap_rows[_ri][1].replace('%',''))
                        _metric = _cap_rows[_ri][0]
                        _good = (_v >= 100) if "Up" in _metric else (_v <= 100)
                        _crs.add("TEXTCOLOR",(1,_ri),(1,_ri), PDF_GREEN if _good else PDF_RED)
                        _crs.add("FONTNAME",(1,_ri),(1,_ri),"Helvetica-Bold")
                    except Exception:
                        pass
                _crt.setStyle(_crs)
                elements.append(_crt)
                elements.append(Spacer(1,0.25*cm))
    except Exception as _ce:
        pass

    ## ── Correlation Table ────────────────────────────────────────────────────
    try:
        if port_returns is not None and benchmark_data is not None:
            _port_r2 = port_returns.dropna()
            _bm_r2   = benchmark_data.pct_change().dropna()
            _common2 = _port_r2.index.intersection(_bm_r2.index)
            if len(_common2) > 20:
                _corr_val = _port_r2.loc[_common2].corr(_bm_r2.loc[_common2])
                _roll_w2  = min(60, max(10, len(_common2) // 3))
                _roll_c   = _port_r2.loc[_common2].rolling(_roll_w2).corr(_bm_r2.loc[_common2]).dropna()
                _avg_corr = _roll_c.mean() if len(_roll_c) > 0 else _corr_val
                _corr_sig = ("High" if abs(_corr_val) > 0.7
                             else "Moderate" if abs(_corr_val) > 0.4
                             else "Low")
                elements.append(Paragraph("Portfolio Correlation with Benchmark", H_SUB))
                elements.append(Spacer(1,0.1*cm))
                _corr_hdr = ["Metric","Value","Interpretation"]
                _corr_rows = [
                    _corr_hdr,
                    ["Full Period Correlation", f"{_corr_val:.4f}",
                     f"{_corr_sig} correlation with {benchmark_name}. "
                     f"{'Portfolio moves closely with index.' if abs(_corr_val)>0.7 else 'Portfolio has meaningful independent movement from index.'}"],
                    ["60-Day Rolling Avg Corr", f"{_avg_corr:.4f}",
                     "Average rolling 60-day correlation — shows stability of relationship over time."],
                ]
                _cort = Table(_corr_rows, colWidths=[CW*0.28, CW*0.12, CW*0.60])
                _cors = _base_ts()
                _cors.add("ALIGN",(1,0),(1,-1),"CENTER")
                _cort.setStyle(_cors)
                elements.append(_cort)
    except Exception:
        pass

    ## PAGE 4 — RISK ANALYTICS
    elements.append(PageBreak())
    elements+=_section_block("SECTION 3 — RISK ANALYTICS","Drawdown | Return distribution | Fat tail analysis")
    elements.append(_fig2img(_chart_drawdown(),17.4,6.2))
    elements.append(_cap_box(f"Portfolio drawdown: % decline from running peak at each point in time. Max drawdown {max_dd:.2%}. Recovery: {rec_str}. Sustained periods below the zero line indicate unrecovered losses."))
    elements.append(Spacer(1,0.15*cm))
    elements.append(_info_box(f"<b>Drawdown:</b> A max drawdown of <b>{max_dd:.2%}</b> means the portfolio fell {abs(max_dd):.2%} from its peak. {'Recovery was achieved by ' + rec_str if recovery_date else 'Portfolio remains below its previous peak — still in drawdown.'}","warning" if abs(max_dd)>0.10 else "info"))
    elements.append(Spacer(1,0.2*cm))
    elements.append(_fig2img(_chart_kde(),17.4,5.5))
    elements.append(_cap_box(f"Daily return distribution (blue) vs normal distribution (gold dashed). Red = VaR tail. Kurtosis {portfolio_stats.get('kurtosis',3):.2f} confirms fat tails — extreme moves more frequent than normal assumption."))
    elements.append(Spacer(1,0.15*cm))
    elements.append(_info_box(f"<b>Fat Tail Risk:</b> Kurtosis {portfolio_stats.get('kurtosis',3):.2f} vs 3.00 for normal distribution. Extreme daily moves occur {(portfolio_stats.get('kurtosis',3)/3):.1f}x more frequently than a normal model predicts. Use Historical VaR (not Parametric) as the conservative risk measure for this portfolio.","info"))

    ## PAGE 5 — STRESS TEST ANALYSIS
    elements.append(PageBreak())
    elements.extend(_section_block("SECTION 3 — STRESS TEST ANALYSIS",
                                   "Estimated portfolio impact under market and historical shock scenarios"))

    if stress_scenarios and portfolio_beta and total_current_val:
        ## Use existing PDF palette variables (defined at top of function)

        ## ── Context info box ──────────────────────────────────────────────────
        elements.append(_info_box(
            f"Portfolio Beta vs {benchmark_name}: {portfolio_beta:.4f}  |  "
            f"Current Portfolio Value: Rs.{total_current_val:,.0f}  |  "
            f"Beta < 1.0 = defensive (moves less than index)",
            style="info"
        ))
        elements.append(Spacer(1, 2*mm))

        ## ── Market shock table ────────────────────────────────────────────────
        elements.append(Paragraph(f"Market Shock Scenarios ({benchmark_name})", H_SUB))
        elements.append(Spacer(1, 1*mm))
        mkt_shocks = [-0.05, -0.10, -0.15, -0.20, -0.30, -0.40, -0.50]
        mkt_labels = ["-5%", "-10%", "-15%", "-20% (Severe)",
                      "-30% Nifty Crash", "-40% Nifty Crisis", "-50% Nifty Black Swan"]
        mkt_data = [["Nifty Shock", "Portfolio Impact", "Est. Loss (Rs.)", "New Value (Rs.)"]]
        for shock, label in zip(mkt_shocks, mkt_labels):
            port_impact = shock * portfolio_beta
            loss_rs     = port_impact * total_current_val
            new_val     = total_current_val + loss_rs
            mkt_data.append([
                Paragraph(label, BODY_S),
                Paragraph(f"{port_impact:.2%}", BODY_S),
                Paragraph(f"Rs.{loss_rs:,.0f}", _S("RL",
                    fontSize=8.5, fontName="Helvetica",
                    textColor=rl_colors.HexColor("#8B1A1A"))),
                Paragraph(f"Rs.{new_val:,.0f}", BODY_S),
            ])
        mkt_ts = _base_ts()
        mkt_tbl = Table(mkt_data, colWidths=[CW*0.26, CW*0.22, CW*0.26, CW*0.26])
        mkt_tbl.setStyle(mkt_ts)
        elements.append(mkt_tbl)
        elements.append(Spacer(1, 2*mm))

        ## ── Historical scenarios table ────────────────────────────────────────
        elements.append(Paragraph("Historical Market Crash Scenarios", H_SUB))
        elements.append(Spacer(1, 1*mm))
        hist_data = [["Event", "Period", "Nifty Drop", "Est. Portfolio Drop", "Est. Loss (Rs.)"]]
        for event, period, nifty_drop in stress_scenarios:
            port_drop = nifty_drop * portfolio_beta
            loss_rs   = port_drop * total_current_val
            hist_data.append([
                Paragraph(event, BODY_S),
                Paragraph(period, CAPTION_S),
                Paragraph(f"{nifty_drop:.1%}", BODY_S),
                Paragraph(f"{port_drop:.1%}", _S("PD",
                    fontSize=8.5, fontName="Helvetica-Bold",
                    textColor=rl_colors.HexColor("#8B1A1A"))),
                Paragraph(f"Rs.{loss_rs:,.0f}", _S("PL",
                    fontSize=8.5, fontName="Helvetica",
                    textColor=rl_colors.HexColor("#8B1A1A"))),
            ])
        hist_tbl = Table(hist_data, colWidths=[CW*0.24, CW*0.20, CW*0.16, CW*0.20, CW*0.20])
        hist_tbl.setStyle(_base_ts())
        elements.append(hist_tbl)
        elements.append(Spacer(1, 2*mm))

        ## ── Interpretation box ────────────────────────────────────────────────
        elements.append(Paragraph("Stress Test Interpretation", H_SUB))
        elements.append(Spacer(1, 1*mm))
        worst_mkt_loss = abs(-0.50 * portfolio_beta * total_current_val)
        worst_hist_pct = min(s[2] for s in stress_scenarios)
        worst_hist_rs  = abs(worst_hist_pct * portfolio_beta * total_current_val)
        interp_text = (
            f"<b>Beta {portfolio_beta:.2f}</b> vs {benchmark_name} — portfolio moves approximately "
            f"{portfolio_beta:.2f}x with the index. "
            f"In a severe 50% crash (Black Swan scenario), the estimated portfolio loss is "
            f"<b>Rs.{worst_mkt_loss:,.0f}</b> ({abs(-0.50*portfolio_beta):.1%} of current value). "
            f"The worst historical scenario (worst historical scenario at {worst_hist_pct:.1%}) "
            f"implies an estimated loss of <b>Rs.{worst_hist_rs:,.0f}</b>. "
            f"A Beta below 1.0 provides a defensive buffer vs the index in broad market downturns. "
            f"However, single-stock concentration risk can cause losses independent of market direction. "
            f"All estimates assume constant Beta and linear market response — actual crisis losses "
            f"may differ due to correlation breakdown, liquidity stress, and non-linear dynamics."
        )
        elements.append(_info_box(interp_text, style="warning"))
        elements.append(Spacer(1, 2*mm))

    ## PAGE 6 — VaR & ES
    elements.append(PageBreak())
    elements+=_section_block("SECTION 4 — VALUE AT RISK & EXPECTED SHORTFALL",f"Confidence: {portfolio_stats.get('alpha',0.95):.0%} | Horizon: {portfolio_stats.get('horizon',10)} day(s) | Portfolio: Rs.{total_current:,.0f}")
    var_hdr=["Method","VaR %","VaR (Rs.)","ES %","ES (Rs.)","Methodology"]
    var_rows_=[var_hdr]
    imap={"Historical":"Empirical dist. — no normality assumption","Parametric Method":"Normal dist. — may understate fat tails","Monte Carlo Simulation":f"{portfolio_stats.get('mc_sims',5000):,} simulated paths"}
    for _,row in risk_df.iterrows():
        var_rows_.append([str(row["Method"]),f"{float(row['VaR %'] or 0):.2%}",f"Rs.{float(row['VaR Amount (Rs.)'] or 0):,.2f}",f"{row['ES %']:.2%}",f"Rs.{row['ES Amount (Rs.)']:,.2f}",imap.get(str(row["Method"]),""),])
    vt=Table(var_rows_,colWidths=[3.5*cm,1.7*cm,2.6*cm,1.7*cm,2.6*cm,5.3*cm],repeatRows=1,splitByRow=0)
    vts=_base_ts()
    for col in [1,2,3,4]: vts.add("ALIGN",(col,0),(col,-1),"RIGHT")
    for i in range(1,len(var_rows_)): vts.add("TEXTCOLOR",(1,i),(4,i),PDF_RED); vts.add("BACKGROUND",(1,i),(4,i),RED_LIGHT)
    vt.setStyle(vts); elements.append(vt); elements.append(Spacer(1,0.2*cm))
    var_chart=_chart_var()
    if var_chart is not None:
        ## Height scales with number of methods selected
        n_methods = len(risk_df)
        img_h = 5.5 if n_methods >= 2 else 5.0
        elements.append(_fig2img(var_chart, 17.4, img_h))
        ## Build dynamic caption based on which methods are in risk_df
        methods_in_report = risk_df["Method"].tolist()
        has_hist_c  = any("Historical"  in m for m in methods_in_report)
        has_para_c  = any("Parametric"  in m for m in methods_in_report)
        has_mc_c    = any("Monte Carlo" in m for m in methods_in_report)
        alpha_v_c   = portfolio_stats.get("alpha", 0.95)
        cap_parts   = []
        if has_hist_c:
            cap_parts.append(
                f"Historical VaR: red shaded area = empirical worst "
                f"{(1-alpha_v_c):.0%} of actual trading days"
            )
        if has_para_c:
            cap_parts.append(
                "Parametric VaR: red shaded area under normal curve — "
                "may understate tail risk when kurtosis > 3"
            )
        if has_mc_c:
            cap_parts.append(
                f"Monte Carlo VaR: histogram of {portfolio_stats.get('mc_sims',5000):,} "
                "simulated horizon return paths"
            )
        cap_text = " | ".join(cap_parts)
        if has_hist_c and has_para_c:
            cap_text += (
                f". Historical VaR is typically higher when kurtosis > 3 "
                f"(yours: {portfolio_stats.get('kurtosis',3):.2f})."
            )
        elements.append(_cap_box(cap_text))

        ## ── Dynamic plain-English explanation per selected method ────────────
        elements.append(Spacer(1, 0.2*cm))
        elements.append(Paragraph("What These Numbers Mean For Your Portfolio", H_SUB))
        elements.append(Spacer(1, 0.15*cm))

        alpha_exp   = portfolio_stats.get("alpha", 0.95)
        horizon_exp = portfolio_stats.get("horizon", 1)

        for _, row_r in risk_df.iterrows():
            method_n  = str(row_r["Method"])
            var_pct   = float(row_r["VaR %"])
            es_pct    = float(row_r["ES %"])
            var_rs    = float(row_r["VaR Amount (Rs.)"])
            es_rs     = float(row_r["ES Amount (Rs.)"])

            ## Method-specific explanation
            if "Historical" in method_n:
                method_how = (
                    "This uses your actual past daily returns — no assumptions "
                    "about the shape of the distribution. It looks at the worst "
                    f"{(1-alpha_exp):.0%} of real trading days your portfolio experienced."
                )
            elif "Parametric" in method_n:
                method_how = (
                    "This assumes your returns follow a normal (bell-curve) distribution "
                    "and uses mean and standard deviation to calculate the risk threshold. "
                    f"Your kurtosis is {portfolio_stats.get('kurtosis', 0):.2f} "
                    f"{'(fat tails — this method may understate actual tail risk)' if portfolio_stats.get('kurtosis', 3) > 3 else '(close to normal — this method is reliable for your portfolio)'}."
                )
            else:
                method_how = (
                    f"This runs {portfolio_stats.get('mc_sims', 5000):,} simulated future return scenarios using your "
                    "portfolio's historical mean and volatility, then finds the worst "
                    f"{(1-alpha_exp):.0%} of those simulations."
                )

            explanation = (
                f"<b>{method_n}:</b> "
                f"At <b>{alpha_exp:.0%} confidence</b> over a <b>{horizon_exp}-day horizon</b>, "
                f"your portfolio is not expected to lose more than "
                f"<b>Rs.{var_rs:,.0f} ({var_pct:.2%})</b> on any given day. "
                f"This is your VaR threshold. "
                f"However, on the <b>{(1-alpha_exp):.0%} of worst days</b> when losses do "
                f"exceed this threshold, the <b>average loss is Rs.{es_rs:,.0f} "
                f"({es_pct:.2%})</b> — this is the Expected Shortfall (ES or CVaR). "
                f"ES is the more conservative measure because it tells you not just "
                f"when things go wrong, but <i>how badly</i> they go wrong. "
                f"{method_how}"
            )
            elements.append(_info_box(explanation, style="info"))
            elements.append(Spacer(1, 0.1*cm))

        ## ── Interpretation of color zones ────────────────────────────────────
        zone_text = (
            "<b>Chart color zones:</b> "
            "<b>Dark red</b> = extreme loss zone — returns beyond ES, "
            "the most severe outcomes in the tail. "
            "<b>Light red</b> = VaR tail zone — returns between VaR and ES, "
            "the bad-but-not-worst outcomes. "
            "<b>Blue</b> = normal range — the majority of daily returns "
            "where no VaR breach occurs. "
            "The dashed red line marks the VaR threshold. "
            "The dotted amber line marks the ES threshold (always to the left of VaR)."
        )
        elements.append(_cap_box(zone_text))

    ## PAGE 7 — ATTRIBUTION (was PAGE 6)
    elements.append(PageBreak())
    elements+=_section_block("SECTION 5 — ATTRIBUTION ANALYSIS","Return Contribution = Weight x Stock Return | Sorted highest to lowest")
    if attr_df is not None and not attr_df.empty:
        at_hdr=["Ticker","Weight","Stock Return","Return Contrib.","P&L (Rs.)","P&L Contrib. %"]
        at_rows_=[at_hdr]
        for _,row in attr_df.iterrows():
            at_rows_.append([str(row["Ticker"]),f"{row['Weight']:.2%}",f"{row['Stock Return']:+.2%}",
                f"{row['Return Contribution']:+.2%}",f"Rs.{row['P&L']:+,.2f}",f"{row.get('P&L Contribution %',0):+.1f}%"])
        at_rows_.append(["TOTAL","100.00%","",f"{attr_df['Return Contribution'].sum():+.2%}",f"Rs.{attr_df['P&L'].sum():+,.2f}","100.0%"])
        att=Table(at_rows_,colWidths=[3.5*cm,2.2*cm,2.8*cm,2.8*cm,3.6*cm,2.5*cm],repeatRows=1)
        ats=_base_ts()
        for col in [1,2,3,4,5]: ats.add("ALIGN",(col,0),(col,-1),"RIGHT")
        for i,row in enumerate(attr_df.itertuples(),start=1):
            rc=row._asdict().get("Return Contribution",0); pnl=row._asdict().get("P&L",0)
            ats.add("TEXTCOLOR",(3,i),(3,i),PDF_GREEN if rc>=0 else PDF_RED)
            ats.add("BACKGROUND",(3,i),(3,i),GREEN_LIGHT if rc>=0 else RED_LIGHT)
            ats.add("TEXTCOLOR",(4,i),(4,i),PDF_GREEN if pnl>=0 else PDF_RED)
        tr=len(at_rows_)-1
        ats.add("BACKGROUND",(0,tr),(-1,tr),DARK_NAVY); ats.add("TEXTCOLOR",(0,tr),(-1,tr),WHITE); ats.add("FONTNAME",(0,tr),(-1,tr),"Helvetica-Bold")
        att.setStyle(ats); elements.append(att); elements.append(Spacer(1,0.2*cm))
        br_=attr_df.loc[attr_df["Return Contribution"].idxmax()]; wr_=attr_df.loc[attr_df["Return Contribution"].idxmin()]
        conc_=attr_df[attr_df["Weight"]>0.40]
        ins=f"<b>{br_['Ticker']}</b> largest contributor ({br_['Return Contribution']:+.2%}, P&L Rs.{br_['P&L']:+,.0f}). <b>{wr_['Ticker']}</b> largest drag ({wr_['Return Contribution']:+.2%}, P&L Rs.{wr_['P&L']:+,.0f})."
        if not conc_.empty:
            c_=conc_.iloc[0]; ins+=f" <b>Concentration Alert:</b> {c_['Ticker']} = {c_['Weight']:.1%} of portfolio — amplifies gains and losses."
        elements.append(_info_box(ins,"warning")); elements.append(Spacer(1,0.2*cm))
    elements.append(_fig2img(_chart_waterfall(),17.4,6.5))
    elements.append(_cap_box("Each bar shows a position's return contribution (weight x return). Green bars add to portfolio return, red bars subtract. The final navy bar is the net portfolio return contribution."))

    ## PAGE 8 — BHB ATTRIBUTION
    elements.append(PageBreak())
    elements+=_section_block(f"SECTION 6 — BRINSON-HOOD-BEEBOWER ATTRIBUTION",f"Allocation = sector bet vs {benchmark_name} | Selection = stock pick within sector | Interaction = combined")
    _etf_note = ""
    if sector_df is not None and not sector_df.empty and "ETF Tier" in sector_df.columns:
        _t1 = sector_df[sector_df["ETF Tier"] == 1]
        _t2 = sector_df[sector_df["ETF Tier"] == 2]
        _t3 = sector_df[sector_df["ETF Tier"] == 3]
        _t4 = sector_df[sector_df["ETF Tier"] == 4]
        if not _t3.empty:
            _t3_tickers = ", ".join(_t3["Ticker"].str.replace(".NS","").str.replace(".BO","").tolist())
            _etf_note += f" <b>Thematic ETFs ({_t3_tickers}):</b> Attribution benchmark = Nifty 500 (proxy — no direct index on Yahoo Finance)."
        if not _t4.empty:
            _t4_tickers = ", ".join(_t4["Ticker"].str.replace(".NS","").str.replace(".BO","").tolist())
            _etf_note += f" <b>Excluded ETFs ({_t4_tickers}):</b> Non-equity or unrecognised — excluded from BHB."
        if not _t2.empty:
            _t2_tickers = ", ".join(_t2["Ticker"].str.replace(".NS","").str.replace(".BO","").tolist())
            _etf_note += f" <b>Broad Market ETFs ({_t2_tickers}):</b> Track benchmark — near-zero active contribution."

    elements.append(_info_box(
        f"<b>BHB Framework:</b> Allocation Effect — did you overweight sectors that outperformed {benchmark_name}? "
        f"Selection Effect — within your sector, did your stock beat the sector index? "
        f"Interaction Effect — did good allocation and stock selection reinforce each other? "
        f"Total Active Return = sum of all three effects. "
        f"<b>ETFs:</b> Selection Effect = N/A (passive vehicles).{_etf_note}",
        "info"
    ))
    elements.append(Spacer(1,0.15*cm))
    if sector_df is not None and not sector_df.empty:
        bhb_hdr=["Ticker","Type","Sector","Wt Port","Wt Nifty","Active Wt","Alloc","Select","Interact","Total Active"]
        bhb_rows_=[bhb_hdr]
        for _,row in sector_df.iterrows():
            bhb_rows_.append([
                str(row["Ticker"]).replace(".NS",""),str(row.get("Type","Stock")),str(row["Sector"])[:26],
                f"{row['Weight (Port)']:.2%}",f"{row.get(f'Weight ({benchmark_name})', row.get('Weight (Nifty)', 0)):.2%}",f"{row['Active Weight']:+.2%}",
                f"{row['Allocation Effect']:+.2%}" if not pd.isna(row.get("Allocation Effect",np.nan)) else "N/A",
                f"{row['Selection Effect']:+.2%}" if not pd.isna(row.get("Selection Effect",np.nan)) else "N/A",
                f"{row['Interaction Effect']:+.2%}" if not pd.isna(row.get("Interaction Effect",np.nan)) else "N/A",
                f"{row['Total Active Return']:+.2%}" if not pd.isna(row.get("Total Active Return",np.nan)) else "N/A",
            ])
        bhb_rows_.append(["TOTAL","","","100%","",f"{sector_df['Active Weight'].sum():+.2%}",
            f"{sector_df['Allocation Effect'].sum(skipna=True):+.2%}",
            f"{sector_df['Selection Effect'].sum(skipna=True):+.2%}",
            f"{sector_df['Interaction Effect'].sum(skipna=True):+.2%}",
            f"{sector_df['Total Active Return'].sum(skipna=True):+.2%}"])
        bht=Table(bhb_rows_,colWidths=[2.0*cm,1.0*cm,3.0*cm,1.2*cm,1.2*cm,1.5*cm,1.8*cm,1.8*cm,1.8*cm,2.1*cm],repeatRows=1,splitByRow=0)
        bhts=_base_ts()
        for col in range(3,10): bhts.add("ALIGN",(col,0),(col,-1),"RIGHT")
        for i,row in enumerate(sector_df.itertuples(),start=1):
            aw=row._asdict().get("Active Weight",0)
            tot=row._asdict().get("Total Active Return",0) if not pd.isna(row._asdict().get("Total Active Return",np.nan)) else 0
            bhts.add("TEXTCOLOR",(5,i),(5,i),PDF_GREEN if aw>0 else PDF_RED); bhts.add("FONTNAME",(5,i),(5,i),"Helvetica-Bold")
            bhts.add("TEXTCOLOR",(9,i),(9,i),PDF_GREEN if tot>0 else (PDF_RED if tot<0 else GREY_500)); bhts.add("FONTNAME",(9,i),(9,i),"Helvetica-Bold")
        tr=len(bhb_rows_)-1
        bhts.add("BACKGROUND",(0,tr),(-1,tr),DARK_NAVY); bhts.add("TEXTCOLOR",(0,tr),(-1,tr),WHITE); bhts.add("FONTNAME",(0,tr),(-1,tr),"Helvetica-Bold")
        bht.setStyle(bhts); elements.append(bht); elements.append(Spacer(1,0.15*cm))
        ta_=sector_df["Allocation Effect"].sum(skipna=True); ts_=sector_df["Selection Effect"].sum(skipna=True); tt_=sector_df["Total Active Return"].sum(skipna=True)
        dom_="sector allocation" if abs(ta_)>abs(ts_) else "stock selection"; dv_=ta_ if abs(ta_)>abs(ts_) else ts_
        bhi=f"<b>Total Active Return: {tt_:+.2%}</b> vs {benchmark_name}. {'Positive excess return.' if tt_>0 else 'Underperformed passive Nifty 50.'} Dominant driver: <b>{dom_}</b> ({dv_:+.2%}). "
        bhi+=("Overweight in underperforming sectors." if dv_<0 and dom_=="sector allocation" else "Stock picks underperformed sector benchmarks." if dv_<0 else "Strong sector positioning." if dom_=="sector allocation" else "Strong stock-picking alpha.")
        elements.append(_info_box(bhi,"risk" if tt_<0 else "success")); elements.append(Spacer(1,0.15*cm))
    bhb_ch=_chart_bhb()
    if bhb_ch is not None:
        elements.append(_fig2img(bhb_ch,17.4,5.5))
        elements.append(_cap_box("Left: Three BHB effects per stock — Allocation (navy), Selection (gold), Interaction (teal). Right: Total active return per stock. Green = outperformed benchmark. Red = underperformed."))
    elements.append(Spacer(1,0.25*cm))
    sec_ch=_chart_sector_vs_stock()
    if sec_ch is not None:
        elements.append(_fig2img(sec_ch,17.4,4.5))
        elements.append(_cap_box(f"Bar = stock return. Diamond = sector index return. Gold line = {benchmark_name} total return. Gap between bar and diamond = stock-specific alpha or drag. Large gaps mean performance was driven by the individual stock, not the broader sector."))

    ## PAGE 9 — KEY FINDINGS + AUTO-GENERATED INSIGHTS
    elements.append(PageBreak())
    elements.extend(_section_block("KEY FINDINGS & INVESTMENT INSIGHTS",
                                   "Auto-generated analysis based on portfolio data"))

    try:
        _findings = []
        _alpha_val = portfolio_stats.get("alpha", 0.95)
        _sharpe_v  = portfolio_stats.get("sharpe", 0)
        _beta_v    = portfolio_stats.get("beta", 1.0)
        _vol_v     = portfolio_stats.get("ann_vol", 0.15)
        _kurt_v    = portfolio_stats.get("kurtosis", 3)

        ## 1. Return vs benchmark
        try:
            if benchmark_data is not None and len(benchmark_data) > 1:
                _bm_ret = (float(benchmark_data.iloc[-1])/float(benchmark_data.iloc[0])-1)*100
                _exc    = total_ret - _bm_ret
                if _exc >= 0:
                    _findings.append(f"<b>Outperformed {benchmark_name}:</b> Portfolio returned "
                                     f"{total_ret:+.2f}% vs {benchmark_name} {_bm_ret:+.2f}%, "
                                     f"generating excess return of <b>+{_exc:.2f}%</b>.")
                else:
                    _findings.append(f"<b>Underperformed {benchmark_name}:</b> Portfolio returned "
                                     f"{total_ret:+.2f}% vs {benchmark_name} {_bm_ret:+.2f}%, "
                                     f"a shortfall of <b>{_exc:.2f}%</b>. Dominant driver: sector allocation.")
        except Exception:
            pass

        ## 2. Best/Worst contributor
        try:
            if attr_df is not None and not attr_df.empty and "Return Contribution" in attr_df.columns:
                _bi = attr_df["Return Contribution"].idxmax()
                _wi = attr_df["Return Contribution"].idxmin()
                _bt = attr_df.loc[_bi,"Ticker"].replace(".NS","").replace(".BO","")
                _wt = attr_df.loc[_wi,"Ticker"].replace(".NS","").replace(".BO","")
                _bc = attr_df.loc[_bi,"Return Contribution"]
                _wc = attr_df.loc[_wi,"Return Contribution"]
                ## Show both return contribution AND stock return for clarity
                _bt_stock_ret = float(attr_df.loc[_bi, "Stock Return"] * 100) if "Stock Return" in attr_df.columns else 0
                _wt_stock_ret = float(attr_df.loc[_wi, "Stock Return"] * 100) if "Stock Return" in attr_df.columns else 0
                ## Check if best/worst are ETFs — use "return" not "stock return"
                _bt_is_etf = (
                    attr_df.loc[_bi, "Type"].startswith("ETF")
                    if "Type" in attr_df.columns else False
                )
                _wt_is_etf = (
                    attr_df.loc[_wi, "Type"].startswith("ETF")
                    if "Type" in attr_df.columns else False
                )
                _bt_ret_label = "ETF return" if _bt_is_etf else "stock return"
                _wt_ret_label = "ETF return" if _wt_is_etf else "stock return"
                _findings.append(
                    f"<b>Best Contributor:</b> {_bt} added <b>{_bc:+.2f}%</b> to portfolio return "
                    f"({_bt_ret_label}: {_bt_stock_ret:+.2f}%). "
                    f"<b>Biggest Drag:</b> {_wt} subtracted <b>{abs(_wc):.2f}%</b> "
                    f"({_wt_ret_label}: {_wt_stock_ret:+.2f}%)."
                )
        except Exception:
            pass

        ## 3. Concentration
        try:
            if attr_df is not None and not attr_df.empty and "Weight" in attr_df.columns:
                _max_w  = attr_df["Weight"].max()
                _max_tk = attr_df.loc[attr_df["Weight"].idxmax(),"Ticker"].replace(".NS","").replace(".BO","")
                if _max_w > 0.5:
                    _findings.append(f"<b>Concentration Risk:</b> {_max_tk} = {_max_w:.1%} of portfolio. "
                                     f"This single position dominates both returns and volatility. "
                                     f"Institutional guidelines typically cap single-stock exposure at 10-20%.")
                elif _max_w > 0.3:
                    _findings.append(f"<b>Moderate Concentration:</b> {_max_tk} = {_max_w:.1%}. "
                                     f"Above typical institutional thresholds — monitor closely.")
        except Exception:
            pass

        ## 4. Risk-adjusted return
        if _sharpe_v < 0:
            _findings.append(f"<b>Sharpe Ratio {_sharpe_v:.4f} (Negative):</b> Portfolio is generating "
                             f"returns below the risk-free rate ({RISK_FREE_ANNUAL:.2%} RBI Repo) on a "
                             f"risk-adjusted basis. Cash or FDs would have outperformed on a risk-adjusted basis.")
        elif _sharpe_v < 0.5:
            _findings.append(f"<b>Sharpe Ratio {_sharpe_v:.4f} (Below target):</b> Risk-adjusted return "
                             f"is positive but below the 0.5 institutional threshold. Return per unit of risk needs improvement.")
        else:
            _findings.append(f"<b>Sharpe Ratio {_sharpe_v:.4f} (Strong):</b> Portfolio is generating "
                             f"good risk-adjusted returns above the institutional threshold of 0.5.")

        ## 5. Beta / defensive or aggressive
        if _beta_v < 0.8:
            _findings.append(f"<b>Defensive Portfolio (Beta {_beta_v:.4f}):</b> Portfolio moves "
                             f"{abs(1-_beta_v)*100:.1f}% less than {benchmark_name}. "
                             f"Provides partial protection in market downturns but may underperform in strong rallies.")
        elif _beta_v > 1.2:
            _findings.append(f"<b>Aggressive Portfolio (Beta {_beta_v:.4f}):</b> Portfolio amplifies "
                             f"market moves by {(_beta_v-1)*100:.1f}%. Higher potential return "
                             f"but higher drawdown risk in downturns.")
        else:
            _findings.append(f"<b>Market-Aligned (Beta {_beta_v:.4f}):</b> Portfolio moves broadly "
                             f"in line with {benchmark_name}.")

        ## 6. Fat tails / VaR recommendation
        if _kurt_v > 4:
            _findings.append(f"<b>Fat Tail Risk (Kurtosis {_kurt_v:.2f}):</b> Extreme daily moves occur "
                             f"{_kurt_v/3:.1f}x more often than a normal distribution predicts. "
                             f"Use Historical VaR as the conservative measure — Parametric VaR "
                             f"will understate actual tail risk for this portfolio.")
        elif _kurt_v > 3:
            _findings.append(f"<b>Mild Fat Tails (Kurtosis {_kurt_v:.2f}):</b> Slight excess kurtosis — "
                             f"tail events are marginally more frequent than normal. Both Historical and "
                             f"Parametric VaR are reasonable estimates.")

        ## 7. VaR summary
        try:
            if risk_df is not None and not risk_df.empty:
                _vr = risk_df.iloc[0]
                _findings.append(f"<b>Downside Risk (VaR):</b> At {_alpha_val:.0%} confidence, "
                                  f"the maximum 1-day loss is estimated at Rs.{_vr['VaR Amount (Rs.)']:,.0f} "
                                  f"({_vr['VaR %']:.2f}%). On the worst days beyond this threshold, "
                                  f"average loss is Rs.{_vr['ES Amount (Rs.)']:,.0f} (ES/CVaR).")
        except Exception:
            pass
        ## Render findings
        for _fi, _finding in enumerate(_findings):
            elements.append(Paragraph(
                f"• {_finding}",
                _S(f"F{_fi}", fontSize=9, fontName="Helvetica",
                   textColor=rl_colors.HexColor("#2C3E50"), leading=14,
                   spaceAfter=4, leftIndent=0)
            ))
            elements.append(Spacer(1, 0.05*cm))

    except Exception as _fe:
        elements.append(_cap_box(f"Key findings unavailable: {_fe}"))

    elements.append(Spacer(1, 0.15*cm))
    elements.append(HRFlowable(width=CW, thickness=0.5, color=BORDER, spaceAfter=0.1*cm))

    ## PAGE 10 — METHODOLOGY + DISCLAIMERS
    elements+=_section_block("SECTION 7 — METHODOLOGY & DISCLAIMERS")
    for mt,md in [
        ("Risk-Free Rate",f"RBI Repo Rate {RISK_FREE_ANNUAL:.2%} p.a. Daily: (1+{RISK_FREE_ANNUAL:.4f})^(1/252)-1. Applied to Sharpe, Sortino, Treynor."),
        ("Sharpe Ratio",f"(mu_p - Rf)/sigma_p x sqrt(252). Rf={RISK_FREE_ANNUAL:.2%}. Source: Sharpe (1966)."),
        ("Sortino Ratio","Downside deviation replaces total sigma. Source: Sortino & Price (1994)."),
        ("Treynor Ratio","(mu_p - Rf) x 252 / Beta. Return per unit market risk. Source: Treynor (1965)."),
        ("Beta",f"Cov(Rp,Rm)/Var(Rm). {portfolio_stats.get('beta_horizon','1Y')} lookback. Benchmark = {benchmark_name}."),
        ("XIRR","Extended IRR for irregular cash flows. Brent-Dekker solver. Uses actual buy dates."),
        ("VaR / ES",f"Max loss at {portfolio_stats.get('alpha',0.95):.0%} confidence over {portfolio_stats.get('horizon',10)} day(s). Historical, Parametric (normal), Monte Carlo ({portfolio_stats.get('mc_sims',5000):,} paths)."),
        ("TWR","GIPS/SEBI standard. Eliminates cash flow timing. Each day weighted by actual portfolio composition."),
        ("BHB Attribution","Brinson-Hood-Beebower (1986). Allocation=(Wp-Wb)(Rs-Rb); Selection=Wb(Rp-Rs); Interaction=(Wp-Wb)(Rp-Rs)."),
        ("ETF Handling","ETFs classified into 4 tiers: Tier 1 Sector ETF (direct sector index benchmark); Tier 2 Broad Market ETF (tracks portfolio benchmark — Allocation Effect = 0 by definition); Tier 3 Thematic ETF (Nifty 500 proxy — no direct index on Yahoo Finance); Tier 4 Non-Equity ETF (Gold/Liquid/International — excluded from BHB entirely). Selection and Interaction Effects = N/A for all ETFs (passive vehicles). P&L and Return Contribution included for all ETFs regardless of tier."),
        ("Sector Data",f"Yahoo Finance sector indices. Fallback to {benchmark_name} when sector index unavailable."),
    ]:
        mr=Table([[Paragraph(mt,ParagraphStyle("MT",fontSize=7.5,fontName="Helvetica-Bold",textColor=STEEL,leading=10)),Paragraph(md,DISC_S)]],colWidths=[3.5*cm,CW-3.5*cm])
        mr.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),("LINEBELOW",(0,0),(-1,-1),0.3,BORDER),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4)]))
        elements.append(mr)
    elements.append(Spacer(1,0.2*cm))
    dt=Table([[Paragraph("<br/>".join(f"&#8226; {d}" for d in [
        "NOT INVESTMENT ADVICE. For informational and analytical purposes only. Not a recommendation or offer to buy or sell any security.",
        "DATA ACCURACY. All data from Yahoo Finance. Accuracy not guaranteed. Prices may be delayed.",
        "PAST PERFORMANCE. Past performance does not indicate future results. All investments involve risk including loss of principal.",
        "MODEL LIMITATIONS. VaR, ES, BHB models have inherent limitations and may not capture all market risks.",
        f"RISK-FREE RATE. {RISK_FREE_ANNUAL:.2%} RBI Repo Rate subject to change with monetary policy.",
        f"GENERATED: {gen_date}.",
    ]),DISC_LIGHT)]],colWidths=[CW])
    dt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),DARK_NAVY),("LEFTPADDING",(0,0),(-1,-1),12),("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10)]))
    elements.append(dt)

    doc.build(elements,canvasmaker=_NC)
    buffer.seek(0)
    return buffer


## ── Determine benchmark ───────────────────────────────────────────────────
if BENCHMARK_OPTIONS[selected_benchmark_label] is None:
    ## Auto-detect
    enriched_rows      = list(enriched_info.values())
    auto_bm_ticker, auto_bm_name, auto_bm_exchange = detect_portfolio_benchmark(enriched_rows)
    BENCHMARK_TICKER   = auto_bm_ticker
    BENCHMARK_NAME     = auto_bm_name
    _auto_detected     = True
else:
    BENCHMARK_TICKER, BENCHMARK_NAME = BENCHMARK_OPTIONS[selected_benchmark_label]
    _auto_detected     = False

## ── Update sector weights AND sector index tickers to match benchmark ────────
## Default fallback: Nifty 100 weights (correct SEBI Large Cap benchmark)
_bsw_default = BENCHMARK_SECTOR_WEIGHTS.get("^CNX100", BENCHMARK_SECTOR_WEIGHTS["^NSEI"])
NIFTY_SECTOR_WEIGHTS = BENCHMARK_SECTOR_WEIGHTS.get(
    BENCHMARK_TICKER, _bsw_default
)["weights"]

## Switch sector index tickers based on exchange
## BSE benchmarks → BSE sector indices | NSE benchmarks → NSE sector indices
_is_bse_benchmark = BENCHMARK_TICKER in (
    "^BSESN", "BSE-MIDCAP.BO", "BSE-500.BO", "BSE-SMLCAP.BO"
)
SECTOR_INDEX_TICKERS = (
    BSE_SECTOR_INDEX_TICKERS if _is_bse_benchmark
    else NSE_SECTOR_INDEX_TICKERS
)

_bm_weight_info = BENCHMARK_SECTOR_WEIGHTS.get(
    BENCHMARK_TICKER,
    BENCHMARK_SECTOR_WEIGHTS["^NSEI"]
)

## Show detection result
_cap_raw     = holdings_df["Cap"].value_counts().to_dict()
_cap_display = {
    ("ETF"        if k == "ETF"       else
     "Large Cap"  if k == "Large Cap" else
     "Mid Cap"    if k == "Mid Cap"   else
     "Small Cap"  if k == "Small Cap" else
     "Unclassified"): v
    for k, v in _cap_raw.items()
}
_exch_counts  = holdings_df["Exchange"].value_counts().to_dict()
_exch_display = {k: v for k, v in _exch_counts.items() if k in ("NSE", "BSE")}
_unresolved   = sum(v for k, v in _exch_counts.items() if k not in ("NSE", "BSE"))
_cap_str  = " · ".join([f"{v} {k}" for k, v in _cap_display.items()])
_exch_str = " · ".join([f"{v} {k}" for k, v in _exch_display.items()])
if _unresolved:
    _exch_str += f" · {_unresolved} Unresolved"
_detect_msg = (
    f"{'Auto-detected' if _auto_detected else 'Selected'} benchmark: "
    f"<strong style='color:#C8A951'>{BENCHMARK_NAME}</strong> ({BENCHMARK_TICKER})"
    f" &nbsp;|&nbsp; Sector weights: {_bm_weight_info['name']} ({_bm_weight_info['updated']})"
    f" &nbsp;|&nbsp; {_cap_str}"
    f" &nbsp;|&nbsp; {_exch_str}"
)
st.markdown(
    f"<div style='padding:8px 14px;background:#0D1723;border:1px solid #21262D;"
    f"border-left:3px solid #C8A951;border-radius:0 8px 8px 0;"
    f"font-size:12px;color:#C9D1D9;margin-bottom:12px'>{_detect_msg}</div>",
    unsafe_allow_html=True
)



## ─────────────────────────────────────────────────────────────────────────────
# %%  MAIN ANALYSIS
## ─────────────────────────────────────────────────────────────────────────────
try:
    prices = fetch_price_history(tickers, portfolio_start_date, portfolio_end_date)





    if prices.empty:
        st.error(
            "⚠️ No price data returned for your tickers. "
            "Possible causes: \n"
            "1. Invalid ticker symbol — ensure it ends with .NS or .BO \n"
            "2. Buy date too recent — market may not have data yet \n"
            "3. Temporary Yahoo Finance outage — try refreshing in a minute \n"
            f"Tickers attempted: {', '.join(tickers)}"
        )
        st.stop()

    returns = compute_returns(prices)

    ## ── Holdings calculations ─────────────────────────────────────────────────
    latest_prices_series = fetch_latest_prices(tickers)
    holdings_df["Current Price"] = holdings_df["Ticker"].map(latest_prices_series)
    ## If some tickers have no current price — warn but continue with available data
    _bad_tickers = holdings_df[holdings_df["Current Price"].isna()]["Ticker"].tolist()
    if _bad_tickers:
        _bad_str = ", ".join([t.replace(".NS","").replace(".BO","") for t in _bad_tickers])
        st.warning(
            f"⚠️ Could not fetch current price for: **{_bad_str}**. "
            f"These holdings will be excluded from calculations. "
            f"Check that the ticker symbol is valid on Yahoo Finance."
        )
        ## Drop the bad tickers from holdings_df
        holdings_df = holdings_df[holdings_df["Current Price"].notna()].copy()
        tickers     = holdings_df["Ticker"].tolist()
        if holdings_df.empty:
            st.error("No valid tickers remaining. Please check your portfolio.")
            st.stop()
        ## CRITICAL: re-align prices and returns to valid tickers only
        ## Otherwise np.dot(returns, weights) shape mismatch crashes
        valid_cols = [t for t in tickers if t in prices.columns]
        prices     = prices[valid_cols]
        returns    = compute_returns(prices)

    holdings_df["Invested Value"] = holdings_df["Buy Price"] * holdings_df["Quantity"]
    holdings_df["Current Value"]  = holdings_df["Current Price"] * holdings_df["Quantity"]
    total_invested                = holdings_df["Invested Value"].sum()
    total_current                 = holdings_df["Current Value"].sum()
    holdings_df["P&L"]            = holdings_df["Current Value"] - holdings_df["Invested Value"]
    ## Guard: if total current value is zero, stop gracefully
    if total_current <= 0:
        st.error("⚠️ Current portfolio value is zero or negative. Please check your holdings.")
        st.stop()

    holdings_df["Return %"]       = (holdings_df["P&L"] / holdings_df["Invested Value"]) * 100
    total_pnl                     = holdings_df["P&L"].sum()
    portfolio_value               = total_current
    weights                       = (holdings_df["Current Value"] / total_current).values

    ## Ensure returns columns are in same order as holdings_df tickers
    ## to guarantee np.dot(returns, weights) alignment
    _aligned_tickers = [t for t in holdings_df["Ticker"].tolist() if t in returns.columns]
    returns          = returns[_aligned_tickers]

    weight_map   = dict(zip(holdings_df["Ticker"], weights))
    value_map    = dict(zip(holdings_df["Ticker"], holdings_df["Current Value"]))
    invested_map = dict(zip(holdings_df["Ticker"], holdings_df["Invested Value"]))
    pnl_map      = dict(zip(holdings_df["Ticker"], holdings_df["P&L"]))

    ## ── XIRR ─────────────────────────────────────────────────────────────────
    cf_port = [-v for v in holdings_df["Invested Value"].values]
    dt_port = list(pd.to_datetime(holdings_df["Buy Date"]))
    cf_port.append(total_current)
    dt_port.append(pd.Timestamp.today())
    portfolio_xirr = xirr(cf_port, dt_port)

    ## ── Returns ──────────────────────────────────────────────────────────────
    port_returns     = pd.Series(portfolio_series(returns, weights), index=returns.index)
    port_returns_twr = compute_twr_portfolio_returns(prices, holdings_df)
    port_returns_bm  = port_returns_twr if not port_returns_twr.empty else port_returns

    ## ── Beta ─────────────────────────────────────────────────────────────────
    benchmark_series = fetch_benchmark(BENCHMARK_TICKER, portfolio_start_date, portfolio_end_date)
    portfolio_beta   = np.nan
    if not benchmark_series.empty:
        bm_ret           = benchmark_series.pct_change().dropna()
        combined         = pd.concat([port_returns_bm, bm_ret], axis=1).dropna()
        combined.columns = ["Portfolio", "Benchmark"]
        combined         = combined.tail(beta_period_map[beta_horizon])
        portfolio_beta   = (
            combined["Portfolio"].cov(combined["Benchmark"]) /
            combined["Benchmark"].var()
        )
    ## ── Store in session_state so PDF button can access on rerun ─────────────
    st.session_state["_pdf_benchmark_series"] = benchmark_series
    st.session_state["_pdf_portfolio_beta"]   = portfolio_beta

    ## ── Ratios ───────────────────────────────────────────────────────────────
    _r                = port_returns_bm  ## TWR — correct series for all risk metrics
    mu_daily          = _r.mean()
    sigma_daily       = _r.std()
    _excess_daily     = _r - risk_free_daily
    _neg_excess       = _excess_daily[_excess_daily < 0]
    downside          = np.sqrt((_neg_excess ** 2).mean()) if len(_neg_excess) > 0 else np.nan
    annual_volatility = sigma_daily * np.sqrt(252)

    sharpe  = (((mu_daily - risk_free_daily) / sigma_daily) * np.sqrt(252)
               if sigma_daily > 0 else np.nan)
    sortino = (((mu_daily - risk_free_daily) / downside) * np.sqrt(252)
               if (downside is not None and not np.isnan(downside) and downside > 0) else np.nan)
    treynor = (((mu_daily - risk_free_daily) * 252) / portfolio_beta
               if (not np.isnan(portfolio_beta) and portfolio_beta != 0) else np.nan)

    ## ── CAGR ─────────────────────────────────────────────────────────────────
    last_252 = port_returns_bm.dropna().tail(252)
    if len(last_252) >= 2:
        total_returns_1y = (1 + last_252).prod() - 1
        years            = len(last_252) / 252
        if years >= 1.0:
            cagr_1y    = (1 + total_returns_1y) ** (1 / years) - 1
            cagr_label = "CAGR (1Y)"
        else:
            cagr_1y    = total_returns_1y
            months     = max(1, round(len(last_252) / 21))
            cagr_label = f"Return ({months}M)"
    else:
        cagr_1y    = np.nan
        cagr_label = "CAGR (1Y)"

    portfolio_stats = {
        "alpha":        alpha,
        "horizon":      horizon_days,
        "mc_sims":      mc_sims,
        "beta":         portfolio_beta,
        "beta_horizon": beta_horizon,
        "sharpe":       sharpe,
        "sortino":      sortino,
        "treynor":      treynor,
        "cagr":         cagr_1y,
        "ann_vol":      annual_volatility,
        "skewness":     skew(port_returns_bm),
        "kurtosis":     kurtosis(port_returns_bm, fisher=False),
    }
    ## ── Store ALL PDF variables in session_state ─────────────────────────────
    ## Every variable below is defined inside this try block
    ## When PDF button is clicked, Streamlit reruns — this block is skipped
    ## session_state persists across reruns — only reliable way to pass these
    st.session_state["_pdf_portfolio_stats"]   = portfolio_stats
    st.session_state["_pdf_benchmark_series"]  = benchmark_series
    st.session_state["_pdf_portfolio_beta"]    = portfolio_beta
    st.session_state["_pdf_holdings_df"]       = holdings_df.copy()
    st.session_state["_pdf_total_invested"]    = total_invested
    st.session_state["_pdf_total_current"]     = total_current
    st.session_state["_pdf_total_pnl"]         = total_pnl
    st.session_state["_pdf_portfolio_xirr"]    = portfolio_xirr
    st.session_state["_pdf_max_dd"]            = max_dd
    st.session_state["_pdf_recovery_date"]     = recovery_date
    st.session_state["_pdf_port_returns_bm"]   = port_returns_bm
    st.session_state["_pdf_benchmark_name"]    = BENCHMARK_NAME
    st.session_state["_pdf_benchmark_ticker"]  = BENCHMARK_TICKER
    st.session_state["_pdf_is_bse"]            = _is_bse_benchmark
    st.session_state["_pdf_display_name"]      = _display_name


    ## ── Tab navigation ────────────────────────────────────────────────────────────
    st.markdown("""
<style>
[data-testid="stTabs"] {
    margin-top: -8px;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #161B22;
    border-bottom: 2px solid #C8A951;
    border-radius: 8px 8px 0 0;
    padding: 4px 8px 0 8px;
    gap: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px 6px 0 0;
    color: #8B949E;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 16px;
    border: none;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #1E3A5F !important;
    color: #C8A951 !important;
    border-bottom: 2px solid #C8A951;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #C9D1D9 !important;
    background: #21262D !important;
}
[data-testid="stTabPanel"] {
    background: #0E1117;
    border: 1px solid #21262D;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 16px 20px;
}
</style>
""", unsafe_allow_html=True)

    ## Initialize sector_df before tabs so tab7 can access it
    ## even if tab3 hasn't been visited yet
    sector_df = None
    attr_df   = None

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "&#9632;  Overview",
        "&#9632;  Performance",
        "&#9632;  Attribution",
        "&#9632;  Risk & VaR",
        "&#9632;  Stress Test",
        "&#9632;  Market Data",
        "&#9632;  Market Intelligence",
    ])

    ## ── TAB 1 — OVERVIEW (Holdings + Composition) ──────────────────────────────
    with tab1:
        ## ═════════════════════════════════════════════════════════════════════════
        ##  SECTION 1 — HOLDINGS SUMMARY
        ## ═════════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header"><h3>&#9632; Section 1 — Holdings Summary</h3><p>Current positions, P&L, and annualised XIRR per holding</p></div>', unsafe_allow_html=True)


        ## ── Styled KPI cards ──────────────────────────────────────────────────────
        pnl_pct   = (total_pnl / total_invested) * 100
        xirr_str2 = f"{portfolio_xirr:.2%}" if (portfolio_xirr is not None and not np.isnan(float(portfolio_xirr))) else "N/A"
        xirr_col  = "green" if (not np.isnan(portfolio_xirr) and portfolio_xirr > 0) else "red"
        pnl_col   = "green" if total_pnl >= 0 else "red"
        val_col   = "green" if total_current >= total_invested else "red"

        st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px">
      <div class="kpi-card blue">
        <div class="kpi-label">Total Invested</div>
        <div class="kpi-value">&#8377;{total_invested:,.0f}</div>
        <div class="kpi-sub">{len(holdings_df)} positions</div>
      </div>
      <div class="kpi-card {val_col}">
        <div class="kpi-label">Current Value</div>
        <div class="kpi-value {val_col}">&#8377;{total_current:,.0f}</div>
        <div class="kpi-sub">Mark-to-market</div>
      </div>
      <div class="kpi-card {pnl_col}">
        <div class="kpi-label">Total P&L</div>
        <div class="kpi-value {pnl_col}">&#8377;{total_pnl:+,.0f}</div>
        <div class="kpi-sub">{pnl_pct:+.2f}% on invested</div>
      </div>
      <div class="kpi-card {xirr_col}">
        <div class="kpi-label">Portfolio XIRR</div>
        <div class="kpi-value {xirr_col}">{xirr_str2}</div>
        <div class="kpi-sub">Annualised return</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

        st.dataframe(
            holdings_df.assign(**{"Buy Date": holdings_df["Buy Date"].dt.strftime("%Y-%m-%d") if hasattr(holdings_df["Buy Date"].dt, "strftime") else holdings_df["Buy Date"]}).style.format({
                "Buy Price":      "₹{:.2f}",
                "Quantity":       "{:.0f}",
                "Current Price":  "₹{:.2f}",
                "Invested Value": "₹{:.2f}",
                "Current Value":  "₹{:.2f}",
                "P&L":            "₹{:.2f}",
                "Return %":       "{:.2f}%"
            }).map(lambda v: "color:#C8A951;font-weight:600"
                   if v in ("Large Cap","Mid Cap","Small Cap") else
                   ("color:#58A6FF" if v == "NSE" else
                    ("color:#3FB950" if v == "BSE" else "")),
                   subset=[c for c in ["Exchange","Cap"]
                           if c in holdings_df.columns]),
            width='stretch'
        )
        st.markdown("<hr style='border:none;border-top:1px solid #21262D;margin:24px 0'>", unsafe_allow_html=True)


        ## ═════════════════════════════════════════════════════════════════════════
        ##  SECTION 2 — PORTFOLIO COMPOSITION (TREEMAP)
        ## ═════════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header"><h3>&#9632; Section 2 — Portfolio Composition</h3><p>Weight distribution and XIRR treemap</p></div>', unsafe_allow_html=True)


        ticker_xirr       = {}
        ticker_data_notes = {}

        for _, row in holdings_df.iterrows():
            t              = row["Ticker"]
            buy_date       = pd.to_datetime(row["Buy Date"])
            buy_price      = float(row["Buy Price"])
            quantity       = float(row["Quantity"])
            invested_value = buy_price * quantity

            if t not in latest_prices_series.index:
                ticker_xirr[t]       = 0.0
                ticker_data_notes[t] = f"Could not fetch current price for **{t}**."
                continue

            current_price  = float(latest_prices_series[t])
            current_value  = current_price * quantity
            xirr_val       = xirr([-invested_value, current_value],
                                   [buy_date, pd.Timestamp.today()])
            ticker_xirr[t] = xirr_val if not np.isnan(xirr_val) else 0.0

            if t in prices.columns:
                ps = prices[t].dropna()
                if len(ps) > 0:
                    earliest = ps.index[0].date()
                    if buy_date.date() < earliest:
                        ticker_data_notes[t] = (
                            f"**{t}** — Yahoo Finance data only available from **{earliest}** "
                            f"(likely demerger / IPO / corporate action). "
                            f"XIRR uses entered Buy Price ₹{buy_price:,.2f} — still accurate."
                        )
                    else:
                        ticker_data_notes[t] = None
                else:
                    ticker_data_notes[t] = None
            else:
                ticker_data_notes[t] = f"No price history for **{t}**. XIRR from entered Buy Price only."

        treemap_df             = pd.DataFrame({"Ticker": tickers})
        treemap_df["Weight"]   = treemap_df["Ticker"].map(weight_map)
        treemap_df["XIRR"]     = treemap_df["Ticker"].map(ticker_xirr)
        treemap_df["Value"]    = treemap_df["Ticker"].map(value_map)
        treemap_df["Invested"] = treemap_df["Ticker"].map(invested_map)
        treemap_df["P&L"]      = treemap_df["Ticker"].map(pnl_map)
        treemap_df             = treemap_df.fillna(0)

        def xirr_category(x):
            if x >= 0.40:   return "Excellent (>40% p.a.)"
            elif x >= 0.20: return "Good (20-40% p.a.)"
            elif x >= 0:    return "Moderate (0-20% p.a.)"
            else:           return "Negative (<0% p.a.)"

        treemap_df["Category"]   = treemap_df["XIRR"].apply(xirr_category)
        treemap_df["HasWarning"] = treemap_df["Ticker"].apply(
            lambda t: "⚠️ data note" if ticker_data_notes.get(t) is not None else ""
        )

        if _HAS_PLOTLY:
            fig_tm = go.Figure(go.Treemap(
                labels     = treemap_df["Ticker"].tolist(),
                parents    = [""] * len(treemap_df),
                values     = treemap_df["Value"].tolist(),
                marker     = dict(
                    colors = treemap_df["Category"].map({
                        "Excellent (>40% p.a.)": "#00C853",
                        "Good (20-40% p.a.)":    "#2979FF",
                        "Moderate (0-20% p.a.)": "#FF9100",
                        "Negative (<0% p.a.)":   "#FF1744"
                    }).tolist(),
                    line = dict(width=2, color="black")
                ),
                customdata = treemap_df[
                    ["Weight", "XIRR", "P&L", "Value", "HasWarning", "Invested"]
                ].values,
                texttemplate = (
                    "<b>%{label}</b><br>"
                    "%{customdata[0]:.1%} weight<br>"
                    "XIRR %{customdata[1]:.1%} p.a.<br>"
                    "%{customdata[4]}<br>"
                    "₹%{customdata[2]:,.0f} P&L<br>"
                    "₹%{customdata[3]:,.0f}"
                ),
                hovertemplate = (
                    "<b>%{label}</b><br>"
                    "Weight: %{customdata[0]:.2%}<br>"
                    "XIRR: %{customdata[1]:.2%} p.a.<br>"
                    "P&L: ₹%{customdata[2]:,.0f}<br>"
                    "Current: ₹%{customdata[3]:,.0f}<br>"
                    "Invested: ₹%{customdata[5]:,.0f}<br>"
                    "%{customdata[4]}<extra></extra>"
                ),
                textinfo = "text",
            ))
            fig_tm.update_layout(height=500, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig_tm, width='stretch')
        else:
            ## Plotly not available — matplotlib bar chart fallback
            _fig2, _ax2 = plt.subplots(figsize=(10, 4))
            _ax2.barh(treemap_df["Ticker"], treemap_df["Weight"],
                      color=["#00C853","#2979FF","#FF9100","#FF1744"] * len(treemap_df))
            _ax2.set_xlabel("Portfolio Weight")
            _ax2.set_title("Portfolio Composition by Weight")
            apply_dark_theme(_ax2, _fig2)
            plt.tight_layout()
            st.pyplot(_fig2)
            plt.close(_fig2)

        has_notes = any(v is not None for v in ticker_data_notes.values())
        if has_notes:
            for ticker, note in ticker_data_notes.items():
                if note is not None:
                    with st.expander(f"📌 {ticker} — Data Note", expanded=False):
                        st.info(note)
        st.markdown("<hr style='border:none;border-top:1px solid #21262D;margin:24px 0'>", unsafe_allow_html=True)



    ## ── TAB 2 — PERFORMANCE (Stats + Benchmark) ────────────────────────────────
    with tab2:
        ## ═════════════════════════════════════════════════════════════════════════
        ##  SECTION 4 — PORTFOLIO SUMMARY STATISTICS
        ## ═════════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header"><h3>&#9632; Section 4 — Portfolio Summary Statistics</h3><p>Risk-adjusted ratios, beta, volatility, skewness, kurtosis</p></div>', unsafe_allow_html=True)
        st.markdown(f"""
    <div style="font-size:12px;color:#8B949E;margin-bottom:12px;padding:8px 12px;
    background:#161B22;border-radius:8px;border:1px solid #21262D">
       All ratios annualised
      &nbsp;|&nbsp; Beta lookback: <span style="color:#C8A951">{beta_horizon}</span>
      &nbsp;|&nbsp; Benchmark: <span style="color:#C8A951">{BENCHMARK_NAME} ({BENCHMARK_TICKER})</span>
    </div>
    """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
        c1.metric(f"Beta ({beta_horizon})",
                  f"{portfolio_beta:.4f}" if not np.isnan(portfolio_beta) else "N/A",
                  help=f">1 = more volatile than {BENCHMARK_NAME}")
        c2.metric("Sharpe Ratio",
                  f"{sharpe:.4f}" if not np.isnan(sharpe) else "N/A",
                  help=f"Excess return over RBI repo ({RISK_FREE_ANNUAL:.2%}) per unit volatility")
        c3.metric("Sortino Ratio",
                  f"{sortino:.4f}" if not np.isnan(sortino) else "N/A",
                  help="Like Sharpe but only penalises downside volatility")
        c4.metric("Treynor Ratio",
                  f"{treynor:.4f}" if not np.isnan(treynor) else "N/A",
                  help="Excess return per unit of market risk (Beta)")
        c5.metric(cagr_label,
                  f"{cagr_1y:.2%}" if not np.isnan(cagr_1y) else "N/A")
        c6.metric("Annual Volatility",
                  f"{annual_volatility:.2%}" if not np.isnan(annual_volatility) else "N/A")
        c7.metric("Skewness",          f"{skew(port_returns_bm):.4f}",
                  help="Negative = left tail risk")
        c8.metric("Kurtosis (Pearson)",f"{kurtosis(port_returns_bm, fisher=False):.4f}",
                  help=">3 = fat tails")
        st.markdown("<hr style='border:none;border-top:1px solid #21262D;margin:24px 0'>", unsafe_allow_html=True)


        ## ═════════════════════════════════════════════════════════════════════════
        ##  SECTION 5 — BENCHMARK COMPARISON
        ## ═════════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header"><h3>&#9632; Section 5 — Benchmark Comparison (Nifty 50)</h3><p>TWR performance, CAPM Alpha, capture ratios, rolling metrics</p></div>', unsafe_allow_html=True)

        if not benchmark_series.empty:
            bm_returns_full = benchmark_series.pct_change().dropna()
            aligned         = pd.concat([port_returns_bm, bm_returns_full], axis=1).dropna()
            aligned.columns = ["Portfolio", "Benchmark"]
            port_ret_a      = aligned["Portfolio"]
            bench_ret_a     = aligned["Benchmark"]

            with st.expander("📌 Methodology Notes — Read Before Interpreting", expanded=False):
                ## Detect any stocks with limited price history
                _limited_stocks = []
                for _, _row in holdings_df.iterrows():
                    _tk  = _row["Ticker"]
                    _buy = pd.to_datetime(_row["Buy Date"])
                    if _tk in prices.columns:
                        _first_available = prices[_tk].dropna().index[0] if not prices[_tk].dropna().empty else _buy
                        ## If first available price is more than 5 days after buy date → limited history
                        if (_first_available - _buy).days > 5:
                            _limited_stocks.append(
                                f"{_tk.replace('.NS','').replace('.BO','')} "
                                f"(data from {_first_available.strftime('%b %Y')})"
                            )

                _limited_note = (
                    f"**Recently listed stocks with limited history:** {', '.join(_limited_stocks)}. "
                    f"Data not available from purchase date — likely IPO / demerger / corporate action."
                    if _limited_stocks else
                    "All stocks have price history from their purchase date."
                )

                st.markdown(f"""
**Comparison period:** {aligned.index[0].date()} to {aligned.index[-1].date()} ({len(aligned)} trading days)

**TWR (Time-Weighted Return)** — GIPS/SEBI industry standard. Each stock's returns included only from its actual purchase date.

{_limited_note}

**Portfolio Return ≠ P&L Return.** TWR daily returns vs simple P&L/invested.

**CAPM Alpha is annualised.** Short holding periods amplify numbers.

*For informational purposes only — not investment advice.*
                """)

            ## Rebase both to 100 at the same start date — same logic as PDF chart
            port_cum  = (1 + port_ret_a).cumprod()
            bench_cum = (1 + bench_ret_a).cumprod()
            port_cum  = (port_cum  / port_cum.iloc[0])  * 100
            bench_cum = (bench_cum / bench_cum.iloc[0]) * 100
            port_total  = port_cum.iloc[-1]  - 100
            bench_total = bench_cum.iloc[-1] - 100
            excess      = port_total - bench_total

            n_years    = len(port_ret_a) / 252
            port_ann   = (port_cum.iloc[-1]  / 100) ** (1/n_years) - 1 if n_years > 0 else np.nan
            bench_ann  = (bench_cum.iloc[-1] / 100) ** (1/n_years) - 1 if n_years > 0 else np.nan
            _rf_ann    = (1 + RISK_FREE_ANNUAL) - 1
            capm_alpha = (port_ann - _rf_ann - portfolio_beta * (bench_ann - _rf_ann)
                          if not np.isnan(portfolio_beta) else np.nan)

            excess_series  = port_ret_a - bench_ret_a
            tracking_error = excess_series.std() * np.sqrt(252)
            info_ratio     = (excess_series.mean() * 252 / tracking_error
                              if tracking_error > 0 else np.nan)

            up_days   = bench_ret_a > 0
            down_days = bench_ret_a < 0

            def capture_ratio(port, bench, mask):
                p = (1 + port[mask]).prod() - 1
                b = (1 + bench[mask]).prod() - 1
                return (p / b * 100) if b != 0 else np.nan

            up_capture   = capture_ratio(port_ret_a, bench_ret_a, up_days)
            down_capture = capture_ratio(port_ret_a, bench_ret_a, down_days)

            def period_returns(cum_series):
                today = cum_series.index[-1]
                cuts  = {"1M": today - pd.DateOffset(months=1),
                         "3M": today - pd.DateOffset(months=3),
                         "6M": today - pd.DateOffset(months=6),
                         "1Y": today - pd.DateOffset(years=1),
                         "All": cum_series.index[0]}
                return {p: (cum_series[cum_series.index >= s].iloc[-1] /
                            cum_series[cum_series.index >= s].iloc[0] - 1) * 100
                        if len(cum_series[cum_series.index >= s]) > 1 else np.nan
                        for p, s in cuts.items()}

            period_df = pd.DataFrame({
                "Portfolio (%)": period_returns(port_cum),
                f"{BENCHMARK_NAME} (%)": period_returns(bench_cum),
            })
            period_df["Excess (%)"] = period_df["Portfolio (%)"] - period_df[f"{BENCHMARK_NAME} (%)"]

            rolling_w       = min(60, len(port_ret_a) // 2)
            rolling_beta_bm = (port_ret_a.rolling(rolling_w).cov(bench_ret_a) /
                               bench_ret_a.rolling(rolling_w).var()).dropna()
            rolling_corr    = port_ret_a.rolling(rolling_w).corr(bench_ret_a).dropna()
            rolling_exc_cum = (port_cum / bench_cum - 1) * 100

            st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Performance vs Nifty 50</p>", unsafe_allow_html=True)
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Portfolio Return (TWR)", f"{port_total:.2f}%")
            st.metric(f"{BENCHMARK_NAME} Return", f"{bench_total:.2f}%")
            m3.metric("Excess Return", f"{excess:+.2f}%",
                      delta="Outperforming" if excess > 0 else "Underperforming",
                      delta_color="normal" if excess > 0 else "inverse")
            m4.metric("CAPM Alpha (Ann.)",
                      f"{capm_alpha:+.2f}%" if not np.isnan(capm_alpha) else "N/A",
                      help="Excess return after Beta adjustment. Positive = genuine skill.")
            m5.metric("Up Capture",
                      f"{up_capture:.1f}%" if not np.isnan(up_capture) else "N/A",
                      help=">100% = gains more than market on up days")
            m6.metric("Down Capture",
                      f"{down_capture:.1f}%" if not np.isnan(down_capture) else "N/A",
                      delta="Defensive" if (not np.isnan(down_capture) and down_capture < 100) else "Aggressive",
                      delta_color="normal" if (not np.isnan(down_capture) and down_capture < 100) else "inverse",
                      help="<100% = loses less than market on down days")

            r1, r2 = st.columns(2)
            r1.metric("Tracking Error (Ann.)",
                      f"{tracking_error:.2%}" if not np.isnan(tracking_error) else "N/A",
                      help=f"Lower = closer to {BENCHMARK_NAME}")
            r2.metric("Information Ratio",
                      f"{info_ratio:.4f}" if not np.isnan(info_ratio) else "N/A",
                      help=">0.5 good. >1.0 excellent.")

            st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Returns by Period</p>", unsafe_allow_html=True)
            st.dataframe(
                period_df.style
                .format("{:+.2f}%")
                .background_gradient(subset=["Excess (%)"], cmap="RdYlGn", vmin=-10, vmax=10),
                width='stretch'
            )

            st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Cumulative Performance — Base 100 (TWR)</p>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(port_cum.index,  port_cum.values,  lw=2, label="Portfolio (TWR)", color="#2979FF")
            ax.plot(bench_cum.index, bench_cum.values, lw=2, label=BENCHMARK_NAME, color="#FF9100", linestyle="--")
            ax.axhline(100, color="white", linestyle=":", lw=0.8, alpha=0.5, label="Base 100")
            ax.fill_between(port_cum.index, port_cum.values, bench_cum.values,
                            where=(port_cum.values >= bench_cum.values), alpha=0.15, color="#00C853", label="Outperforming")
            ax.fill_between(port_cum.index, port_cum.values, bench_cum.values,
                            where=(port_cum.values < bench_cum.values),  alpha=0.15, color="#FF1744", label="Underperforming")
            ax.set_title(f"Portfolio (TWR) vs {BENCHMARK_NAME} — Indexed Performance", fontsize=13, fontweight="bold")
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Indexed Return", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            apply_dark_theme(ax, fig)
            legend = ax.legend(fontsize=9)
            legend.get_frame().set_facecolor("#0E1117")
            for text in legend.get_texts():
                text.set_color("white")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Rolling Metrics</p>", unsafe_allow_html=True)
            rtab1, rtab2, rtab3 = st.tabs(["📈 Excess Return", "📊 Rolling Beta", "🔗 Rolling Correlation"])

            with rtab1:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(rolling_exc_cum.index, rolling_exc_cum.values, color="#2979FF", lw=1.5)
                ax.axhline(0, color="white", linestyle="--", lw=0.8)
                ax.fill_between(rolling_exc_cum.index, rolling_exc_cum.values, 0,
                                where=(rolling_exc_cum.values >= 0), alpha=0.2, color="#00C853")
                ax.fill_between(rolling_exc_cum.index, rolling_exc_cum.values, 0,
                                where=(rolling_exc_cum.values < 0),  alpha=0.2, color="#FF1744")
                ax.set_title(f"Cumulative Excess Return vs {BENCHMARK_NAME} (%)", fontsize=11, color="white")
                apply_dark_theme(ax, fig)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with rtab2:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(rolling_beta_bm.index, rolling_beta_bm.values, color="#FF9100", lw=1.5, label="Rolling Beta")
                ax.axhline(1, color="white", linestyle="--", lw=0.8, label="Beta = 1")
                ax.axhline(0, color="gray",  linestyle=":", lw=0.6, alpha=0.5)
                ax.set_title(f"Rolling {rolling_w}-Day Beta vs {BENCHMARK_NAME}", fontsize=11, color="white")
                legend = ax.legend(fontsize=8)
                legend.get_frame().set_facecolor("#0E1117")
                for t in legend.get_texts():
                    t.set_color("white")
                apply_dark_theme(ax, fig)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with rtab3:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(rolling_corr.index, rolling_corr.values, color="#5DCAA5", lw=1.5, label="Rolling Correlation")
                ax.axhline(0,  color="white", linestyle="--", lw=0.8)
                ax.axhline(1,  color="gray",  linestyle=":", lw=0.6, alpha=0.5)
                ax.axhline(-1, color="gray",  linestyle=":", lw=0.6, alpha=0.5)
                ax.set_ylim(-1.1, 1.1)
                ax.set_title(f"Rolling {rolling_w}-Day Correlation with {BENCHMARK_NAME}", fontsize=11, color="white")
                legend = ax.legend(fontsize=8)
                legend.get_frame().set_facecolor("#0E1117")
                for t in legend.get_texts():
                    t.set_color("white")
                apply_dark_theme(ax, fig)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Benchmark Interpretation</p>", unsafe_allow_html=True)
            if excess > 0:
                st.success(f"Portfolio **outperformed {BENCHMARK_NAME} by {excess:.2f}%** (TWR basis).")
            else:
                st.error(f"Portfolio **underperformed {BENCHMARK_NAME} by {abs(excess):.2f}%** (TWR basis).")
            if not np.isnan(capm_alpha):
                if capm_alpha > 0:
                    st.success(f"CAPM Alpha **{capm_alpha:+.2f}%** — genuine skill after Beta adjustment.")
                else:
                    st.warning(f"CAPM Alpha **{capm_alpha:+.2f}%** — excess return explained by market risk, not skill.")
            if not np.isnan(up_capture) and not np.isnan(down_capture):
                if up_capture > 100 and down_capture < 100:
                    st.success(f"Excellent — up {up_capture:.1f}%, down {down_capture:.1f}%. Ideal positioning.")
                elif up_capture < 100 and down_capture < 100:
                    st.info(f"Defensive — up {up_capture:.1f}%, down {down_capture:.1f}%. Protects capital.")
                elif up_capture > 100 and down_capture > 100:
                    st.warning(f"Aggressive — up {up_capture:.1f}%, down {down_capture:.1f}%. Amplifies both.")
                else:
                    st.error(f"Unfavourable — up {up_capture:.1f}%, down {down_capture:.1f}%. Missing gains, absorbing losses.")
            if not np.isnan(info_ratio):
                if info_ratio > 1.0:   st.success(f"IR {info_ratio:.2f} — excellent active management.")
                elif info_ratio > 0.5: st.info(f"IR {info_ratio:.2f} — good consistent excess return.")
                elif info_ratio > 0:   st.warning(f"IR {info_ratio:.2f} — marginal active return.")
                else:                  st.error(f"IR {info_ratio:.2f} — passive index more efficient.")
        else:
            st.warning(f"Benchmark data ({BENCHMARK_NAME}) could not be fetched.")

        st.markdown("<hr style='border:none;border-top:1px solid #21262D;margin:24px 0'>", unsafe_allow_html=True)



    ## ── TAB 3 — ATTRIBUTION ────────────────────────────────────────────────────
    with tab3:
        ## ═════════════════════════════════════════════════════════════════════════
        ##  SECTION 3 — ATTRIBUTION ANALYSIS
        ## ═════════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header"><h3>&#9632; Section 3 — Attribution Analysis</h3><p>Return contribution, BHB sector attribution, stock vs sector comparison</p></div>', unsafe_allow_html=True)


        attr_df                  = pd.DataFrame({"Ticker": tickers, "Weight": weights})
        attr_df["P&L"]           = attr_df["Ticker"].map(pnl_map)
        attr_df["Current Value"] = attr_df["Ticker"].map(value_map)
        attr_df["Invested"]      = attr_df["Ticker"].map(invested_map)
        attr_df                  = attr_df.fillna(0)

        attr_df["Stock Return"] = np.where(
            attr_df["Invested"] > 0,
            attr_df["P&L"] / attr_df["Invested"],
            0.0
        )
        attr_df["Return Contribution"] = attr_df["Weight"] * attr_df["Stock Return"]

        _total_rc = attr_df["Return Contribution"].sum()
        attr_df["P&L Contribution %"] = np.where(
            abs(_total_rc) > 0.0001,
            (attr_df["Return Contribution"] / abs(_total_rc)) * 100,
            0.0
        )
        attr_df = attr_df.sort_values("Return Contribution", ascending=False).reset_index(drop=True)
        ## Store for PDF — attr_df only exists when user visits Attribution tab
        st.session_state["_pdf_attr_df"] = attr_df.copy()

        best_stock  = attr_df.iloc[0]
        worst_stock = attr_df.iloc[-1]

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Biggest Contributor", best_stock["Ticker"],
                  f"{best_stock['Return Contribution']:.2%}")
        a2.metric("Biggest Drag", worst_stock["Ticker"],
                  f"{worst_stock['Return Contribution']:.2%}", delta_color="inverse")
        a3.metric("Positive Contributors",
                  f"{(attr_df['P&L'] > 0).sum()} of {len(attr_df)}")
        a4.metric("Concentration Risk",
                  f"{attr_df['Weight'].max():.1%}",
                  delta="High" if attr_df["Weight"].max() > 0.40 else "Acceptable",
                  delta_color="inverse" if attr_df["Weight"].max() > 0.40 else "normal")

        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Stock-Level Attribution</p>", unsafe_allow_html=True)
        display_df = attr_df[[
            "Ticker", "Weight", "Stock Return",
            "Return Contribution", "P&L", "P&L Contribution %"
        ]].copy().rename(columns={
            "Stock Return":        "Return %",
            "Return Contribution": "Return Contrib.",
            "P&L Contribution %":  "P&L Contrib. %"
        })

        st.dataframe(
            display_df.style
            .format({
                "Weight":          "{:.2%}",
                "Return %":        "{:.2%}",
                "Return Contrib.": "{:.2%}",
                "P&L":             "₹{:,.2f}",
                "P&L Contrib. %":  "{:+.1f}%"
            })
            .background_gradient(subset=["Return Contrib."], cmap="RdYlGn", vmin=-0.05, vmax=0.05)
            .background_gradient(subset=["P&L Contrib. %"],  cmap="RdYlGn", vmin=-100,  vmax=100),
            width='stretch'
        )

        ## Waterfall
        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>P&L Waterfall</p>", unsafe_allow_html=True)
        waterfall_df = attr_df.sort_values("P&L", ascending=False).copy()
        pnl_vals_sorted = waterfall_df["P&L"].tolist() + [total_pnl]
        x_labels = waterfall_df["Ticker"].tolist() + ["Net Result"]
        ## Standalone bars — each from zero, positive up, negative down
        bar_colors = ["#1D9E75" if v >= 0 else "#E24B4A"
                      for v in waterfall_df["P&L"]] + ["#2979FF"]

        fig, ax = plt.subplots(figsize=(9, 4.2))
        bars = ax.bar(range(len(x_labels)), pnl_vals_sorted,
                      color=bar_colors, edgecolor="black", linewidth=0.8, width=0.6, zorder=3)
        y_max = max(abs(v) for v in pnl_vals_sorted) * 0.04
        for bar, val in zip(bars, pnl_vals_sorted):
            y_pos = val + y_max if val >= 0 else val - y_max
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"₹{val:+.0f}", ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=9, color="white")
        ax.axhline(0, color="white", linewidth=0.8, alpha=0.5, zorder=2)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel("P&L (₹)", fontsize=10)
        ax.set_title("P&L Attribution — Waterfall Chart", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
        legend = ax.legend(handles=[
            Patch(facecolor="#1D9E75", label="Positive"),
            Patch(facecolor="#E24B4A", label="Negative"),
            Patch(facecolor="#2979FF", label="Net result"),
        ], fontsize=9)
        legend.get_frame().set_facecolor("#0E1117")
        for text in legend.get_texts():
            text.set_color("white")
        apply_dark_theme(ax, fig)
        ax.tick_params(axis="x", colors="white", labelsize=10)
        ax.tick_params(axis="y", colors="white", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        ## Return contribution
        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Return Contribution (Weight × Stock Return)</p>", unsafe_allow_html=True)
        rc_sorted = attr_df.sort_values("Return Contribution", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        rc_colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in rc_sorted["Return Contribution"]]
        bars = ax.barh(rc_sorted["Ticker"], rc_sorted["Return Contribution"] * 100,
                       color=rc_colors, edgecolor="black", linewidth=0.8, height=0.6)
        for bar, val in zip(bars, rc_sorted["Return Contribution"] * 100):
            x_lbl = bar.get_width()
            ax.text(x_lbl + (0.05 if val >= 0 else -0.05),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}%", ha="left" if val >= 0 else "right",
                    va="center", fontsize=9, color="white")
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Return Contribution (%)", fontsize=10)
        ax.set_title("Return Contribution per Stock", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        apply_dark_theme(ax, fig)
        ax.tick_params(colors="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        ## Attribution interpretation
        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Attribution Interpretation</p>", unsafe_allow_html=True)
        worst_rc     = attr_df.loc[attr_df["Return Contribution"].idxmin()]
        best_rc      = attr_df.loc[attr_df["Return Contribution"].idxmax()]
        concentrated = attr_df[attr_df["Weight"] > 0.40]

        if total_pnl > 0:
            st.success(f"Portfolio up **₹{total_pnl:,.2f}**. "
                       f"**{best_rc['Ticker']}** is the biggest driver "
                       f"({best_rc['Return Contribution']:.2%} contribution).")
        else:
            st.error(f"Portfolio down **₹{abs(total_pnl):,.2f}**. "
                     f"**{worst_rc['Ticker']}** is the biggest drag "
                     f"({worst_rc['Return Contribution']:.2%} contribution).")

        if not concentrated.empty:
            for _, crow in concentrated.iterrows():
                st.warning(
                    f"**{crow['Ticker']}** at **{crow['Weight']:.1%}** weight. "
                    f"Concentrated position amplifies both gains and losses. "
                    f"Contributed **{crow['Return Contribution']:.2%}** to portfolio return."
                )

        gainers = attr_df[attr_df["P&L"] > 0]
        losers  = attr_df[attr_df["P&L"] < 0]
        if not gainers.empty and not losers.empty:
            st.info(
                f"**Gaining:** {', '.join(gainers['Ticker'].tolist())} "
                f"(combined ₹{gainers['P&L'].sum():,.2f})  |  "
                f"**Losing:** {', '.join(losers['Ticker'].tolist())} "
                f"(combined ₹{losers['P&L'].sum():,.2f})"
            )

        ## ── SECTOR ATTRIBUTION — BHB MODEL ───────────────────────────────────────
        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Sector Attribution — Brinson-Hood-Beebower Model</p>", unsafe_allow_html=True)
        st.caption(
            "Sector return from each stock's own buy date. "
            "ETFs treated as sector allocation decisions."
        )

        with st.spinner("Fetching sector data..."):
            sector_rows = []

            for _, row in holdings_df.iterrows():
                t        = row["Ticker"]
                buy_date = pd.to_datetime(row["Buy Date"])

                stock_ret_arr = attr_df.loc[attr_df["Ticker"] == t, "Stock Return"].values
                if len(stock_ret_arr) == 0:
                    continue
                stock_ret = float(stock_ret_arr[0])
                weight    = float(weight_map.get(t, 0))

                ## ── ETF Detection: check enriched_info first, then ETF_DEFINITIONS ──
                _enriched = enriched_info.get(
                    t, enriched_info.get(t.replace(".NS","").replace(".BO",""), {})
                )
                _etf_info_enriched = _enriched.get("etf_info", None)

                ## Also check manual ETF_DEFINITIONS override
                if t in ETF_DEFINITIONS:
                    _etf_info_enriched = ETF_DEFINITIONS[t]

                if _etf_info_enriched and _etf_info_enriched.get("is_etf"):
                    etf_info = _etf_info_enriched
                    sector   = etf_info.get("sector", "Unknown ETF")
                    industry = etf_info.get("industry", f"ETF — {t}")
                    is_etf   = True
                    etf_tier = etf_info.get("etf_tier", 4)
                else:
                    is_etf   = False
                    etf_tier = 0
                    ## Priority 1: use sector already fetched by enrich_info
                    ## (avoids duplicate Yahoo API call and uses cached result)
                    sector   = _enriched.get("sector", None)
                    industry = _enriched.get("display_name", t)

                    ## Priority 2: fresh fetch if enrich_info didn't have it
                    if sector in ["Unknown", "N/A", "", None]:
                        _info = fetch_sector_info(t)
                        sector   = _info.get("sector", "Unknown")
                        industry = _info.get("industry", "Unknown")

                    ## Priority 3: manual override map for known problem tickers
                    if sector in ["Unknown", "N/A", "", None]:
                        sector = MANUAL_SECTOR_MAP.get(t, MANUAL_SECTOR_MAP.get(
                            t.replace(".NS","").replace(".BO",""), "N/A"
                        ))
                    if industry in ["Unknown", "N/A", "", None]:
                        industry = "N/A"

                ## ── Sector return: tiered logic ──────────────────────────────
                sector_ret    = np.nan
                sector_index  = None
                _attr_bm_label = ""

                if is_etf:
                    etf_bm = (etf_info or {}).get("attribution_bm", None)
                    _attr_bm_label = (etf_info or {}).get("attribution_bm_label", "")

                    if etf_tier == 4 or etf_bm is None:
                        ## Tier 4 or unknown → exclude from BHB
                        sector_ret = np.nan
                    elif etf_tier == 2:
                        ## Tier 2 broad market → ETF tracks benchmark index
                        ## Fetch benchmark return from the ETF's buy_date to today
                        sector_ret = fetch_sector_return(
                            BENCHMARK_TICKER, buy_date, portfolio_end_date
                        )
                        ## If fetch fails, try Nifty 500 as fallback
                        if np.isnan(sector_ret):
                            sector_ret = fetch_sector_return(
                                "NIFTYMIDCAP150.NS", buy_date, portfolio_end_date
                            )
                    else:
                        ## Tier 1 (sector) or Tier 3 (thematic proxy)
                        sector_ret = fetch_sector_return(
                            etf_bm, buy_date, portfolio_end_date
                        )
                        if np.isnan(sector_ret):
                            ## fallback to Nifty 500 if sector index fails
                            sector_ret = fetch_sector_return(
                                "NIFTYMIDCAP150.NS", buy_date, portfolio_end_date
                            )
                else:
                    ## Regular stock — existing logic unchanged
                    sector_index = SECTOR_INDEX_TICKERS.get(sector, None)
                    if sector_index:
                        sector_ret = fetch_sector_return(
                            sector_index, buy_date, portfolio_end_date
                        )
                        if np.isnan(sector_ret):
                            sector_ret = fetch_sector_return(
                                BENCHMARK_TICKER, buy_date, portfolio_end_date
                            )

                ## Per-holding benchmark return — SAME date range as sector_ret
                ## Ensures Allocation Effect uses consistent periods for all holdings
                ## For Tier 2 ETF: bm_ret_holding = sector_ret → Allocation = exactly 0
                bm_ret_holding = fetch_sector_return(
                    BENCHMARK_TICKER, buy_date, portfolio_end_date
                )

                ## Benchmark weight — fetched from live sector weights, NOT hardcoded
                bench_weight = NIFTY_SECTOR_WEIGHTS.get(sector, 0.0)

                ## ETF tier label for display
                _etf_type_label = "Stock"
                if is_etf:
                    _tier_labels = {
                        1: "ETF (Sector)",
                        2: "ETF (Broad)",
                        3: "ETF (Thematic)",
                        4: "ETF (Excluded)",
                    }
                    _etf_type_label = _tier_labels.get(etf_tier, "ETF")

                sector_rows.append({
                    "Ticker":         t,
                    "Type":           _etf_type_label,
                    "Sector":         sector if sector not in ["Unknown", None] else "N/A",
                    "Industry":       industry,
                    "Weight (Port)":  weight,
                    f"Weight ({BENCHMARK_NAME})": bench_weight,
                    "Active Weight":  weight - bench_weight,
                    "Stock Return":   stock_ret,
                    "Sector Return":  sector_ret,
                    "BM Return":      bm_ret_holding,
                    "Difference":     (stock_ret - sector_ret)
                                      if not np.isnan(sector_ret) else np.nan,
                    "Buy Date":       buy_date.date(),
                    "Is ETF":         is_etf,
                    "ETF Tier":       etf_tier if is_etf else 0,
                    "Attribution BM": _attr_bm_label if is_etf else "",
                })

        sector_df = pd.DataFrame(sector_rows)

        try:
            bm_return = bench_total / 100
        except NameError:
            bm_return = 0.0

        if not sector_df.empty:
            ## Use per-holding BM Return so sector_ret and bm_ret share the same date range
            ## For Tier 2 ETF: sector_ret == bm_ret_holding → Allocation = exactly 0
            _bm_ref = sector_df["BM Return"].fillna(bm_return) if "BM Return" in sector_df.columns else bm_return
            sector_df["Allocation Effect"] = np.where(
                sector_df["Sector Return"].notna(),
                sector_df["Active Weight"] * (sector_df["Sector Return"] - _bm_ref),
                np.nan
            )
            sector_df["Selection Effect"] = np.where(
                sector_df["Sector Return"].notna(),
                sector_df[f"Weight ({BENCHMARK_NAME})"] * (sector_df["Stock Return"] - sector_df["Sector Return"]),
                np.nan
            )
            sector_df["Interaction Effect"] = np.where(
                sector_df["Sector Return"].notna(),
                sector_df["Active Weight"] * (sector_df["Stock Return"] - sector_df["Sector Return"]),
                np.nan
            )
            sector_df["Total Active Return"] = (
                sector_df["Allocation Effect"].fillna(0) +
                sector_df["Selection Effect"].fillna(0) +
                sector_df["Interaction Effect"].fillna(0)
            )
            ## Store for PDF — sector_df only exists when user visits Attribution tab
            st.session_state["_pdf_sector_df"] = sector_df.copy()

            ## Stock vs Sector table
            st.markdown("<p style='font-size:12px;font-weight:600;color:#C9D1D9;margin:12px 0 6px 0'>Stock vs Sector Comparison</p>", unsafe_allow_html=True)
            ## Build dynamic caption based on what is actually in the portfolio
            _has_tier2 = (
                'sector_df' in dir() and sector_df is not None and
                not sector_df.empty and "ETF Tier" in sector_df.columns and
                (sector_df["ETF Tier"] == 2).any()
            )
            _has_tier4 = (
                'sector_df' in dir() and sector_df is not None and
                not sector_df.empty and "ETF Tier" in sector_df.columns and
                (sector_df["ETF Tier"] == 4).any()
            )
            _has_any_etf = (
                'sector_df' in dir() and sector_df is not None and
                not sector_df.empty and "Is ETF" in sector_df.columns and
                sector_df["Is ETF"].any()
            )
            ## Get earliest stock (non-ETF) buy date for the example
            _earliest_ticker = ""
            _earliest_date   = ""
            if 'sector_df' in dir() and sector_df is not None and not sector_df.empty:
                _stocks_df = sector_df[~sector_df["Is ETF"]] if "Is ETF" in sector_df.columns else sector_df
                if not _stocks_df.empty and "Buy Date" in _stocks_df.columns:
                    _ei = _stocks_df.loc[_stocks_df["Buy Date"].idxmin()]
                    _earliest_ticker = str(_ei["Ticker"]).replace(".NS","").replace(".BO","")
                    _earliest_date   = str(_ei["Buy Date"])

            _caption_parts = [
                "📌 **BHB Methodology Note:** "
                "All returns measured from each holding's own buy date — not the portfolio start date. "
                + (
                    f"For example: **{_earliest_ticker}** (bought {_earliest_date}) is compared "
                    f"against its sector from {_earliest_date}, not from when a later holding was bought. "
                    if _earliest_ticker else
                    "This ensures like-for-like comparison across all holdings. "
                )
            ]
            if _has_tier2:
                ## Get actual Tier 2 ETF names
                _t2_names = ", ".join(
                    sector_df[sector_df["ETF Tier"]==2]["Ticker"]
                    .str.replace(".NS","").str.replace(".BO","").tolist()
                )
                _caption_parts.append(
                    f"**Broad Market ETFs ({_t2_names}):** Allocation Effect = 0 — "
                    "holding the benchmark index itself is not an active allocation decision (professional BHB standard). "
                )
            if _has_tier4:
                _t4_names = ", ".join(
                    sector_df[sector_df["ETF Tier"]==4]["Ticker"]
                    .str.replace(".NS","").str.replace(".BO","").tolist()
                )
                _caption_parts.append(
                    f"**Non-Equity ETFs ({_t4_names}):** Fully excluded — no equity benchmark applies. "
                )
            if _has_any_etf:
                _caption_parts.append(
                    "**All ETFs:** Selection & Interaction Effect = N/A — passive vehicles make no stock selection."
                )

            st.caption("".join(_caption_parts))

            def color_diff(val):
                if pd.isna(val): return ""
                if val > 0.05:    return "color: #1D9E75; font-weight: bold"
                elif val < -0.05: return "color: #E24B4A; font-weight: bold"
                else:             return "color: #FF9100"

            st.dataframe(
                sector_df[[
                    "Ticker", "Type", "Sector", "Industry", "Buy Date",
                    "Stock Return", "Sector Return", "Difference"
                ]].style
                .format({
                    "Stock Return":  "{:.2%}",
                    "Sector Return": lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A",
                    "Difference":    lambda x: f"{x:+.2%}" if not pd.isna(x) else "N/A",
                })
                .map(color_diff, subset=["Difference"])
                .background_gradient(subset=["Stock Return"], cmap="RdYlGn", vmin=-0.3, vmax=0.3),
                width='stretch'
            )

            ## BHB table
            st.markdown("<p style='font-size:12px;font-weight:600;color:#C9D1D9;margin:12px 0 6px 0'>BHB Attribution Decomposition</p>", unsafe_allow_html=True)
            st.caption(
                f"Allocation = overweight right sector vs {BENCHMARK_NAME} | "
                "Selection = right stock within sector | "
                "Interaction = both combined"
            )
            st.dataframe(
                sector_df[[
                    "Ticker", "Type", "Sector",
                    "Weight (Port)", f"Weight ({BENCHMARK_NAME})", "Active Weight",
                    "Allocation Effect", "Selection Effect",
                    "Interaction Effect", "Total Active Return"
                ]].style
                .format({
                    "Weight (Port)":      "{:.2%}",
                    f"Weight ({BENCHMARK_NAME})":     "{:.2%}",
                    "Active Weight":      "{:+.2%}",
                    "Allocation Effect":  lambda x: f"{x:+.2%}" if not pd.isna(x) else "N/A",
                    "Selection Effect":   lambda x: f"{x:+.2%}" if not pd.isna(x) else "N/A",
                    "Interaction Effect": lambda x: f"{x:+.2%}" if not pd.isna(x) else "N/A",
                    "Total Active Return":lambda x: f"{x:+.2%}" if not pd.isna(x) else "N/A",
                })
                .background_gradient(subset=["Total Active Return"], cmap="RdYlGn", vmin=-0.05, vmax=0.05)
                .background_gradient(subset=["Active Weight"],       cmap="RdYlGn", vmin=-0.3,  vmax=0.3),
                width='stretch'
            )

            ## BHB summary metrics
            total_alloc    = sector_df["Allocation Effect"].sum(skipna=True)
            total_select   = sector_df["Selection Effect"].sum(skipna=True)
            total_interact = sector_df["Interaction Effect"].sum(skipna=True)
            total_active   = total_alloc + total_select + total_interact

            sa1, sa2, sa3, sa4 = st.columns(4)
            sa1.metric("Allocation Effect",   f"{total_alloc:+.2%}",
                       help="Return from over/underweighting sectors vs Nifty 50")
            sa2.metric("Selection Effect",    f"{total_select:+.2%}",
                       help="Return from picking stocks that beat their sector")
            sa3.metric("Interaction Effect",  f"{total_interact:+.2%}",
                       help="Combined allocation and selection effect")
            sa4.metric("Total Active Return", f"{total_active:+.2%}",
                       help="Sum of all three BHB effects")

            ## BHB interpretation
            st.markdown("<p style='font-size:12px;font-weight:600;color:#C9D1D9;margin:12px 0 6px 0'>BHB Interpretation</p>", unsafe_allow_html=True)
            if total_active > 0:
                st.success(f"Total active return **{total_active:+.2%}** vs {BENCHMARK_NAME}. "
                           f"Portfolio decisions generated positive excess return.")
            else:
                st.error(f"Total active return **{total_active:+.2%}** vs {BENCHMARK_NAME}. "
                         f"A passive Nifty index fund would have performed better.")

            dominant     = "stock selection" if abs(total_select) > abs(total_alloc) else "sector allocation"
            dominant_val = total_select      if abs(total_select) > abs(total_alloc) else total_alloc
            if dominant_val > 0:
                st.info(
                    f"Primary driver: **{dominant}** ({dominant_val:+.2%}). "
                    f"{'Strong stock-picking within sectors.' if dominant == 'stock selection' else 'Effective sector allocation vs Nifty 50.'}"
                )
            else:
                st.warning(
                    f"Primary drag: **{dominant}** ({dominant_val:+.2%}). "
                    f"{'Stock picks underperformed their sector benchmarks.' if dominant == 'stock selection' else 'Overweight in underperforming sectors.'}"
                )

            ## Per-ticker interpretation
            for _, row in sector_df.iterrows():
                ticker_name = row["Ticker"]
                is_etf      = row.get("Is ETF", False)

                if is_etf:
                    _tier     = row.get("ETF Tier", 4)
                    _attr_bm  = row.get("Attribution BM", "")
                    _tracks   = (ETF_DEFINITIONS.get(ticker_name, {}).get("tracks")
                                 or enriched_info.get(ticker_name, {}).get("etf_info", {}).get("tracks","an index")
                                 if isinstance(enriched_info.get(ticker_name, {}).get("etf_info"), dict)
                                 else "an index")

                    if _tier == 4:
                        st.warning(
                            f"**{ticker_name}** — ETF not mapped to an attribution benchmark. "
                            f"Position excluded from BHB analysis. "
                            f"(Could be a non-equity ETF such as Gold, Bond, or International.)"
                        )
                    elif _tier == 2:
                        st.info(
                            f"**{ticker_name}** is a **Broad Market ETF** tracking *{_tracks}*. "
                            f"This ETF tracks the portfolio benchmark — active contribution is near zero. "
                            f"Selection Effect: N/A. Interaction Effect: N/A."
                        )
                    elif _tier == 1:
                        st.info(
                            f"**{ticker_name}** is a **Sector ETF** tracking *{_tracks}*. "
                            f"Attribution Benchmark: **{_attr_bm}** (direct sector index). "
                            f"Selection Effect: N/A — ETF tracks index mechanically. "
                            f"Allocation Effect shows whether this sector bet added value vs {BENCHMARK_NAME}."
                        )
                    else:
                        ## Tier 3 — thematic
                        st.info(
                            f"**{ticker_name}** is a **Thematic ETF** tracking *{_tracks}*. "
                            f"Attribution Benchmark: **{_attr_bm}** — "
                            f"no direct index available on Yahoo Finance for this theme. "
                            f"Selection Effect: N/A. Allocation Effect measured vs broad market proxy."
                        )
                    continue

                if pd.isna(row["Difference"]):
                    st.info(f"**{ticker_name}** ({row['Sector']}) — "
                            f"sector benchmark data unavailable for comparison.")
                    continue

                diff   = row["Difference"]
                sr     = row["Stock Return"]
                secr   = row["Sector Return"]
                sector_name = row["Sector"]

                if diff < -0.10:
                    st.error(
                        f"**{ticker_name}** ({sector_name}) — stock {sr:.2%} vs sector {secr:.2%} "
                        f"(diff: {diff:+.2%}). **Stock-specific underperformance** — not a sector issue."
                    )
                elif diff > 0.10:
                    st.success(
                        f"**{ticker_name}** ({sector_name}) — stock {sr:.2%} vs sector {secr:.2%} "
                        f"(diff: {diff:+.2%}). **Stock-specific alpha** — outperformed its sector."
                    )
                elif sr < 0 and secr < 0:
                    st.info(
                        f"**{ticker_name}** ({sector_name}) — both stock ({sr:.2%}) and "
                        f"sector ({secr:.2%}) declined. **Sector-wide decline** — not stock-specific."
                    )
                else:
                    st.info(
                        f"**{ticker_name}** ({sector_name}) — {sr:.2%} vs sector {secr:.2%} "
                        f"(diff: {diff:+.2%}). Performance broadly in line with sector."
                    )

        st.markdown("<hr style='border:none;border-top:1px solid #21262D;margin:24px 0'>", unsafe_allow_html=True)



    ## ── TAB 4 — RISK & VaR ─────────────────────────────────────────────────────
    with tab4:
        ## ═════════════════════════════════════════════════════════════════════════
        ##  SECTION 6 — RISK ANALYTICS
        ## ═════════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header"><h3>&#9632; Section 6 — Risk Analytics</h3><p>Return distribution, drawdown analysis, annual returns</p></div>', unsafe_allow_html=True)

        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Return Distribution</p>", unsafe_allow_html=True)
        plot_kde(port_returns_bm)

        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Drawdown Analysis</p>", unsafe_allow_html=True)
        max_dd, recovery_date = plot_drawdown(port_returns_bm)  ## TWR — starts from earliest buy date
        dd1, dd2 = st.columns(2)
        dd1.metric("Max Drawdown", f"{max_dd:.2%}")
        dd2.metric("Recovery",
                   f"{recovery_date.strftime('%b %Y')}" if recovery_date else "Not yet recovered")

        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Annual Returns</p>", unsafe_allow_html=True)
        annual_returns       = (1 + port_returns_bm).resample("YE").prod() - 1
        annual_returns.index = annual_returns.index.year
        fig, ax = plt.subplots(figsize=(10, 4))
        colors  = ["#00C853" if r >= 0 else "#FF1744" for r in annual_returns.values]
        bars    = ax.bar(annual_returns.index.astype(str), annual_returns.values * 100,
                         color=colors, edgecolor="White", linewidth=1.2, alpha=0.9)
        for bar, value in zip(bars, annual_returns.values):
            height = bar.get_height()
            if value >= 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f"{value*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="white")
            else:
                ax.text(bar.get_x() + bar.get_width()/2, height - 1,
                        f"{value*100:.1f}%", ha="center", va="top", fontsize=11, fontweight="bold", color="white")
        ax.axhline(0, color="white", linewidth=1)
        ax.set_title("Portfolio Returns (Annual)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Year", fontsize=11)
        ax.set_ylabel("Annual Return (%)", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        apply_dark_theme(ax, fig)
        ax.tick_params(axis="x", labelsize=11, colors="white")
        ax.tick_params(axis="y", labelsize=11, colors="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("<hr style='border:none;border-top:1px solid #21262D;margin:24px 0'>", unsafe_allow_html=True)


        ## ═════════════════════════════════════════════════════════════════════════
        ##  SECTION 7 — VaR & ES
        ## ═════════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header"><h3>&#9632; Section 7 — Value at Risk &amp; Expected Shortfall</h3><p>Historical, Parametric, Monte Carlo — selected methods only</p></div>', unsafe_allow_html=True)
        st.markdown(f"""
    <div style="font-size:12px;color:#8B949E;margin-bottom:12px;padding:8px 12px;
    background:#161B22;border-radius:8px;border:1px solid #21262D">
      Confidence: <span style="color:#C8A951">{alpha:.0%}</span>
      &nbsp;|&nbsp; Horizon: <span style="color:#C8A951">{horizon_days} day(s)</span>
      &nbsp;|&nbsp; Portfolio value: <span style="color:#C8A951">&#8377;{portfolio_value:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)

        results            = []
        distribution_plots = []
        T                  = horizon_days
        mu                 = port_returns_bm.mean()
        sigma              = port_returns_bm.std()
        z                  = norm.ppf(1 - alpha)
        mu_h               = mu * T
        sigma_h            = sigma * np.sqrt(T)

        horizon_returns = (
            (1 + port_returns_bm).rolling(T).apply(np.prod, raw=True) - 1
        ).dropna()

        if use_hist:
            var_val = historical_var(horizon_returns, alpha)
            es_val  = historical_es(horizon_returns, alpha)
            results.append(["Historical", var_val, var_val*portfolio_value, es_val, es_val*portfolio_value])
            distribution_plots.append(("Historical", horizon_returns, var_val, es_val, None, None))

        if use_para:
            var_val = -(mu_h + z * sigma_h)
            es_val  = -(mu_h - sigma_h * (norm.pdf(z) / (1 - alpha)))
            results.append(["Parametric Method", var_val, var_val*portfolio_value, es_val, es_val*portfolio_value])
            distribution_plots.append(("Parametric Method", None, var_val, es_val, mu_h, sigma_h))

        if use_mc:
            np.random.seed(42)
            sim_daily           = np.random.normal(mu, sigma, (mc_sims, T))
            sim_horizon_returns = (1 + sim_daily).prod(axis=1) - 1
            var_val             = float(-np.percentile(sim_horizon_returns, (1-alpha)*100))
            cutoff              = np.percentile(sim_horizon_returns, (1-alpha)*100)
            es_val              = float(-sim_horizon_returns[sim_horizon_returns <= cutoff].mean())
            results.append(["Monte Carlo Simulation", var_val, var_val*portfolio_value, es_val, es_val*portfolio_value])
            distribution_plots.append(("Monte Carlo Simulation", sim_horizon_returns, var_val, es_val, None, None))

        risk_df = pd.DataFrame(
            results,
            columns=["Method", "VaR %", "VaR Amount (Rs.)", "ES %", "ES Amount (Rs.)"]
        )
        ## Store for PDF — risk_df only exists when user visits Risk & VaR tab
        st.session_state["_pdf_risk_df"] = risk_df.copy()

        if risk_df.empty:
            st.markdown("""
<div style="padding:12px 16px;background:#161B22;border:1px solid #C8A951;
border-left:4px solid #C8A951;border-radius:0 8px 8px 0;font-size:13px;color:#C9D1D9">
  <strong style="color:#C8A951">Select a VaR method in the sidebar</strong>
  &nbsp; — tick Historical, Parametric, or Monte Carlo to see risk analysis.
</div>
""", unsafe_allow_html=True)

        st.dataframe(
            risk_df.style.format({
                "VaR %":            "{:.2%}",
                "ES %":             "{:.2%}",
                "VaR Amount (Rs.)": "₹{:,.2f}",
                "ES Amount (Rs.)":  "₹{:,.2f}",
            }),
            width='stretch'
        )

        if len(risk_df) > 0:
            r0 = risk_df.iloc[0]
            st.info(
                f"At **{alpha:.0%} confidence** over **{horizon_days} day(s)**: "
                f"there is a {1-alpha:.0%} probability of losing more than "
                f"**₹{r0['VaR Amount (Rs.)']:,.2f}** ({r0['VaR %']:.2%}). "
                f"If breached, average expected loss = "
                f"**₹{r0['ES Amount (Rs.)']:,.2f}** ({r0['ES %']:.2%})."
            )

        ncols = min(3, len(distribution_plots))
        cols  = st.columns(ncols)
        for i, (method_name, hr, var_val, es_val, mh, sh) in enumerate(distribution_plots):
            with cols[i % ncols]:
                plot_var_distribution(hr, var_val, es_val, method_name, horizon_days, mh, sh)
        st.markdown("<hr style='border:none;border-top:1px solid #21262D;margin:24px 0'>", unsafe_allow_html=True)



    ## ── TAB 5 — STRESS TEST ────────────────────────────────────────────────────
    with tab5:
        st.markdown('<div class="section-header"><h3>&#9632; Stress Test — Portfolio Shock Analysis</h3><p>Estimate portfolio impact under hypothetical market and stock-specific shock scenarios</p></div>', unsafe_allow_html=True)

        ## ── Context banner ───────────────────────────────────────────────────
        st.markdown(f"""
<div style="padding:10px 14px;background:#161B22;border:1px solid #21262D;border-left:4px solid #C8A951;
border-radius:0 8px 8px 0;font-size:12px;color:#C9D1D9;margin-bottom:16px">
  <strong style="color:#C8A951">How this works:</strong>
  &nbsp; Market shocks use your portfolio Beta ({portfolio_beta:.4f}) to estimate portfolio impact.
  &nbsp; Stock shocks apply directly to each holding's weight.
  &nbsp; All results are estimates — actual impact depends on real-time correlations.
</div>
""", unsafe_allow_html=True)

        ## ══════════════════════════════════════════════════════════════════════
        ## TYPE 1 — MARKET SHOCK SCENARIOS (Nifty 50 shocks via Beta)
        ## ══════════════════════════════════════════════════════════════════════
        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Type 1 — Market Shock Scenarios (Nifty 50)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px;color:#8B949E;margin-bottom:12px'>How much would your portfolio lose if Nifty 50 dropped by these amounts? Uses Beta = " + f"{portfolio_beta:.4f}" + " to estimate portfolio sensitivity.</p>", unsafe_allow_html=True)

        market_shocks = [-0.05, -0.10, -0.15, -0.20, -0.30, -0.40, -0.50]
        shock_labels  = ["−5%", "−10%", "−15%", "−20% Nifty Severe", "−30% Nifty Crash", "−40% Nifty Crisis", "−50% Nifty Black Swan"]
        shock_colors  = ["#FFB74D", "#FF9100", "#FF6D00", "#FF1744", "#D50000", "#B71C1C", "#7F0000"]

        mkt_rows = []
        for shock, label in zip(market_shocks, shock_labels):
            port_impact   = shock * portfolio_beta
            port_loss_rs  = port_impact * total_current
            new_value     = total_current + port_loss_rs
            mkt_rows.append({
                "Nifty Shock":       label,
                "Nifty Drop (%)":    f"{shock:.0%}",
                "Portfolio Impact (%)": f"{port_impact:.2%}",
                "Loss (₹)":          f"₹{port_loss_rs:,.0f}",
                "New Value (₹)":     f"₹{new_value:,.0f}",
            })


        mkt_df = pd.DataFrame(mkt_rows)

        ## Styled display
        def color_impact(val):
            try:
                v = float(val.replace('%','').replace('₹','').replace(',',''))
                if '₹' in val and v < 0:
                    return 'color: #F85149; font-weight: 600'
                if '%' in val and v < 0:
                    return 'color: #F85149; font-weight: 600'
            except:
                pass
            return ''

        st.dataframe(
            mkt_df.style.map(color_impact, subset=["Portfolio Impact (%)", "Loss (₹)"]),
            width='stretch', hide_index=True
        )

        ## Visual bar chart of market shocks
        fig_ms, ax_ms = plt.subplots(figsize=(12, 4))
        port_impacts_pct = [s * portfolio_beta * 100 for s in market_shocks]
        port_impacts_rs  = [s * portfolio_beta * total_current for s in market_shocks]
        bar_cols = shock_colors
        bars = ax_ms.barh(shock_labels, port_impacts_rs, color=bar_cols,
                          edgecolor="white", height=0.6)
        for bar, val_rs, val_pct in zip(bars, port_impacts_rs, port_impacts_pct):
            ## Label on the RIGHT side of the chart (outside the bar, past zero)
            ax_ms.text(4,
                       bar.get_y() + bar.get_height()/2,
                       f"₹{val_rs:,.0f}  ({val_pct:.1f}%)",
                       ha='left', va='center', fontsize=8.5,
                       color='#C9D1D9', fontweight='500')
        ax_ms.axvline(0, color='white', lw=0.8, alpha=0.4)
        ## Give extra space on the right side for labels
        cur_xlim = ax_ms.get_xlim()
        ax_ms.set_xlim(cur_xlim[0], abs(cur_xlim[0]) * 0.35)
        ax_ms.set_xlabel("Estimated Portfolio Loss (₹)", fontsize=10)
        ax_ms.set_title("Estimated Portfolio Loss by Nifty 50 Shock Scenario",
                        fontsize=12, fontweight='bold')
        apply_dark_theme(ax_ms, fig_ms)
        plt.tight_layout()
        st.pyplot(fig_ms)
        plt.close(fig_ms)

        ## ══════════════════════════════════════════════════════════════════════
        ## TYPE 2 — STOCK-SPECIFIC SHOCKS
        ## ══════════════════════════════════════════════════════════════════════
        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:20px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Type 2 — Stock-Specific Shock Scenarios</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px;color:#8B949E;margin-bottom:12px'>What if an individual stock drops sharply? Shows the direct portfolio impact of each holding falling by fixed percentages.</p>", unsafe_allow_html=True)

        stock_shocks = [-0.10, -0.20, -0.30]
        stock_cols   = st.columns(len(stock_shocks))

        for col_idx, shock in enumerate(stock_shocks):
            with stock_cols[col_idx]:
                st.markdown(f"<p style='font-size:12px;font-weight:600;color:#E6EDF3;text-align:center;margin-bottom:8px'>If stock drops {shock:.0%}</p>", unsafe_allow_html=True)
                rows = []
                for _, row in holdings_df.iterrows():
                    ticker  = row["Ticker"].replace(".NS","")
                    w       = row["Current Value"] / total_current
                    impact  = shock * w
                    loss_rs = shock * row["Current Value"]
                    rows.append({"Stock": ticker,
                                 "Weight": f"{w:.1%}",
                                 "Portfolio Impact": f"{impact:.2%}",
                                 "Loss (₹)": f"₹{loss_rs:,.0f}"})
                ## Total row
                total_impact = sum(shock * row["Current Value"] / total_current
                                   for _, row in holdings_df.iterrows())
                rows.append({"Stock": "TOTAL (all drop)",
                             "Weight": "100%",
                             "Portfolio Impact": f"{total_impact:.2%}",
                             "Loss (₹)": f"₹{total_impact*total_current:,.0f}"})

                df_s = pd.DataFrame(rows)
                st.dataframe(df_s.style.map(
                    lambda v: 'color:#F85149;font-weight:600' if (
                        isinstance(v, str) and v.startswith('-')) else '',
                    subset=["Portfolio Impact", "Loss (₹)"]
                ), width='stretch', hide_index=True)

        ## ══════════════════════════════════════════════════════════════════════
        ## TYPE 3 — HISTORICAL SCENARIOS
        ## ══════════════════════════════════════════════════════════════════════
        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:20px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Type 3 — Historical Market Crash Scenarios</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px;color:#8B949E;margin-bottom:12px'>How would your current portfolio have fared during past market crashes? Estimated using Nifty 50 peak-to-trough decline × portfolio Beta.</p>", unsafe_allow_html=True)

        historical_scenarios = [
            ("COVID Crash (NSE)",        "Feb–Mar 2020",      -0.385, "Nifty fell 38.5% in 40 days — fastest crash in NSE history. Circuit breakers triggered twice."),
            ("2008 Global Crisis (NSE)", "Jan–Oct 2008",      -0.605, "Nifty fell 60.5% in 9 months. FII outflows of ₹52,000 Cr. Worst bear market in NSE history."),
            ("2022 Rate Selloff (NSE)",  "Oct 2021–Jun 2022", -0.175, "Nifty fell 17.5%. RBI raised repo rate 250bps. FII sold ₹2.8 lakh Cr in Indian equities."),
            ("2024 Election Shock (NSE)","Apr–Jun 2024",      -0.062, "Nifty fell 6.2% on June 4. Unexpected coalition outcome. Recovered within 1 month."),
            ("2015–16 Correction (NSE)", "Mar 2015–Feb 2016",-0.261, "Nifty fell 26.1% over 11 months. Global commodity rout hit Indian metals and IT."),
            ("Dot-Com Bust (Sensex)",    "Feb 2000–Sep 2001", -0.560, "Sensex fell 56% over 19 months. Indian tech stocks crashed 70–90%. Pre-2008 worst."),
        ]

        hist_rows = []
        for event, period, nifty_drop, desc in historical_scenarios:
            port_impact  = nifty_drop * portfolio_beta
            loss_rs      = port_impact * total_current
            new_val      = total_current + loss_rs
            hist_rows.append({
                "Event":                event,
                "Period":               period,
                "Nifty 50 Drop":        f"{nifty_drop:.1%}",
                "Est. Portfolio Drop":  f"{port_impact:.1%}",
                "Est. Loss (₹)":        f"₹{loss_rs:,.0f}",
                "Portfolio Would Be":   f"₹{new_val:,.0f}",
                "Notes":                desc
            })

        hist_df = pd.DataFrame(hist_rows)
        st.dataframe(
            hist_df.style.map(
                lambda v: 'color:#F85149;font-weight:600' if (
                    isinstance(v, str) and v.startswith('-')) else '',
                subset=["Est. Portfolio Drop", "Est. Loss (₹)"]
            ),
            width='stretch', hide_index=True
        )

        ## Visual comparison chart
        fig_hs, ax_hs = plt.subplots(figsize=(12, 4.5))
        events       = [r["Event"] for r in hist_rows]
        nifty_drops  = [s[2]*100 for s in historical_scenarios]
        port_drops   = [s[2]*portfolio_beta*100 for s in historical_scenarios]
        x            = np.arange(len(events))
        w_bar        = 0.38
        b1 = ax_hs.bar(x - w_bar/2, nifty_drops,  width=w_bar, color="#C8A951",
                       alpha=0.85, label="Nifty 50 drop", edgecolor="white")
        b2 = ax_hs.bar(x + w_bar/2, port_drops,   width=w_bar, color="#FF1744",
                       alpha=0.85, label="Est. portfolio drop", edgecolor="white")
        for bar, val in zip(b1, nifty_drops):
            ax_hs.text(bar.get_x()+bar.get_width()/2, bar.get_height()-1.5,
                       f"{val:.0f}%", ha='center', va='top',
                       fontsize=8, color='white', fontweight='bold')
        for bar, val in zip(b2, port_drops):
            ax_hs.text(bar.get_x()+bar.get_width()/2, bar.get_height()-1.5,
                       f"{val:.0f}%", ha='center', va='top',
                       fontsize=8, color='white', fontweight='bold')
        ax_hs.axhline(0, color='white', lw=0.8, alpha=0.4)
        ax_hs.set_xticks(x)
        ax_hs.set_xticklabels(events, fontsize=9, rotation=12, ha='right')
        ax_hs.set_ylabel("Decline (%)", fontsize=10)
        ax_hs.set_title("Historical Crash Scenarios — Nifty 50 vs Estimated Portfolio Impact",
                        fontsize=12, fontweight='bold')
        legend = ax_hs.legend(fontsize=9, framealpha=0.85)
        legend.get_frame().set_facecolor("#0E1117")
        legend.get_frame().set_edgecolor("#21262D")
        for txt in legend.get_texts():
            txt.set_color("white")
        apply_dark_theme(ax_hs, fig_hs)
        plt.tight_layout()
        st.pyplot(fig_hs)
        plt.close(fig_hs)

        ## ══════════════════════════════════════════════════════════════════════
        ## CONCENTRATION STRESS — BHARTIARTL specific
        ## ══════════════════════════════════════════════════════════════════════
        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:20px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Concentration Risk — Largest Holding Stress</p>", unsafe_allow_html=True)

        ## Find the largest holding dynamically
        max_holding     = holdings_df.loc[holdings_df["Current Value"].idxmax()]
        max_ticker      = max_holding["Ticker"]
        max_weight      = max_holding["Current Value"] / total_current
        max_curr_val    = max_holding["Current Value"]

        st.markdown(f"""
<div style="padding:10px 14px;background:#1A0000;border:1px solid #F85149;border-left:4px solid #F85149;
border-radius:0 8px 8px 0;font-size:12px;color:#C9D1D9;margin-bottom:12px">
  <strong style="color:#F85149">&#9888; Concentration Alert:</strong>
  &nbsp; <strong>{max_ticker}</strong> is {max_weight:.1%} of your portfolio.
  Because of this concentration, a shock to this single stock has an outsized effect on the whole portfolio.
</div>
""", unsafe_allow_html=True)

        conc_shocks = [-0.05, -0.10, -0.15, -0.20, -0.30, -0.40, -0.50]
        conc_rows   = []
        for shock in conc_shocks:
            direct_impact = shock * max_weight
            loss_rs       = shock * max_curr_val
            new_total     = total_current + loss_rs
            conc_rows.append({
                f"{max_ticker.replace('.NS','')} Drop": f"{shock:.0%}",
                "Direct Portfolio Impact": f"{direct_impact:.2%}",
                "Loss (₹)":               f"₹{loss_rs:,.0f}",
                "Portfolio Value After":  f"₹{new_total:,.0f}",
            })

        conc_df = pd.DataFrame(conc_rows)
        st.dataframe(
            conc_df.style.map(
                lambda v: 'color:#F85149;font-weight:600' if (
                    isinstance(v, str) and v.startswith('-')) else '',
                subset=["Direct Portfolio Impact", "Loss (₹)"]
            ),
            width='stretch', hide_index=True
        )

        ## Disclaimer
        st.markdown("""
<div style="margin-top:16px;padding:10px 14px;background:#161B22;border:1px solid #21262D;
border-radius:8px;font-size:11px;color:#8B949E">
  <strong style="color:#C8A951">Important:</strong>
  Stress test results are estimates based on Beta and portfolio weights.
  Actual losses may differ due to changing correlations, liquidity, and non-linear market dynamics.
  Historical scenarios use Nifty 50 peak-to-trough declines — your portfolio did not exist during most of these events.
  This is for risk awareness only — not a prediction or investment advice.
</div>
""", unsafe_allow_html=True)


    ## ── TAB 6 — MARKET DATA ────────────────────────────────────────────────────
    with tab6:
        ## ═════════════════════════════════════════════════════════════════════════
        ##  SECTION 8 — PRICE HISTORY & CORRELATION
        ## ═════════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header"><h3>&#9632; Section 8 — Price History &amp; Correlation</h3><p>Adjusted close prices and correlation matrix</p></div>', unsafe_allow_html=True)

        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Adjusted Close Prices</p>", unsafe_allow_html=True)
        st.line_chart(prices)

        st.markdown("<p style='font-size:13px;font-weight:600;color:#C8A951;margin:16px 0 8px 0;border-bottom:1px solid #21262D;padding-bottom:6px'>Correlation Matrix</p>", unsafe_allow_html=True)
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "corr_cmap", ["#D32F2F", "#FFFFFF", "#1B5E20"], N=256
        )
        corr_matrix = returns.corr()
        st.dataframe(
            corr_matrix.style
            .background_gradient(cmap=custom_cmap, vmin=-1, vmax=1)
            .format("{:.2f}"),
            width='stretch'
        )
        st.markdown("<hr style='border:none;border-top:1px solid #21262D;margin:24px 0'>", unsafe_allow_html=True)



    ## ── TAB 7 — MARKET INTELLIGENCE ────────────────────────────────────────────
    with tab7:
        st.markdown(
            '<div class="section-header"><h3>&#9632; Market Intelligence</h3>'
            '<p>Top 5 sector stocks by market cap + latest news for your portfolio sectors</p></div>',
            unsafe_allow_html=True
        )

        ## ── Collect unique sectors from portfolio ────────────────────────────
        _ETF_PSEUDO_SECTORS = {
            "Non-Equity ETF","Broad Market ETF","ETF","Defence & Aerospace",
            "PSU Theme","Mid Cap","Small Cap","Factor ETF","Consumption Theme",
            "Multi Asset","Unknown ETF",
        }
        _portfolio_sectors = []
        _etf_holdings      = []
        _ticker_sector_map = {}
        if 'sector_df' in dir() and sector_df is not None and not sector_df.empty:
            for _, _row in sector_df.iterrows():
                _sec  = _row.get("Sector","N/A")
                _tk   = _row.get("Ticker","")
                _is_e = _row.get("Is ETF", False)
                _tier = _row.get("ETF Tier", 0)
                if _is_e:
                    _etf_holdings.append({
                        "ticker": _tk, "sector": _sec, "tier": _tier,
                        "type": _row.get("Type","ETF"), "attrBM": _row.get("Attribution BM",""),
                    })
                elif _sec and _sec not in ["N/A","Unknown"] and _sec not in _ETF_PSEUDO_SECTORS:
                    if _sec not in _portfolio_sectors:
                        _portfolio_sectors.append(_sec)
                if _tk:
                    _ticker_sector_map[_tk] = _sec
        else:
            ## Fallback — fetch sectors from enriched_info (sector stored during enrich step)
            for _tk, _ei in enriched_info.items():
                _sec = _ei.get("sector", None)
                ## If not in enriched_info, fetch from Yahoo directly
                if not _sec or _sec in ["Unknown", "N/A"]:
                    _sec_info = fetch_sector_info(_tk)
                    _sec = _sec_info.get("sector", "Unknown")
                if _sec and _sec not in ["Unknown","N/A"] and _sec not in _portfolio_sectors:
                    _portfolio_sectors.append(_sec)
                _ticker_sector_map[_tk] = _sec


        ## ── ETF Holdings Summary ─────────────────────────────────────────────────
        if _etf_holdings:
            st.markdown(
                "<p style='font-size:13px;font-weight:600;color:#C8A951;margin:12px 0 6px 0'>"
                "&#9632; ETF Holdings in your Portfolio</p>",
                unsafe_allow_html=True
            )
            _tier_desc = {
                1: ("Sector ETF",
                    "Tracks a specific sector index. Attribution benchmark = sector index."),
                2: ("Broad Market ETF",
                    "Tracks the portfolio benchmark. Allocation Effect = 0 — "
                    "holding the benchmark itself is not an active bet."),
                3: ("Thematic ETF",
                    "Tracks a thematic index (Defence, PSU, Midcap etc). "
                    "Attribution benchmark = Nifty 500 proxy."),
                4: ("Excluded ETF",
                    "Non-equity ETF (Gold, Silver, Liquid, International). "
                    "Excluded from BHB attribution — no equity benchmark applies."),
            }
            for _etf in _etf_holdings:
                _t      = _etf["ticker"].replace(".NS","").replace(".BO","")
                _ticker = _etf["ticker"]
                _tier   = _etf.get("tier", 4)
                _sec    = _etf.get("sector", "ETF")
                _attrBM = _etf.get("attrBM", "")
                _color  = {1:"#1D9E75",2:"#2979FF",3:"#C8A951",4:"#8B949E"}.get(_tier,"#8B949E")
                _etf_def = ETF_DEFINITIONS.get(_ticker, {})
                _tracks  = _etf_def.get("tracks", "")
                _label, _ = _tier_desc.get(_tier, ("ETF",""))
                if _tier == 1:
                    _desc = (
                        f"Sector: <b>{_sec}</b>. "
                        + (f"Tracks: <b>{_tracks}</b>. " if _tracks else "")
                        + f"Attribution benchmark: <b>{_attrBM}</b> (direct sector index). "
                        + "Selection Effect = N/A."
                    )
                elif _tier == 2:
                    _desc = (
                        f"Tracks: <b>{_tracks or 'Broad Market Index'}</b>. "
                        + "Allocation Effect = <b>0.00%</b> — holding the benchmark "
                        + "is not an active decision (professional BHB standard). "
                        + "Selection and Interaction = N/A."
                    )
                elif _tier == 3:
                    _desc = (
                        f"Theme: <b>{_sec}</b>. "
                        + (f"Tracks: <b>{_tracks}</b>. " if _tracks else "")
                        + f"Attribution benchmark: <b>{_attrBM or 'Nifty 500 (Proxy)'}</b>. "
                        + "Selection and Interaction = N/A."
                    )
                else:
                    _desc = (
                        f"Category: <b>{_sec}</b>. "
                        + (f"Tracks: <b>{_tracks}</b>. " if _tracks else "")
                        + "<b>Excluded from BHB attribution</b> — no equity benchmark applies. "
                        + "P&L and Return Contribution still included in all portfolio calculations."
                    )
                st.markdown(
                    f"<div style='padding:10px 14px;margin-bottom:8px;"
                    f"background:#161B22;border-left:3px solid {_color};"
                    f"border-radius:0 8px 8px 0'>"
                    f"<div style='display:flex;align-items:center;gap:10px'>"
                    f"<span style='color:#F3F6FA;font-weight:700;font-size:13px'>{_t}</span>"
                    f"<span style='color:{_color};font-size:11px;background:rgba(255,255,255,0.05);"
                    f"padding:2px 8px;border-radius:10px'>Tier {_tier} — {_label}</span>"
                    f"</div>"
                    f"<div style='color:#8B949E;font-size:11px;margin-top:4px'>{_desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            st.markdown("---")


        if not _portfolio_sectors:
            st.info("Sector data not available. Please ensure tickers are valid NSE/BSE stocks.")
        else:
            ## Determine exchange for top5 lookup
            _exch = "BSE" if _is_bse_benchmark else "NSE"

            ## ── Loop over each sector ────────────────────────────────────────
            for _sector in _portfolio_sectors:
                ## Stocks in this sector from the portfolio
                _stocks_in_sector = [t for t, s in _ticker_sector_map.items()
                                     if s == _sector]

                st.markdown(
                    f"<div style='margin:20px 0 10px 0;padding:10px 16px;"
                    f"background:linear-gradient(90deg,#1E3A5F,#0D1723);"
                    f"border-left:4px solid #C8A951;border-radius:0 10px 10px 0'>"
                    f"<span style='font-size:14px;font-weight:700;color:#F3F6FA'>"
                    f"&#9632; {_sector}</span>"
                    f"<span style='font-size:11px;color:#8EA1B4;margin-left:12px'>"
                    f"Your holdings: {', '.join([t.replace('.NS','').replace('.BO','') for t in _stocks_in_sector])}"
                    f"</span></div>",
                    unsafe_allow_html=True
                )

                _col_top5, _col_news = st.columns([1, 1])

                ## ── LEFT: Top 5 stocks by market cap ─────────────────────────
                with _col_top5:
                    st.markdown(
                        "<p style='font-size:12px;font-weight:600;color:#C8A951;"
                        "margin-bottom:8px'>Top 5 by Market Cap</p>",
                        unsafe_allow_html=True
                    )
                    with st.spinner(f"Loading top {_sector} stocks..."):
                        _top5 = fetch_top5_in_sector(
                            _sector,
                            exchange=_exch,
                            exclude_tickers=tickers
                        )

                    if _top5 is not None and not _top5.empty:
                        for _, _r in _top5.iterrows():
                            _in_port   = _r.get("In Portfolio", False)
                            _chg_1d    = _r.get("1D Change %", 0)
                            _ret_52w   = _r.get("52W Return %", 0)
                            _price     = _r.get("Price", 0)
                            _name      = _r.get("Name", _r["Ticker"])
                            _ticker_d  = _r["Ticker"].replace(".NS","").replace(".BO","")
                            _chg_col   = "#00E676" if _chg_1d >= 0 else "#FF5252"
                            _r52_col   = "#00E676" if _ret_52w >= 0 else "#FF5252"
                            _bg        = "rgba(200,169,81,0.08)" if _in_port else "transparent"
                            _badge     = (" <span style='font-size:9px;padding:1px 5px;"
                                          "background:rgba(200,169,81,0.2);color:#C8A951;"
                                          "border-radius:4px'>YOUR HOLDING</span>"
                                          if _in_port else "")
                            st.markdown(
                                f"<div style='padding:8px 10px;background:{_bg};"
                                f"border:1px solid rgba(255,255,255,0.05);"
                                f"border-radius:8px;margin-bottom:5px'>"
                                f"<div style='display:flex;justify-content:space-between;"
                                f"align-items:center'>"
                                f"<div style='font-size:12px;font-weight:600;color:#F3F6FA'>"
                                f"#{int(_r['Rank'])} {_ticker_d}{_badge}</div>"
                                f"<div style='font-size:12px;color:#F3F6FA'>"
                                f"&#8377;{_price:,.2f}</div></div>"
                                f"<div style='display:flex;justify-content:space-between;"
                                f"margin-top:3px'>"
                                f"<div style='font-size:10px;color:#8EA1B4'>{_name[:28]}</div>"
                                f"<div style='font-size:10px'>"
                                f"<span style='color:{_chg_col}'>{_chg_1d:+.2f}% 1D</span>"
                                f" &nbsp; "
                                f"<span style='color:{_r52_col}'>{_ret_52w:+.1f}% 52W</span>"
                                f"</div></div></div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption("No data available for this sector.")

                ## ── RIGHT: News ───────────────────────────────────────────────
                with _col_news:
                    st.markdown(
                        "<p style='font-size:12px;font-weight:600;color:#C8A951;"
                        "margin-bottom:8px'>Latest News</p>",
                        unsafe_allow_html=True
                    )

                    _all_news = []

                    ## ── Stock news — user's actual holdings in this sector ────
                    for _tk in _stocks_in_sector:
                        with st.spinner(f"Loading news for {_tk.replace('.NS','').replace('.BO','')}..."):
                            _news = fetch_ticker_news(_tk, news_type="stock")
                        _all_news.extend(_news)

                    ## ── Sector news — 2 proxy tickers (exclude held stocks) ──
                    _held_set    = set(_stocks_in_sector)
                    _proxies     = SECTOR_NEWS_PROXIES.get(_sector, [])
                    _sector_proxies = [p for p in _proxies if p not in _held_set][:2]
                    for _px in _sector_proxies:
                        with st.spinner(f"Loading {_sector} sector news..."):
                            _px_news = fetch_ticker_news(_px, news_type="sector")
                        _all_news.extend(_px_news)

                    ## ── Deduplicate by title, sort newest first ───────────────
                    _seen_titles = set()
                    _unique_news = []
                    for _n in sorted(_all_news, key=lambda x: x.get("pub_ts", 0), reverse=True):
                        _t = _n.get("title", "")
                        if _t and _t not in _seen_titles:
                            _seen_titles.add(_t)
                            _unique_news.append(_n)

                    if _unique_news:
                        ## Legend
                        st.markdown(
                            "<div style='display:flex;gap:12px;margin-bottom:8px'>"
                            "<span style='font-size:10px;color:#8EA1B4'>"
                            "<span style='color:#C8A951;font-weight:700'>━</span>"
                            " Your holdings</span>"
                            "<span style='font-size:10px;color:#8EA1B4'>"
                            "<span style='color:#58A6FF;font-weight:700'>━</span>"
                            " Sector context</span>"
                            "</div>",
                            unsafe_allow_html=True
                        )
                        for _news_item in _unique_news[:8]:
                            _nt      = _news_item.get("title",     "")
                            _ns      = _news_item.get("source",    "Yahoo Finance")
                            _np      = _news_item.get("published", "")
                            _nl      = _news_item.get("link",      "#")
                            _ntype   = _news_item.get("query_type","sector")
                            _dot_col = "#C8A951" if _ntype == "stock" else "#58A6FF"
                            st.markdown(
                                f"<div style='padding:7px 10px;border:1px solid "
                                f"rgba(255,255,255,0.05);border-left:3px solid {_dot_col};"
                                f"border-radius:0 8px 8px 0;margin-bottom:5px;"
                                f"background:linear-gradient(145deg,#0F1C2B,#0D1723)'>"
                                f"<a href='{_nl}' target='_blank' style='font-size:11.5px;"
                                f"color:#E6EDF3;text-decoration:none;line-height:1.4'>"
                                f"{_nt[:90]}{'...' if len(_nt)>90 else ''}</a>"
                                f"<div style='margin-top:4px;font-size:10px;color:#8EA1B4'>"
                                f"{_ns} &nbsp;·&nbsp; {_np}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown(
                            "<div style='padding:10px;background:#0D1723;"
                            "border:1px solid #21262D;border-radius:8px;"
                            "font-size:12px;color:#8EA1B4'>No news available "
                            "for this sector from Yahoo Finance.</div>",
                            unsafe_allow_html=True
                        )

                st.markdown(
                    "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.05);"
                    "margin:16px 0'>",
                    unsafe_allow_html=True
                )

        ## ── Disclaimer ───────────────────────────────────────────────────────
        st.markdown(
            "<div style='margin-top:12px;padding:8px 14px;background:#0D1723;"
            "border:1px solid #21262D;border-radius:8px;font-size:11px;color:#8EA1B4'>"
            "<strong style='color:#C8A951'>Note:</strong> "
            "Top 5 stocks sourced from a curated Nifty 500 list, ranked by market cap. "
            "News fetched from Yahoo Finance — content is from third-party sources. "
            "Updates follow the auto-refresh interval set in the sidebar. "
            "Not investment advice.</div>",
            unsafe_allow_html=True
        )

    ## ── PDF button — all variables now available after tabs run ────────────────
    _stress_scenarios = [
        ("COVID Crash (NSE)",        "Feb-Mar 2020",       -0.385),
        ("2008 Global Crisis (NSE)", "Jan-Oct 2008",       -0.605),
        ("2022 Rate Selloff (NSE)",  "Oct 2021-Jun 2022",  -0.175),
        ("2024 Election Shock (NSE)","Apr-Jun 2024",       -0.062),
        ("2015-16 Correction (NSE)", "Mar 2015-Feb 2016",  -0.261),
        ("Dot-Com Bust (Sensex)",    "Feb 2000-Sep 2001",  -0.560),
    ]
    try:
        ## ── Read ALL variables from session_state ────────────────────────────
        ## Every variable needed by PDF is stored in session_state when computed
        ## On PDF button rerun, the try/submitted block is skipped — session_state
        ## is the ONLY reliable source for these values
        _ss              = st.session_state
        portfolio_stats  = _ss.get("_pdf_portfolio_stats",  {})
        portfolio_beta   = _ss.get("_pdf_portfolio_beta",   np.nan)
        _pdf_holdings    = _ss.get("_pdf_holdings_df",      holdings_df if 'holdings_df' in locals() else pd.DataFrame())
        _pdf_ti          = _ss.get("_pdf_total_invested",   total_invested   if 'total_invested'   in locals() else 0)
        _pdf_tc          = _ss.get("_pdf_total_current",    total_current    if 'total_current'    in locals() else 0)
        _pdf_pnl         = _ss.get("_pdf_total_pnl",        total_pnl        if 'total_pnl'        in locals() else 0)
        _pdf_xirr        = _ss.get("_pdf_portfolio_xirr",   portfolio_xirr   if 'portfolio_xirr'   in locals() else np.nan)
        _pdf_maxdd       = _ss.get("_pdf_max_dd",            max_dd           if 'max_dd'           in locals() else np.nan)
        _pdf_recov       = _ss.get("_pdf_recovery_date",    recovery_date    if 'recovery_date'    in locals() else None)
        _pdf_ret         = _ss.get("_pdf_port_returns_bm",  port_returns_bm  if 'port_returns_bm'  in locals() else pd.Series(dtype=float))
        _pdf_bm_name     = _ss.get("_pdf_benchmark_name",   BENCHMARK_NAME   if 'BENCHMARK_NAME'   in locals() else "Nifty 50")
        _pdf_bm_ticker   = _ss.get("_pdf_benchmark_ticker", BENCHMARK_TICKER if 'BENCHMARK_TICKER' in locals() else "^NSEI")
        _pdf_is_bse      = _ss.get("_pdf_is_bse",           _is_bse_benchmark if '_is_bse_benchmark' in locals() else False)
        _pdf_dname       = _ss.get("_pdf_display_name",     _display_name    if '_display_name'    in locals() else "")

        ## attr_df / risk_df / sector_df: stored in session_state if user visited Attribution tab
        _attr_df_pdf  = _ss.get("_pdf_attr_df",   attr_df   if 'attr_df'   in locals() else pd.DataFrame())
        _risk_df_pdf  = _ss.get("_pdf_risk_df",   risk_df   if 'risk_df'   in locals() else pd.DataFrame())
        _sect_df_pdf  = _ss.get("_pdf_sector_df", sector_df if 'sector_df' in locals() else None)

        ## Benchmark price series for BM Return KPI card
        try:
            _bm_raw = _ss.get("_pdf_benchmark_series", None)
            if _bm_raw is not None and not _bm_raw.empty:
                _bm_tmp = _bm_raw.dropna()
                if isinstance(_bm_tmp, pd.DataFrame):
                    _bm_tmp = _bm_tmp.iloc[:, 0]
                _bm_pdf = _bm_tmp.squeeze() if len(_bm_tmp) > 1 else None
            else:
                _bm_pdf = None
        except Exception:
            _bm_pdf = None

        try:
            _quote_text, _quote_author = get_quote_of_day()
        except Exception:
            _quote_text, _quote_author = "", ""

        _pdf_buf = create_institutional_pdf(
            holdings_df=_pdf_holdings,
            risk_df=_risk_df_pdf,
            portfolio_stats=portfolio_stats,
            attr_df=_attr_df_pdf,
            total_invested=_pdf_ti,
            total_current=_pdf_tc,
            total_pnl=_pdf_pnl,
            portfolio_xirr=_pdf_xirr,
            max_dd=_pdf_maxdd,
            recovery_date=_pdf_recov,
            RISK_FREE_ANNUAL=RISK_FREE_ANNUAL,
            port_returns=_pdf_ret,
            benchmark_data=_bm_pdf,
            sector_df=_sect_df_pdf,
            stress_scenarios=_stress_scenarios,
            portfolio_beta=portfolio_beta,
            total_current_val=_pdf_tc,
            benchmark_name=_pdf_bm_name,
            benchmark_ticker=_pdf_bm_ticker,
            portfolio_exchange="BSE" if _pdf_is_bse else "NSE",
            client_name=_pdf_dname,
            quote_text=_quote_text,
            quote_author=_quote_author,
        )
        with _pdf_placeholder:
            st.download_button(
                label="📄 Download PDF Report",
                data=_pdf_buf,
                file_name=f"portfolio_risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                width='stretch',
            )
    except Exception as _pdf_err:
        with _pdf_placeholder:
            st.caption(f"PDF unavailable: {_pdf_err}")

    ## ── FOOTER — always rendered after tabs ────────────────────────────────────
    last_updated   = pd.Timestamp.now().strftime("%d %b %Y  %H:%M:%S")
    refresh_status = f"ON — every {refresh_minutes} min" if auto_refresh else "OFF"
    st.markdown(f"""
<div style="margin-top:24px;padding:12px 20px;background:#0D1723;
border-top:1px solid rgba(255,255,255,0.05);border-radius:0 0 12px 12px;
display:flex;justify-content:space-between;align-items:center">
  <div style="font-size:11px;color:#8EA1B4">
    <span style="color:#00E676">&#9679;</span>
    &nbsp; Last updated: <span style="color:#F3F6FA">{last_updated}</span>
    &nbsp;|&nbsp; Auto-refresh: <span style="color:#C8A951">{refresh_status}</span>
    &nbsp;|&nbsp; Data: <span style="color:#F3F6FA">Yahoo Finance</span>
  </div>
  <div style="font-size:11px;color:#8EA1B4">
    Past performance is not indicative of future results. Not investment advice.
  </div>
</div>
""", unsafe_allow_html=True)


except Exception as e:
    import traceback
    st.error(f"**Dashboard Error:** {e}")
    with st.expander("🔍 Full Error Details (share with developer)", expanded=False):
        st.code(traceback.format_exc(), language="python")
    st.info("Try refreshing the page. If the error persists, check that all ticker symbols are valid (e.g. RELIANCE.NS not RELIANCE).")