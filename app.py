import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# PAGE CONFIG & DARK THEME CSS
# ============================================================
st.set_page_config(layout="wide", page_title="RSI Oversold Scanner", page_icon="🔍")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-card h3 { color: #8b949e; font-size: 0.8rem; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card p { color: #e6edf3; font-size: 1.8rem; font-weight: 700; margin: 4px 0 0 0; }
    .status-extreme { color: #ff4b4b; font-weight: 700; }
    .status-strong { color: #ff8c00; font-weight: 700; }
    .status-normal { color: #ffd700; font-weight: 700; }
    .tfs-badge { background-color: #21262d; padding: 2px 8px; border-radius: 12px; color: #e6edf3; font-size: 0.8rem; }
    div[data-testid="stSidebar"] { background-color: #161b22; }
    .stSelectbox label, .stMultiSelect label, .stTextInput label { color: #8b949e !important; }
    .ticker-up { color: #3fb950; font-weight: 700; }
    .ticker-down { color: #f85149; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TIMEFRAME CONFIG
# ============================================================
TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "45m": "45m",
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
}

# ============================================================
# RSI CALCULATION (Wilder's, pure pandas/numpy)
# ============================================================
def compute_wilder_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute Wilder's RSI from close prices. Returns the last value."""
    if len(closes) < period + 1:
        return None
    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# ============================================================
# COINGECKO MARKET CAP RANK (cached 1h)
# ============================================================
@st.cache_data(ttl=3600)
def fetch_coingecko_ranks():
    """Fetch top 1000 coins from CoinGecko and return {SYMBOL: RANK}."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        ranks = {}
        for idx, coin in enumerate(data):
            sym = coin.get("symbol", "").upper()
            if sym:
                ranks[sym] = idx + 1
        # Fetch pages 2-4 for top 1000
        for page in [2, 3, 4]:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page},
                timeout=15,
            )
            if resp.status_code == 200:
                for idx, coin in enumerate(resp.json()):
                    sym = coin.get("symbol", "").upper()
                    if sym and sym not in ranks:
                        ranks[sym] = (page - 1) * 250 + idx + 1
            time.sleep(0.3)  # rate limit courtesy
        return ranks
    except Exception:
        return {}

# ============================================================
# KRAKEN EXCHANGE SETUP
# ============================================================
@st.cache_resource
def get_kraken_exchange():
    """Initialize Kraken futures/perp exchange via ccxt."""
    ex = ccxt.kraken({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
    })
    return ex

@st.cache_resource
def get_coinbase_exchange():
    """Initialize Coinbase for symbol validation (spot only)."""
    ex = ccxt.coinbase({"enableRateLimit": True})
    return ex

# ============================================================
# FETCH LIQUID PERPETUAL PAIRS FROM KRAKEN
# ============================================================
@st.cache_data(ttl=180)
def fetch_liquid_pairs(min_volume: float = 5_000_000):
    """Fetch Kraken perpetual markets and filter by 24h quote volume."""
    ex = get_kraken_exchange()
    try:
        markets = ex.load_markets()
        tickers = ex.fetch_tickers()
    except Exception as e:
        st.error(f"Failed to load Kraken markets: {e}")
        return []

    liquid = []
    for sym, mkt in markets.items():
        if not mkt.get("swap") or not mkt.get("active"):
            continue
        base = mkt.get("base", "")
        quote = mkt.get("quote", "")
        if quote != "USD" and quote != "USDT":
            continue
        ticker = tickers.get(sym, {})
        vol = ticker.get("quoteVolume") or 0
        if vol >= min_volume:
            liquid.append({
                "symbol": sym,
                "base": base,
                "quote": quote,
                "volume": vol,
            })
    return liquid

# ============================================================
# FETCH OHLCV FOR A SINGLE PAIR & TIMEFRAME
# ============================================================
def fetch_ohlcv_safe(ex, symbol: str, timeframe: str, limit: int = 100):
    """Fetch OHLCV and return close prices array."""
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 20:
            return None
        closes = np.array([c[4] for c in ohlcv], dtype=float)
        return closes
    except Exception:
        return None

def resample_45m(closes_15m: np.ndarray, timestamps_15m: list) -> np.ndarray:
    """Resample 15m closes into 45m closes by taking every 3rd close."""
    if closes_15m is None or len(closes_15m) < 6:
        return None
    # Take every 3rd element as the 45m close
    return closes_15m[2::3]

# ============================================================
# SCAN A SINGLE PAIR ACROSS ALL TIMEFRAMES
# ============================================================
def scan_pair(pair_info: dict, cg_ranks: dict):
    """Scan one pair across all timeframes and return oversold signals."""
    ex = get_kraken_exchange()
    symbol = pair_info["symbol"]
    base = pair_info["base"]

    results = []
    oversold_tfs = []

    for tf_label, tf_ccxt in TIMEFRAMES.items():
        if tf_label == "45m":
            # Resample from 15m
            closes_15m = fetch_ohlcv_safe(ex, symbol, "15m", limit=150)
            if closes_15m is None:
                continue
            closes = resample_45m(closes_15m, None)
        else:
            closes = fetch_ohlcv_safe(ex, symbol, tf_ccxt, limit=100)

        if closes is None or len(closes) < 16:
            continue

        # Closed candle RSI (iloc[-2])
        closed_rsi = compute_wilder_rsi(closes[:-1])
        # Live candle RSI (iloc[-1])
        live_rsi = compute_wilder_rsi(closes)

        if closed_rsi is None:
            continue

        rsi_val = closed_rsi
        is_oversold = rsi_val < 30

        if is_oversold:
            oversold_tfs.append(tf_label)
            # Determine status
            if rsi_val < 20:
                status = "Extremely Oversold"
            elif rsi_val <= 25:
                status = "Strongly Oversold"
            else:
                status = "Oversold"

            last_candle_ts = None
            try:
                ohlcv = ex.fetch_ohlcv(symbol, tf_ccxt if tf_label != "45m" else "15m", limit=3)
                if ohlcv and len(ohlcv) >= 2:
                    last_candle_ts = ohlcv[-2][0]
            except Exception:
                pass

            results.append({
                "base": base,
                "symbol": symbol,
                "tf": tf_label,
                "rsi_closed": round(rsi_val, 2),
                "rsi_live": round(live_rsi, 2) if live_rsi else None,
                "dist_to_50": round(rsi_val - 50, 2),
                "status": status,
                "last_candle_ts": last_candle_ts,
                "mcap_rank": cg_ranks.get(base, None),
            })

        time.sleep(0.05)  # rate limit

    # Attach confluence count to each result
    tf_count = len(oversold_tfs)
    tf_list = " · ".join(oversold_tfs)
    for r in results:
        r["tfs_oversold"] = f"{tf_count} | {tf_list}" if tf_count > 0 else "0"

    return results

# ============================================================
# MAIN SCAN ORCHESTRATOR
# ============================================================
@st.cache_data(ttl=180, show_spinner=False)
def run_scan(min_volume: float = 5_000_000):
    """Run the full scan and return DataFrame + metadata."""
    start_time = time.time()
    cg_ranks = fetch_coingecko_ranks()
    pairs = fetch_liquid_pairs(min_volume)

    if not pairs:
        return pd.DataFrame(), 0, 0, time.time() - start_time

    all_results = []
    # Limit concurrent workers to avoid rate limits
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(scan_pair, p, cg_ranks): p for p in pairs}
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    all_results.extend(res)
            except Exception:
                pass

    elapsed = time.time() - start_time
    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return df, len(pairs), len(all_results), elapsed

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🔍 RSI Scanner")
    st.caption("Kraken Perpetuals · CoinGecko Ranks")

    if st.button("🔄 Refresh Scan Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Live majors ticker
    st.subheader("Live Majors")
    try:
        cb = get_coinbase_exchange()
        for sym_label, cb_sym in [("BTC", "BTC/USD"), ("ETH", "ETH/USD")]:
            try:
                t = cb.fetch_ticker(cb_sym)
                pct = t.get("percentage") or 0
                cls = "ticker-up" if pct >= 0 else "ticker-down"
                st.markdown(
                    f"<span style='color:#8b949e'>{sym_label}</span> "
                    f"<span class='{cls}'>{pct:+.2f}%</span>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.caption(f"{sym_label}: --")
    except Exception:
        st.caption("Ticker unavailable")

    st.divider()

    # Watchlist (visual only in this version)
    watchlist = st.multiselect(
        "⭐ My Watchlist",
        options=["BTC", "ETH", "SOL", "UNI", "AAVE", "ARB", "LINK", "MATIC"],
        default=[],
    )

# ============================================================
# TOP METRICS (placeholder before scan)
# ============================================================
metric_cols = st.columns(4)

# ============================================================
# RUN SCAN
# ============================================================
with st.spinner("Scanning Kraken perpetuals across 7 timeframes..."):
    df_raw, pairs_scanned, total_signals, scan_seconds = run_scan(5_000_000)

# ============================================================
# UPDATE METRICS
# ============================================================
if not df_raw.empty:
    unique_coins = df_raw["base"].nunique()
    confluence_2plus = df_raw.groupby("base")["tf"].nunique()
    confluence_count = (confluence_2plus >= 2).sum()
else:
    unique_coins = 0
    confluence_count = 0

last_scan_str = datetime.now(timezone.utc).strftime("%d %b, %H:%M UTC")
scan_time_str = f"Last scan: {scan_seconds:.0f}s"

metric_cols[0].markdown(
    f'<div class="metric-card"><h3>Coins Scanned</h3><p>{pairs_scanned}</p></div>',
    unsafe_allow_html=True,
)
metric_cols[1].markdown(
    f'<div class="metric-card"><h3>Oversold Signals</h3><p>{total_signals}</p></div>',
    unsafe_allow_html=True,
)
metric_cols[2].markdown(
    f'<div class="metric-card"><h3>Multi-TF Confluence</h3><p>{confluence_count}</p></div>',
    unsafe_allow_html=True,
)
metric_cols[3].markdown(
    f'<div class="metric-card"><h3>Last Successful Scan</h3><p style="font-size:1rem">{last_scan_str}</p><p style="font-size:0.8rem;color:#8b949e">{scan_time_str}</p></div>',
    unsafe_allow_html=True,
)

# ============================================================
# FILTER BAR
# ============================================================
st.markdown("---")
filter_cols = st.columns([2, 1, 1, 2, 1.5, 1, 0.8])

with filter_cols[0]:
    search = st.text_input("Search Symbol", placeholder="e.g. AAVE")

with filter_cols[1]:
    vol_filter = st.selectbox("Min 24h Vol", ["$1M", "$5M", "$10M", "$25M", "$50M"], index=1)

with filter_cols[2]:
    rank_filter = st.selectbox("Max MCAP Rank", ["All Ranks", "Top 50", "Top 100", "Top 250", "Top 500"], index=0)

with filter_cols[3]:
    tf_filter = st.multiselect("TF Filter", list(TIMEFRAMES.keys()), default=list(TIMEFRAMES.keys()))

with filter_cols[4]:
    class_filter = st.multiselect("Class", ["< 20", "20-25", "25-30"], default=["< 20", "20-25", "25-30"])

with filter_cols[5]:
    conf_filter = st.selectbox("Confluence", ["At least 1 TF", "At least 2 TFs", "At least 3 TFs"], index=0)

with filter_cols[6]:
    live_preview = st.checkbox("Live preview", value=False)

# ============================================================
# APPLY FILTERS
# ============================================================
df = df_raw.copy() if not df_raw.empty else pd.DataFrame()

if not df.empty:
    # Symbol search
    if search:
        df = df[df["base"].str.upper().str.contains(search.upper())]

    # Rank filter
    if rank_filter != "All Ranks":
        max_rank = int(rank_filter.replace("Top ", ""))
        df = df[df["mcap_rank"].notna() & (df["mcap_rank"] <= max_rank)]

    # TF filter
    if tf_filter:
        df = df[df["tf"].isin(tf_filter)]

    # Class filter
    if class_filter:
        masks = []
        if "< 20" in class_filter:
            masks.append(df["rsi_closed"] < 20)
        if "20-25" in class_filter:
            masks.append((df["rsi_closed"] >= 20) & (df["rsi_closed"] <= 25))
        if "25-30" in class_filter:
            masks.append((df["rsi_closed"] > 25) & (df["rsi_closed"] <= 30))
        if masks:
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m
            df = df[combined]

    # Live preview filter
    if live_preview and "rsi_live" in df.columns:
        df = df[(df["rsi_closed"] < 30) | (df["rsi_live"] < 30)]

    # Confluence filter
    if conf_filter == "At least 2 TFs":
        conf_counts = df.groupby("base")["tf"].nunique()
        valid_bases = conf_counts[conf_counts >= 2].index
        df = df[df["base"].isin(valid_bases)]
    elif conf_filter == "At least 3 TFs":
        conf_counts = df.groupby("base")["tf"].nunique()
        valid_bases = conf_counts[conf_counts >= 3].index
        df = df[df["base"].isin(valid_bases)]

# ============================================================
# DISPLAY TABLE
# ============================================================
if df.empty:
    st.info("No coins match the current filters. Try relaxing the criteria or refresh the scan.")
else:
    # Sort by confluence (desc) then RSI (asc)
    df["conf_count"] = df["tfs_oversold"].str.split(" | ").str[0].astype(int)
    df = df.sort_values(["conf_count", "rsi_closed"], ascending=[False, True])

    # Format for display
    display_df = df[[
        "mcap_rank", "base", "tf", "rsi_closed", "rsi_live", "dist_to_50", "status", "tfs_oversold"
    ]].copy()
    display_df.columns = ["MCAP RANK", "SYMBOL", "TF", "RSI 14", "LIVE RSI", "DIST. TO 50", "STATUS", "TFS OVERSOLD"]

    # Format rank
    display_df["MCAP RANK"] = display_df["MCAP RANK"].apply(
        lambda x: f"#{int(x)}" if pd.notna(x) else "Unranked"
    )

    # Format symbol
    display_df["SYMBOL"] = display_df["SYMBOL"] + " /USDT"

    # Format dist
    display_df["DIST. TO 50"] = display_df["DIST. TO 50"].apply(lambda x: f"{x:+.2f}")

    # Style status
    def color_status(val):
        if val == "Extremely Oversold":
            return "color: #ff4b4b; font-weight: 700"
        elif val == "Strongly Oversold":
            return "color: #ff8c00; font-weight: 700"
        else:
            return "color: #ffd700; font-weight: 700"

    styled = display_df.style.map(color_status, subset=["STATUS"])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    # CSV Export
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export CSV", csv, "rsi_oversold_scan.csv", "text/csv")

# ============================================================
# TRADINGVIEW CHART DRAWER
# ============================================================
st.markdown("---")
st.subheader("📈 TradingView Chart")

if not df.empty:
    coin_options = sorted(df["base"].unique())
    selected_coin = st.selectbox("Select coin for chart", coin_options)

    if selected_coin:
        # Kraken perpetual symbol format for TradingView
        tv_symbol = f"KRAKEN:{selected_coin}USD.P"
        tv_html = f"""
        <div class="tradingview-widget-container" style="height:500px;width:100%">
            <div id="tradingview_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({{
                "autosize": true,
                "symbol": "{tv_symbol}",
                "interval": "60",
                "timezone": "Etc/UTC",
                "theme": "dark",
                "style": "1",
                "locale": "en",
                "toolbar_bg": "#f1f3f6",
                "enable_publishing": false,
                "allow_symbol_change": true,
                "container_id": "tradingview_chart"
            }});
            </script>
        </div>
        """
        st.components.v1.html(tv_html, height=520)
else:
    st.caption("Run a scan and apply filters to select a coin for charting.")