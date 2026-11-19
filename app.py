import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit.components.v1 as components

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
    .metric-card h3 {
        color: #8b949e; font-size: 0.75rem; margin: 0;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .metric-card p {
        color: #e6edf3; font-size: 1.7rem;
        font-weight: 700; margin: 4px 0 0 0;
    }
    .metric-card small { color: #6e7681; font-size: 0.75rem; }
    div[data-testid="stSidebar"] { background-color: #161b22; }
    .ticker-up { color: #3fb950; font-weight: 700; }
    .ticker-down { color: #f85149; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
TF_LABELS = ["5m", "15m", "30m", "45m", "1H", "2H", "4H"]
TF_CCXT = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "45m": None,      # resampled from 15m
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
}
TF_MS = {
    "5m": 5 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "45m": 45 * 60 * 1000,
    "1H": 60 * 60 * 1000,
    "2H": 2 * 60 * 60 * 1000,
    "4H": 4 * 60 * 60 * 1000,
}
VOL_OPTIONS = {
    "$1M": 1_000_000,
    "$5M": 5_000_000,
    "$10M": 10_000_000,
    "$25M": 25_000_000,
    "$50M": 50_000_000,
}

_fetch_lock = threading.Lock()

# ============================================================
# WILDER RSI (pure numpy)
# ============================================================
def wilder_rsi(closes, period=14):
    arr = np.asarray(closes, dtype=float)
    if len(arr) < period + 1:
        return None
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# ============================================================
# EXCHANGE SETUP (OKX primary → KuCoin Futures fallback)
# ============================================================
@st.cache_resource
def get_exchange():
    for ex_id in ["okx", "kucoinfutures"]:
        try:
            ex_class = getattr(ccxt, ex_id)
            ex = ex_class({
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            })
            ex.load_markets()
            if ex.markets:
                return ex, ex_id
        except Exception:
            continue
    return None, None

# ============================================================
# COINGECKO TOP-1000 RANK (cached 1 hour)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_coingecko_ranks():
    ranks = {}
    try:
        for page in range(1, 5):
            r = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": page,
                },
                timeout=20,
            )
            if r.status_code != 200:
                break
            for i, coin in enumerate(r.json()):
                sym = (coin.get("symbol") or "").upper()
                if sym and sym not in ranks:
                    ranks[sym] = (page - 1) * 250 + i + 1
            time.sleep(0.3)
    except Exception:
        pass
    return ranks

# ============================================================
# SAFE OHLCV FETCH (thread-safe)
# ============================================================
def fetch_ohlcv_safe(ex, symbol, timeframe, limit=200):
    if ex is None:
        return None
    try:
        with _fetch_lock:
            return ex.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception:
        return None

# ============================================================
# SCAN ONE TIMEFRAME (incl. 45m resampling via pandas)
# ============================================================
def scan_timeframe(ex, symbol, tf_label):
    if tf_label == "45m":
        ohlcv = fetch_ohlcv_safe(ex, symbol, "15m", limit=200)
        if not ohlcv or len(ohlcv) < 60:
            return None
        df = pd.DataFrame(
            ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df["close_dt"] = pd.to_datetime(
            df["ts"] + 15 * 60 * 1000, unit="ms", utc=True
        )
        df = df.set_index("close_dt")
        try:
            res = df.resample("45min", label="right", closed="right").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
        except Exception:
            return None
        if len(res) < 16:
            return None
        closes = res["close"].values.astype(float)
        last_ts = (
            int(res.index[-2].timestamp() * 1000)
            if len(res) >= 2 else None
        )
    else:
        ohlcv = fetch_ohlcv_safe(
            ex, symbol, TF_CCXT[tf_label], limit=200
        )
        if not ohlcv or len(ohlcv) < 16:
            return None
        closes = np.array([c[4] for c in ohlcv], dtype=float)
        last_ts = (
            int(ohlcv[-2][0]) + TF_MS[tf_label]
            if len(ohlcv) >= 2 else None
        )

    if len(closes) < 16:
        return None

    rsi_c = wilder_rsi(closes[:-1])   # closed candle
    rsi_l = wilder_rsi(closes)        # live candle

    if rsi_c is None:
        return None

    return {
        "rsi_closed": round(float(rsi_c), 2),
        "rsi_live": round(float(rsi_l), 2) if rsi_l is not None else None,
        "last_close_ts": last_ts,
    }

# ============================================================
# SCAN ONE PAIR ACROSS ALL TIMEFRAMES
# ============================================================
def scan_pair(ex, pair_info, cg_ranks):
    symbol = pair_info["symbol"]
    base = pair_info["base"]
    results = []
    closed_os_tfs = []

    for tf in TF_LABELS:
        d = scan_timeframe(ex, symbol, tf)
        if d is None:
            continue
        rsi_c = d["rsi_closed"]
        rsi_l = d["rsi_live"]
        if rsi_c is None:
            continue

        is_closed_os = rsi_c < 30
        is_live_os = rsi_l is not None and rsi_l < 30

        if not (is_closed_os or is_live_os):
            continue

        disp_rsi = rsi_c if is_closed_os else rsi_l

        if disp_rsi < 20:
            status = "Extremely Oversold"
        elif disp_rsi <= 25:
            status = "Strongly Oversold"
        else:
            status = "Oversold"

        if is_closed_os:
            closed_os_tfs.append(tf)

        results.append({
            "base": base,
            "symbol": symbol,
            "tf": tf,
            "rsi_closed": rsi_c,
            "rsi_live": rsi_l,
            "disp_rsi": round(disp_rsi, 2),
            "is_closed_os": is_closed_os,
            "dist_to_50": round(disp_rsi - 50, 2),
            "status": status,
            "last_close_ts": d["last_close_ts"],
            "mcap_rank": cg_ranks.get(base),
        })

    if results:
        cnt = len(closed_os_tfs)
        tf_str = (
            f"{cnt} | " + " · ".join(closed_os_tfs)
            if closed_os_tfs else "0"
        )
        for r in results:
            r["tfs_oversold"] = tf_str

    return results

# ============================================================
# FULL SCAN (cached 3 minutes, keyed on min_vol_value)
# ============================================================
@st.cache_data(ttl=180, show_spinner=False)
def run_full_scan(min_vol_value):
    start = time.time()
    cg_ranks = fetch_coingecko_ranks()
    ex, _ = get_exchange()

    if ex is None:
        return pd.DataFrame(), 0, 0, round(time.time() - start, 1)

    try:
        tickers = ex.fetch_tickers()
    except Exception:
        return pd.DataFrame(), 0, 0, round(time.time() - start, 1)

      liquid = []
    for sym, mkt in ex.markets.items():
        if not mkt.get("swap") or not mkt.get("active"):
            continue
        if mkt.get("quote") != "USDT":
            continue
        t = tickers.get(sym) or {}
        vol = t.get("quoteVolume")
        if not vol:
            base_vol = float(t.get("baseVolume") or 0)
            last_px = float(t.get("last") or 0)
            contract_sz = float(mkt.get("contractSize") or 1)
            vol = base_vol * last_px * contract_sz
        if float(vol or 0) >= min_vol_value:
            liquid.append({
                "symbol": sym,
                "base": mkt.get("base", ""),
                "volume": float(vol),
            })

    liquid.sort(key=lambda x: x["volume"], reverse=True)
    liquid = liquid[:150]

    if not liquid:
        return pd.DataFrame(), 0, 0, round(time.time() - start, 1)

    all_results = []
    with ThreadPoolExecutor(max_workers=4) as exe:
        futures = {
            exe.submit(scan_pair, ex, p, cg_ranks): p
            for p in liquid
        }
        for f in as_completed(futures):
            try:
                r = f.result()
                if r:
                    all_results.extend(r)
            except Exception:
                continue

    elapsed = round(time.time() - start, 1)
    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return df, len(liquid), len(all_results), elapsed

# ============================================================
# LIVE MAJORS TICKER (cached 60s)
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_majors_ticker():
    ex, _ = get_exchange()
    if ex is None:
        return {}
    out = {}
    for base in ["BTC", "ETH"]:
        sym = f"{base}/USDT:USDT"
        try:
            if sym in ex.markets:
                t = ex.fetch_ticker(sym)
                out[base] = t.get("percentage")
            else:
                out[base] = None
        except Exception:
            out[base] = None
    return out

# ============================================================
# FORMAT HELPERS
# ============================================================
def fmt_ts(ts_ms):
    if ts_ms is None or (isinstance(ts_ms, float) and pd.isna(ts_ms)):
        return "—"
    try:
        return datetime.fromtimestamp(
            int(ts_ms) / 1000, tz=timezone.utc
        ).strftime("%d %b, %H:%M")
    except Exception:
        return "—"

# ============================================================
# SIDEBAR
# ============================================================
ex_ui, ex_id_ui = get_exchange()

with st.sidebar:
    st.title("🔍 RSI Scanner")
    st.caption(f"Exchange: {(ex_id_ui or 'N/A').upper()}")

    if st.button("🔄 Refresh Scan Now", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("Live Majors")
    majors = get_majors_ticker()
    for base in ["BTC", "ETH"]:
        pct = majors.get(base)
        if pct is None:
            st.markdown(f"**{base}**: —")
        else:
            cls = "ticker-up" if pct >= 0 else "ticker-down"
            st.markdown(
                f"**{base}**: <span class='{cls}'>{pct:+.2f}%</span>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.multiselect(
        "⭐ My Watchlist",
        ["BTC", "ETH", "SOL", "UNI", "AAVE", "ARB", "LINK", "OP"],
        default=[],
    )

# ============================================================
# CONNECTION GUARD
# ============================================================
if ex_ui is None:
    st.error(
        "❌ Could not connect to OKX or KuCoin Futures. "
        "Check network / API availability and click Refresh."
    )
    st.stop()

# ============================================================
# SCAN VOLUME CONTROL → run_full_scan
# ============================================================
ctl = st.columns([1, 4])
with ctl[0]:
    vol_label = st.selectbox(
        "Min 24h Vol", list(VOL_OPTIONS.keys()), index=1
    )
min_vol_value = VOL_OPTIONS[vol_label]

with st.spinner("Scanning USDT perpetuals across 7 timeframes..."):
    df_raw, n_pairs, n_signals, elapsed = run_full_scan(min_vol_value)

# ============================================================
# SUMMARY METRICS
# ============================================================
n_confluence = 0
if not df_raw.empty:
    closed_df = df_raw[df_raw["is_closed_os"]]
    n_confluence = int(
        (closed_df.groupby("base")["tf"].nunique() >= 2).sum()
    )

m1, m2, m3, m4 = st.columns(4)
m1.markdown(
    f'<div class="metric-card"><h3>Coins Scanned</h3>'
    f'<p>{n_pairs}</p></div>',
    unsafe_allow_html=True,
)
m2.markdown(
    f'<div class="metric-card"><h3>Oversold Signals</h3>'
    f'<p>{n_signals}</p></div>',
    unsafe_allow_html=True,
)
m3.markdown(
    f'<div class="metric-card"><h3>Multi-TF Confluence</h3>'
    f'<p>{n_confluence}</p></div>',
    unsafe_allow_html=True,
)

last_scan = datetime.now(timezone.utc).strftime("%d %b, %H:%M UTC")
m4.markdown(
    f'<div class="metric-card"><h3>Last Successful Scan</h3>'
    f'<p style="font-size:1.1rem">{last_scan}</p>'
    f'<small>Last scan: {elapsed:.1f}s</small></div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================
# FILTER BAR
# ============================================================
fc1 = st.columns([2, 1, 2, 3, 1.5])
search = fc1[0].text_input("Search Symbol", placeholder="e.g. AAVE")
rank_sel = fc1[1].selectbox(
    "Max MCAP Rank",
    ["All Ranks", "Top 50", "Top 100", "Top 250", "Top 500"],
    index=0,
)
class_sel = fc1[2].multiselect(
    "Class (RSI)", ["< 20", "20-25", "25-30"],
    default=["< 20", "20-25", "25-30"],
)
tf_sel = fc1[3].multiselect(
    "TF Filter", TF_LABELS, default=TF_LABELS
)
conf_sel = fc1[4].selectbox(
    "Confluence",
    ["At least 1 TF", "At least 2 TFs", "At least 3 TFs"],
    index=0,
)

live_preview = st.checkbox(
    "Live-candle preview "
    "(include rows where LIVE RSI < 30 even if RSI 14 ≥ 30)",
    value=False,
)

# ============================================================
# APPLY FILTERS
# ============================================================
df = df_raw.copy() if not df_raw.empty else pd.DataFrame()

if not df.empty:
    if not live_preview:
        df = df[df["is_closed_os"]]

    if search:
        df = df[
            df["base"].str.contains(
                search, case=False, na=False, regex=False
            )
        ]

    if rank_sel != "All Ranks":
        max_rank = int(rank_sel.replace("Top ", ""))
        df = df[
            df["mcap_rank"].notna() & (df["mcap_rank"] <= max_rank)
        ]

    if tf_sel:
        df = df[df["tf"].isin(tf_sel)]

    if class_sel:
        mask = pd.Series(False, index=df.index)
        if "< 20" in class_sel:
            mask |= (df["disp_rsi"] < 20)
        if "20-25" in class_sel:
            mask |= (df["disp_rsi"] >= 20) & (df["disp_rsi"] <= 25)
        if "25-30" in class_sel:
            mask |= (df["disp_rsi"] > 25) & (df["disp_rsi"] <= 30)
        df = df[mask]

    if conf_sel != "At least 1 TF":
        min_conf = 2 if conf_sel == "At least 2 TFs" else 3
        conf_counts = df.groupby("base")["tf"].nunique()
        valid = conf_counts[conf_counts >= min_conf].index
        df = df[df["base"].isin(valid)]

# ============================================================
# TABLE + EXPORT + TRADINGVIEW
# ============================================================
if df.empty:
    st.info(
        "No coins match the current filters. "
        "Try relaxing criteria or click Refresh Scan Now."
    )
else:
    df["conf_count"] = (
        df["tfs_oversold"].str.split(" | ").str[0].astype(int)
    )
    df = df.sort_values(
        ["conf_count", "rsi_closed"], ascending=[False, True]
    )

    display_df = pd.DataFrame({
        "MCAP RANK": df["mcap_rank"].apply(
            lambda x: f"#{int(x)}" if pd.notna(x) else "Unranked"
        ),
        "SYMBOL": df["base"] + " /USDT",
        "TF": df["tf"],
        "RSI 14": df["rsi_closed"],
        "LIVE RSI": df["rsi_live"],
        "DIST. TO 50": df["dist_to_50"].apply(lambda x: f"{x:+.2f}"),
        "STATUS": df["status"],
        "TFS OVERSOLD": df["tfs_oversold"],
        "LAST CANDLE CLOSE": df["last_close_ts"].apply(fmt_ts),
    })

    def style_status(v):
        if v == "Extremely Oversold":
            return "color: #ff4b4b; font-weight: bold"
        elif v == "Strongly Oversold":
            return "color: #ff8c00; font-weight: bold"
        return "color: #ffd700; font-weight: bold"

    def style_rsi(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return ""
        if f < 20:
            return "color: #ff4b4b; font-weight: bold"
        if f < 25:
            return "color: #ff8c00; font-weight: bold"
        if f < 30:
            return "color: #ffd700; font-weight: bold"
        return "color: #c9d1d9"

    styled = (
        display_df.style
        .map(style_status, subset=["STATUS"])
        .map(style_rsi, subset=["RSI 14", "LIVE RSI"])
    )

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=500,
    )

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export CSV",
        csv,
        "rsi_oversold_scan.csv",
        "text/csv",
    )

    st.markdown("---")
    st.subheader("📈 TradingView Chart")

    coin_sel = st.selectbox(
        "Select coin for chart", sorted(df["base"].unique())
    )

    if coin_sel:
        tv_prefix = "OKX" if ex_id_ui == "okx" else "KUCOIN"
        tv_symbol = f"{tv_prefix}:{coin_sel}USDT.P"

        tv_html = f"""
        <div id="tv_chart" style="height:520px"></div>
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>
        new TradingView.widget({{
            "container_id": "tv_chart",
            "symbol": "{tv_symbol}",
            "interval": "60",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "timezone": "Etc/UTC",
            "autosize": true,
            "hide_side_toolbar": false,
            "allow_symbol_change": true
        }});
        </script>
        """
        components.html(tv_html, height=540)
