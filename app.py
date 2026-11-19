import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    div[data-testid="stSidebar"] { background-color: #161b22; }
    .stSelectbox label, .stMultiSelect label, .stTextInput label { color: #8b949e !important; }
    .ticker-up { color: #3fb950; font-weight: 700; }
    .ticker-down { color: #f85149; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "45m": "45m",
    "1H": "1h",
    "2H": "2h",
    "4H": "4h",
}

VOL_MAP = {
    "$1M": 1_000_000,
    "$5M": 5_000_000,
    "$10M": 10_000_000,
    "$25M": 25_000_000,
    "$50M": 50_000_000,
}

def compute_wilder_rsi(closes: np.ndarray, period: int = 14):
    if closes is None or len(closes) < period + 2:
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
    return float(100.0 - (100.0 / (1.0 + rs)))

@st.cache_data(ttl=3600)
def fetch_coingecko_ranks():
    ranks = {}
    try:
        for page in range(1, 5):
            resp = requests.get(
                "https://coingecko.com",
                params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page},
                timeout=12,
            )
            if resp.status_code != 200:
                break
            for idx, coin in enumerate(resp.json()):
                sym = coin.get("symbol", "").upper()
                if sym and sym not in ranks:
                    ranks[sym] = (page - 1) * 250 + idx + 1
            time.sleep(0.25)
    except Exception:
        pass
    return ranks

def create_exchange(exchange_id: str):
    if exchange_id == "okx":
        return ccxt.okx({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    elif exchange_id == "kucoinfutures":
        return ccxt.kucoinfutures({"enableRateLimit": True})
    else:
        return ccxt.krakenfutures({"enableRateLimit": True})

@st.cache_data(ttl=300)
def get_active_perpetual_markets():
    for ex_id in ("okx", "kucoinfutures", "krakenfutures"):
        try:
            ex = create_exchange(ex_id)
            markets = ex.load_markets()
            tickers = ex.fetch_tickers()
            pairs = list()
            for sym, mkt in markets.items():
                if not mkt.get("active"):
                    continue
                is_swap = mkt.get("swap") or mkt.get("future") or mkt.get("linear")
                if not is_swap:
                    continue
                base = mkt.get("base", "").upper()
                quote = mkt.get("quote", "").upper()
                if quote not in ("USDT", "USD"):
                    continue

                t = tickers.get(sym, {})
                vol = t.get("quoteVolume")
                if not vol and t.get("baseVolume") and t.get("last"):
                    vol = float(t.get("baseVolume")) * float(t.get("last"))
                vol = float(vol or 0)

                pairs.append({
                    "exchange": ex_id,
                    "symbol": sym,
                    "base": base,
                    "quote": quote,
                    "volume": vol,
                })
            if len(pairs) > 20:
                return ex_id, pairs
        except Exception:
            continue
    return None, list()

def resample_to_45m(ohlcv_15m: list):
    if not ohlcv_15m or len(ohlcv_15m) < 45:
        return None
    cols = ("ts", "open", "high", "low", "close", "vol")
    df = pd.DataFrame(ohlcv_15m, columns=list(cols))
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")
    resampled = df.resample("45min").agg({
        "ts": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum"
    }).dropna()
    return resampled[list(cols)].values.tolist()

def scan_single_pair(pair_info: dict, cg_ranks: dict):
    ex = create_exchange(pair_info.get("exchange"))
    symbol = pair_info.get("symbol")
    base = pair_info.get("base")

    tf_data = {}
    ohlcv_15m_cache = None

    for tf_label, tf_ccxt in TIMEFRAMES.items():
        try:
            if tf_label == "45m":
                if ohlcv_15m_cache is None:
                    ohlcv_15m_cache = ex.fetch_ohlcv(symbol, "15m", limit=240)
                ohlcv = resample_to_45m(ohlcv_15m_cache)
            elif tf_label == "15m":
                ohlcv_15m_cache = ex.fetch_ohlcv(symbol, "15m", limit=240)
                ohlcv = ohlcv_15m_cache
            else:
                ohlcv = ex.fetch_ohlcv(symbol, tf_ccxt, limit=100)

            if not ohlcv or len(ohlcv) < 20:
                continue

            closes = np.array([c[4] for c in ohlcv], dtype=float)
            closed_rsi = compute_wilder_rsi(closes[:-1])
            live_rsi = compute_wilder_rsi(closes)

            if closed_rsi is None:
                continue

            last_closed_ts = ohlcv[-2][0]
            tf_data[tf_label] = {
                "closed_rsi": round(closed_rsi, 2),
                "live_rsi": round(live_rsi, 2) if live_rsi is not None else round(closed_rsi, 2),
                "last_ts": last_closed_ts,
            }
        except Exception:
            continue

    oversold_closed_tfs = [tf for tf, d in tf_data.items() if d.get("closed_rsi") < 30]
    tf_count = len(oversold_closed_tfs)
    tf_list_str = " · ".join(oversold_closed_tfs)

    results = list()
    for tf_label, d in tf_data.items():
        c_rsi = d.get("closed_rsi")
        l_rsi = d.get("live_rsi")
        if c_rsi < 30 or l_rsi < 30:
            ref_rsi = c_rsi if c_rsi < 30 else l_rsi
            if ref_rsi < 20:
                status = "Extremely Oversold"
            elif ref_rsi <= 25:
                status = "Strongly Oversold"
            else:
                status = "Oversold"

            close_dt = datetime.fromtimestamp(d.get("last_ts") / 1000, tz=timezone.utc).strftime("%d %b, %H:%M")
            results.append({
                "base": base,
                "symbol": symbol,
                "volume": pair_info.get("volume"),
                "tf": tf_label,
                "rsi_closed": c_rsi,
                "rsi_live": l_rsi,
                "dist_to_50": round(c_rsi - 50.0, 2),
                "status": status,
                "tfs_oversold": f"{tf_count} | {tf_list_str}" if tf_count > 0 else "0 (Live only)",
                "conf_count": tf_count,
                "last_candle_close": close_dt,
                "mcap_rank": cg_ranks.get(base, None),
            })

    return results

@st.cache_data(ttl=180, show_spinner=False)
def run_full_scan(min_vol_usd: float):
    start_time = time.time()
    cg_ranks = fetch_coingecko_ranks()
    ex_id, all_pairs = get_active_perpetual_markets()

    if not all_pairs:
        return pd.DataFrame(), 0, time.time() - start_time, "None"

    filtered_pairs = [p for p in all_pairs if p.get("volume") >= min_vol_usd]
    filtered_pairs.sort(key=lambda x: x.get("volume"), reverse=True)
    filtered_pairs = filtered_pairs[:180]

    all_results = list()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(scan_single_pair, p, cg_ranks) for p in filtered_pairs]
        for future in as_completed(futures):
            try:
                res = future.result()
                if res:
                    all_results.extend(res)
            except Exception:
                pass

    elapsed = time.time() - start_time
    df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return df, len(filtered_pairs), elapsed, ex_id.upper()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🔍 RSI Scanner")
    st.caption("USDT Perpetuals · CoinGecko Ranks")

    if st.button("🔄 Refresh Scan Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.subheader("Live Majors")
    try:
        ex_ticker = create_exchange("okx")
        for label, pair in (("BTC", "BTC/USDT:USDT"), ("ETH", "ETH/USDT:USDT")):
            try:
                t = ex_ticker.fetch_ticker(pair)
                pct = float(t.get("percentage") or 0)
                cls = "ticker-up" if pct >= 0 else "ticker-down"
                st.markdown(
                    f"<span style='color:#8b949e'>{label}</span> "
                    f"<span class='{cls}'>{pct:+.2f}%</span>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.caption(f"{label}: --")
    except Exception:
        st.caption("Majors ticker unavailable")

    st.divider()
    watchlist = st.multiselect(
        "⭐ Filter by Watchlist",
        options=list(("BTC", "ETH", "SOL", "UNI", "AAVE", "ARB", "LINK", "SUI", "AVAX", "DOGE")),
        default=list(),
    )

# ============================================================

FILTER BAR


============================================================

filter_cols = st.columns((2, 1, 1, 2, 1.5, 1, 0.8))
with filter_cols[0]:
search = st.text_input("Search Symbol", placeholder="e.g. UNI, AAVE")
with filter_cols[1]:
vol_label = st.selectbox("Min 24h Vol", list(VOL_MAP.keys()), index=1)
min_vol_value = VOL_MAP.get(vol_label)
with filter_cols[2]:
rank_filter = st.selectbox("Max MCAP Rank", list(("All Ranks", "Top 50", "Top 100", "Top 250", "Top 500")), index=0)
with filter_cols[3]:
tf_filter = st.multiselect("TF Filter", list(TIMEFRAMES.keys()), default=list(TIMEFRAMES.keys()))
with filter_cols[4]:
class_filter = st.multiselect("Class", list(("< 20", "20-25", "25-30")), default=list(("< 20", "20-25", "25-30")))
with filter_cols[5]:
conf_filter = st.selectbox("Confluence", list(("At least 1 TF", "At least 2 TFs", "At least 3 TFs")), index=0)
with filter_cols[6]:
st.write("")
live_preview = st.checkbox("Live preview", value=False)

============================================================


EXECUTE SCAN


============================================================

with st.spinner(f"Scanning perpetual pairs (Min Vol {vol_label}) across 7 timeframes..."):
df_raw, pairs_scanned, scan_seconds, active_exchange = run_full_scan(min_vol_value)

============================================================


APPLY FILTERS


============================================================

df = df_raw.copy() if not df_raw.empty else pd.DataFrame()
if not df.empty:
if not live_preview:
df = df[df["rsi_closed"] < 30]
else:
df = df[(df["rsi_closed"] < 30) | (df["rsi_live"] < 30)]
if watchlist:
df = df[df["base"].isin(watchlist)]
if search:
df = df[df["base"].str.contains(search.strip().upper(), na=False)]
if rank_filter != "All Ranks":
max_rank = int(rank_filter.replace("Top ", ""))
df = df[df["mcap_rank"].notna() & (df["mcap_rank"] <= max_rank)]
if tf_filter:
df = df[df["tf"].isin(tf_filter)]
if class_filter:
active_rsi = df["rsi_closed"] if not live_preview else np.minimum(df["rsi_closed"], df["rsi_live"])
masks = list()
if "< 20" in class_filter:
masks.append(active_rsi < 20)
if "20-25" in class_filter:
masks.append((active_rsi >= 20) & (active_rsi <= 25))
if "25-30" in class_filter:
masks.append((active_rsi > 25) & (active_rsi <= 30))
if masks:
combined = masks[0]
for m in masks[1:]:
combined = combined | m
df = df[combined]
if conf_filter == "At least 2 TFs":
df = df[df["conf_count"] >= 2]
elif conf_filter == "At least 3 TFs":
df = df[df["conf_count"] >= 3]

============================================================


TOP 4 SUMMARY METRIC CARDS


============================================================

total_signals = len(df) if not df.empty else 0
confluence_2plus = df[df["conf_count"] >= 2]["base"].nunique() if not df.empty else 0
last_scan_str = datetime.now(timezone.utc).strftime("%d %b, %H:%M UTC")
m_cols = st.columns(4)
m_cols[0].markdown(
f'Coins Scanned ({active_exchange}){pairs_scanned}',
unsafe_allow_html=True,
)
m_cols[1].markdown(
f'Oversold Signals{total_signals}',
unsafe_allow_html=True,
)
m_cols[2].markdown(
f'Multi-TF Confluence{confluence_2plus}',
unsafe_allow_html=True,
)
m_cols[3].markdown(
f'Last Successful Scan{last_scan_str}Scan time: {scan_seconds:.1f}s',
unsafe_allow_html=True,
)
st.markdown("---")

============================================================


DISPLAY DATA TABLE


============================================================

if df.empty:
st.warning(f"No oversold coins matched your filters on {active_exchange} ({pairs_scanned} pairs scanned). Try lowering Min 24h Vol to $1M or enabling Live preview.")
else:
sort_cols = list(("conf_count", "rsi_closed"))
sort_asc = list((False, True))
df = df.sort_values(sort_cols, ascending=sort_asc)
display_cols = list((
"mcap_rank", "base", "tf", "rsi_closed", "rsi_live", "dist_to_50", "status", "tfs_oversold", "last_candle_close"
))
display_df = df[display_cols].copy()
display_df.columns = list((
"MCAP RANK", "SYMBOL", "TF", "RSI 14", "LIVE RSI", "DIST. TO 50", "STATUS", "TFS OVERSOLD", "LAST CANDLE CLOSE"
))
display_df["MCAP RANK"] = display_df["MCAP RANK"].apply(lambda x: f"#{int(x)}" if pd.notna(x) else "Unranked")
display_df["SYMBOL"] = display_df["SYMBOL"] + " /USDT"
display_df["DIST. TO 50"] = display_df["DIST. TO 50"].apply(lambda x: f"{x:+.2f}")
def color_status(val):
if val == "Extremely Oversold":
return "color: #ff4b4b; font-weight: 700"
elif val == "Strongly Oversold":
return "color: #ff8c00; font-weight: 700"
return "color: #ffd700; font-weight: 700"
styled = display_df.style.map(color_status, subset=list(("STATUS",)))
st.dataframe(styled, use_container_width=True, hide_index=True, height=520)
csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Export CSV", csv, "rsi_oversold_scan.csv", "text/csv")

============================================================


TRADINGVIEW CHART DRAWER


============================================================

st.markdown("---")
st.subheader("📈 TradingView Chart")
if not df.empty:
coin_options = sorted(df["base"].unique())
selected_coin = st.selectbox("Select coin for live chart", coin_options)
if selected_coin:
tv_symbol = f"BYBIT:{selected_coin}USDT.P"
tv_html = f"""




new TradingView.widget({{
"autosize": true,
"symbol": "{tv_symbol}",
"interval": "15",
"timezone": "Etc/UTC",
"theme": "dark",
"style": "1",
"locale": "en",
"enable_publishing": false,
"allow_symbol_change": true,
"container_id": "tradingview_chart"
}});


"""
st.components.v1.html(tv_html, height=540)
