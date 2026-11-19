import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import requests
import threading
import time
from collections import deque
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIG & DARK THEME
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
    "5m": "5m", "15m": "15m", "30m": "30m",
    "45m": None,  # synthetic resample from 15m
    "1H": "1h", "2H": "2h", "4H": "4h",
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
    "$500K": 500_000,
    "$1M": 1_000_000,
    "$5M": 5_000_000,
    "$10M": 10_000_000,
    "$25M": 25_000_000,
}

TOP_N_PAIRS = 180
INITIAL_LIMIT = 100
DELTA_LIMIT = 3
RSI_PERIOD = 14
STORE_MAXLEN = 100
WORKERS = 8
REFRESH_SECS = 60

# ---------- Crypto exclusion sets (stocks, fiat, stables) ----------
EXCLUDE_STOCKS = {
    "NVDA", "GOOG", "GOOGL", "AAPL", "TSLA", "MSFT", "AMZN", "META",
    "NFLX", "SPY", "QQQ", "ARKK", "GLD", "SLV", "JPM", "V", "MA",
    "KO", "PEP", "DIS", "BA", "INTC", "AMD", "PYPL", "SNAP", "BRK",
    "COIN", "MSTR", "HOOD", "PLTR", "RIOT", "MARA",
}
EXCLUDE_FIAT = {
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "HKD",
    "SGD", "NZD", "SEK", "NOK", "DKK", "PLN", "ZAR", "MXN",
    "BRL", "INR", "KRW", "RUB", "TRY", "ILS",
}
EXCLUDE_STABLES = {
    "USDT", "USDC", "DAI", "TUSD", "BUSD", "FRAX", "USDD",
    "USDP", "GUSD", "UST", "USTC", "PYUSD", "FDUSD",
    "PAXG", "XAUT",
}
MEME_PREFIXES = ["1000000", "100000", "10000", "1000", "100", "1M"]

_fetch_lock = threading.Lock()

# ============================================================
# WILDER RSI(14) — pure numpy
# ============================================================
def wilder_rsi(closes, period=RSI_PERIOD):
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
# VOLUME CALC (handles None quoteVolume, contract size)
# ============================================================
def calc_volume_usdt(ticker, market):
    qv = ticker.get("quoteVolume")
    if qv:
        try:
            f = float(qv)
            if f > 0:
                return f
        except (TypeError, ValueError):
            pass
    try:
        base_vol = float(ticker.get("baseVolume") or 0)
    except (TypeError, ValueError):
        base_vol = 0.0
    try:
        last_price = float(ticker.get("last") or 0)
    except (TypeError, ValueError):
        last_price = 0.0
    try:
        contract_size = float(market.get("contractSize") or 1)
    except (TypeError, ValueError):
        contract_size = 1.0
    return base_vol * last_price * contract_size

# ============================================================
# EXCHANGE — Binance USDT-M Perpetual → Bybit USDT Perpetual
# ============================================================
@st.cache_resource
def get_exchange():
    # Binance USDⓈ-M perpetuals
    try:
        ex = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        ex.load_markets()
        if ex.markets:
            return ex, "binance"
    except Exception:
        pass

    # Bybit USDT perpetuals
    try:
        ex = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })
        ex.load_markets()
        if ex.markets:
            return ex, "bybit"
    except Exception:
        pass

    return None, None

# ============================================================
# COINGECKO top-1500 (verified crypto symbols + ranks)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_coingecko_set():
    symbols = set()
    ranks = {}
    try:
        for page in range(1, 7):
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
                if sym:
                    symbols.add(sym)
                    if sym not in ranks:
                        ranks[sym] = (page - 1) * 250 + i + 1
            time.sleep(0.3)
    except Exception:
        pass
    return symbols, ranks

# ============================================================
# CRYPTO-ONLY FILTER
# ============================================================
def normalize_base(base):
    b = (base or "").upper()
    for p in MEME_PREFIXES:
        if b.startswith(p) and len(b) > len(p) + 1:
            return b[len(p):]
    return b

def is_crypto(base, cg_symbols):
    b = (base or "").upper()
    if not b:
        return False
    if b in EXCLUDE_STOCKS or b in EXCLUDE_FIAT or b in EXCLUDE_STABLES:
        return False
    if b in cg_symbols:
        return True
    return normalize_base(b) in cg_symbols

# ============================================================
# PERSISTENT CANDLE STORE
# ============================================================
@st.cache_resource
def get_candle_store():
    # dict[symbol][tf] = deque(maxlen=100) of [ts, o, h, l, c, v]
    return {}

# ============================================================
# CANDLE FETCH & IN-MEMORY MERGE
# ============================================================
def _fetch_candles(ex, symbol, tf, limit):
    if ex is None:
        return []
    try:
        with _fetch_lock:
            result = ex.fetch_ohlcv(symbol, tf, limit=limit)
        return result or []
    except Exception:
        return []

def _update_deque(dq, new_candles):
    if not new_candles:
        return
    existing = {c[0]: i for i, c in enumerate(dq)}
    for nc in new_candles:
        ts = nc[0]
        if ts in existing:
            dq[existing[ts]] = nc
        else:
            dq.append(nc)

def refresh_pair(ex, symbol, store):
    if symbol not in store:
        store[symbol] = {}
    pair_store = store[symbol]
    for tf_label, tf_ccxt in TF_CCXT.items():
        if tf_ccxt is None:
            continue
        dq = pair_store.get(tf_label)
        if dq is None:
            candles = _fetch_candles(ex, symbol, tf_ccxt, INITIAL_LIMIT)
            pair_store[tf_label] = deque(candles, maxlen=STORE_MAXLEN)
        else:
            candles = _fetch_candles(ex, symbol, tf_ccxt, DELTA_LIMIT)
            _update_deque(dq, candles)

def refresh_all(ex, pairs, store):
    with ThreadPoolExecutor(max_workers=WORKERS) as exe:
        futures = [
            exe.submit(refresh_pair, ex, p["symbol"], store)
            for p in pairs
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass

# ============================================================
# PAIR DISCOVERY (crypto-only perpetuals, top 180 liquid)
# ============================================================
def _discover_pairs_impl(ex, cg_symbols, min_vol):
    if ex is None:
        return []
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        return []
    candidates = []
    for sym, mkt in ex.markets.items():
        if not mkt.get("active"):
            continue
        # Keep only linear USDT-margined perpetual swaps
        if not mkt.get("swap"):
            continue
        if not mkt.get("linear"):
            continue
        if mkt.get("quote") != "USDT":
            continue
        base = (mkt.get("base") or "").upper()
        if not is_crypto(base, cg_symbols):
            continue
        t = tickers.get(sym)
        if not t:
            continue
        vol = calc_volume_usdt(t, mkt)
        if vol >= min_vol:
            candidates.append({
                "symbol": sym,
                "base": base,
                "quote": "USDT",
                "volume": vol,
            })
    candidates.sort(key=lambda x: x["volume"], reverse=True)
    return candidates[:TOP_N_PAIRS]

@st.cache_data(ttl=60, show_spinner=False)
def discover_pairs_cached(ex_id, min_vol):
    ex, _ = get_exchange()
    if ex is None:
        return []
    cg_symbols, _ = fetch_coingecko_set()
    return _discover_pairs_impl(ex, cg_symbols, min_vol)

# ============================================================
# 45m RESAMPLE (from in-memory 15m deque)
# ============================================================
def build_45m_closes(candles_15m):
    if not candles_15m or len(candles_15m) < 30:
        return None, None
    try:
        df = pd.DataFrame(
            list(candles_15m),
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        df["dt"] = pd.to_datetime(
            df["ts"] + 15 * 60 * 1000, unit="ms", utc=True
        )
        df = df.set_index("dt")
        res = df.resample("45min", label="right", closed="right").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        if len(res) < 16:
            return None, None
        closes = res["close"].values.astype(float)
        last_ts = (
            int(res.index[-2].timestamp() * 1000)
            if len(res) >= 2 else None
        )
        return closes, last_ts
    except Exception:
        return None, None

# ============================================================
# COMPUTE RSI TABLE FROM IN-MEMORY STORE
# ============================================================
def compute_rsi_table(store, pairs, cg_ranks):
    all_rows = []
    for p in pairs:
        symbol = p["symbol"]
        base = p["base"]
        if symbol not in store:
            continue
        pair_store = store[symbol]
        pair_rows = []
        closed_os_tfs = []

        for tf in TF_LABELS:
            closes = None
            last_ts = None

            if tf == "45m":
                dq_15 = pair_store.get("15m")
                if not dq_15 or len(dq_15) < 30:
                    continue
                closes, last_ts = build_45m_closes(list(dq_15))
            else:
                dq = pair_store.get(tf)
                if not dq or len(dq) < 16:
                    continue
                candles = list(dq)
                closes = np.array([c[4] for c in candles], dtype=float)
                last_ts = (
                    int(candles[-2][0]) + TF_MS[tf]
                    if len(candles) >= 2 else None
                )

            if closes is None or len(closes) < 16:
                continue

            rsi_c = wilder_rsi(closes[:-1])   # closed-candle RSI
            rsi_l = wilder_rsi(closes)        # live forming-candle RSI
            if rsi_c is None:
                continue

            is_closed_os = rsi_c < 30
            is_live_os = rsi_l is not None and rsi_l < 30
            if not (is_closed_os or is_live_os):
                continue

            disp = rsi_c if is_closed_os else rsi_l
            if disp < 20:
                status = "Extremely Oversold"
            elif disp <= 25:
                status = "Strongly Oversold"
            else:
                status = "Oversold"

            if is_closed_os:
                closed_os_tfs.append(tf)

            pair_rows.append({
                "base": base,
                "symbol": symbol,
                "quote": p["quote"],
                "tf": tf,
                "rsi_closed": round(float(rsi_c), 2),
                "rsi_live": (
                    round(float(rsi_l), 2) if rsi_l is not None else None
                ),
                "disp_rsi": round(float(disp), 2),
                "is_closed_os": is_closed_os,
                "dist_to_50": round(float(disp) - 50, 2),
                "status": status,
                "last_close_ts": last_ts,
                "mcap_rank": cg_ranks.get(base),
            })

        if pair_rows:
            cnt = len(closed_os_tfs)
            tf_str = (
                f"{cnt} | " + " · ".join(closed_os_tfs)
                if closed_os_tfs else "0"
            )
            for r in pair_rows:
                r["tfs_oversold"] = tf_str
            all_rows.extend(pair_rows)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

# ============================================================
# LIVE MAJORS TICKER
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_majors_ticker():
    ex, _ = get_exchange()
    if ex is None:
        return {}
    out = {}
    for base in ["BTC", "ETH"]:
        # On Binance futures / Bybit swap the symbol is BASE/USDT:USDT
        sym = f"{base}/USDT:USDT"
        try:
            if sym in ex.markets:
                t = ex.fetch_ticker(sym)
                out[base] = t.get("percentage")
        except Exception:
            continue
    return out

# ============================================================
# FORMATTERS
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
# SIDEBAR (persistent, outside fragment)
# ============================================================
ex_ui, ex_id_ui = get_exchange()

with st.sidebar:
    st.title("🔍 RSI Scanner")
    st.caption(f"Exchange: {(ex_id_ui or 'N/A').upper()} Perpetuals")
    if st.button("🔄 Refresh Scan Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.subheader("Live Majors")
    majors = get_majors_ticker()
    for base in ["BTC", "ETH"]:
        pct = majors.get(base)
        try:
            if pct is None or (isinstance(pct, float) and np.isnan(pct)):
                st.markdown(f"**{base}**: —")
            else:
                f = float(pct)
                cls = "ticker-up" if f >= 0 else "ticker-down"
                st.markdown(
                    f"**{base}**: <span class='{cls}'>{f:+.2f}%</span>",
                    unsafe_allow_html=True,
                )
        except (TypeError, ValueError):
            st.markdown(f"**{base}**: —")
    st.divider()
    st.multiselect(
        "⭐ My Watchlist",
        ["BTC", "ETH", "SOL", "UNI", "AAVE", "ARB", "LINK", "OP"],
        default=[], key="watchlist",
    )

if ex_ui is None:
    st.error("❌ Could not connect to Binance Futures or Bybit Perpetuals. Click Refresh.")
    st.stop()

# ============================================================
# AUTO-REFRESHING SCANNER FRAGMENT
# ============================================================
@st.fragment(run_every=REFRESH_SECS)
def auto_scanner():
    t_start = time.time()
    ex, ex_id = get_exchange()
    cg_symbols, cg_ranks = fetch_coingecko_set()
    store = get_candle_store()

    # ---- Volume control ----
    vol_label = st.selectbox(
        "Min 24h Vol",
        list(VOL_OPTIONS.keys()),
        index=2,
        key="vol_selector",
    )
    min_vol = VOL_OPTIONS[vol_label]

    # ---- Discover crypto perpetual pairs ----
    pairs = discover_pairs_cached(ex_id, min_vol)

    # ---- Decide refresh: 60s tick OR pair set changed ----
    prev_symbols = st.session_state.get("_last_pair_symbols")
    prev_ts = st.session_state.get("_last_refresh_ts", 0)
    pair_symbols = frozenset(p["symbol"] for p in pairs)
    symbols_changed = prev_symbols != pair_symbols
    time_elapsed = (time.time() - prev_ts) > (REFRESH_SECS - 10)
    is_first_load = not store

    if symbols_changed or time_elapsed or is_first_load:
        label = "Initial load" if is_first_load else "Delta refresh"
        with st.spinner(f"{label}: {len(pairs)} perpetual pairs..."):
            refresh_all(ex, pairs, store)
        st.session_state["_last_pair_symbols"] = pair_symbols
        st.session_state["_last_refresh_ts"] = time.time()

    # ---- Compute RSI table ----
    df_raw = compute_rsi_table(store, pairs, cg_ranks)
    elapsed = time.time() - t_start

    # ---- Summary metrics ----
    n_pairs = len(pairs)
    n_signals = len(df_raw) if not df_raw.empty else 0
    n_confluence = 0
    if not df_raw.empty:
        closed = df_raw[df_raw["is_closed_os"]]
        n_confluence = int(
            (closed.groupby("base")["tf"].nunique() >= 2).sum()
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f'<div class="metric-card"><h3>Coins Scanned</h3>'
        f'<p>{n_pairs}</p></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><h3>Oversold Signals</h3>'
        f'<p>{n_signals}</p></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><h3>Multi-TF Confluence</h3>'
        f'<p>{n_confluence}</p></div>',
        unsafe_allow_html=True,
    )
    last_scan = datetime.now(timezone.utc).strftime("%d %b, %H:%M UTC")
    c4.markdown(
        f'<div class="metric-card"><h3>Last Successful Scan</h3>'
        f'<p style="font-size:1.05rem">{last_scan}</p>'
        f'<small>Scan: {elapsed:.1f}s</small></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- Filter bar ----
    f1 = st.columns([2, 1.2, 2, 3, 1.5])
    search = f1[0].text_input(
        "Search Symbol", key="f_search", placeholder="e.g. AAVE"
    )
    rank_sel = f1[1].selectbox(
        "Max MCAP Rank",
        ["All Ranks", "Top 50", "Top 100", "Top 250", "Top 500"],
        index=0,
        key="f_rank",
    )
    class_sel = f1[2].multiselect(
        "Class (RSI)",
        ["< 20", "20-25", "25-30"],
        default=["< 20", "20-25", "25-30"],
        key="f_class",
    )
    tf_sel = f1[3].multiselect(
        "TF Filter", TF_LABELS, default=TF_LABELS, key="f_tf"
    )
    conf_sel = f1[4].selectbox(
        "Confluence",
        ["At least 1 TF", "At least 2 TFs", "At least 3 TFs"],
        index=0,
        key="f_conf",
    )
    live_preview = st.checkbox(
        "Live-candle preview "
        "(include LIVE RSI < 30 even if RSI 14 ≥ 30)",
        value=False,
        key="f_live",
    )

    # ---- Apply filters ----
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

    # ---- Render table ----
    if df.empty:
        st.info(
            "No coins match the current filters. "
            "Try relaxing criteria or click Refresh."
        )
        return

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
        "SYMBOL": df["base"] + " /" + df["quote"],
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
        if v == "Strongly Oversold":
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
        styled, use_container_width=True, hide_index=True, height=500
    )

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export CSV",
        csv,
        "rsi_oversold_scan.csv",
        "text/csv",
        key="csv_dl",
    )

    # ---- TradingView chart (USDT perpetual) ----
    st.markdown("---")
    st.subheader("📈 TradingView Chart (USDT Perpetual)")

    df["_base_quote"] = df["base"] + "|" + df["quote"]
    options = sorted(df["_base_quote"].unique())
    sel = st.selectbox("Select coin", options, key="chart_sel")
    if sel:
        b, q = sel.split("|")
        # Binance USDT-M perpetual  →  BINANCE:BTCUSDT.P
        # Bybit USDT perpetual      →  BYBIT:BTCUSDT.P
        if ex_id == "binance":
            tv_symbol = f"BINANCE:{b}{q}.P"
        else:
            tv_symbol = f"BYBIT:{b}{q}.P"

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

# ============================================================
# RUN
# ============================================================
auto_scanner()
