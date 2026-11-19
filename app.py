import streamlit as st
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
st.set_page_config(layout="wide", page_title="OKX RSI Oversold Scanner", page_icon="🔍")

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
OKX_API = "https://www.okx.com/api/v5"

TF_LABELS = ["5m", "15m", "30m", "45m", "1H", "2H", "4H"]
TF_OKX_BAR = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "45m": None,     # synthetic — resampled from 15m
    "1H": "1H",
    "2H": "2H",
    "4H": "4H",
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

TOP_N_PAIRS = 150
RSI_PERIOD = 14
STORE_MAXLEN = 100
WORKERS = 20
REFRESH_SECS = 60

_fetch_lock = threading.Lock()

# ============================================================
# WILDER'S RSI(14) — pure numpy
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
# OKX V5 REST HELPERS
# ============================================================
def fetch_swap_tickers():
    """Single call — all OKX swap tickers."""
    try:
        r = requests.get(
            f"{OKX_API}/market/tickers",
            params={"instType": "SWAP"},
            timeout=15,
        )
        if r.status_code != 200:
            return []
        payload = r.json()
        if payload.get("code") != "0":
            return []
        return payload.get("data", [])
    except Exception:
        return []

def fetch_candles(inst_id, bar, limit, retries=2):
    """Fetch OKX candles. OKX returns newest-first → reverse to oldest-first."""
    for attempt in range(retries):
        try:
            r = requests.get(
                f"{OKX_API}/market/candles",
                params={"instId": inst_id, "bar": bar, "limit": limit},
                timeout=10,
            )
            if r.status_code == 429:
                time.sleep(0.3 * (attempt + 1))
                continue
            if r.status_code != 200:
                return []
            payload = r.json()
            if payload.get("code") != "0":
                return []
            data = payload.get("data", [])
            if not data:
                return []
            out = []
            for row in data[::-1]:  # reverse to chronological order
                try:
                    out.append([
                        int(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    ])
                except (ValueError, IndexError):
                    continue
            return out
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.3)
                continue
            return []
    return []

# ============================================================
# PAIR DISCOVERY (1-call tickers, cached 60s)
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def discover_pairs(min_vol):
    tickers = fetch_swap_tickers()
    pairs = []
    for t in tickers:
        inst_id = t.get("instId", "")
        if not inst_id.endswith("-USDT-SWAP"):
            continue
        try:
            vol_ccy = float(t.get("volCcy24h") or 0)
            last = float(t.get("last") or 0)
            vol = vol_ccy * last   # base coins × last price = USDT volume
        except (TypeError, ValueError):
            continue
        if vol >= min_vol:
            pairs.append({
                "instId": inst_id,
                "base": inst_id.split("-")[0],
                "volume": vol,
            })
    pairs.sort(key=lambda x: x["volume"], reverse=True)
    return pairs[:TOP_N_PAIRS]

# ============================================================
# ROLLING CANDLE STORE (persistent, cache_resource)
# ============================================================
@st.cache_resource
def get_candle_store():
    # {instId: {tf_label: deque(maxlen=100) of [ts,o,h,l,c,v]}}
    return {}

def update_deque(dq, new_candles):
    """Merge new candles into deque — replace on same ts, append otherwise."""
    if not new_candles:
        return
    existing = {c[0]: i for i, c in enumerate(dq)}
    for nc in new_candles:
        ts = nc[0]
        if ts in existing:
            dq[existing[ts]] = nc
        else:
            dq.append(nc)

def refresh_pair(inst_id, store):
    if inst_id not in store:
        store[inst_id] = {}
    pair_store = store[inst_id]
    for tf_label, tf_bar in TF_OKX_BAR.items():
        if tf_bar is None:
            continue
        dq = pair_store.get(tf_label)
        if dq is None:
            candles = fetch_candles(inst_id, tf_bar, 100)
            pair_store[tf_label] = deque(candles, maxlen=STORE_MAXLEN)
        else:
            candles = fetch_candles(inst_id, tf_bar, 3)
            update_deque(dq, candles)

def refresh_all(pairs, store):
    with ThreadPoolExecutor(max_workers=WORKERS) as exe:
        futures = [exe.submit(refresh_pair, p["instId"], store) for p in pairs]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass

# ============================================================
# COINGECKO RANKS (cached 1 hour)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_coingecko_ranks():
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
                if sym and sym not in ranks:
                    ranks[sym] = (page - 1) * 250 + i + 1
            time.sleep(0.3)
    except Exception:
        pass
    return ranks

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
        df["dt"] = pd.to_datetime(df["ts"] + 15 * 60 * 1000, unit="ms", utc=True)
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
        last_ts = int(res.index[-2].timestamp() * 1000) if len(res) >= 2 else None
        return closes, last_ts
    except Exception:
        return None, None

# ============================================================
# RSI TABLE FROM IN-MEMORY STORE
# ============================================================
def compute_rsi_table(store, pairs, cg_ranks):
    all_rows = []
    for p in pairs:
        inst_id = p["instId"]
        base = p["base"]
        if inst_id not in store:
            continue
        pair_store = store[inst_id]
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
                "instId": inst_id,
                "tf": tf,
                "rsi_closed": round(float(rsi_c), 2),
                "rsi_live": round(float(rsi_l), 2) if rsi_l is not None else None,
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
# LIVE MAJORS (cached 60s)
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def get_majors_ticker():
    out = {}
    for base in ["BTC", "ETH"]:
        try:
            r = requests.get(
                f"{OKX_API}/market/ticker",
                params={"instId": f"{base}-USDT-SWAP"},
                timeout=10,
            )
            if r.status_code == 200:
                payload = r.json()
                if payload.get("code") == "0" and payload.get("data"):
                    t = payload["data"][0]
                    last = float(t.get("last") or 0)
                    open24h = float(t.get("open24h") or 0)
                    if open24h > 0 and last > 0:
                        out[base] = (last - open24h) / open24h * 100.0
                    else:
                        out[base] = None
                else:
                    out[base] = None
            else:
                out[base] = None
        except Exception:
            out[base] = None
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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🔍 OKX RSI Scanner")
    st.caption("OKX V5 REST · USDT Perpetuals")

    if st.button("🔄 Refresh Scan Now", use_container_width=True):
        st.cache_data.clear()
        get_candle_store().clear()
        st.session_state.pop("_last_refresh_ts", None)
        st.session_state.pop("_last_pair_set", None)
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
        key="watchlist",
    )

# ============================================================
# AUTO-REFRESHING SCANNER FRAGMENT
# ============================================================
@st.fragment(run_every=REFRESH_SECS)
def auto_scanner():
    t_start = time.time()
    store = get_candle_store()
    cg_ranks = fetch_coingecko_ranks()

    # ---- Volume filter (feeds scan) ----
    vol_label = st.selectbox(
        "Min 24h Vol",
        list(VOL_OPTIONS.keys()),
        index=2,   # $5M default
        key="vol_selector",
    )
    min_vol = VOL_OPTIONS[vol_label]

    # ---- Discovery (1-call tickers, cached) ----
    pairs = discover_pairs(min_vol)

    # ---- Decide whether to run initial load / delta refresh ----
    prev_ts = st.session_state.get("_last_refresh_ts", 0)
    prev_pairs = st.session_state.get("_last_pair_set")
    pair_set = frozenset(p["instId"] for p in pairs)
    pairs_changed = prev_pairs != pair_set
    time_elapsed = (time.time() - prev_ts) > (REFRESH_SECS - 10)
    is_first = not store

    if pairs_changed or time_elapsed or is_first:
        label = "Initial load" if is_first else "Delta refresh"
        with st.spinner(f"{label}: {len(pairs)} OKX USDT perpetuals..."):
            refresh_all(pairs, store)
        st.session_state["_last_pair_set"] = pair_set
        st.session_state["_last_refresh_ts"] = time.time()

    # ---- Compute RSI table from in-memory deques ----
    df_raw = compute_rsi_table(store, pairs, cg_ranks)
    elapsed = time.time() - t_start

    # ---- Summary metrics ----
    n_pairs = len(pairs)
    n_signals = len(df_raw) if not df_raw.empty else 0
    n_confluence = 0
    if not df_raw.empty:
        closed = df_raw[df_raw["is_closed_os"]]
        n_confluence = int((closed.groupby("base")["tf"].nunique() >= 2).sum())

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
        "Live-candle preview (include LIVE RSI < 30 rows)",
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
            df = df[df["mcap_rank"].notna() & (df["mcap_rank"] <= max_rank)]
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

    df["conf_count"] = df["tfs_oversold"].str.split(" | ").str[0].astype(int)
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

    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export CSV",
        csv,
        "rsi_oversold_scan.csv",
        "text/csv",
        key="csv_dl",
    )

    # ---- TradingView OKX USDT Perpetual chart ----
    st.markdown("---")
    st.subheader("📈 TradingView Chart (OKX USDT Perpetual)")

    coin_sel = st.selectbox(
        "Select coin for chart",
        sorted(df["base"].unique()),
        key="chart_sel",
    )
    if coin_sel:
        tv_symbol = f"OKX:{coin_sel}USDT.P"
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
