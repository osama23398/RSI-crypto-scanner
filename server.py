import os
import json
import time
import math
import threading
import requests
import pandas as pd
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================
ALL_TFS = ['3m','5m','15m','30m','45m','1H','2H','4H','6H','12H','1D','1W']
MAGIC_TFS = ['15m','1H','4H','6H','12H','1D']
CONF_TF = {'15m':'30m','1H':'2H','4H':'6H','6H':'12H','12H':'1D','1D':'1W'}
MAGIC_LOOKBACK = 100
MAGIC_RSI_LEVEL = 30
MAGIC_CONF_BARS = 3
OVERSOLD_TFS = ['5m','15m','30m','45m','1H','2H','4H']
RETEST_TFS = ['3m','5m','15m','30m','45m','1H','4H']
# Change from 16 to 3 or 4. 
# This spaces out the requests enough to stay under the radar.
CONCURRENCY = 10 
SCAN_INTERVAL_SEC = 180
MAX_SYMBOLS = 2000
MIN_VOL = 1000000
MAGIC_MIN_VOL = 5000000

TF_BYBIT = {'3m':'3','5m':'5','15m':'15','30m':'30','1H':'60','2H':'120','4H':'240','6H':'360','12H':'720','1D':'D','1W':'W'}
TF_BINANCE = {'3m':'3m','5m':'5m','15m':'15m','30m':'30m','1H':'1h','2H':'2h','4H':'4h','6H':'6h','12H':'12h','1D':'1d','1W':'1w'}
TF_MS = {'3m':180000,'5m':300000,'15m':900000,'30m':1800000,'45m':2700000,'1H':3600000,'2H':7200000,'4H':14400000,'6H':21600000,'12H':43200000,'1D':86400000,'1W':604800000}

EXCLUDED_SYMBOLS = {'QQQUSDT','XAUUSDT','XAGUSDT','CLUSDT','EURUSDT','GBPUSDT','AMZNUSDT','NVDAUSDT','AAPLUSDT','TSLAUSDT','GOOGUSDT','MSFTUSDT','METAUSDT','NFLXUSDT','COINUSDT'}
COMM_SYMBOLS = ['QQQUSDT','XAUUSDT','XAGUSDT','CLUSDT']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKED_FILE = os.path.join(BASE_DIR, 'tracked.json')
RANKS_FILE = os.path.join(BASE_DIR, 'ranks.json')
SIGNALS_FILE = os.path.join(BASE_DIR, 'signals.json')

# ==========================================
# STATE MANAGEMENT
# ==========================================
STATE = {
    'rows': {'magic': [], 'magicrec': [], 'retest': [], 'oversold': [], 'funding': []},
    'tickers': [],
    'ranks': {},
    'commodities': [],
    'alts': {'bull': [], 'bear': []},
    'scanStats': {'coins': 0, 'calls': 0},
    'lastScan': None,
    'nextScanTs': None,
    'loading': False,
    'progress': 0
}
STATE_LOCK = threading.Lock()
SCAN_EVENT = threading.Event()
TRACKED_ZONES = []
KLINE_CACHE = {}   # <--- ADD THIS LINE
# ==========================================
# SAFE CONVERSION HELPERS
# ==========================================
def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return default
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    try:
        return int(safe_float(val, default))
    except (ValueError, TypeError):
        return default

# ==========================================
# MATH ENGINE
# ==========================================
def wilder_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0: gains += d
        else: losses -= d
    ag, al = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + (d if d > 0 else 0)) / period
        al = (al * (period - 1) + (-d if d < 0 else 0)) / period
    if al == 0: return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)

def rsi_series(closes, period=14):
    if len(closes) < period + 2: return []
    out = []
    g, l = 0.0, 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0: g += d
        else: l -= d
    ag, al = g / period, l / period
    out.append(100.0 - 100.0 / (1.0 + ag / (al if al > 0 else 1e-9)))
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + (d if d > 0 else 0)) / period
        al = (al * (period - 1) + (-d if d < 0 else 0)) / period
        out.append(100.0 - 100.0 / (1.0 + ag / (al if al > 0 else 1e-9)))
    return out

def normalize_base(base):
    if not base: return ''
    b = base.upper()
    for p in ['1000000', '100000', '10000', '1000', '1M']:
        if b.startswith(p): return b[len(p):]
    return b

# ==========================================
# DATA LAYER
# ==========================================
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

# Add these 3 lines near the top of server.py (under CONCURRENCY)
MAX_RPS = 40
RATE_LOCK = threading.Lock()
RATE_STATE = {'count': 0, 'window_start': time.time()}

def fetch_json(url, timeout=12, max_retries=3):
    global RATE_STATE
    for attempt in range(max_retries):
        with RATE_LOCK:
            now = time.time()
            # Reset counter if 1 second has passed
            if now - RATE_STATE['window_start'] >= 1.0:
                RATE_STATE['count'] = 0
                RATE_STATE['window_start'] = now
            
            # If we hit 40 requests, sleep until the next second
            if RATE_STATE['count'] >= MAX_RPS:
                sleep_time = 1.0 - (now - RATE_STATE['window_start'])
                if sleep_time > 0: time.sleep(sleep_time)
                RATE_STATE['count'] = 0
                RATE_STATE['window_start'] = time.time()
            
            RATE_STATE['count'] += 1

        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                sleep_time = (2 ** attempt) + 1
                print(f"Rate limited by {url.split('/')[2]}. Sleeping {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                break
        except Exception:
            time.sleep(1)
    return None

def bybit_instruments():
    d = fetch_json('https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000')
    if not d or 'result' not in d: return []
    return [{'symbol': i['symbol'], 'base': i['baseCoin'], 'exchange': 'BYBIT'} for i in d['result'].get('list', []) if i.get('quoteCoin') == 'USDT' and i.get('status') == 'Trading' and i.get('contractType') == 'LinearPerpetual']

def bybit_tickers():
    d = fetch_json('https://api.bybit.com/v5/market/tickers?category=linear')
    if not d or 'result' not in d: return []
    return [{
        'symbol': t['symbol'],
        'last': safe_float(t.get('lastPrice')),
        'change24h': safe_float(t.get('price24hPcnt')) * 100,
        'vol': safe_float(t.get('turnover24h')),
        'funding': safe_float(t.get('fundingRate')),
        'nextFunding': safe_int(t.get('nextFundingTime')),
        'mark': safe_float(t.get('markPrice')),
        'exchange': 'BYBIT'
    } for t in d['result'].get('list', [])]

def bybit_klines(symbol, tf, limit=1000):
    iv = TF_BYBIT.get(tf)
    if not iv: return []
    d = fetch_json(f'https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={iv}&limit={limit}')
    if not d or 'result' not in d: return []
    return [{
        'ts': safe_int(k[0]),
        'open': safe_float(k[1]),
        'high': safe_float(k[2]),
        'low': safe_float(k[3]),
        'close': safe_float(k[4]),
        'volume': safe_float(k[5])
    } for k in reversed(d['result'].get('list', []))]

def binance_instruments():
    d = fetch_json('https://fapi.binance.com/fapi/v1/exchangeInfo')
    if not d: return []
    return [{'symbol': s['symbol'], 'base': s['baseAsset'], 'exchange': 'BINANCE'} for s in d.get('symbols', []) if s.get('contractType') == 'PERPETUAL' and s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING']

def binance_tickers():
    d = fetch_json('https://fapi.binance.com/fapi/v1/ticker/24hr')
    if not d: return []
    return [{
        'symbol': t['symbol'],
        'last': safe_float(t.get('lastPrice')),
        'change24h': safe_float(t.get('priceChangePercent')),
        'vol': safe_float(t.get('quoteVolume')),
        'funding': None,
        'nextFunding': None,
        'mark': safe_float(t.get('lastPrice')),
        'exchange': 'BINANCE'
    } for t in d if t.get('symbol', '').endswith('USDT')]

def binance_funding():
    d = fetch_json('https://fapi.binance.com/fapi/v1/premiumIndex')
    if not d: return {}
    return {x['symbol']: {
        'funding': safe_float(x.get('lastFundingRate')),
        'nextFunding': safe_int(x.get('nextFundingTime')),
        'mark': safe_float(x.get('markPrice'))
    } for x in d if x.get('symbol')}

def binance_klines(symbol, tf, limit=1000):
    iv = TF_BINANCE.get(tf)
    if not iv: return []
    d = fetch_json(f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={iv}&limit={limit}')
    if not d: return []
    return [{
        'ts': safe_int(k[0]),
        'open': safe_float(k[1]),
        'high': safe_float(k[2]),
        'low': safe_float(k[3]),
        'close': safe_float(k[4]),
        'volume': safe_float(k[5])
    } for k in d]

def resample_45m(c15):
    if not c15: return []
    buckets = {}
    for c in c15:
        b = (c['ts'] // 2700000) * 2700000
        if b not in buckets: buckets[b] = dict(c, ts=b)
        else:
            buckets[b]['high'] = max(buckets[b]['high'], c['high'])
            buckets[b]['low'] = min(buckets[b]['low'], c['low'])
            buckets[b]['close'] = c['close']
            buckets[b]['volume'] += c['volume']
    return [buckets[k] for k in sorted(buckets.keys())]

def get_cg_ranks():
    if os.path.exists(RANKS_FILE) and time.time() - os.path.getmtime(RANKS_FILE) < 21600:
        try:
            with open(RANKS_FILE, 'r') as f: return json.load(f)
        except: pass
    ranks = {}
    try:
        for p in range(1, 7):
            d = fetch_json(f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={p}&sparkline=false')
            if not d: break
            for c in d: ranks[c['symbol'].upper()] = c['market_cap_rank']
            if len(d) < 250: break
            time.sleep(1)
        with open(RANKS_FILE, 'w') as f: json.dump(ranks, f)
    except: pass
    return ranks

def fetch_all_klines(symbol, exchange, tfs):
    out = {}
    now = time.time() * 1000
    
    for tf in tfs:
        cache_key = f"{symbol}_{exchange}_{tf}"
        
        # Determine how long we can safely cache this timeframe
        if tf == '45m':
            ttl = 900000  # Cache 45m for 15 mins
        else:
            # Cache for half the timeframe duration (e.g., 1H caches for 30 mins)
            # But minimum 60 seconds so we don't spam 3m/5m charts
            ttl = max(TF_MS[tf] * 0.5, 60000) 
            
        if cache_key in KLINE_CACHE:
            cached_time, cached_data = KLINE_CACHE[cache_key]
            if (now - cached_time) < ttl:
                out[tf] = cached_data
                continue # Skip fetching, use cache
                
        # If no cache or expired, fetch from API
        try:
            if tf == '45m':
                c15 = bybit_klines(symbol, '15m', 240) if exchange == 'BYBIT' else binance_klines(symbol, '15m', 240)
                data = resample_45m(c15)
            else:
                data = bybit_klines(symbol, tf) if exchange == 'BYBIT' else binance_klines(symbol, tf)
                
            KLINE_CACHE[cache_key] = (now, data)
            out[tf] = data
        except:
            out[tf] = []
            
    return out
# ==========================================
# STRATEGIES
# ==========================================
def detect_magic(candles, tf, conf_candles):
    if tf not in MAGIC_TFS or not candles or len(candles) < 110: return []
    n = len(candles)
    req = 3 if tf in ['15m', '1H'] else 2
    if n < MAGIC_LOOKBACK + req + 5: return []

    period = 14
    closes = [c['close'] for c in candles]
    rsi_raw = rsi_series(closes, period)
    rsi = [None] * n
    for k in range(len(rsi_raw)): rsi[k + period] = rsi_raw[k]

    conf_closed, conf_rsi = [], []
    if conf_candles and len(conf_candles) > period + 5:
        conf_closed = conf_candles
        conf_rsi = rsi_series([c['close'] for c in conf_closed], period)

    current_close = candles[-1]['close']
    live = candles[-1]['close']
    out = []
    start_idx = max(period + 1, MAGIC_LOOKBACK)

    for i in range(start_idx, n):
        if rsi[i] is None or rsi[i] >= MAGIC_RSI_LEVEL: continue
        if i > 0 and rsi[i-1] is not None and rsi[i-1] < MAGIC_RSI_LEVEL: continue
        if candles[i]['close'] >= candles[i]['open']: continue

        run_ok, mn = True, float('inf')
        for j in range(i, i + req):
            if j >= n or rsi[j] is None or rsi[j] >= MAGIC_RSI_LEVEL:
                run_ok = False; break
            if rsi[j] < mn: mn = rsi[j]
        if not run_ok: continue

        prev = candles[i-1]
        top, bot, bp = prev['high'], prev['low'], candles[i]['close']
        if top <= bot: continue

        peak_high, peak_idx = -float('inf'), i - 1
        lb0 = max(0, i - MAGIC_LOOKBACK)
        for k in range(lb0, i):
            if candles[k]['high'] > peak_high:
                peak_high = candles[k]['high']; peak_idx = k
        pt = candles[peak_idx]['ts']

        confirmed = False
        if conf_closed:
            count = 0
            for j in range(len(conf_closed)):
                if conf_closed[j]['ts'] <= candles[i]['ts']: continue
                cR = conf_rsi[j - period] if j >= period else None
                if cR is None: continue
                if cR < MAGIC_RSI_LEVEL:
                    count += 1
                    if count > MAGIC_CONF_BARS: confirmed = True; break
                else: count = 0
        if not confirmed: continue

        grade = 'B'
        if conf_closed:
            best_j = -1
            for j in range(len(conf_closed)):
                if conf_closed[j]['ts'] <= candles[i]['ts']: best_j = j
                else: break
            if best_j >= period and conf_rsi[best_j - period] is not None and conf_rsi[best_j - period] < MAGIC_RSI_LEVEL:
                grade = 'A'

        dp = ((peak_high - bp) / peak_high) * 100 if peak_high > 0 else 0
        v3, c3, v20, c20 = 0, 0, 0, 0
        for k in range(i, min(i + 3, n)): v3 += candles[k]['volume']; c3 += 1
        for k in range(max(0, i - 20), i): v20 += candles[k]['volume']; c20 += 1
        vr = (v3 / c3) / (v20 / c20) if c3 > 0 and c20 > 0 and v20 > 0 else 0

        if current_close > top: continue
        below_bot_pct = ((bot - current_close) / bot * 100) if current_close < bot else 0
        if below_bot_pct > 5: continue

        dist = below_bot_pct if current_close < bot else abs(current_close - top) / top * 100

        out.append({
            't': candles[i]['ts'], 'cf': candles[min(i + req - 1, n - 1)]['ts'],
            'pt': pt, 'top': top, 'bot': bot, 'bp': bp, 'mn': mn,
            'g': grade, 'grade': grade, 'tt': None, 'bk': None, 'r': None,
            'dp': round(dp, 2), 'vr': round(vr, 2), 'dist': round(dist, 2),
            'live': live, 'drop': round(dp, 2), 'spike': round(vr, 2), 'note': ''
        })
    return out[-1:]

def detect_oversold(candles, tf):
    if not candles or len(candles) < 20: return None
    closes = [c['close'] for c in candles]
    r = wilder_rsi(closes[:-1])
    rl = wilder_rsi(closes)
    if r is None or rl is None: return None
    is_closed, is_live = r < 30, rl < 30
    if not is_closed and not is_live: return None
    val = r if is_closed else rl
    tier = 'OS3' if val < 20 else 'OS2' if val <= 25 else 'OS1'
    return {'r': r, 'rl': rl, 'tier': tier}

def detect_retest(candles, tf):
    if not candles or len(candles) < 20: return None
    rsi = rsi_series([c['close'] for c in candles[:-1]])
    if len(rsi) < 6: return None
    bars = 0
    for i in range(len(rsi) - 1, -1, -1):
        if rsi[i] > 70: bars += 1
        else: break
    if bars < 3: return None
    cur = rsi[-1]
    if cur < 49: return None
    stage = None
    if cur <= 51: stage = 'Retest Zone'
    elif cur <= 52: stage = 'Alert'
    elif cur <= 55: stage = 'Near Retest'
    elif cur <= 60: stage = 'Early Warning'
    elif cur <= 65: stage = 'Monitoring'
    return {'rsi': cur, 'stage': stage, 'bars': bars} if stage else None

# ==========================================
# SCAN WORKER
# ==========================================
def run_scan():
    global TRACKED_ZONES
    with STATE_LOCK:
        STATE['loading'] = True
        STATE['progress'] = 0

    try:
        with ThreadPoolExecutor(max_workers=10) as ex:
            f_ins = ex.submit(bybit_instruments)
            f_ins2 = ex.submit(binance_instruments)
            f_tk = ex.submit(bybit_tickers)
            f_tk2 = ex.submit(binance_tickers)
            f_fund = ex.submit(binance_funding)
            f_cg = ex.submit(get_cg_ranks)

            by_ins, bn_ins = f_ins.result(), f_ins2.result()
            by_tk, bn_tk = f_tk.result(), f_tk2.result()
            bn_fund, cg = f_fund.result(), f_cg.result()
        tk_map = {}
        # 1. Load Binance tickers first
        for t in bn_tk:
            if t['symbol'] in bn_fund: 
                t.update(bn_fund[t['symbol']])
            tk_map[t['symbol']] = t
            
        # 2. Load Bybit tickers ONLY if the coin doesn't exist on Binance
        for t in by_tk:
            if t['symbol'] not in tk_map:
                tk_map[t['symbol']] = t

        with STATE_LOCK: STATE['tickers'] = list(tk_map.values())

        comm = [{'symbol': s, **tk_map.get(s, {})} for s in COMM_SYMBOLS if s in tk_map and tk_map[s].get('last')]
        with STATE_LOCK: STATE['commodities'] = comm

        seen, dedup = set(), []
        # 1. Add Binance USDT.P instruments first
        for i in bn_ins:
            if i['symbol'] not in seen:
                seen.add(i['symbol']); 
                dedup.append(i)
        # 2. Add Bybit USDT.P instruments ONLY if not already on Binance
        for i in by_ins:
            if i['symbol'] not in seen:
                seen.add(i['symbol']); 
                dedup.append(i)

        raw_cands = []
        for i in dedup:
            t = tk_map.get(i['symbol'])
            if not t or i['symbol'] in EXCLUDED_SYMBOLS or t.get('vol', 0) < MIN_VOL: continue
            nbase = normalize_base(i['base'])
            raw_cands.append({**i, 'rank': cg.get(nbase) or cg.get(i['base']), 'vol': t['vol'], 'last': t['last'], 'change': t['change24h'], 'ticker': t})

        has_rank = len(cg) > 200
        strict = [i for i in raw_cands if i['rank'] is not None] if has_rank else raw_cands
        candidates = sorted((strict if len(strict) >= 30 else raw_cands), key=lambda x: x['vol'], reverse=True)[:MAX_SYMBOLS]

        with STATE_LOCK: STATE['scanStats']['coins'] = len(candidates)

        magic, magicrec, retest, oversold, funding = [], [], [], [], []
        calls = 0
        done = 0

        def process_candidate(inst):
            nonlocal calls
            try:
                klines = fetch_all_klines(inst['symbol'], inst['exchange'], ALL_TFS)
                calls += len(klines)
            except Exception as e:
                print(f"SCAN ERROR on {inst['symbol']}: {e}")
                return

            for tf in MAGIC_TFS:
                if len(klines.get(tf, [])) >= 110:
                    conf = klines.get(CONF_TF.get(tf))
                    ms = detect_magic(klines[tf], tf, conf)
                    for m in ms:
                        magic.append({**inst, 'tf': tf, **m})
                        if not any(z['sym'] == inst['symbol'] and z['tf'] == tf and z['t'] == m['t'] for z in TRACKED_ZONES):
                            TRACKED_ZONES.append({'sym': inst['symbol'], 'tf': tf, 'top': m['top'], 'bot': m['bot'], 't': m['t'], 'ts': int(time.time()*1000), 'tp': 0, 'dd': 0, 'done': False})

            for tf in OVERSOLD_TFS:
                if len(klines.get(tf, [])) >= 20:
                    o = detect_oversold(klines[tf], tf)
                    if o: oversold.append({**inst, 'tf': tf, **o})

            for tf in RETEST_TFS:
                if len(klines.get(tf, [])) >= 20:
                    r = detect_retest(klines[tf], tf)
                    if r: retest.append({**inst, 'tf': tf, **r})

            t = inst['ticker']
            if t.get('funding') is not None and t['funding'] < 0:
                funding.append({**inst, 'rate': t['funding'], 'next': t.get('nextFunding'), 'mark': t.get('mark')})

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = [ex.submit(process_candidate, inst) for inst in candidates]
            for f in as_completed(futures):
                done += 1
                with STATE_LOCK: STATE['progress'] = int(done / len(candidates) * 100)

        # Update Tracked Zones
        for z in TRACKED_ZONES:
            if z['done']: continue
            t = tk_map.get(z['sym'])
            if not t: continue
            if not z.get('retested') and z['bot'] <= t['last'] <= z['top']: z['retested'] = True
            if z.get('retested'):
                dd = (z['bot'] - t['last']) / z['bot'] * 100
                if dd > z['dd']: z['dd'] = dd
                if t['last'] > z['top']: z['done'] = True
                elif dd >= 1: z['tp'] = max(z['tp'], 3 if dd >= 5 else 2 if dd >= 3 else 1)

        magicrec = [{'symbol': z['sym'], 'sym': z['sym'], 'exchange': 'BYBIT', 'tf': z['tf'], **z} for z in TRACKED_ZONES if z.get('retested') and not z['done'] and z['tp'] > 0]
        TRACKED_ZONES = [z for z in TRACKED_ZONES if not z['done']][-200:]

        with open(TRACKED_FILE, 'w') as f: json.dump(TRACKED_ZONES, f)

        # Fetch Alts
        movers = sorted([{'symbol': c['symbol'], 'exchange': c['exchange'], 'vol': c['vol'], 'chg': c.get('change', 0)} for c in candidates if c['vol'] >= 2000000], key=lambda x: x['chg'], reverse=True)
        bull_c, bear_c = movers[:15], movers[-15:][::-1]

        def get_alt_rsi(c):
            try:
                k = fetch_all_klines(c['symbol'], c['exchange'], ['1H'])
                return {**c, 'r1': wilder_rsi([x['close'] for x in k['1H'][:-1]])}
            except: return {**c, 'r1': None}

        bull_r = [get_alt_rsi(c) for c in bull_c]
        bear_r = [get_alt_rsi(c) for c in bear_c]

        # Save Signals for OpenClaw
        signals_out = {
            'timestamp': int(time.time() * 1000),
            'magic': [{k: v for k, v in m.items() if k != 'ticker'} for m in magic[:20]],
            'oversold': [{k: v for k, v in o.items() if k != 'ticker'} for o in oversold[:20]],
            'retest': [{k: v for k, v in r.items() if k != 'ticker'} for r in retest[:20]]
        }
        with open(SIGNALS_FILE, 'w') as f: json.dump(signals_out, f)

        with STATE_LOCK:
            STATE['rows'] = {'magic': magic, 'magicrec': magicrec, 'retest': retest, 'oversold': oversold, 'funding': funding}
            STATE['alts'] = {
                'bull': [c for c in bull_r if c['r1'] is not None and c['r1'] >= 48][:15],
                'bear': [c for c in bear_r if c['r1'] is not None and c['r1'] <= 52][:15]
            }
            STATE['ranks'] = cg
            STATE['lastScan'] = int(time.time() * 1000)
            STATE['nextScanTs'] = int(time.time() * 1000) + (SCAN_INTERVAL_SEC * 1000)
            STATE['scanStats']['calls'] = calls
            STATE['progress'] = 100

    except Exception as e:
        print(f"Scan Error: {e}")
    finally:
        with STATE_LOCK: STATE['loading'] = False

def scan_loop():
    global TRACKED_ZONES
    if os.path.exists(TRACKED_FILE):
        try:
            with open(TRACKED_FILE, 'r') as f: TRACKED_ZONES = json.load(f)
        except: pass
    while True:
        run_scan()
        SCAN_EVENT.wait(timeout=SCAN_INTERVAL_SEC)
        SCAN_EVENT.clear()

# ==========================================
# FASTAPI APP
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup_event():
    threading.Thread(target=scan_loop, daemon=True).start()

@app.get("/")
def serve_index(): return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.get("/api/data")
def get_data():
    with STATE_LOCK: return JSONResponse(STATE)

@app.post("/api/scan-now")
def trigger_scan():
    SCAN_EVENT.set()
    return {"status": "triggered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
