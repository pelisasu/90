import os
import json
import time
import math
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("APEX-QUANT-OMNI")

# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================
CONFIG = {
    "VERSION": "v12.1-SOVEREIGN-REFACTORED",
    "SYMBOLS": ["BTCUSD", "ETHUSD"],
    "TIMEFRAMES": {
        "5m": {"sec": 300, "bars": 350},
        "15m": {"sec": 900, "bars": 350},
        "1h": {"sec": 3600, "bars": 350},
        "4h": {"sec": 14400, "bars": 720},
    },
    # Broker price offset calibration (sesuaikan dengan selisih quote MT5 Anda)
    "BROKER_OFFSET": {
        "BTCUSD": float(os.getenv("OFFSET_BTCUSD", "0.0")),
        "ETHUSD": float(os.getenv("OFFSET_ETHUSD", "0.0")),
    },
    # Routing strategi berdasarkan Forensic Audit Log:
    # ACTIVE = kirim sinyal Telegram | SHADOW = audit log saja | QUARANTINE = nonaktif
    "STRATEGY_MATRIX": {
        "BTCUSD_SWING_SNIPER": {"tier": "ACTIVE", "atr_len": 20, "z_thresh": 1.5, "min_score": 72.0},
        "BTCUSD_SWING_PRO":    {"tier": "ACTIVE", "atr_len": 20, "z_thresh": 1.5, "min_score": 68.0},
        "ETHUSD_SWING_PRO":    {"tier": "SHADOW", "atr_len": 20, "z_thresh": 1.5, "min_score": 70.0},
        "BTCUSD_SCALP_PRO":    {"tier": "QUARANTINE", "atr_len": 10, "z_thresh": 1.5, "min_score": 75.0},
        "BTCUSD_SCALP_SNIPER": {"tier": "QUARANTINE", "atr_len": 10, "z_thresh": 1.5, "min_score": 78.0},
        "ETHUSD_SCALP_PRO":    {"tier": "QUARANTINE", "atr_len": 20, "z_thresh": 2.0, "min_score": 75.0},
        "ETHUSD_SCALP_SNIPER": {"tier": "QUARANTINE", "atr_len": 20, "z_thresh": 2.0, "min_score": 78.0},
        "ETHUSD_SWING_SNIPER": {"tier": "QUARANTINE", "atr_len": 20, "z_thresh": 1.5, "min_score": 78.0},
    },
    "STATE_FILE": "apex_state.json",
    "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", ""),
    "REQUEST_TIMEOUT": 12,
}

# ==============================================================================
# MULTI-SOURCE PUBLIC MARKET DATA ENGINE (NO API KEY REQUIRED)
# ==============================================================================
class MarketDataEngine:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "APEX-QUANT-OMNI/12.1"})

    def _fetch_coinbase(self, symbol: str, tf_sec: int, limit: int) -> Optional[pd.DataFrame]:
        pair = "BTC-USD" if symbol == "BTCUSD" else "ETH-USD"
        url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
        params = {"granularity": tf_sec}
        resp = self.session.get(url, params=params, timeout=CONFIG["REQUEST_TIMEOUT"])
        resp.raise_for_status()
        raw = resp.json()
        if not raw or not isinstance(raw, list):
            return None
        df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.sort_values("time").drop_duplicates("time").tail(limit).reset_index(drop=True)
        return df[["time", "open", "high", "low", "close", "volume"]].astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )

    def _fetch_kraken(self, symbol: str, tf_sec: int, limit: int) -> Optional[pd.DataFrame]:
        pair = "XXBTZUSD" if symbol == "BTCUSD" else "XETHZUSD"
        interval_min = tf_sec // 60
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": pair, "interval": interval_min}
        resp = self.session.get(url, params=params, timeout=CONFIG["REQUEST_TIMEOUT"])
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return None
        result_key = [k for k in data["result"].keys() if k != "last"][0]
        raw = data["result"][result_key]
        df = pd.DataFrame(
            raw, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"]
        )
        df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
        df = df.sort_values("time").drop_duplicates("time").tail(limit).reset_index(drop=True)
        return df[["time", "open", "high", "low", "close", "volume"]].astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )

    def _fetch_kucoin(self, symbol: str, tf_name: str, limit: int) -> Optional[pd.DataFrame]:
        pair = "BTC-USDT" if symbol == "BTCUSD" else "ETH-USDT"
        tf_map = {"5m": "5min", "15m": "15min", "1h": "1hour", "4h": "4hour"}
        url = "https://api.kucoin.com/api/v1/market/candles"
        params = {"type": tf_map[tf_name], "symbol": pair}
        resp = self.session.get(url, params=params, timeout=CONFIG["REQUEST_TIMEOUT"])
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "200000" or not data.get("data"):
            return None
        raw = data["data"]
        df = pd.DataFrame(raw, columns=["time", "open", "close", "high", "low", "volume", "turnover"])
        df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
        df = df.sort_values("time").drop_duplicates("time").tail(limit).reset_index(drop=True)
        return df[["time", "open", "high", "low", "close", "volume"]].astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )

    def get_ohlcv(self, symbol: str, tf_name: str) -> Tuple[Optional[pd.DataFrame], str, bool]:
        """
        Mengambil data OHLCV dengan fallback chain & stale-data guard.
        Mengembalikan: (df_closed_bars, source_name, is_fresh)
        """
        tf_sec = CONFIG["TIMEFRAMES"][tf_name]["sec"]
        limit = CONFIG["TIMEFRAMES"][tf_name]["bars"]
        now_utc = datetime.now(timezone.utc)

        # Urutan prioritas sumber: 4h utama Kraken (history panjang), lainnya Coinbase -> Kraken -> KuCoin
        sources = (
            ["KRAKEN", "COINBASE", "KUCOIN"]
            if tf_name == "4h"
            else ["COINBASE", "KRAKEN", "KUCOIN"]
        )

        best_df = None
        best_src = "NONE"

        for src in sources:
            try:
                if src == "COINBASE":
                    df = self._fetch_coinbase(symbol, tf_sec, limit)
                elif src == "KRAKEN":
                    df = self._fetch_kraken(symbol, tf_sec, limit)
                else:
                    df = self._fetch_kucoin(symbol, tf_name, limit)

                if df is None or len(df) < 100:
                    continue

                # Terapkan kalibrasi broker offset
                offset = CONFIG["BROKER_OFFSET"].get(symbol, 0.0)
                if offset != 0.0:
                    for col in ["open", "high", "low", "close"]:
                        df[col] = df[col] + offset

                # Evaluasi hanya CLOSED BARS (buang candle terakhir jika belum tutup)
                last_open_ts = df["time"].iloc[-1]
                bar_close_ts = last_open_ts + pd.Timedelta(seconds=tf_sec)
                if now_utc < bar_close_ts:
                    df_closed = df.iloc[:-1].copy().reset_index(drop=True)
                else:
                    df_closed = df.copy().reset_index(drop=True)

                last_closed_ts = df_closed["time"].iloc[-1]
                lag_sec = (now_utc - last_closed_ts).total_seconds()
                is_fresh = lag_sec <= (tf_sec * 2.2)

                if is_fresh:
                    logger.info(
                        f"[{symbol} {tf_name}] loaded from {src} ({len(df_closed)} closed bars, last={last_closed_ts})"
                    )
                    return df_closed, f"{src}:{symbol}", True

                if best_df is None:
                    best_df, best_src = df_closed, f"{src}:{symbol}(STALE)"

            except Exception as e:
                logger.warning(f"[{symbol} {tf_name}] {src} failed: {e}")

        if best_df is not None:
            logger.warning(
                f"[{symbol} {tf_name}] using fallback stale feed {best_src} (last={best_df['time'].iloc[-1]})"
            )
            return best_df, best_src, False

        return None, "FAILED", False


# ==============================================================================
# QUANTITATIVE INDICATORS & REGIME ANALYTICS
# ==============================================================================
def compute_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def compute_ker(close: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman Efficiency Ratio (0 = noise/choppy, 1 = pure directional trend)"""
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (change / volatility.replace(0, np.nan)).fillna(0.0)


def compute_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


# ==============================================================================
# ADAPTIVE SIGNAL & CANDLE MICROSTRUCTURE ENGINE
# ==============================================================================
def analyze_swing_setup(
    symbol: str,
    mode: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    cfg: Dict,
) -> Dict:
    """
    Evaluasi setup SWING berbasis keselarasan 4H + 1H + 15M dengan kalibrasi filter adaptif.
    """
    atr_len = cfg["atr_len"]
    z_thresh = cfg["z_thresh"]
    min_score = cfg["min_score"]

    # Hitung indikator pada 1H (anchor swing) & 4H (macro bias)
    df_1h = df_1h.copy()
    df_1h["ema21"] = df_1h["close"].ewm(span=21, adjust=False).mean()
    df_1h["ema55"] = df_1h["close"].ewm(span=55, adjust=False).mean()
    df_1h["atr"] = compute_atr(df_1h, atr_len)
    df_1h["adx"] = compute_adx(df_1h, 14)
    df_1h["ker"] = compute_ker(df_1h["close"], 10)
    df_1h["z"] = compute_zscore(df_1h["close"], 20)

    df_4h = df_4h.copy()
    df_4h["ema21"] = df_4h["close"].ewm(span=21, adjust=False).mean()
    df_4h["ema55"] = df_4h["close"].ewm(span=55, adjust=False).mean()

    last_1h = df_1h.iloc[-1]
    last_4h = df_4h.iloc[-1]
    last_15m = df_15m.iloc[-1]

    px = float(last_15m["close"])
    atr = float(last_1h["atr"])
    adx = float(last_1h["adx"])
    ker = float(last_1h["ker"])
    z_val = float(last_1h["z"])

    # Tentukan Rezim Pasar
    bull_4h = last_4h["close"] > last_4h["ema21"] > last_4h["ema55"]
    bear_4h = last_4h["close"] < last_4h["ema21"] < last_4h["ema55"]
    bull_1h = last_1h["close"] > last_1h["ema21"] > last_1h["ema55"]
    bear_1h = last_1h["close"] < last_1h["ema21"] < last_1h["ema55"]

    if adx >= 25 and bear_1h:
        regime = "TREND_BEAR"
    elif adx >= 25 and bull_1h:
        regime = "TREND_BULL"
    else:
        regime = "RANGE_CHOP"

    # Mikrostruktur candle terakhir 1H
    c_open = float(last_1h["open"])
    c_high = float(last_1h["high"])
    c_low = float(last_1h["low"])
    c_close = float(last_1h["close"])
    c_range = max(c_high - c_low, 1e-9)

    body_ratio = abs(c_close - c_open) / c_range
    lower_wick_ratio = (min(c_open, c_close) - c_low) / c_range
    upper_wick_ratio = (c_high - max(c_open, c_close)) / c_range

    # Kalibrasi ambang adaptif berdasarkan kekuatan ADX & KER
    strong_trend = (adx >= 35.0 and ker >= 0.55)
    min_body_req = 0.22 if strong_trend else 0.35
    max_wick_trap = 0.65 if strong_trend else 0.52

    # Scoring System (0 - 100)
    score = 40.0
    direction = "NONE"
    reason = "NO_SETUP"

    # Evaluasi arah SELL (Bearish Continuation / Pullback)
    if regime == "TREND_BEAR" and bear_4h:
        direction = "SELL"
        score += 18.0  # Macro 4H + 1H aligned
        if adx >= 30:
            score += 10.0
        if ker >= 0.50:
            score += 8.0
        # Pullback sehat ke area EMA21 atau momentum Z-score terkendali (tidak ekstrem oversold)
        if -z_thresh <= z_val <= 0.35:
            score += 12.0
        elif z_val < -2.4:
            score -= 10.0  # Hindari mengejar harga yang sudah terlalu jauh di bawah rata-rata

        # Filter mikrostruktur adaptif
        if c_close >= c_open:
            return _build_result(symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, "COUNTER_CANDLE", atr)
        if body_ratio < min_body_req:
            return _build_result(
                symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, f"BODY_WEAK({body_ratio:.2f})", atr
            )
        if lower_wick_ratio > max_wick_trap:
            return _build_result(
                symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, f"WICK_TRAP_DN({lower_wick_ratio:.2f})", atr
            )

    # Evaluasi arah BUY (Bullish Continuation / Pullback)
    elif regime == "TREND_BULL" and bull_4h:
        direction = "BUY"
        score += 18.0
        if adx >= 30:
            score += 10.0
        if ker >= 0.50:
            score += 8.0
        if -0.35 <= z_val <= z_thresh:
            score += 12.0
        elif z_val > 2.4:
            score -= 10.0

        if c_close <= c_open:
            return _build_result(symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, "COUNTER_CANDLE", atr)
        if body_ratio < min_body_req:
            return _build_result(
                symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, f"BODY_WEAK({body_ratio:.2f})", atr
            )
        if upper_wick_ratio > max_wick_trap:
            return _build_result(
                symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, f"WICK_TRAP_UP({upper_wick_ratio:.2f})", atr
            )
    else:
        return _build_result(symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, "MTF_UNALIGNED", atr)

    # Filter Sniper lebih ketat pada efisiensi tren
    if mode == "SNIPER" and (ker < 0.55 or adx < 32.0):
        return _build_result(symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, "SNIPER_QUALITY_GATE", atr)

    if score < min_score:
        return _build_result(
            symbol, mode, "NO_SIGNAL", "NONE", px, score, regime, ker, adx, f"LOW_SCORE({score:.1f}<{min_score})", atr
        )

    return _build_result(
        symbol,
        mode,
        "SIGNAL_READY",
        direction,
        px,
        score,
        regime,
        ker,
        adx,
        "CONFIRMED",
        atr,
        bar_ts=str(last_1h["time"]),
    )


def _build_result(
    symbol: str,
    mode: str,
    status: str,
    direction: str,
    px: float,
    score: float,
    regime: str,
    ker: float,
    adx: float,
    reason: str,
    atr: float,
    bar_ts: str = "",
) -> Dict:
    sl, tp1, tp2 = 0.0, 0.0, 0.0
    if status == "SIGNAL_READY" and direction in ("BUY", "SELL"):
        sl_mult = 1.35 if mode == "SNIPER" else 1.50
        if direction == "BUY":
            sl = px - (atr * sl_mult)
            tp1 = px + (atr * 1.80)
            tp2 = px + (atr * 3.00)
        else:
            sl = px + (atr * sl_mult)
            tp1 = px - (atr * 1.80)
            tp2 = px - (atr * 3.00)

    return {
        "symbol": symbol,
        "mode": mode,
        "status": status,
        "direction": direction,
        "px": round(px, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "score": round(score, 1),
        "regime": regime,
        "ker": round(ker, 2),
        "adx": round(adx, 1),
        "reason": reason,
        "bar_ts": bar_ts,
    }


# ==============================================================================
# STATE DEDUPLICATION & TELEGRAM DISPATCHER
# ==============================================================================
def load_state() -> Dict:
    if os.path.exists(CONFIG["STATE_FILE"]):
        try:
            with open(CONFIG["STATE_FILE"], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"sent_hashes": []}
    return {"sent_hashes": []}


def save_state(state: Dict) -> None:
    state["sent_hashes"] = state.get("sent_hashes", [])[-100:]
    with open(CONFIG["STATE_FILE"], "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def send_telegram_alert(sig: Dict, tier: str) -> bool:
    token = CONFIG["TELEGRAM_BOT_TOKEN"]
    chat_id = CONFIG["TELEGRAM_CHAT_ID"]
    if not token or not chat_id:
        logger.warning("Telegram credentials missing; skipping alert dispatch.")
        return False

    icon = "🔴" if sig["direction"] == "SELL" else "🟢"
    msg = (
        f"{icon} *APEX-QUANT-OMNI {CONFIG['VERSION']}*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• *Pair / Mode:* `{sig['symbol']} SWING {sig['mode']}` ({tier})\n"
        f"• *Action:* *{sig['direction']}* @ `{sig['px']}`\n"
        f"• *Stop Loss:* `{sig['sl']}`\n"
        f"• *TP 1 (1.2R):* `{sig['tp1']}`\n"
        f"• *TP 2 (2.0R):* `{sig['tp2']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• *Score:* `{sig['score']}/100` | *Regime:* `{sig['regime']}`\n"
        f"• *ADX / KER:* `{sig['adx']}` / `{sig['ker']}`\n"
        f"• *Candle UTC:* `{sig['bar_ts']}`"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


# ==============================================================================
# MAIN ENGINE EXECUTION
# ==============================================================================
def main():
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    engine = MarketDataEngine()
    state = load_state()
    sent_hashes = set(state.get("sent_hashes", []))

    market_cache: Dict[str, Dict] = {}
    feed_status: Dict[str, Dict] = {}

    for sym in CONFIG["SYMBOLS"]:
        market_cache[sym] = {}
        feed_status[sym] = {"sources": {}, "all_fresh": True}
        for tf in CONFIG["TIMEFRAMES"].keys():
            df, src, fresh = engine.get_ohlcv(sym, tf)
            market_cache[sym][tf] = df
            feed_status[sym]["sources"][tf] = src
            if not fresh or df is None:
                feed_status[sym]["all_fresh"] = False

    print("=" * 92)
    print(f"APEX-QUANT-OMNI — SOVEREIGN AUDIT LOG ({CONFIG['VERSION']})")
    print("=" * 92)
    print(f"Timestamp UTC : {now_utc} | Active Focus=BTCUSD SWING (PRO & SNIPER)")

    for sym in CONFIG["SYMBOLS"]:
        df_5m = market_cache[sym]["5m"]
        spot = float(df_5m["close"].iloc[-1]) if df_5m is not None else 0.0
        est_spread = round(spot * 0.0003, 2)
        print(
            f"[{sym} FEEDS] : Spot=${spot:.2f} | EstSpread=${est_spread:.2f} | "
            f"Fresh={feed_status[sym]['all_fresh']} | {json.dumps(feed_status[sym]['sources'])}"
        )

    print("-" * 92)

    dispatched_count = 0
    for strat_key, cfg in CONFIG["STRATEGY_MATRIX"].items():
        parts = strat_key.split("_")
        sym, style, mode = parts[0], parts[1], parts[2]
        tier = cfg["tier"]

        if tier == "QUARANTINE":
            print(f"[LIVE {sym} {style} {mode}]: SKIPPED (QUARANTINED_LOW_PF)")
            continue

        # Blokir eksekusi jika data feed pada pair tersebut tidak sinkron/stale
        if not feed_status[sym]["all_fresh"]:
            print(f"[LIVE {sym} {style} {mode}]: BLOCKED (STALE_OR_INCOMPLETE_FEED)")
            continue

        res = analyze_swing_setup(
            symbol=sym,
            mode=mode,
            df_15m=market_cache[sym]["15m"],
            df_1h=market_cache[sym]["1h"],
            df_4h=market_cache[sym]["4h"],
            cfg=cfg,
        )

        print(
            f"[LIVE {sym} {style} {mode} ({tier})]: {res['status']} ({res['direction']}) | "
            f"Px={res['px']} | Score={res['score']} | Regime={res['regime']} | "
            f"KER={res['ker']} | ADX={res['adx']} | Reason={res['reason']}"
        )

        if res["status"] == "SIGNAL_READY" and tier == "ACTIVE":
            sig_fingerprint = f"{sym}_{style}_{mode}_{res['direction']}_{res['bar_ts']}"
            sig_hash = hashlib.sha256(sig_fingerprint.encode()).hexdigest()[:16]

            if sig_hash not in sent_hashes:
                if send_telegram_alert(res, tier):
                    sent_hashes.add(sig_hash)
                    dispatched_count += 1
                    logger.info(f"Dispatched Telegram alert for {sig_fingerprint}")
            else:
                logger.info(f"Duplicate signal suppressed: {sig_fingerprint}")

    state["sent_hashes"] = list(sent_hashes)
    save_state(state)
    print("-" * 92)
    print(f"[SUMMARY] : Alerts Dispatched={dispatched_count} | State Cached={len(sent_hashes)} hashes")
    print("=" * 92)


if __name__ == "__main__":
    main()
