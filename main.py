#!/usr/bin/env python3
"""
Quantitative Multi-Asset Dual-Mode Trading Signal Engine
Instruments: XAUUSD, NAS100, BTCUSD, ETHUSD
Modes: SCALP & SWING
Execution: Serverless GitHub Actions + SQLite Persistence + Telegram Interactive Commands
"""

import os
import sys
import time
import math
import sqlite3
import logging
import traceback
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ==============================================================================
# 1. LOGGING & KONFIGURASI GLOBAL (KONSISTEN UTC)
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [UTC] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger("QuantSignalEngine")

DB_PATH = os.getenv("DB_PATH", "trading_signals.db")
CHART_PATH = os.getenv("CHART_PATH", "backtest_performance.png")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000.0"))
RISK_PER_TRADE_SCALP = float(os.getenv("RISK_PER_TRADE_SCALP", "0.0075"))  # 0.75%
RISK_PER_TRADE_SWING = float(os.getenv("RISK_PER_TRADE_SWING", "0.0125"))  # 1.25%

NAS100_PRICE_MULTIPLIER = float(os.getenv("NAS100_PRICE_MULTIPLIER", "1.0"))
NAS100_PRICE_OFFSET = float(os.getenv("NAS100_PRICE_OFFSET", "0.0"))
XAUUSD_PRICE_MULTIPLIER = float(os.getenv("XAUUSD_PRICE_MULTIPLIER", "1.0"))
XAUUSD_PRICE_OFFSET = float(os.getenv("XAUUSD_PRICE_OFFSET", "0.0"))

FORCE_MARKET_OPEN = os.getenv("FORCE_MARKET_OPEN", "false").strip().lower() in ("1", "true", "yes", "on")

REQUEST_TIMEOUT = 15
API_SLEEP_SEC = 0.65


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    display_name: str
    asset_class: str  # "METALS", "INDEX", "CRYPTO"
    primary_tickers: List[str]
    coinbase_product: Optional[str]
    cryptocompare_fsym: Optional[str]
    coingecko_id: Optional[str]
    price_decimals: int
    contract_size: float
    pip_size: float
    spread_points: float
    slippage_points: float
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 50.0
    price_multiplier: float = 1.0
    price_offset: float = 0.0
    active_hours_utc: Optional[Tuple[int, int]] = None  # None = 24/7


INSTRUMENTS: Dict[str, InstrumentSpec] = {
    "XAUUSD": InstrumentSpec(
        symbol="XAUUSD",
        display_name="XAUUSD",
        asset_class="METALS",
        primary_tickers=["GC=F"],
        coinbase_product=None,
        cryptocompare_fsym=None,
        coingecko_id=None,
        price_decimals=2,
        contract_size=100.0,
        pip_size=0.10,
        spread_points=0.25,
        slippage_points=0.10,
        price_multiplier=XAUUSD_PRICE_MULTIPLIER,
        price_offset=XAUUSD_PRICE_OFFSET,
        active_hours_utc=(6, 21),
    ),
    "NAS100": InstrumentSpec(
        symbol="NAS100",
        display_name="NAS100",
        asset_class="INDEX",
        primary_tickers=["^NDX", "QQQ"],
        coinbase_product=None,
        cryptocompare_fsym=None,
        coingecko_id=None,
        price_decimals=2,
        contract_size=1.0,
        pip_size=1.0,
        spread_points=1.80,
        slippage_points=0.75,
        price_multiplier=NAS100_PRICE_MULTIPLIER,
        price_offset=NAS100_PRICE_OFFSET,
        active_hours_utc=(12, 21),
    ),
    "BTCUSD": InstrumentSpec(
        symbol="BTCUSD",
        display_name="BTCUSD",
        asset_class="CRYPTO",
        primary_tickers=["BTC-USD"],
        coinbase_product="BTC-USD",
        cryptocompare_fsym="BTC",
        coingecko_id="bitcoin",
        price_decimals=2,
        contract_size=1.0,
        pip_size=1.0,
        spread_points=18.0,
        slippage_points=8.0,
        price_multiplier=1.0,
        price_offset=0.0,
        active_hours_utc=None,
    ),
    "ETHUSD": InstrumentSpec(
        symbol="ETHUSD",
        display_name="ETHUSD",
        asset_class="CRYPTO",
        primary_tickers=["ETH-USD"],
        coinbase_product="ETH-USD",
        cryptocompare_fsym="ETH",
        coingecko_id="ethereum",
        price_decimals=2,
        contract_size=1.0,
        pip_size=0.10,
        spread_points=1.50,
        slippage_points=0.60,
        price_multiplier=1.0,
        price_offset=0.0,
        active_hours_utc=None,
    ),
}


@dataclass(frozen=True)
class ModeConfig:
    mode: str  # "SCALP" atau "SWING"
    htf_interval: str
    htf_period: str
    ltf_interval: str
    ltf_period: str
    atr_period: int
    sl_atr_mult: float
    tp1_r: float
    tp2_r: float
    tp3_r: float
    min_score: int
    min_efficiency_ratio: float
    cooldown_minutes: int
    risk_pct: float


MODE_CONFIGS: Dict[str, ModeConfig] = {
    "SCALP": ModeConfig(
        mode="SCALP",
        htf_interval="1h",
        htf_period="30d",
        ltf_interval="15m",
        ltf_period="14d",
        atr_period=14,
        sl_atr_mult=1.35,
        tp1_r=1.20,
        tp2_r=2.00,
        tp3_r=2.80,
        min_score=65,
        min_efficiency_ratio=0.25,
        cooldown_minutes=45,
        risk_pct=RISK_PER_TRADE_SCALP,
    ),
    "SWING": ModeConfig(
        mode="SWING",
        htf_interval="1d",
        htf_period="365d",
        ltf_interval="4h",
        ltf_period="90d",
        atr_period=14,
        sl_atr_mult=1.85,
        tp1_r=1.50,
        tp2_r=2.60,
        tp3_r=4.00,
        min_score=68,
        min_efficiency_ratio=0.28,
        cooldown_minutes=240,
        risk_pct=RISK_PER_TRADE_SWING,
    ),
}


@dataclass
class SignalCandidate:
    symbol: str
    mode: str
    direction: str  # "BUY", "SELL", atau "NO_SIGNAL"
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    lot_size: float = 0.01
    score: int = 0
    regime: str = "NEUTRAL"
    efficiency_ratio: float = 0.0
    z_score: float = 0.0
    atr: float = 0.0
    candle_ts: str = ""
    reason: str = ""
    best_params: Dict[str, Any] = field(default_factory=dict)
    backtest_stats: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# 2. SQLITE PERSISTENCE LAYER
# ==============================================================================

class SignalDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=20.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry REAL NOT NULL,
                    sl REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    tp3 REAL NOT NULL,
                    lot_size REAL NOT NULL,
                    score INTEGER NOT NULL,
                    regime TEXT NOT NULL,
                    efficiency_ratio REAL NOT NULL,
                    z_score REAL NOT NULL,
                    candle_ts TEXT NOT NULL,
                    status TEXT DEFAULT 'ACTIVE'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS run_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at_utc TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    regime TEXT NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    note TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_lookup
                ON signals(symbol, mode, created_at_utc)
            """)
            conn.commit()

    def get_state(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bot_state(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            conn.commit()

    def is_duplicate_or_in_cooldown(
        self, symbol: str, mode: str, direction: str, candle_ts: str, cooldown_minutes: int
    ) -> Tuple[bool, str]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT direction, candle_ts, created_at_utc
                FROM signals
                WHERE symbol = ? AND mode = ?
                ORDER BY id DESC LIMIT 1
                """,
                (symbol, mode),
            ).fetchone()

            if not row:
                return False, "OK"

            if row["direction"] == direction and row["candle_ts"] == candle_ts:
                return True, f"Duplikat candle_ts ({candle_ts})"

            if row["direction"] == direction:
                last_time = datetime.fromisoformat(row["created_at_utc"])
                elapsed_min = (datetime.now(timezone.utc) - last_time).total_seconds() / 60.0
                if elapsed_min < cooldown_minutes:
                    return True, f"Cooldown arah {direction} ({elapsed_min:.1f}/{cooldown_minutes}m)"

            return False, "OK"

    def save_signal(self, sig: SignalCandidate) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    created_at_utc, symbol, mode, direction, entry, sl, tp1, tp2, tp3,
                    lot_size, score, regime, efficiency_ratio, z_score, candle_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_utc,
                    sig.symbol,
                    sig.mode,
                    sig.direction,
                    sig.entry,
                    sig.sl,
                    sig.tp1,
                    sig.tp2,
                    sig.tp3,
                    sig.lot_size,
                    sig.score,
                    sig.regime,
                    sig.efficiency_ratio,
                    sig.z_score,
                    sig.candle_ts,
                ),
            )
            conn.commit()

    def save_audit(self, sig: SignalCandidate) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        stats = sig.backtest_stats or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_audits (
                    run_at_utc, symbol, mode, outcome, score, regime,
                    win_rate, profit_factor, max_drawdown_pct, total_trades, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_utc,
                    sig.symbol,
                    sig.mode,
                    sig.direction,
                    sig.score,
                    sig.regime,
                    float(stats.get("win_rate", 0.0)),
                    float(stats.get("profit_factor", 0.0)),
                    float(stats.get("max_dd_pct", 0.0)),
                    int(stats.get("total_trades", 0)),
                    sig.reason[:240],
                ),
            )
            conn.commit()


# ==============================================================================
# 3. MULTI-SOURCE DATA ENGINE & SANITY GUARD (ANTI-451 & SUFFICIENT HISTORY)
# ==============================================================================

def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    return df


def _fetch_coinbase_candles(product_id: str, interval: str) -> pd.DataFrame:
    granularity_map = {
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    gran = granularity_map.get(interval)
    if not gran:
        return pd.DataFrame()

    now_dt = datetime.now(timezone.utc)
    window_sec = gran * 295
    start_dt = now_dt - timedelta(seconds=window_sec)

    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "granularity": gran,
        "start": start_dt.isoformat(),
        "end": now_dt.isoformat(),
    }
    headers = {"User-Agent": "QuantSignalBot/1.0", "Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    raw = resp.json()
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()

    rows = []
    for item in raw:
        if len(item) >= 6:
            rows.append({
                "Datetime": pd.to_datetime(int(item[0]), unit="s", utc=True),
                "Low": float(item[1]),
                "High": float(item[2]),
                "Open": float(item[3]),
                "Close": float(item[4]),
                "Volume": float(item[5]),
            })
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Datetime").sort_index()


def _fetch_cryptocompare_ohlc(fsym: str, interval: str, limit: int = 180) -> pd.DataFrame:
    """
    Failover kripto untuk riwayat panjang (terutama timeframe 1d / 4h / 1h) via CryptoCompare.
    """
    endpoint_map = {
        "15m": ("histominute", 15, min(limit, 300)),
        "1h": ("histohour", 1, min(limit, 300)),
        "4h": ("histohour", 4, min(limit * 4, 600)),
        "1d": ("histoday", 1, max(limit, 120)),
    }
    cfg = endpoint_map.get(interval)
    if not cfg:
        return pd.DataFrame()

    endpoint, agg_n, req_limit = cfg
    url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
    params = {
        "fsym": fsym,
        "tsym": "USD",
        "limit": req_limit,
        "aggregate": agg_n,
    }
    headers = {"User-Agent": "QuantSignalBot/1.0", "Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    data = ((payload or {}).get("Data") or {}).get("Data") or []
    if not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        ts = int(item.get("time", 0))
        o = float(item.get("open", 0.0))
        h = float(item.get("high", 0.0))
        l = float(item.get("low", 0.0))
        c = float(item.get("close", 0.0))
        v = float(item.get("volumeto", 0.0))
        if ts > 0 and o > 0 and h > 0 and l > 0 and c > 0:
            rows.append({
                "Datetime": pd.to_datetime(ts, unit="s", utc=True),
                "Open": o,
                "High": h,
                "Low": l,
                "Close": c,
                "Volume": v,
            })
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("Datetime").sort_index()


def _fetch_coingecko_ohlc(coin_id: str, interval: str) -> pd.DataFrame:
    days = "180" if interval == "1d" else ("90" if interval == "4h" else "30")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    headers = {"User-Agent": "QuantSignalBot/1.0", "Accept": "application/json"}
    resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    raw = resp.json()
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()

    rows = []
    for item in raw:
        if len(item) >= 5:
            rows.append({
                "Datetime": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                "Open": float(item[1]),
                "High": float(item[2]),
                "Low": float(item[3]),
                "Close": float(item[4]),
                "Volume": 1.0,
            })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("Datetime").sort_index()
    if interval == "1d":
        df = df.resample("1d").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=["Open", "High", "Low", "Close"])
    return df


def _resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    if df_1h.empty:
        return df_1h
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    return df_1h.resample("4h").agg(agg).dropna(subset=["Open", "High", "Low", "Close"])


def _sanitize_ohlcv(
    df: pd.DataFrame,
    spec: InstrumentSpec,
    interval: str,
    apply_price_calibration: bool = True,
) -> Tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return pd.DataFrame(), "DataFrame kosong"

    df = _flatten_yfinance_columns(df.copy())
    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame(), f"Kolom OHLC tidak lengkap: {list(df.columns)}"

    if "Volume" not in df.columns:
        df["Volume"] = 1.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = df["Volume"].fillna(0.0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]
    df = df[df["High"] >= df["Low"]]

    min_bars = 30 if interval == "1d" else 35
    if len(df) < min_bars:
        return pd.DataFrame(), f"Bar valid terlalu sedikit ({len(df)} < {min_bars})"

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[~df.index.duplicated(keep="last")].sort_index()

    if apply_price_calibration:
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col] * spec.price_multiplier + spec.price_offset

    # Sanity guard terhadap spike anomali ekstrem (> 12x median range 20-bar)
    bar_range = (df["High"] - df["Low"]).replace(0, np.nan)
    med_range = bar_range.rolling(20, min_periods=5).median().bfill().ffill()
    spike_mask = bar_range > (med_range * 12.0)
    if spike_mask.any():
        logger.warning(f"[{spec.symbol} {interval}] Menghapus {int(spike_mask.sum())} outlier spike bar.")
        df = df[~spike_mask]

    if len(df) < min_bars:
        return pd.DataFrame(), "Bar tersisa setelah filter spike tidak cukup"

    # Cek staleness
    now_utc = datetime.now(timezone.utc)
    last_ts = df.index[-1].to_pydatetime()
    age_hours = (now_utc - last_ts).total_seconds() / 3600.0
    max_stale_hours = 96.0 if spec.asset_class in ("METALS", "INDEX") else (36.0 if interval == "1d" else 12.0)
    if age_hours > max_stale_hours:
        return pd.DataFrame(), f"Data stale: bar terakhir {last_ts.isoformat()} ({age_hours:.1f}j lalu)"

    # Gunakan candle tertutup bila bar terakhir masih sangat baru
    interval_minutes = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(interval, 60)
    if (now_utc - last_ts).total_seconds() < (interval_minutes * 60 * 0.85) and len(df) > (min_bars + 10):
        df = df.iloc[:-1]

    return df, "OK"


def fetch_market_data(spec: InstrumentSpec, interval: str, period: str) -> Tuple[pd.DataFrame, str]:
    yf_interval = "1h" if interval == "4h" else interval
    yf_period = "120d" if interval == "4h" else period

    last_error = "Tidak ada sumber tersedia"
    for ticker in spec.primary_tickers:
        try:
            time.sleep(API_SLEEP_SEC)
            raw = yf.download(
                tickers=ticker,
                period=yf_period,
                interval=yf_interval,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=REQUEST_TIMEOUT,
            )
            df = _flatten_yfinance_columns(raw)
            if interval == "4h" and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                elif df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                df = _resample_to_4h(df)

            apply_cal = True
            if spec.symbol == "NAS100" and ticker == "QQQ" and spec.price_multiplier == 1.0:
                for col in ["Open", "High", "Low", "Close"]:
                    df[col] = df[col] * 41.5 + spec.price_offset
                apply_cal = False

            clean_df, status = _sanitize_ohlcv(df, spec, interval, apply_price_calibration=apply_cal)
            if not clean_df.empty:
                src_label = f"yfinance:{ticker}" if ticker != "GC=F" else "yfinance:GC=F(gold_proxy)"
                return clean_df, src_label
            last_error = f"{ticker}: {status}"
        except Exception as exc:
            last_error = f"{ticker} error: {exc}"
            logger.warning(f"[{spec.symbol} {interval}] Gagal ambil {ticker}: {exc}")

    # Failover 1 untuk Kripto: Coinbase Exchange API
    if spec.coinbase_product:
        try:
            time.sleep(API_SLEEP_SEC)
            cb_df = _fetch_coinbase_candles(spec.coinbase_product, interval)
            clean_df, status = _sanitize_ohlcv(cb_df, spec, interval, apply_price_calibration=True)
            if not clean_df.empty:
                return clean_df, f"coinbase:{spec.coinbase_product}"
            last_error = f"coinbase: {status}"
        except Exception as exc:
            last_error = f"coinbase error: {exc}"
            logger.warning(f"[{spec.symbol} {interval}] Failover Coinbase gagal: {exc}")

    # Failover 2 untuk Kripto: CryptoCompare OHLC API (riwayat harian panjang)
    if spec.cryptocompare_fsym:
        try:
            time.sleep(API_SLEEP_SEC)
            cc_df = _fetch_cryptocompare_ohlc(spec.cryptocompare_fsym, interval, limit=180)
            clean_df, status = _sanitize_ohlcv(cc_df, spec, interval, apply_price_calibration=True)
            if not clean_df.empty:
                return clean_df, f"cryptocompare:{spec.cryptocompare_fsym}"
            last_error = f"cryptocompare: {status}"
        except Exception as exc:
            last_error = f"cryptocompare error: {exc}"
            logger.warning(f"[{spec.symbol} {interval}] Failover CryptoCompare gagal: {exc}")

    # Failover 3 untuk Kripto: CoinGecko OHLC API
    if spec.coingecko_id:
        try:
            time.sleep(API_SLEEP_SEC)
            cg_df = _fetch_coingecko_ohlc(spec.coingecko_id, interval)
            clean_df, status = _sanitize_ohlcv(cg_df, spec, interval, apply_price_calibration=True)
            if not clean_df.empty:
                return clean_df, f"coingecko:{spec.coingecko_id}"
            last_error = f"coingecko: {status}"
        except Exception as exc:
            last_error = f"coingecko error: {exc}"
            logger.warning(f"[{spec.symbol} {interval}] Failover CoinGecko gagal: {exc}")

    return pd.DataFrame(), last_error


# ==============================================================================
# 4. INDIKATOR KUANTITATIF, STRUKTUR HARGA & REZIM PASAR
# ==============================================================================

def compute_indicators(df: pd.DataFrame, ema_fast: int = 21, ema_slow: int = 55, atr_period: int = 14) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    open_ = out["Open"]
    vol = out["Volume"].replace(0, np.nan).fillna(1.0)

    out["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()
    ema_200_span = min(200, max(25, len(out) // 2))
    out["ema_200"] = close.ewm(span=ema_200_span, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / atr_period, adjust=False).mean().bfill()

    # Rolling Z-Score dengan proteksi pembagian nol (+ 1e-8)
    z_win = 20
    roll_mean = close.rolling(z_win, min_periods=8).mean()
    roll_std = close.rolling(z_win, min_periods=8).std()
    out["z_score"] = ((close - roll_mean) / (roll_std + 1e-8)).fillna(0.0)

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = gain / (loss + 1e-8)
    out["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    # Kaufman Efficiency Ratio (ER)
    er_win = 14
    net_change = (close - close.shift(er_win)).abs()
    volatility_sum = close.diff().abs().rolling(er_win, min_periods=er_win).sum()
    out["er"] = (net_change / (volatility_sum + 1e-8)).fillna(0.0)

    # Candle Anatomy & Wick Filter
    full_range = (high - low) + 1e-8
    body = (close - open_).abs()
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    out["body_ratio"] = body / full_range
    out["upper_wick_ratio"] = upper_wick / full_range
    out["lower_wick_ratio"] = lower_wick / full_range

    vol_ma = vol.rolling(20, min_periods=5).mean()
    out["rel_vol"] = (vol / (vol_ma + 1e-8)).fillna(1.0)

    # Swing High / Swing Low Structure (BOS)
    swing_lookback = 8
    out["prev_swing_high"] = high.shift(1).rolling(swing_lookback, min_periods=4).max()
    out["prev_swing_low"] = low.shift(1).rolling(swing_lookback, min_periods=4).min()

    return out


def detect_htf_bias(htf_df: pd.DataFrame) -> Tuple[str, float]:
    ind = compute_indicators(htf_df, ema_fast=21, ema_slow=55, atr_period=14)
    last = ind.iloc[-1]
    prev = ind.iloc[-2]

    bull_points = 0.0
    bear_points = 0.0

    if last["ema_fast"] > last["ema_slow"]:
        bull_points += 1.5
    elif last["ema_fast"] < last["ema_slow"]:
        bear_points += 1.5

    if last["Close"] > last["ema_fast"]:
        bull_points += 1.0
    elif last["Close"] < last["ema_fast"]:
        bear_points += 1.0

    if last["Close"] > last["ema_200"]:
        bull_points += 0.5
    else:
        bear_points += 0.5

    if last["ema_fast"] > prev["ema_fast"]:
        bull_points += 0.5
    elif last["ema_fast"] < prev["ema_fast"]:
        bear_points += 0.5

    if bull_points >= 2.0 and bull_points > bear_points + 0.5:
        return "BULLISH", float(last["er"])
    if bear_points >= 2.0 and bear_points > bull_points + 0.5:
        return "BEARISH", float(last["er"])
    return "NEUTRAL", float(last["er"])


def check_session_and_regime(
    spec: InstrumentSpec,
    mode_cfg: ModeConfig,
    ltf_ind: pd.DataFrame,
    htf_bias: str,
) -> Tuple[bool, str, str]:
    now_utc = datetime.now(timezone.utc)
    hour_utc = now_utc.hour
    weekday = now_utc.weekday()

    if not FORCE_MARKET_OPEN:
        if spec.asset_class in ("METALS", "INDEX"):
            if weekday == 5 or (weekday == 6 and hour_utc < 22) or (weekday == 4 and hour_utc >= 22):
                return False, "CLOSED", "Pasar tutup akhir pekan"

        if mode_cfg.mode == "SCALP" and spec.active_hours_utc is not None:
            start_h, end_h = spec.active_hours_utc
            if not (start_h <= hour_utc <= end_h):
                return False, "OFF_SESSION", f"Di luar jam likuiditas SCALP ({start_h:02d}-{end_h:02d} UTC)"

    last = ltf_ind.iloc[-1]
    er = float(last["er"])
    atr_pct = float(last["atr"] / (last["Close"] + 1e-8))
    atr_pct_median = float((ltf_ind["atr"] / (ltf_ind["Close"] + 1e-8)).tail(60).median())

    if atr_pct < atr_pct_median * 0.40:
        return False, "DEAD_VOL", "Volatilitas terlalu rendah (konsolidasi sempit)"
    if atr_pct > atr_pct_median * 3.50:
        return False, "EXTREME_SPIKE", "Lonjakan volatilitas ekstrem (risiko slippage tinggi)"

    if er < mode_cfg.min_efficiency_ratio:
        return False, "CHOPPY", f"Pasar choppy (ER={er:.2f} < {mode_cfg.min_efficiency_ratio:.2f})"

    if htf_bias == "NEUTRAL":
        return False, "CONFLICT_MTF", "Arah tren HTF netral/bertabrakan"

    regime_tag = "FORCE_TREND" if FORCE_MARKET_OPEN else "TREND"
    return True, regime_tag, "OK"


# ==============================================================================
# 5. BACKTEST ENGINE, PARAMETER OPTIMIZATION & LOT SIZING
# ==============================================================================

def calculate_position_lot(
    spec: InstrumentSpec,
    balance: float,
    risk_pct: float,
    entry: float,
    sl: float,
) -> float:
    sl_distance = abs(entry - sl)
    if sl_distance <= 0:
        return spec.min_lot

    risk_amount = balance * risk_pct
    loss_per_standard_lot = sl_distance * spec.contract_size
    if loss_per_standard_lot <= 0:
        return spec.min_lot

    raw_lot = risk_amount / loss_per_standard_lot
    stepped_lot = math.floor(raw_lot / spec.lot_step) * spec.lot_step
    final_lot = max(spec.min_lot, min(spec.max_lot, round(stepped_lot, 2)))
    return final_lot


def run_realistic_backtest(
    df_ind: pd.DataFrame,
    spec: InstrumentSpec,
    mode_cfg: ModeConfig,
    htf_bias_series: pd.Series,
    ema_fast: int,
    ema_slow: int,
    z_thresh: float,
) -> Tuple[Dict[str, float], pd.Series]:
    ind = compute_indicators(df_ind, ema_fast=ema_fast, ema_slow=ema_slow, atr_period=mode_cfg.atr_period)
    cost_price = (spec.spread_points + spec.slippage_points) * spec.pip_size

    equity = ACCOUNT_BALANCE
    peak_equity = equity
    max_dd_pct = 0.0
    equity_curve = []

    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    in_pos = False
    pos_dir = ""
    entry_p = 0.0
    sl_p = 0.0
    tp1_p = 0.0
    tp2_p = 0.0
    tp1_hit = False
    risk_dollars = 0.0

    start_idx = min(35, max(15, len(ind) // 4))
    for i in range(start_idx, len(ind)):
        row = ind.iloc[i]
        ts = ind.index[i]
        bias = htf_bias_series.iloc[i] if i < len(htf_bias_series) else "NEUTRAL"

        if in_pos:
            if pos_dir == "BUY":
                if row["Low"] <= sl_p:
                    pnl_r = 0.45 if tp1_hit else -1.0
                    pnl = risk_dollars * pnl_r
                    equity += pnl
                    if pnl >= 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        losses += 1
                        gross_loss += abs(pnl)
                    in_pos = False
                else:
                    if not tp1_hit and row["High"] >= tp1_p:
                        tp1_hit = True
                        sl_p = entry_p + (cost_price * 0.5)
                    if row["High"] >= tp2_p:
                        pnl = risk_dollars * (0.5 * mode_cfg.tp1_r + 0.5 * mode_cfg.tp2_r)
                        equity += pnl
                        wins += 1
                        gross_profit += pnl
                        in_pos = False
            else:
                if row["High"] >= sl_p:
                    pnl_r = 0.45 if tp1_hit else -1.0
                    pnl = risk_dollars * pnl_r
                    equity += pnl
                    if pnl >= 0:
                        wins += 1
                        gross_profit += pnl
                    else:
                        losses += 1
                        gross_loss += abs(pnl)
                    in_pos = False
                else:
                    if not tp1_hit and row["Low"] <= tp1_p:
                        tp1_hit = True
                        sl_p = entry_p - (cost_price * 0.5)
                    if row["Low"] <= tp2_p:
                        pnl = risk_dollars * (0.5 * mode_cfg.tp1_r + 0.5 * mode_cfg.tp2_r)
                        equity += pnl
                        wins += 1
                        gross_profit += pnl
                        in_pos = False

        if not in_pos and i < len(ind) - 1:
            er_ok = row["er"] >= (mode_cfg.min_efficiency_ratio * 0.85)
            atr_val = row["atr"]
            if er_ok and atr_val > 0:
                buy_cond = (
                    bias == "BULLISH"
                    and row["ema_fast"] > row["ema_slow"]
                    and row["Close"] > row["ema_fast"]
                    and 0.10 <= row["z_score"] <= z_thresh
                    and row["upper_wick_ratio"] < 0.50
                )
                sell_cond = (
                    bias == "BEARISH"
                    and row["ema_fast"] < row["ema_slow"]
                    and row["Close"] < row["ema_fast"]
                    and -z_thresh <= row["z_score"] <= -0.10
                    and row["lower_wick_ratio"] < 0.50
                )
                if buy_cond:
                    in_pos = True
                    pos_dir = "BUY"
                    tp1_hit = False
                    entry_p = float(row["Close"] + cost_price)
                    sl_dist = atr_val * mode_cfg.sl_atr_mult
                    sl_p = entry_p - sl_dist
                    tp1_p = entry_p + sl_dist * mode_cfg.tp1_r
                    tp2_p = entry_p + sl_dist * mode_cfg.tp2_r
                    risk_dollars = equity * mode_cfg.risk_pct
                elif sell_cond:
                    in_pos = True
                    pos_dir = "SELL"
                    tp1_hit = False
                    entry_p = float(row["Close"] - cost_price)
                    sl_dist = atr_val * mode_cfg.sl_atr_mult
                    sl_p = entry_p + sl_dist
                    tp1_p = entry_p - sl_dist * mode_cfg.tp1_r
                    tp2_p = entry_p - sl_dist * mode_cfg.tp2_r
                    risk_dollars = equity * mode_cfg.risk_pct

        peak_equity = max(peak_equity, equity)
        dd_pct = ((peak_equity - equity) / (peak_equity + 1e-8)) * 100.0
        max_dd_pct = max(max_dd_pct, dd_pct)
        equity_curve.append((ts, equity))

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    profit_factor = (gross_profit / (gross_loss + 1e-8)) if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
    net_return_pct = ((equity - ACCOUNT_BALANCE) / ACCOUNT_BALANCE) * 100.0

    stability_score = net_return_pct - (max_dd_pct * 1.35) + (min(profit_factor, 3.5) * 2.5)
    if total_trades < 3:
        stability_score -= 15.0

    stats = {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "net_return_pct": round(net_return_pct, 2),
        "stability_score": round(stability_score, 3),
    }
    eq_series = pd.Series(
        data=[v for _, v in equity_curve],
        index=pd.DatetimeIndex([t for t, _ in equity_curve]),
    )
    return stats, eq_series


def optimize_parameters(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    spec: InstrumentSpec,
    mode_cfg: ModeConfig,
) -> Tuple[Dict[str, Any], Dict[str, float], pd.Series]:
    htf_ind = compute_indicators(htf_df, ema_fast=21, ema_slow=55, atr_period=14)
    htf_trend = pd.Series("NEUTRAL", index=htf_ind.index)
    htf_trend[htf_ind["ema_fast"] > htf_ind["ema_slow"]] = "BULLISH"
    htf_trend[htf_ind["ema_fast"] < htf_ind["ema_slow"]] = "BEARISH"
    aligned_htf_bias = htf_trend.reindex(ltf_df.index, method="ffill").fillna("NEUTRAL")

    param_grid = [
        {"ema_fast": 13, "ema_slow": 34, "z_thresh": 2.2},
        {"ema_fast": 21, "ema_slow": 55, "z_thresh": 2.5},
        {"ema_fast": 21, "ema_slow": 89, "z_thresh": 2.8},
    ]

    best_params = param_grid[1]
    best_stats: Dict[str, float] = {}
    best_eq = pd.Series(dtype=float)
    best_obj = -999999.0

    for params in param_grid:
        stats, eq_series = run_realistic_backtest(
            df_ind=ltf_df,
            spec=spec,
            mode_cfg=mode_cfg,
            htf_bias_series=aligned_htf_bias,
            ema_fast=params["ema_fast"],
            ema_slow=params["ema_slow"],
            z_thresh=params["z_thresh"],
        )
        if stats["stability_score"] > best_obj:
            best_obj = stats["stability_score"]
            best_params = params
            best_stats = stats
            best_eq = eq_series

    return best_params, best_stats, best_eq


# ==============================================================================
# 6. SIGNAL GENERATOR & VALIDATOR HARGA KETAT
# ==============================================================================

def validate_price_levels(
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    tp3: float,
    atr: float,
) -> Tuple[bool, str]:
    if any(math.isnan(x) or math.isinf(x) or x <= 0 for x in [entry, sl, tp1, tp2, tp3, atr]):
        return False, "Nilai harga/ATR tidak valid (NaN/Inf/<=0)"

    sl_dist = abs(entry - sl)
    if sl_dist < (0.4 * atr) or sl_dist > (4.5 * atr):
        return False, f"Jarak SL ({sl_dist:.4f}) di luar batas aman ATR ({atr:.4f})"

    if direction == "BUY":
        if not (sl < entry < tp1 < tp2 < tp3):
            return False, f"Urutan BUY tidak valid: SL={sl}, E={entry}, TP1={tp1}, TP2={tp2}, TP3={tp3}"
    elif direction == "SELL":
        if not (sl > entry > tp1 > tp2 > tp3):
            return False, f"Urutan SELL tidak valid: SL={sl}, E={entry}, TP1={tp1}, TP2={tp2}, TP3={tp3}"
    else:
        return False, f"Arah tidak dikenal: {direction}"

    return True, "OK"


def evaluate_instrument_mode(spec: InstrumentSpec, mode_cfg: ModeConfig) -> Tuple[SignalCandidate, pd.Series]:
    htf_df, htf_src = fetch_market_data(spec, mode_cfg.htf_interval, mode_cfg.htf_period)
    if htf_df.empty:
        return SignalCandidate(
            symbol=spec.symbol,
            mode=mode_cfg.mode,
            direction="NO_SIGNAL",
            reason=f"Data HTF gagal ({htf_src})",
        ), pd.Series(dtype=float)

    ltf_df, ltf_src = fetch_market_data(spec, mode_cfg.ltf_interval, mode_cfg.ltf_period)
    if ltf_df.empty:
        return SignalCandidate(
            symbol=spec.symbol,
            mode=mode_cfg.mode,
            direction="NO_SIGNAL",
            reason=f"Data LTF gagal ({ltf_src})",
        ), pd.Series(dtype=float)

    htf_bias, _ = detect_htf_bias(htf_df)
    best_params, bt_stats, eq_series = optimize_parameters(ltf_df, htf_df, spec, mode_cfg)

    ltf_ind = compute_indicators(
        ltf_df,
        ema_fast=best_params["ema_fast"],
        ema_slow=best_params["ema_slow"],
        atr_period=mode_cfg.atr_period,
    )

    regime_ok, regime_label, regime_reason = check_session_and_regime(spec, mode_cfg, ltf_ind, htf_bias)
    last = ltf_ind.iloc[-1]
    prev = ltf_ind.iloc[-2]
    candle_ts = ltf_ind.index[-1].isoformat()

    if not regime_ok:
        return SignalCandidate(
            symbol=spec.symbol,
            mode=mode_cfg.mode,
            direction="NO_SIGNAL",
            regime=regime_label,
            efficiency_ratio=round(float(last["er"]), 3),
            z_score=round(float(last["z_score"]), 2),
            atr=float(last["atr"]),
            candle_ts=candle_ts,
            reason=regime_reason,
            best_params=best_params,
            backtest_stats=bt_stats,
        ), eq_series

    bull_score = 0
    bear_score = 0

    if htf_bias == "BULLISH":
        bull_score += 30
    elif htf_bias == "BEARISH":
        bear_score += 30

    if last["Close"] > last["ema_fast"] > last["ema_slow"] and last["ema_fast"] > prev["ema_fast"]:
        bull_score += 25
    elif last["Close"] < last["ema_fast"] < last["ema_slow"] and last["ema_fast"] < prev["ema_fast"]:
        bear_score += 25

    if last["Close"] > last["prev_swing_high"] or (
        prev["Low"] <= prev["ema_fast"] and last["Close"] > prev["High"]
    ):
        bull_score += 20
    if last["Close"] < last["prev_swing_low"] or (
        prev["High"] >= prev["ema_fast"] and last["Close"] < prev["Low"]
    ):
        bear_score += 20

    z_val = float(last["z_score"])
    rsi_val = float(last["rsi"])
    z_cap = best_params["z_thresh"]
    if 0.10 <= z_val <= z_cap and 50.0 <= rsi_val <= 75.0:
        bull_score += 15
    elif -z_cap <= z_val <= -0.10 and 25.0 <= rsi_val <= 50.0:
        bear_score += 15

    if last["Close"] > last["Open"] and last["body_ratio"] >= 0.42 and last["upper_wick_ratio"] <= 0.35:
        bull_score += 10
    elif last["Close"] < last["Open"] and last["body_ratio"] >= 0.42 and last["lower_wick_ratio"] <= 0.35:
        bear_score += 10

    if bull_score >= mode_cfg.min_score and bear_score >= 30:
        return SignalCandidate(
            symbol=spec.symbol,
            mode=mode_cfg.mode,
            direction="NO_SIGNAL",
            score=max(bull_score, bear_score),
            regime=regime_label,
            efficiency_ratio=round(float(last["er"]), 3),
            z_score=round(z_val, 2),
            atr=float(last["atr"]),
            candle_ts=candle_ts,
            reason="Konflik indikator bullish vs bearish",
            best_params=best_params,
            backtest_stats=bt_stats,
        ), eq_series

    if bull_score >= mode_cfg.min_score:
        direction = "BUY"
        final_score = bull_score
    elif bear_score >= mode_cfg.min_score:
        direction = "SELL"
        final_score = bear_score
    else:
        return SignalCandidate(
            symbol=spec.symbol,
            mode=mode_cfg.mode,
            direction="NO_SIGNAL",
            score=max(bull_score, bear_score),
            regime=regime_label,
            efficiency_ratio=round(float(last["er"]), 3),
            z_score=round(z_val, 2),
            atr=float(last["atr"]),
            candle_ts=candle_ts,
            reason=f"Konfluensi kurang (Bull={bull_score}, Bear={bear_score} < {mode_cfg.min_score})",
            best_params=best_params,
            backtest_stats=bt_stats,
        ), eq_series

    atr_val = float(last["atr"])
    close_price = float(last["Close"])
    half_spread = (spec.spread_points * spec.pip_size) * 0.5
    sl_dist = atr_val * mode_cfg.sl_atr_mult

    if direction == "BUY":
        entry = round(close_price + half_spread, spec.price_decimals)
        sl = round(entry - sl_dist, spec.price_decimals)
        tp1 = round(entry + sl_dist * mode_cfg.tp1_r, spec.price_decimals)
        tp2 = round(entry + sl_dist * mode_cfg.tp2_r, spec.price_decimals)
        tp3 = round(entry + sl_dist * mode_cfg.tp3_r, spec.price_decimals)
    else:
        entry = round(close_price - half_spread, spec.price_decimals)
        sl = round(entry + sl_dist, spec.price_decimals)
        tp1 = round(entry - sl_dist * mode_cfg.tp1_r, spec.price_decimals)
        tp2 = round(entry - sl_dist * mode_cfg.tp2_r, spec.price_decimals)
        tp3 = round(entry - sl_dist * mode_cfg.tp3_r, spec.price_decimals)

    valid_prices, price_msg = validate_price_levels(direction, entry, sl, tp1, tp2, tp3, atr_val)
    if not valid_prices:
        return SignalCandidate(
            symbol=spec.symbol,
            mode=mode_cfg.mode,
            direction="NO_SIGNAL",
            score=final_score,
            regime=regime_label,
            efficiency_ratio=round(float(last["er"]), 3),
            z_score=round(z_val, 2),
            atr=atr_val,
            candle_ts=candle_ts,
            reason=f"Validasi harga gagal: {price_msg}",
            best_params=best_params,
            backtest_stats=bt_stats,
        ), eq_series

    lot_size = calculate_position_lot(spec, ACCOUNT_BALANCE, mode_cfg.risk_pct, entry, sl)

    return SignalCandidate(
        symbol=spec.symbol,
        mode=mode_cfg.mode,
        direction=direction,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        lot_size=lot_size,
        score=final_score,
        regime=regime_label,
        efficiency_ratio=round(float(last["er"]), 3),
        z_score=round(z_val, 2),
        atr=atr_val,
        candle_ts=candle_ts,
        reason=f"Valid ({htf_src} + {ltf_src})",
        best_params=best_params,
        backtest_stats=bt_stats,
    ), eq_series


# ==============================================================================
# 7. VISUALISASI GRAFIK PERFORMA BACKTEST (MATPLOTLIB AGG)
# ==============================================================================

def generate_performance_chart(
    equity_curves: Dict[str, pd.Series],
    results: List[SignalCandidate],
    output_path: str = CHART_PATH,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.2, 1]})
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    colors = [
        "#38bdf8", "#f59e0b", "#10b981", "#a855f7",
        "#fb7185", "#2dd4bf", "#facc15", "#818cf8",
    ]
    plotted = 0
    for idx, (label, series) in enumerate(equity_curves.items()):
        if series is not None and not series.empty and len(series) >= 2:
            ax1.plot(
                series.index,
                series.values,
                label=label,
                color=colors[idx % len(colors)],
                linewidth=1.6,
                alpha=0.92,
            )
            plotted += 1

    if plotted == 0:
        now = datetime.now(timezone.utc)
        ax1.plot([now - timedelta(hours=1), now], [ACCOUNT_BALANCE, ACCOUNT_BALANCE], color="#38bdf8", label="Baseline")

    ax1.axhline(ACCOUNT_BALANCE, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.set_title(
        f"Quantitative Strategy Equity Curves (UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')})",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_ylabel("Simulated Equity ($)", color="#cbd5e1", fontsize=10)
    ax1.grid(True, color="#334155", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", ncol=4, fontsize=8, facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    labels = [f"{r.symbol}\n{r.mode}" for r in results]
    win_rates = [r.backtest_stats.get("win_rate", 0.0) for r in results]
    bar_colors = ["#10b981" if r.direction in ("BUY", "SELL") else "#64748b" for r in results]

    if labels:
        bars = ax2.bar(labels, win_rates, color=bar_colors, width=0.55)
        ax2.axhline(50.0, color="#f59e0b", linestyle="--", linewidth=1.0, alpha=0.7)
        for bar, wr in zip(bars, win_rates):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1.5,
                f"{wr:.1f}%",
                ha="center",
                va="bottom",
                color="#f8fafc",
                fontsize=8,
            )
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Backtest Win Rate (%)", color="#cbd5e1", fontsize=10)
    ax2.grid(True, axis="y", color="#334155", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Grafik backtest disimpan ke {output_path}")


# ==============================================================================
# 8. TELEGRAM & DISCORD NOTIFIER (EXPONENTIAL BACKOFF + /CHECK HANDLER)
# ==============================================================================

def _http_post_with_backoff(url: str, payload: Dict[str, Any], max_retries: int = 4) -> bool:
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return True
            if resp.status_code == 429:
                retry_after = 2 ** attempt
                try:
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", retry_after))
                except Exception:
                    pass
                logger.warning(f"Rate limit (429). Tidur {retry_after}s sebelum retry...")
                time.sleep(retry_after + 0.5)
                continue
            logger.warning(f"HTTP {resp.status_code} saat kirim notifikasi: {resp.text[:200]}")
        except Exception as exc:
            logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {exc}")

        time.sleep((2 ** attempt) + 0.5)
    return False


def send_telegram_message(text: str, chat_id: Optional[str] = None) -> bool:
    target_chat = (chat_id or TELEGRAM_CHAT_ID).strip()
    if not TELEGRAM_BOT_TOKEN or not target_chat:
        logger.info("TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID kosong; melewati pengiriman Telegram.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": target_chat,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    return _http_post_with_backoff(url, payload)


def send_discord_message(text: str) -> bool:
    if not DISCORD_WEBHOOK_URL:
        return False
    clean_text = (
        text.replace("<b>", "**")
        .replace("</b>", "**")
        .replace("<code>", "`")
        .replace("</code>", "`")
    )
    return _http_post_with_backoff(DISCORD_WEBHOOK_URL, {"content": clean_text})


def format_signal_telegram(sig: SignalCandidate) -> str:
    spec = INSTRUMENTS[sig.symbol]
    dec = spec.price_decimals
    now_hhmm = datetime.now(timezone.utc).strftime("%H:%M UTC")
    trail_note = "Aktif @ TP1 (BE)" if sig.mode == "SCALP" else "Aktif @ TP1 (BE) + ATR Trail"
    wr = sig.backtest_stats.get("win_rate", 0.0)
    pf = sig.backtest_stats.get("profit_factor", 0.0)

    return (
        f"⚡ <b>{sig.symbol} | {sig.mode} | {sig.direction}</b>\n"
        f"🎯 Entry: <code>{sig.entry:.{dec}f}</code> | 🛑 SL: <code>{sig.sl:.{dec}f}</code>\n"
        f"✅ TP1: <code>{sig.tp1:.{dec}f}</code> | TP2: <code>{sig.tp2:.{dec}f}</code> | TP3: <code>{sig.tp3:.{dec}f}</code>\n"
        f"📐 Lot: <code>{sig.lot_size:.2f}</code> | Skor: <code>{sig.score}/100</code> | Regime: <code>{sig.regime}</code>\n"
        f"🔄 Trail: {trail_note} | 🕒 <code>{now_hhmm}</code>\n"
        f"📊 ER: <code>{sig.efficiency_ratio:.2f}</code> | Z: <code>{sig.z_score:+.2f}</code> | BT WR: <code>{wr:.0f}%</code> (PF {pf:.2f})"
    )


def process_telegram_commands(db: SignalDatabase, latest_results: List[SignalCandidate]) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    last_update_str = db.get_state("last_telegram_update_id", "0")
    try:
        offset = int(last_update_str) + 1 if int(last_update_str) > 0 else None
    except ValueError:
        offset = None

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params: Dict[str, Any] = {"timeout": 3, "allowed_updates": ["message"]}
    if offset is not None:
        params["offset"] = offset

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            return
        data = resp.json()
        if not data.get("ok"):
            return

        updates = data.get("result", [])
        if not updates:
            return

        max_update_id = int(last_update_str) if last_update_str.isdigit() else 0
        for upd in updates:
            upd_id = int(upd.get("update_id", 0))
            if upd_id > max_update_id:
                max_update_id = upd_id

            msg = upd.get("message") or {}
            chat_id = str((msg.get("chat") or {}).get("id", "")).strip()
            text = (msg.get("text") or "").strip()

            if chat_id != TELEGRAM_CHAT_ID or not text.startswith("/"):
                continue

            parts = text.upper().split()
            cmd = parts[0].split("@")[0]

            if cmd in ("/CHECK", "/STATUS"):
                filter_tokens = set(parts[1:])
                filtered = latest_results
                if filter_tokens:
                    filtered = [
                        r for r in latest_results
                        if r.symbol in filter_tokens or r.mode in filter_tokens
                    ]
                    if not filtered:
                        filtered = latest_results

                now_hhmm = datetime.now(timezone.utc).strftime("%H:%M UTC")
                lines = [f"📡 <b>ON-DEMAND SCAN ({now_hhmm})</b>"]
                for r in filtered:
                    if r.direction in ("BUY", "SELL"):
                        dec = INSTRUMENTS[r.symbol].price_decimals
                        lines.append(
                            f"• <b>{r.symbol} {r.mode}</b>: <b>{r.direction}</b> @ <code>{r.entry:.{dec}f}</code> "
                            f"(SL <code>{r.sl:.{dec}f}</code> | TP1 <code>{r.tp1:.{dec}f}</code> | Skor {r.score})"
                        )
                    else:
                        lines.append(
                            f"• <b>{r.symbol} {r.mode}</b>: <code>NO_SIGNAL</code> ({r.regime}, Skor {r.score})"
                        )
                send_telegram_message("\n".join(lines), chat_id=chat_id)

        if max_update_id > 0:
            db.set_state("last_telegram_update_id", str(max_update_id))

    except Exception as exc:
        logger.warning(f"Gagal memproses perintah Telegram: {exc}")


# ==============================================================================
# 9. FORENSIC AUDIT LOG & ORCHESTRATOR UTAMA
# ==============================================================================

def forensic_audit_log(results: List[SignalCandidate]) -> None:
    print("\n" + "=" * 96)
    print(f"FORENSIC AUDIT LOG | UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 96)
    header = (
        f"{'SYMBOL':<8} | {'MODE':<6} | {'OUTCOME':<9} | {'SCORE':<5} | {'REGIME':<12} | "
        f"{'ER':<5} | {'Z-SCR':<6} | {'BT_WR':<6} | {'PF':<5} | {'DD%':<5} | NOTE"
    )
    print(header)
    print("-" * 96)
    for r in results:
        st = r.backtest_stats or {}
        print(
            f"{r.symbol:<8} | {r.mode:<6} | {r.direction:<9} | {r.score:<5d} | {r.regime:<12} | "
            f"{r.efficiency_ratio:<5.2f} | {r.z_score:<+6.2f} | {st.get('win_rate', 0.0):<5.1f}% | "
            f"{st.get('profit_factor', 0.0):<5.2f} | {st.get('max_dd_pct', 0.0):<5.1f} | {r.reason}"
        )
    print("=" * 96 + "\n")


def main() -> None:
    db = SignalDatabase(DB_PATH)
    results: List[SignalCandidate] = []
    equity_curves: Dict[str, pd.Series] = {}
    sent_count = 0

    for symbol, spec in INSTRUMENTS.items():
        for mode_name, mode_cfg in MODE_CONFIGS.items():
            try:
                sig, eq_series = evaluate_instrument_mode(spec, mode_cfg)
                results.append(sig)
                equity_curves[f"{symbol}-{mode_name}"] = eq_series
                db.save_audit(sig)

                if sig.direction in ("BUY", "SELL"):
                    is_dup, dup_reason = db.is_duplicate_or_in_cooldown(
                        symbol=sig.symbol,
                        mode=sig.mode,
                        direction=sig.direction,
                        candle_ts=sig.candle_ts,
                        cooldown_minutes=mode_cfg.cooldown_minutes,
                    )
                    if is_dup:
                        logger.info(f"[{sig.symbol} {sig.mode}] Sinyal {sig.direction} ditahan: {dup_reason}")
                        sig.reason = f"Ditahan ({dup_reason})"
                    else:
                        msg = format_signal_telegram(sig)
                        tg_ok = send_telegram_message(msg)
                        send_discord_message(msg)
                        db.save_signal(sig)
                        sent_count += 1
                        logger.info(
                            f"[{sig.symbol} {sig.mode}] SINYAL TERKIRIM: {sig.direction} @ {sig.entry} (TG={tg_ok})"
                        )
            except Exception as item_exc:
                err_msg = f"Error evaluasi {symbol} {mode_name}: {item_exc}"
                logger.error(err_msg)
                results.append(
                    SignalCandidate(
                        symbol=symbol,
                        mode=mode_name,
                        direction="NO_SIGNAL",
                        reason=err_msg[:180],
                    )
                )

    forensic_audit_log(results)
    generate_performance_chart(equity_curves, results, CHART_PATH)
    process_telegram_commands(db, results)
    logger.info(f"Selesai. Total sinyal baru terkirim: {sent_count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_exc:
        tb = traceback.format_exc()
        logger.critical(f"FATAL ENGINE ERROR:\n{tb}")
        try:
            send_telegram_message(
                f"🚨 <b>QUANT BOT ENGINE ERROR</b>\n"
                f"🕒 <code>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</code>\n"
                f"<code>{str(fatal_exc)[:350]}</code>"
            )
        except Exception:
            pass
        sys.exit(1)
