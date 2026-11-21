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
import json
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

# Kalibrasi offset/multiplier harga jika broker CFD memiliki selisih terhadap feed indeks acuan
NAS100_PRICE_MULTIPLIER = float(os.getenv("NAS100_PRICE_MULTIPLIER", "1.0"))
NAS100_PRICE_OFFSET = float(os.getenv("NAS100_PRICE_OFFSET", "0.0"))
XAUUSD_PRICE_OFFSET = float(os.getenv("XAUUSD_PRICE_OFFSET", "0.0"))

REQUEST_TIMEOUT = 15
API_SLEEP_SEC = 0.65


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    display_name: str
    asset_class: str  # "METALS", "INDEX", "CRYPTO"
    primary_tickers: List[str]
    binance_symbol: Optional[str]
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
        primary_tickers=["XAUUSD=X"],  # Prioritas Spot, hindari GC=F
        binance_symbol=None,
        price_decimals=2,
        contract_size=100.0,  # 1 lot standar = 100 oz -> $1 gerak = $100/lot
        pip_size=0.10,
        spread_points=0.25,
        slippage_points=0.10,
        price_multiplier=1.0,
        price_offset=XAUUSD_PRICE_OFFSET,
        active_hours_utc=(6, 21),  # Sesi London - New York aktif
    ),
    "NAS100": InstrumentSpec(
        symbol="NAS100",
        display_name="NAS100",
        asset_class="INDEX",
        primary_tickers=["^NDX", "QQQ"],  # Indeks acuan + fallback ETF likuid
        binance_symbol=None,
        price_decimals=2,
        contract_size=1.0,  # 1 lot standar CFD indeks = $1 per 1.0 poin indeks
        pip_size=1.0,
        spread_points=1.80,
        slippage_points=0.75,
        price_multiplier=NAS100_PRICE_MULTIPLIER,
        price_offset=NAS100_PRICE_OFFSET,
        active_hours_utc=(12, 21),  # Pre-market US hingga penutupan NYSE/NASDAQ
    ),
    "BTCUSD": InstrumentSpec(
        symbol="BTCUSD",
        display_name="BTCUSD",
        asset_class="CRYPTO",
        primary_tickers=["BTC-USD"],
        binance_symbol="BTCUSDT",
        price_decimals=2,
        contract_size=1.0,  # 1 lot = 1 BTC -> $1 gerak = $1/lot
        pip_size=1.0,
        spread_points=18.0,
        slippage_points=8.0,
        price_multiplier=1.0,
        price_offset=0.0,
        active_hours_utc=None,  # 24/7
    ),
    "ETHUSD": InstrumentSpec(
        symbol="ETHUSD",
        display_name="ETHUSD",
        asset_class="CRYPTO",
        primary_tickers=["ETH-USD"],
        binance_symbol="ETHUSDT",
        price_decimals=2,
        contract_size=1.0,  # 1 lot = 1 ETH -> $1 gerak = $1/lot
        pip_size=0.10,
        spread_points=1.50,
        slippage_points=0.60,
        price_multiplier=1.0,
        price_offset=0.0,
        active_hours_utc=None,  # 24/7
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
        min_score=68,
        min_efficiency_ratio=0.28,
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
        min_score=70,
        min_efficiency_ratio=0.30,
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
        """
        Aturan anti-duplikasi cerdas:
        1. Tolak jika sinyal persis pada (symbol, mode, direction, candle_ts) sudah pernah terkirim.
        2. Tolak jika sinyal dengan ARAH YANG SAMA baru dikirim dalam rentang cooldown_minutes.
        3. IZINKAN jika terjadi pembalikan arah yang sah (BUY -> SELL atau sebaliknya) pada candle baru.
        """
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

    def get_recent_signals(self, limit: int = 8) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()


# ==============================================================================
# 3. MULTI-SOURCE DATA ENGINE & SANITY GUARD
# ==============================================================================

def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    return df


def _fetch_binance_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    interval_map = {
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
    }
    bn_interval = interval_map.get(interval)
    if not bn_interval:
        return pd.DataFrame()

    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": bn_interval, "limit": min(limit, 1000)}
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    raw = resp.json()
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()

    records = []
    for k in raw:
        records.append({
            "Datetime": pd.to_datetime(int(k[0]), unit="ms", utc=True),
            "Open": float(k[1]),
            "High": float(k[2]),
            "Low": float(k[3]),
            "Close": float(k[4]),
            "Volume": float(k[5]),
        })
    df = pd.DataFrame(records).set_index("Datetime")
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
    """
    Membersihkan NaN, memvalidasi timestamp segar (anti-stale), dan memfilter outlier spike ekstrem.
    """
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

    if len(df) < 45:
        return pd.DataFrame(), f"Bar valid terlalu sedikit ({len(df)} < 45)"

    # Pastikan index bertipe DatetimeIndex UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()]
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Kalibrasi harga broker (khusus jika QQQ dipakai sebagai fallback NAS100)
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

    if len(df) < 45:
        return pd.DataFrame(), "Bar tersisa setelah filter spike tidak cukup"

    # Cek staleness (apakah data berhenti berhari-hari saat bukan akhir pekan)
    now_utc = datetime.now(timezone.utc)
    last_ts = df.index[-1].to_pydatetime()
    age_hours = (now_utc - last_ts).total_seconds() / 3600.0
    max_stale_hours = 78.0 if spec.asset_class in ("METALS", "INDEX") else 8.0
    if age_hours > max_stale_hours:
        return pd.DataFrame(), f"Data stale: bar terakhir {last_ts.isoformat()} ({age_hours:.1f}j lalu)"

    # Gunakan candle yang sudah tertutup (drop bar terakhir yang masih berjalan / unclosed)
    # Kecuali jika bar terakhir sudah berumur lebih dari durasi intervalnya
    interval_minutes = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}.get(interval, 60)
    if (now_utc - last_ts).total_seconds() < (interval_minutes * 60 * 0.85) and len(df) > 50:
        df = df.iloc[:-1]

    return df, "OK"


def fetch_market_data(spec: InstrumentSpec, interval: str, period: str) -> Tuple[pd.DataFrame, str]:
    """
    Multi-source failover fetcher:
    1. Coba primary_tickers via yfinance (dengan resampling 1h -> 4h jika interval == '4h').
    2. Jika kripto dan yfinance gagal/stale, failover ke Binance Public API.
    """
    yf_interval = "1h" if interval == "4h" else interval
    yf_period = "120d" if interval == "4h" else period

    last_error = "Tidak ada sumber tersedia"
    for idx, ticker in enumerate(spec.primary_tickers):
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

            # Jika fallback ke ETF QQQ untuk NAS100 tanpa custom multiplier, sesuaikan skala rasio ~41.5x
            apply_cal = True
            if spec.symbol == "NAS100" and ticker == "QQQ" and spec.price_multiplier == 1.0:
                for col in ["Open", "High", "Low", "Close"]:
                    df[col] = df[col] * 41.5 + spec.price_offset
                apply_cal = False

            clean_df, status = _sanitize_ohlcv(df, spec, interval, apply_price_calibration=apply_cal)
            if not clean_df.empty:
                return clean_df, f"yfinance:{ticker}"
            last_error = f"{ticker}: {status}"
        except Exception as exc:
            last_error = f"{ticker} error: {exc}"
            logger.warning(f"[{spec.symbol} {interval}] Gagal ambil {ticker}: {exc}")

    # Failover ke Binance untuk aset Kripto
    if spec.binance_symbol:
        try:
            time.sleep(API_SLEEP_SEC)
            bn_df = _fetch_binance_klines(spec.binance_symbol, interval, limit=600)
            clean_df, status = _sanitize_ohlcv(bn_df, spec, interval, apply_price_calibration=True)
            if not clean_df.empty:
                return clean_df, f"binance:{spec.binance_symbol}"
            last_error = f"binance: {status}"
        except Exception as exc:
            last_error = f"binance error: {exc}"
            logger.warning(f"[{spec.symbol} {interval}] Failover Binance gagal: {exc}")

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

    # Exponential Moving Averages
    out["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()
    out["ema_200"] = close.ewm(span=min(200, max(60, len(out) // 2)), adjust=False).mean()

    # True Range & ATR (Wilder Smoothing)
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
    roll_mean = close.rolling(z_win, min_periods=10).mean()
    roll_std = close.rolling(z_win, min_periods=10).std()
    out["z_score"] = ((close - roll_mean) / (roll_std + 1e-8)).fillna(0.0)

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs = gain / (loss + 1e-8)
    out["rsi"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

    # Kaufman Efficiency Ratio (ER) - Filter Anti-Chop
    er_win = 14
    net_change = (close - close.shift(er_win)).abs()
    volatility_sum = close.diff().abs().rolling(er_win, min_periods=er_win).sum()
    out["er"] = (net_change / (volatility_sum + 1e-8)).fillna(0.0)

    # Candle Anatomy & Wick Rejection Filter (Order Flow Proxy)
    full_range = (high - low) + 1e-8
    body = (close - open_).abs()
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    out["body_ratio"] = body / full_range
    out["upper_wick_ratio"] = upper_wick / full_range
    out["lower_wick_ratio"] = lower_wick / full_range

    vol_ma = vol.rolling(20, min_periods=5).mean()
    out["rel_vol"] = (vol / (vol_ma + 1e-8)).fillna(1.0)

    # Swing High / Swing Low Structure (Break of Structure / BOS)
    swing_lookback = 8
    out["prev_swing_high"] = high.shift(1).rolling(swing_lookback, min_periods=4).max()
    out["prev_swing_low"] = low.shift(1).rolling(swing_lookback, min_periods=4).min()

    return out


def detect_htf_bias(htf_df: pd.DataFrame) -> Tuple[str, float]:
    """
    Menentukan arah tren Higher Timeframe (HTF) dan kekuatan momentumnya.
    """
    ind = compute_indicators(htf_df, ema_fast=21, ema_slow=55, atr_period=14)
    last = ind.iloc[-1]
    prev = ind.iloc[-2]

    bull_points = 0
    bear_points = 0

    if last["Close"] > last["ema_fast"] > last["ema_slow"]:
        bull_points += 2
    elif last["Close"] < last["ema_fast"] < last["ema_slow"]:
        bear_points += 2

    if last["Close"] > last["ema_200"]:
        bull_points += 1
    else:
        bear_points += 1

    if last["ema_fast"] > prev["ema_fast"] and last["rsi"] > 52:
        bull_points += 1
    elif last["ema_fast"] < prev["ema_fast"] and last["rsi"] < 48:
        bear_points += 1

    if bull_points >= 3 and bear_points <= 1:
        return "BULLISH", float(last["er"])
    if bear_points >= 3 and bull_points <= 1:
        return "BEARISH", float(last["er"])
    return "NEUTRAL", float(last["er"])


def check_session_and_regime(
    spec: InstrumentSpec,
    mode_cfg: ModeConfig,
    ltf_ind: pd.DataFrame,
    htf_bias: str,
) -> Tuple[bool, str, str]:
    """
    Validasi jam perdagangan aktif, rezim volatilitas, dan filter anti-chop per kelas aset.
    """
    now_utc = datetime.now(timezone.utc)
    hour_utc = now_utc.hour
    weekday = now_utc.weekday()  # 5=Sabtu, 6=Minggu

    # 1. Cek hari libur pasar tradisional (XAUUSD & NAS100 tutup akhir pekan)
    if spec.asset_class in ("METALS", "INDEX"):
        if weekday == 5 or (weekday == 6 and hour_utc < 22) or (weekday == 4 and hour_utc >= 22):
            return False, "CLOSED", "Pasar tutup akhir pekan"

    # 2. Filter sesi aktif khusus mode SCALP pada Metals & Index
    if mode_cfg.mode == "SCALP" and spec.active_hours_utc is not None:
        start_h, end_h = spec.active_hours_utc
        if not (start_h <= hour_utc <= end_h):
            return False, "OFF_SESSION", f"Di luar jam likuiditas SCALP ({start_h:02d}-{end_h:02d} UTC)"

    last = ltf_ind.iloc[-1]
    er = float(last["er"])
    atr_pct = float(last["atr"] / (last["Close"] + 1e-8))
    atr_pct_median = float((ltf_ind["atr"] / (ltf_ind["Close"] + 1e-8)).tail(60).median())

    # 3. Filter volatilitas mati atau ekstrem (flash crash / news spike liar)
    if atr_pct < atr_pct_median * 0.45:
        return False, "DEAD_VOL", "Volatilitas terlalu rendah (konsolidasi sempit)"
    if atr_pct > atr_pct_median * 3.20:
        return False, "EXTREME_SPIKE", "Lonjakan volatilitas ekstrem (risiko slippage tinggi)"

    # 4. Filter rezim choppy menggunakan Kaufman Efficiency Ratio
    if er < mode_cfg.min_efficiency_ratio:
        return False, "CHOPPY", f"Pasar choppy (ER={er:.2f} < {mode_cfg.min_efficiency_ratio:.2f})"

    if htf_bias == "NEUTRAL":
        return False, "CONFLICT_MTF", "Arah tren HTF netral/bertabrakan"

    return True, "TREND", "OK"


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
    # Nilai risiko per 1.0 lot standar = sl_distance * contract_size
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
    """
    Simulasi backtest bar-by-bar dengan biaya spread + slippage, partial TP1 -> Breakeven,
    dan trailing stop pada sisa posisi.
    """
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

    start_idx = min(60, max(25, len(ind) // 4))
    for i in range(start_idx, len(ind)):
        row = ind.iloc[i]
        ts = ind.index[i]
        bias = htf_bias_series.iloc[i] if i < len(htf_bias_series) else "NEUTRAL"

        if in_pos:
            if pos_dir == "BUY":
                # Cek SL terlebih dahulu (konservatif)
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
                        sl_p = entry_p + (cost_price * 0.5)  # Geser ke Breakeven + biaya
                    if row["High"] >= tp2_p:
                        pnl = risk_dollars * (0.5 * mode_cfg.tp1_r + 0.5 * mode_cfg.tp2_r)
                        equity += pnl
                        wins += 1
                        gross_profit += pnl
                        in_pos = False
            else:  # SELL
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
            er_ok = row["er"] >= (mode_cfg.min_efficiency_ratio * 0.90)
            atr_val = row["atr"]
            if er_ok and atr_val > 0:
                buy_cond = (
                    bias == "BULLISH"
                    and row["ema_fast"] > row["ema_slow"]
                    and row["Close"] > row["ema_fast"]
                    and 0.15 <= row["z_score"] <= z_thresh
                    and row["upper_wick_ratio"] < 0.48
                )
                sell_cond = (
                    bias == "BEARISH"
                    and row["ema_fast"] < row["ema_slow"]
                    and row["Close"] < row["ema_fast"]
                    and -z_thresh <= row["z_score"] <= -0.15
                    and row["lower_wick_ratio"] < 0.48
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

    # Skor stabilitas dengan penalti drawdown untuk grid search
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
    """
    Grid search terukur dengan penalti drawdown agar parameter tahan terhadap overfitting.
    """
    htf_ind = compute_indicators(htf_df, ema_fast=21, ema_slow=55, atr_period=14)
    htf_trend = pd.Series("NEUTRAL", index=htf_ind.index)
    htf_trend[htf_ind["ema_fast"] > htf_ind["ema_slow"]] = "BULLISH"
    htf_trend[htf_ind["ema_fast"] < htf_ind["ema_slow"]] = "BEARISH"
    aligned_htf_bias = htf_trend.reindex(ltf_df.index, method="ffill").fillna("NEUTRAL")

    param_grid = [
        {"ema_fast": 13, "ema_slow": 34, "z_thresh": 2.1},
        {"ema_fast": 21, "ema_slow": 55, "z_thresh": 2.3},
        {"ema_fast": 21, "ema_slow": 89, "z_thresh": 2.5},
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
    """
    Pengaman logika harga wajib sebelum sinyal dikeluarkan:
    - BUY : SL < Entry < TP1 < TP2 < TP3
    - SELL: SL > Entry > TP1 > TP2 > TP3
    - Jarak SL harus terukur proporsional terhadap ATR.
    """
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
    """
    Pipeline evaluasi end-to-end untuk 1 pasangan (Instrumen, Mode).
    """
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

    htf_bias, htf_er = detect_htf_bias(htf_df)
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

    # Penilaian Confluences Kuantitatif (Skor 0 - 100)
    bull_score = 0
    bear_score = 0

    # 1. Keselarasan Multi-Timeframe (30 poin)
    if htf_bias == "BULLISH":
        bull_score += 30
    elif htf_bias == "BEARISH":
        bear_score += 30

    # 2. Struktur EMA & Momentum LTF (25 poin)
    if last["Close"] > last["ema_fast"] > last["ema_slow"] and last["ema_fast"] > prev["ema_fast"]:
        bull_score += 25
    elif last["Close"] < last["ema_fast"] < last["ema_slow"] and last["ema_fast"] < prev["ema_fast"]:
        bear_score += 25

    # 3. Break of Structure (BOS) / Pullback Continuation (20 poin)
    if last["Close"] > last["prev_swing_high"] or (
        prev["Low"] <= prev["ema_fast"] and last["Close"] > prev["High"]
    ):
        bull_score += 20
    if last["Close"] < last["prev_swing_low"] or (
        prev["High"] >= prev["ema_fast"] and last["Close"] < prev["Low"]
    ):
        bear_score += 20

    # 4. Statistical Z-Score & RSI Momentum Zone (15 poin)
    z_val = float(last["z_score"])
    rsi_val = float(last["rsi"])
    z_cap = best_params["z_thresh"]
    if 0.15 <= z_val <= z_cap and 52.0 <= rsi_val <= 72.0:
        bull_score += 15
    elif -z_cap <= z_val <= -0.15 and 28.0 <= rsi_val <= 48.0:
        bear_score += 15

    # 5. Order Flow / Candle Body Dominance & Anti-Wick Rejection (10 poin)
    if last["Close"] > last["Open"] and last["body_ratio"] >= 0.45 and last["upper_wick_ratio"] <= 0.32:
        bull_score += 10
    elif last["Close"] < last["Open"] and last["body_ratio"] >= 0.45 and last["lower_wick_ratio"] <= 0.32:
        bear_score += 10

    # Deteksi konflik arah
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

    # Kalkulasi Entry, SL, TP1, TP2, TP3 berbasis ATR
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
    """
    Membuat grafik kurva ekuitas & ringkasan audit backtest.
    Selalu menghasilkan file valid meskipun data pasar terbatas.
    """
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

    # Subplot 2: Bar Chart Win Rate per Instrumen & Mode
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
    """
    Format notifikasi Telegram yang ringkas, padat, siap eksekusi manual di MT5,
    disertai metrik kuantitatif tambahan (ER, Z-Score, dan Backtest Win Rate).
    """
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
    """
    Memeriksa pembaruan perintah Telegram (/check, /check XAUUSD, /check SCALP, /status)
    dengan validasi chat_id dan pelacakan offset update_id di SQLite.
    """
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

            # Validasi otorisasi chat_id
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
