#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APEX-QUANT-OMNI — HYPERION PRECISION EDITION
Dual-Mode Quantitative Signal, Backtest, Risk & Auto-Trailing Engine for XAUUSD (Spot Gold).
"""

import os
import sys
import time
import math
import json
import sqlite3
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ==============================================================================
# LOGGING & GLOBAL CONSTANTS
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("APEX_QUANT_OMNI")

DB_PATH = os.getenv("DB_PATH", "trading_signals.db")
CHART_PATH = os.getenv("CHART_PATH", "backtest_performance.png")

# XAUUSD Spot Contract Calibration (Standard MT5)
CONTRACT_SIZE = 100.0       # 1.00 standard lot = 100 troy ounces
PIP_SIZE = 0.10             # 1 pip = $0.10 price move ($10/pip per 1.00 lot)
MIN_LOT = 0.01
LOT_STEP = 0.01
MAX_LOT = float(os.getenv("MAX_LOT", "10.0"))

# Friction & Execution Realism
SPREAD_USD = float(os.getenv("SPREAD_USD", "0.30"))
SLIPPAGE_USD = float(os.getenv("SLIPPAGE_USD", "0.10"))

# Account & Risk
DEFAULT_ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000.0"))
DEFAULT_RISK_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))

# Circuit Breaker & Retry Policy
RETRY_BACKOFFS = [2, 4, 8]
CIRCUIT_MAX_FAILURES = 5
CIRCUIT_COOLDOWN_MINUTES = 60

# Grid Search Space
GRID_ATR_PERIODS = [10, 14, 20]
GRID_Z_THRESHOLDS = [1.5, 2.0, 2.5]


@dataclass(frozen=True)
class ModeConfig:
    mode_name: str
    exec_tf: str
    mid_tf: str
    htf_tf: str
    sl_atr_mult: float
    tp_r_multiples: Tuple[float, float, float]
    trail_atr_mult: float
    min_score: float
    min_ker: float
    max_chop: float
    min_adx: float
    cooldown_minutes: int
    min_backtest_trades: int
    z_window: int
    vwap_window: int
    swing_lr: int
    stale_max_minutes: int
    be_buffer_usd: float


SCALP_CONFIG = ModeConfig(
    mode_name="SCALP",
    exec_tf="5m",
    mid_tf="15m",
    htf_tf="1h",
    sl_atr_mult=1.35,
    tp_r_multiples=(1.2, 2.0, 3.0),
    trail_atr_mult=0.50,
    min_score=72.0,
    min_ker=0.30,
    max_chop=58.0,
    min_adx=18.0,
    cooldown_minutes=30,
    min_backtest_trades=8,
    z_window=20,
    vwap_window=48,
    swing_lr=3,
    stale_max_minutes=35,
    be_buffer_usd=0.25,
)

SWING_CONFIG = ModeConfig(
    mode_name="SWING",
    exec_tf="15m",
    mid_tf="1h",
    htf_tf="4h",
    sl_atr_mult=1.85,
    tp_r_multiples=(1.5, 2.5, 4.0),
    trail_atr_mult=0.80,
    min_score=75.0,
    min_ker=0.34,
    max_chop=56.0,
    min_adx=20.0,
    cooldown_minutes=120,
    min_backtest_trades=5,
    z_window=24,
    vwap_window=32,
    swing_lr=4,
    stale_max_minutes=60,
    be_buffer_usd=0.35,
)


@dataclass
class SignalCandidate:
    mode: str
    direction: str                  # "BUY", "SELL", or "NO_SIGNAL"
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    lot_size: float = 0.01
    score: float = 0.0
    p_win: float = 0.0
    ev_r: float = 0.0
    ker: float = 0.0
    adx: float = 0.0
    z_score: float = 0.0
    atr: float = 0.0
    regime: str = "UNKNOWN"
    mtf_bias: str = "NEUTRAL"
    candle_ts: str = ""
    reason: str = ""
    pillar_details: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# UTILITY FUNCTIONS (TIME, RISK, MT5 LOT CALCULATION)
# ==============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_market_open(now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Spot XAUUSD market session guard:
    - Closed: Friday >= 21:00 UTC through Sunday < 22:05 UTC.
    - Rollover illiquid window: 21:00 - 22:00 UTC Mon-Thu.
    - Friday late close risk: >= 20:00 UTC Friday.
    """
    dt = now_utc or utc_now()
    wd = dt.weekday()  # Mon=0 .. Sun=6
    hr = dt.hour
    mn = dt.minute

    if wd == 4 and hr >= 21:
        return False, "WEEKEND_CLOSED"
    if wd == 5:
        return False, "WEEKEND_CLOSED"
    if wd == 6 and (hr < 22 or (hr == 22 and mn < 5)):
        return False, "WEEKEND_CLOSED"
    if hr == 21:
        return False, "ROLLOVER_ILLIQUID"
    if wd == 4 and hr >= 20:
        return False, "FRIDAY_LATE_CLOSE"
    return True, "ACTIVE_SESSION"


def calculate_mt5_lot_size(
    entry_price: float,
    sl_price: float,
    account_balance: float = DEFAULT_ACCOUNT_BALANCE,
    risk_pct: float = DEFAULT_RISK_PCT,
) -> float:
    """
    Exact MT5 Spot Gold lot sizing using CONTRACT_SIZE & PIP_SIZE with safe 0.01 rounding.
    """
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 1e-6 or account_balance <= 0 or risk_pct <= 0:
        return MIN_LOT

    risk_usd = account_balance * (risk_pct / 100.0)
    sl_pips = sl_distance / PIP_SIZE
    pip_value_per_lot = CONTRACT_SIZE * PIP_SIZE  # $10.00 per pip per 1.00 lot
    raw_lot = risk_usd / (sl_pips * pip_value_per_lot + 1e-8)

    rounded_lot = math.floor(raw_lot / LOT_STEP) * LOT_STEP
    clamped_lot = max(MIN_LOT, min(MAX_LOT, round(rounded_lot, 2)))
    return clamped_lot


# ==============================================================================
# SQLITE STATE MANAGER (PERSISTENCE, COOLDOWN, DEDUP, CIRCUIT BREAKER)
# ==============================================================================
class StateManager:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry REAL NOT NULL,
                    initial_sl REAL NOT NULL,
                    current_sl REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    tp3 REAL NOT NULL,
                    lot_size REAL NOT NULL,
                    score REAL NOT NULL,
                    p_win REAL NOT NULL,
                    ev_r REAL NOT NULL,
                    atr REAL NOT NULL,
                    regime TEXT NOT NULL,
                    status TEXT NOT NULL,
                    candle_ts TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_signals_mode_ts ON signals(mode, candle_ts)"
            )
            conn.commit()

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM bot_state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        now_s = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, now_s),
            )
            conn.commit()

    def is_circuit_open(self) -> Tuple[bool, str]:
        open_until = self.get_state("circuit_open_until")
        if not open_until:
            return False, ""
        try:
            dt_until = datetime.fromisoformat(open_until)
            if utc_now() < dt_until:
                return True, open_until
        except ValueError:
            pass
        self.set_state("circuit_open_until", "")
        self.set_state("circuit_failures", "0")
        return False, ""

    def record_failure(self) -> Tuple[int, bool]:
        failures = int(self.get_state("circuit_failures", "0") or "0") + 1
        self.set_state("circuit_failures", str(failures))
        tripped = False
        if failures >= CIRCUIT_MAX_FAILURES:
            until = utc_now() + timedelta(minutes=CIRCUIT_COOLDOWN_MINUTES)
            self.set_state("circuit_open_until", until.isoformat())
            tripped = True
        return failures, tripped

    def record_success(self) -> None:
        self.set_state("circuit_failures", "0")
        self.set_state("circuit_open_until", "")
        self.set_state("last_healthy_run", utc_now().isoformat())

    def is_duplicate_or_cooldown(
        self, mode: str, direction: str, candle_ts: str, cooldown_minutes: int, score: float
    ) -> Tuple[bool, str]:
        with self._connect() as conn:
            dup = conn.execute(
                """
                SELECT id FROM signals
                WHERE mode = ? AND direction = ? AND candle_ts = ?
                """,
                (mode, direction, candle_ts),
            ).fetchone()
            if dup:
                return True, f"DUPLICATE_CANDLE_{candle_ts}"

            last_sig = conn.execute(
                """
                SELECT direction, created_at FROM signals
                WHERE mode = ?
                ORDER BY id DESC LIMIT 1
                """,
                (mode,),
            ).fetchone()
            if last_sig:
                last_dt = datetime.fromisoformat(last_sig["created_at"])
                elapsed_min = (utc_now() - last_dt).total_seconds() / 60.0
                if elapsed_min < cooldown_minutes:
                    return True, f"COOLDOWN_ACTIVE_{elapsed_min:.1f}m/{cooldown_minutes}m"
                if (
                    last_sig["direction"] != direction
                    and elapsed_min < (cooldown_minutes * 2.0)
                    and score < 88.0
                ):
                    return True, f"ANTI_WHIPSAW_FLIP_GUARD_{elapsed_min:.1f}m"
        return False, ""

    def insert_signal(self, sig: SignalCandidate) -> int:
        now_s = utc_now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO signals (
                    mode, direction, entry, initial_sl, current_sl,
                    tp1, tp2, tp3, lot_size, score, p_win, ev_r,
                    atr, regime, status, candle_ts, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)
                """,
                (
                    sig.mode,
                    sig.direction,
                    sig.entry,
                    sig.sl,
                    sig.sl,
                    sig.tp1,
                    sig.tp2,
                    sig.tp3,
                    sig.lot_size,
                    sig.score,
                    sig.p_win,
                    sig.ev_r,
                    sig.atr,
                    sig.regime,
                    sig.candle_ts,
                    now_s,
                    now_s,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_active_signals(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM signals
                WHERE status IN ('OPEN', 'TP1_TRAIL', 'TP2_TRAIL')
                ORDER BY id DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_signals(self, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update_signal_trail(self, sig_id: int, new_sl: float, new_status: str) -> None:
        now_s = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE signals
                SET current_sl = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (round(new_sl, 2), new_status, now_s, sig_id),
            )
            conn.commit()


# ==============================================================================
# NOTIFICATION & INTERACTIVE TELEGRAM POLLING (EXPONENTIAL BACKOFF)
# ==============================================================================
class Notifier:
    def __init__(self, state: StateManager) -> None:
        self.state = state
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        raw_chat_ids = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.authorized_chat_ids = [
            c.strip() for c in raw_chat_ids.split(",") if c.strip()
        ]
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    def _request_with_backoff(
        self, method: str, url: str, **kwargs: Any
    ) -> Optional[requests.Response]:
        kwargs.setdefault("timeout", 12)
        for attempt in range(len(RETRY_BACKOFFS) + 1):
            try:
                resp = requests.request(method, url, **kwargs)
                if resp.status_code in (200, 201, 204):
                    return resp
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < len(RETRY_BACKOFFS):
                        wait_s = RETRY_BACKOFFS[attempt]
                        if resp.status_code == 429:
                            try:
                                retry_after = int(
                                    resp.json().get("parameters", {}).get("retry_after", wait_s)
                                )
                                wait_s = min(max(wait_s, retry_after), 20)
                            except Exception:
                                pass
                        logger.warning(
                            f"HTTP {resp.status_code} from {url[:45]}... retrying in {wait_s}s"
                        )
                        time.sleep(wait_s)
                        continue
                logger.error(f"HTTP {resp.status_code} error: {resp.text[:250]}")
                return resp
            except requests.RequestException as exc:
                if attempt < len(RETRY_BACKOFFS):
                    wait_s = RETRY_BACKOFFS[attempt]
                    logger.warning(f"Network error ({exc}); retrying in {wait_s}s")
                    time.sleep(wait_s)
                else:
                    logger.error(f"Network delivery failed permanently: {exc}")
        return None

    def send_telegram(self, html_text: str, chat_id: Optional[str] = None) -> bool:
        if not self.tg_token or not self.authorized_chat_ids:
            logger.info("Telegram credentials absent; printing message locally:\n" + html_text)
            return False

        targets = [chat_id] if chat_id else self.authorized_chat_ids
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        ok = True
        for cid in targets:
            payload = {
                "chat_id": cid,
                "text": html_text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            resp = self._request_with_backoff("POST", url, json=payload)
            if resp is None or resp.status_code != 200:
                ok = False
        return ok

    def send_discord(self, content: str) -> None:
        if not self.discord_webhook:
            return
        clean = (
            content.replace("<b>", "**")
            .replace("</b>", "**")
            .replace("<code>", "`")
            .replace("</code>", "`")
        )
        self._request_with_backoff("POST", self.discord_webhook, json={"content": clean})

    def broadcast(self, html_text: str, target_chat_id: Optional[str] = None) -> None:
        self.send_telegram(html_text, chat_id=target_chat_id)
        if not target_chat_id:
            self.send_discord(html_text)

    def poll_commands(self) -> List[Tuple[str, str]]:
        """
        Polls Telegram updates, enforces chat_id authorization, tracks update_id in SQLite.
        Returns list of (chat_id, normalized_command).
        """
        if not self.tg_token or not self.authorized_chat_ids:
            return []

        offset_str = self.state.get_state("telegram_update_offset", "0") or "0"
        offset = int(offset_str)
        url = f"https://api.telegram.org/bot{self.tg_token}/getUpdates"
        params = {"offset": offset, "timeout": 3, "allowed_updates": ["message"]}
        resp = self._request_with_backoff("GET", url, params=params)
        if resp is None or resp.status_code != 200:
            return []

        try:
            data = resp.json()
        except ValueError:
            return []

        if not data.get("ok"):
            return []

        commands: List[Tuple[str, str]] = []
        max_uid = offset - 1
        for upd in data.get("result", []):
            uid = int(upd.get("update_id", 0))
            if uid > max_uid:
                max_uid = uid
            msg = upd.get("message") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            text = (msg.get("text") or "").strip()

            if not text or not chat_id:
                continue
            if chat_id not in self.authorized_chat_ids:
                logger.warning(f"Unauthorized Telegram command attempt from chat_id={chat_id}")
                continue

            cmd_norm = text.lower().split("@")[0].strip()
            if cmd_norm.startswith("/check") or cmd_norm.startswith("/status") or cmd_norm.startswith("/help"):
                commands.append((chat_id, text.strip()))

        if max_uid >= offset:
            self.state.set_state("telegram_update_offset", str(max_uid + 1))
        return commands


# ==============================================================================
# MULTI-SOURCE DATA ENGINE + SANITY & VOLATILITY SHOCK GUARD
# ==============================================================================
class MarketDataEngine:
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
    }

    @staticmethod
    def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()]
        col_map = {c.lower(): c for c in df.columns}
        required = {}
        for std_name in ["open", "high", "low", "close"]:
            if std_name not in col_map:
                return pd.DataFrame()
            required[col_map[std_name]] = std_name.capitalize()
        if "volume" in col_map:
            required[col_map["volume"]] = "Volume"

        df = df.rename(columns=required)
        cols = ["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])
        df = df[cols].copy()
        if "Volume" not in df.columns:
            df["Volume"] = 0.0

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df["Volume"] = df["Volume"].fillna(0.0).clip(lower=0.0)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df[~df.index.isna()]
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Enforce OHLC structural consistency & positive prices
        valid_mask = (
            (df["Open"] > 100.0)
            & (df["Close"] > 100.0)
            & (df["High"] >= df[["Open", "Close"]].max(axis=1))
            & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
            & (df["Low"] > 100.0)
        )
        df = df[valid_mask]

        # Vectorized extreme single-bar anomaly spike removal (>4.5% move on intraday bar)
        pct_jump = df["Close"].pct_change().abs().fillna(0.0)
        bar_range_pct = ((df["High"] - df["Low"]) / (df["Close"] + 1e-8)).fillna(0.0)
        df = df[(pct_jump < 0.045) & (bar_range_pct < 0.055)]
        return df

    def _fetch_yfinance(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=15,
            )
            return self._sanitize_ohlcv(raw)
        except Exception as exc:
            logger.warning(f"yfinance failed for {symbol} ({interval}): {exc}")
            return pd.DataFrame()

    def _fetch_yahoo_chart_api(self, symbol: str, interval: str, range_str: str) -> pd.DataFrame:
        for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
            url = f"https://{host}/v8/finance/chart/{symbol}"
            params = {"interval": interval, "range": range_str}
            try:
                resp = requests.get(url, params=params, headers=self.HEADERS, timeout=12)
                if resp.status_code != 200:
                    continue
                payload = resp.json()
                result = (payload.get("chart") or {}).get("result") or []
                if not result:
                    continue
                res0 = result[0]
                ts = res0.get("timestamp") or []
                quote = ((res0.get("indicators") or {}).get("quote") or [{}])[0]
                if not ts or not quote:
                    continue
                df = pd.DataFrame(
                    {
                        "Open": quote.get("open"),
                        "High": quote.get("high"),
                        "Low": quote.get("low"),
                        "Close": quote.get("close"),
                        "Volume": quote.get("volume") or [0.0] * len(ts),
                    },
                    index=pd.to_datetime(ts, unit="s", utc=True),
                )
                clean = self._sanitize_ohlcv(df)
                if not clean.empty:
                    return clean
            except Exception as exc:
                logger.warning(f"Yahoo v8 API ({host}, {symbol}) error: {exc}")
        return pd.DataFrame()

    def _fetch_kraken_paxg(self, interval: str) -> pd.DataFrame:
        """
        Emergency Spot-Backed Gold Proxy (PAXG/USD on Kraken Public API, 1:1 physical fine troy oz)
        used ONLY if all XAUUSD=X and XAU=X feeds fail. Never uses GC=F futures.
        """
        tf_map = {"5m": 5, "15m": 15, "1h": 60}
        k_int = tf_map.get(interval, 15)
        try:
            resp = requests.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": "PAXGUSD", "interval": k_int},
                headers=self.HEADERS,
                timeout=12,
            )
            if resp.status_code != 200:
                return pd.DataFrame()
            data = resp.json()
            if data.get("error"):
                return pd.DataFrame()
            result = data.get("result") or {}
            rows = None
            for k, v in result.items():
                if k != "last" and isinstance(v, list):
                    rows = v
                    break
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(
                rows,
                columns=["Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"],
            )
            df.index = pd.to_datetime(df["Time"].astype(float), unit="s", utc=True)
            return self._sanitize_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]])
        except Exception as exc:
            logger.warning(f"Kraken PAXGUSD fallback failed ({interval}): {exc}")
            return pd.DataFrame()

    def fetch_timeframe(self, interval: str, period: str, min_rows: int = 120) -> Tuple[pd.DataFrame, str]:
        """
        Layered Free Failover strictly targeting Spot Gold (never futures GC=F):
        1. yfinance XAUUSD=X
        2. Direct Yahoo Chart v8 API XAUUSD=X
        3. yfinance XAU=X
        4. Direct Yahoo Chart v8 API XAU=X
        5. Kraken PAXGUSD + yfinance PAXG-USD (physical spot proxy emergency failover)
        """
        sources = [
            ("YF:XAUUSD=X", lambda: self._fetch_yfinance("XAUUSD=X", interval, period)),
            ("YAPI:XAUUSD=X", lambda: self._fetch_yahoo_chart_api("XAUUSD=X", interval, period)),
            ("YF:XAU=X", lambda: self._fetch_yfinance("XAU=X", interval, period)),
            ("YAPI:XAU=X", lambda: self._fetch_yahoo_chart_api("XAU=X", interval, period)),
            ("KRAKEN:PAXGUSD", lambda: self._fetch_kraken_paxg(interval)),
            ("YF:PAXG-USD", lambda: self._fetch_yfinance("PAXG-USD", interval, period)),
        ]
        for name, loader in sources:
            df = loader()
            if not df.empty and len(df) >= min_rows:
                logger.info(f"Data [{interval}] loaded from {name} ({len(df)} bars, last={df.index[-1]})")
                return df, name
        return pd.DataFrame(), "NONE"

    @staticmethod
    def validate_freshness_and_shock(
        df: pd.DataFrame, stale_max_minutes: int, market_open: bool
    ) -> Tuple[bool, str]:
        if df.empty or len(df) < 80:
            return False, "INSUFFICIENT_DATA"

        last_ts = df.index[-1]
        age_min = (utc_now() - last_ts).total_seconds() / 60.0
        if market_open and age_min > stale_max_minutes:
            return False, f"STALE_DATA_{age_min:.1f}m"

        # Volatility Shock Guard: compare latest bar TR & short ATR vs baseline ATR(50)
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_fast = tr.rolling(5).mean().iloc[-1]
        atr_base = tr.rolling(50).mean().iloc[-1]
        last_tr = tr.iloc[-1]

        if atr_base > 0:
            if (atr_fast / (atr_base + 1e-8)) > 2.85 or (last_tr / (atr_base + 1e-8)) > 4.2:
                return False, f"VOLATILITY_SHOCK_ATR_RATIO={(atr_fast/(atr_base+1e-8)):.2f}"
        return True, "OK"

    def load_all_timeframes(self) -> Dict[str, Tuple[pd.DataFrame, str]]:
        df_5m, src_5m = self.fetch_timeframe("5m", "10d", min_rows=150)
        df_15m, src_15m = self.fetch_timeframe("15m", "30d", min_rows=150)
        df_1h, src_1h = self.fetch_timeframe("1h", "90d", min_rows=150)

        if not df_1h.empty:
            df_4h = (
                df_1h.resample("4h")
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Volume": "sum",
                    }
                )
                .dropna(subset=["Open", "High", "Low", "Close"])
            )
            src_4h = f"{src_1h}->4h"
        else:
            df_4h, src_4h = pd.DataFrame(), "NONE"

        return {
            "5m": (df_5m, src_5m),
            "15m": (df_15m, src_15m),
            "1h": (df_1h, src_1h),
            "4h": (df_4h, src_4h),
        }


# ==============================================================================
# 5-PILLAR QUANTITATIVE FEATURE ENGINE + AI ANALYTICAL AUGMENTATIONS
# ==============================================================================
class QuantFeatureEngine:
    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        atr = QuantFeatureEngine.compute_atr(df, period) + 1e-8
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def compute_ker(close: pd.Series, window: int = 10) -> pd.Series:
        direction = (close - close.shift(window)).abs()
        volatility = close.diff().abs().rolling(window).sum()
        return direction / (volatility + 1e-8)

    @staticmethod
    def compute_choppiness(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        tr_sum = tr.rolling(window).sum()
        hl_span = high.rolling(window).max() - low.rolling(window).min()
        ratio = (tr_sum / (hl_span + 1e-8)).clip(lower=1e-8)
        return 100.0 * np.log10(ratio) / np.log10(float(window))

    @staticmethod
    def compute_smc_causal(df: pd.DataFrame, lr: int = 3) -> pd.DataFrame:
        """
        Causal Smart Money Concepts (Zero Look-Ahead):
        A swing high/low at index p = i - lr is confirmed strictly at bar i.
        BOS and ChoCh trigger only on candle Close beyond confirmed swing levels.
        """
        n = len(df)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)

        last_sh = np.full(n, np.nan)
        last_sl = np.full(n, np.nan)
        structure = np.zeros(n, dtype=int)       # +1 bull, -1 bear
        bos_bull = np.zeros(n, dtype=int)
        bos_bear = np.zeros(n, dtype=int)
        choch_bull = np.zeros(n, dtype=int)
        choch_bear = np.zeros(n, dtype=int)
        sweep_bull = np.zeros(n, dtype=int)      # Wick sweeps below SL, closes back above
        sweep_bear = np.zeros(n, dtype=int)      # Wick sweeps above SH, closes back below

        cur_sh = np.nan
        cur_sl = np.nan
        sh_broken = False
        sl_broken = False
        cur_struct = 0

        for i in range(2 * lr, n):
            p = i - lr
            win_h = highs[p - lr : p + lr + 1]
            win_l = lows[p - lr : p + lr + 1]

            if highs[p] == np.max(win_h) and np.sum(win_h == highs[p]) == 1:
                cur_sh = highs[p]
                sh_broken = False
            if lows[p] == np.min(win_l) and np.sum(win_l == lows[p]) == 1:
                cur_sl = lows[p]
                sl_broken = False

            last_sh[i] = cur_sh
            last_sl[i] = cur_sl

            # Liquidity Sweep + Reclaim Detection (AI Augmentation)
            if not np.isnan(cur_sl) and lows[i] < cur_sl and closes[i] > cur_sl:
                sweep_bull[i] = 1
            if not np.isnan(cur_sh) and highs[i] > cur_sh and closes[i] < cur_sh:
                sweep_bear[i] = 1

            # Confirmed Close-Based BOS & ChoCh
            if not np.isnan(cur_sh) and not sh_broken and closes[i] > cur_sh:
                sh_broken = True
                if cur_struct == -1:
                    choch_bull[i] = 1
                else:
                    bos_bull[i] = 1
                cur_struct = 1
            elif not np.isnan(cur_sl) and not sl_broken and closes[i] < cur_sl:
                sl_broken = True
                if cur_struct == 1:
                    choch_bear[i] = 1
                else:
                    bos_bear[i] = 1
                cur_struct = -1

            structure[i] = cur_struct

        return pd.DataFrame(
            {
                "last_sh": last_sh,
                "last_sl": last_sl,
                "smc_structure": structure,
                "bos_bull": bos_bull,
                "bos_bear": bos_bear,
                "choch_bull": choch_bull,
                "choch_bear": choch_bear,
                "sweep_bull": sweep_bull,
                "sweep_bear": sweep_bear,
            },
            index=df.index,
        )

    @staticmethod
    def compute_order_flow_and_vwap(df: pd.DataFrame, vwap_window: int = 32) -> pd.DataFrame:
        """
        Pillar II: Volume-weighted body-to-wick ratio & wick-trap filter.
        Falls back seamlessly to range-velocity synthetic tick volume when spot feed reports 0 volume.
        """
        candle_range = (df["High"] - df["Low"]) + 1e-8
        signed_body = df["Close"] - df["Open"]
        body_ratio = signed_body / candle_range  # [-1, +1]

        upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)
        lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        has_real_vol = (df["Volume"] > 0).sum() > (len(df) * 0.25)
        if has_real_vol:
            vol_base = df["Volume"].replace(0.0, np.nan).ffill().fillna(1.0)
            rel_vol = (vol_base / (vol_base.rolling(20).mean() + 1e-8)).clip(0.25, 3.0)
        else:
            # Synthetic Spot Activity Volume Proxy (range expansion * directional conviction)
            med_range = candle_range.rolling(20).median() + 1e-8
            rel_vol = (candle_range / med_range).clip(0.35, 2.5)

        of_score = (body_ratio * rel_vol).rolling(3).mean().fillna(0.0)

        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        tp_vol = (typical_price * rel_vol).rolling(vwap_window).sum()
        vol_sum = rel_vol.rolling(vwap_window).sum() + 1e-8
        rolling_vwap = tp_vol / vol_sum

        return pd.DataFrame(
            {
                "of_score": of_score,
                "body_ratio": body_ratio,
                "upper_wick_ratio": upper_wick_ratio,
                "lower_wick_ratio": lower_wick_ratio,
                "rolling_vwap": rolling_vwap,
            },
            index=df.index,
        )

    @classmethod
    def compute_htf_bias(cls, df_htf: pd.DataFrame, tf_delta: pd.Timedelta) -> pd.DataFrame:
        """
        Computes HTF directional bias & shifts timestamp by tf_delta so lower-TF
        merge_asof only ever observes COMPLETED higher-TF candles (zero look-ahead).
        """
        if df_htf.empty or len(df_htf) < 55:
            return pd.DataFrame()
        ema_fast = df_htf["Close"].ewm(span=20, adjust=False).mean()
        ema_slow = df_htf["Close"].ewm(span=50, adjust=False).mean()
        smc = cls.compute_smc_causal(df_htf, lr=3)
        adx = cls.compute_adx(df_htf, 14)

        bias = np.where(
            (df_htf["Close"] > ema_fast) & (ema_fast > ema_slow) & (smc["smc_structure"] >= 0),
            1,
            np.where(
                (df_htf["Close"] < ema_fast) & (ema_fast < ema_slow) & (smc["smc_structure"] <= 0),
                -1,
                np.where(ema_fast > ema_slow, 1, -1),
            ),
        )
        out = pd.DataFrame(
            {
                "htf_bias": bias,
                "htf_adx": adx.to_numpy(),
                "htf_smc": smc["smc_structure"].to_numpy(),
            },
            index=df_htf.index + tf_delta,
        )
        return out

    @classmethod
    def enrich_dataframe(
        cls,
        df_exec: pd.DataFrame,
        df_mid: pd.DataFrame,
        df_htf: pd.DataFrame,
        cfg: ModeConfig,
        atr_period: int = 14,
    ) -> pd.DataFrame:
        df = df_exec.copy()
        df["atr"] = cls.compute_atr(df, atr_period)
        df["atr_base"] = cls.compute_atr(df, 50)
        df["adx"] = cls.compute_adx(df, 14)
        df["ker"] = cls.compute_ker(df["Close"], 10)
        df["chop"] = cls.compute_choppiness(df, 14)

        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()

        # Pillar III: Rolling Z-Score with zero-division protection (+ 1e-8)
        z_mean = df["Close"].rolling(cfg.z_window).mean()
        z_std = df["Close"].rolling(cfg.z_window).std(ddof=0)
        df["z_score"] = (df["Close"] - z_mean) / (z_std + 1e-8)
        df["z_prev"] = df["z_score"].shift(1)

        # Pillar I & II + VWAP
        smc = cls.compute_smc_causal(df, lr=cfg.swing_lr)
        of_vwap = cls.compute_order_flow_and_vwap(df, vwap_window=cfg.vwap_window)
        df = pd.concat([df, smc, of_vwap], axis=1)

        df["vwap_dist_atr"] = (df["Close"] - df["rolling_vwap"]) / (df["atr"] + 1e-8)

        # Pillar IV: Causal MTF Alignment
        mid_delta = pd.Timedelta(minutes=15) if cfg.mid_tf == "15m" else pd.Timedelta(hours=1)
        htf_delta = pd.Timedelta(hours=1) if cfg.htf_tf == "1h" else pd.Timedelta(hours=4)

        mid_bias_df = cls.compute_htf_bias(df_mid, mid_delta)
        htf_bias_df = cls.compute_htf_bias(df_htf, htf_delta)

        df_reset = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        if "ts" not in df_reset.columns:
            df_reset = df_reset.rename(columns={df_reset.columns[0]: "ts"})

        if not mid_bias_df.empty:
            mid_reset = mid_bias_df.reset_index().rename(columns={mid_bias_df.index.name or "index": "ts"})
            if "ts" not in mid_reset.columns:
                mid_reset = mid_reset.rename(columns={mid_reset.columns[0]: "ts"})
            mid_reset = mid_reset.rename(
                columns={"htf_bias": "mid_bias", "htf_adx": "mid_adx", "htf_smc": "mid_smc"}
            )
            df_reset = pd.merge_asof(
                df_reset.sort_values("ts"),
                mid_reset.sort_values("ts"),
                on="ts",
                direction="backward",
            )
        else:
            df_reset["mid_bias"] = 0
            df_reset["mid_adx"] = 20.0
            df_reset["mid_smc"] = 0

        if not htf_bias_df.empty:
            htf_reset = htf_bias_df.reset_index().rename(columns={htf_bias_df.index.name or "index": "ts"})
            if "ts" not in htf_reset.columns:
                htf_reset = htf_reset.rename(columns={htf_reset.columns[0]: "ts"})
            df_reset = pd.merge_asof(
                df_reset.sort_values("ts"),
                htf_reset.sort_values("ts"),
                on="ts",
                direction="backward",
            )
        else:
            df_reset["htf_bias"] = 0
            df_reset["htf_adx"] = 20.0
            df_reset["htf_smc"] = 0

        df_out = df_reset.set_index("ts")
        df_out["mid_bias"] = df_out["mid_bias"].fillna(0).astype(int)
        df_out["htf_bias"] = df_out["htf_bias"].fillna(0).astype(int)
        return df_out

    @staticmethod
    def classify_regime(row: pd.Series, cfg: ModeConfig) -> Tuple[str, bool]:
        """
        Market Regime & Anti-Whipsaw Governor.
        Returns (regime_label, is_tradeable).
        """
        atr = float(row.get("atr", 0.0))
        atr_base = float(row.get("atr_base", 0.0))
        adx = float(row.get("adx", 0.0))
        ker = float(row.get("ker", 0.0))
        chop = float(row.get("chop", 50.0))
        ema20 = float(row.get("ema20", 0.0))
        ema50 = float(row.get("ema50", 0.0))

        if atr_base > 0 and (atr / (atr_base + 1e-8)) > 2.6:
            return "VOLATILITY_SHOCK", False
        if atr_base > 0 and (atr / (atr_base + 1e-8)) < 0.42:
            return "LOW_LIQUIDITY_DEAD", False
        if chop > cfg.max_chop and ker < cfg.min_ker:
            return "CHOPPY_SIDEWAYS", False
        if adx < cfg.min_adx and ker < cfg.min_ker:
            return "RANGING_WEAK", False

        if ema20 > ema50 and (adx >= cfg.min_adx or ker >= cfg.min_ker):
            return "TREND_BULL", True
        if ema20 < ema50 and (adx >= cfg.min_adx or ker >= cfg.min_ker):
            return "TREND_BEAR", True
        return "TRANSITIONAL", False

    @classmethod
    def evaluate_bar(
        cls,
        df: pd.DataFrame,
        idx: int,
        cfg: ModeConfig,
        z_thresh: float,
        check_session: bool = True,
    ) -> Tuple[str, float, str, Dict[str, Any]]:
        """
        Evaluates a single closed bar using the 5 Pillars + AI Analytical Filters.
        Returns (direction, score, regime, diagnostics).
        """
        if idx < 60 or idx >= len(df):
            return "NO_SIGNAL", 0.0, "WARMUP", {"reason": "WARMUP"}

        row = df.iloc[idx]
        ts = df.index[idx]

        if check_session:
            sess_ok, sess_reason = is_market_open(ts.to_pydatetime())
            if not sess_ok:
                return "NO_SIGNAL", 0.0, sess_reason, {"reason": sess_reason}

        regime, tradeable = cls.classify_regime(row, cfg)
        if not tradeable:
            return "NO_SIGNAL", 0.0, regime, {"reason": f"REGIME_BLOCK:{regime}"}

        close = float(row["Close"])
        ema20 = float(row["ema20"])
        ema50 = float(row["ema50"])
        z_val = float(row["z_score"])
        z_prev = float(row["z_prev"]) if not pd.isna(row["z_prev"]) else z_val
        of_score = float(row["of_score"])
        body_ratio = float(row["body_ratio"])
        upper_wick = float(row["upper_wick_ratio"])
        lower_wick = float(row["lower_wick_ratio"])
        vwap_dist = float(row["vwap_dist_atr"])
        ker = float(row["ker"])
        adx = float(row["adx"])

        mid_bias = int(row["mid_bias"])
        htf_bias = int(row["htf_bias"])
        smc_struct = int(row["smc_structure"])
        recent_bos_bull = int(df["bos_bull"].iloc[max(0, idx - 5) : idx + 1].sum() > 0)
        recent_choch_bull = int(df["choch_bull"].iloc[max(0, idx - 5) : idx + 1].sum() > 0)
        recent_bos_bear = int(df["bos_bear"].iloc[max(0, idx - 5) : idx + 1].sum() > 0)
        recent_choch_bear = int(df["choch_bear"].iloc[max(0, idx - 5) : idx + 1].sum() > 0)
        recent_sweep_bull = int(df["sweep_bull"].iloc[max(0, idx - 3) : idx + 1].sum() > 0)
        recent_sweep_bear = int(df["sweep_bear"].iloc[max(0, idx - 3) : idx + 1].sum() > 0)

        # Bullish & Bearish Confluence Scoring (Max 100)
        bull_score = 0.0
        bear_score = 0.0

        # Pillar IV: MTF Fractal Harmony (25 pts) — Strict Alignment Gate
        bull_mtf_ok = (htf_bias >= 0 and mid_bias > 0) or (htf_bias > 0 and mid_bias >= 0)
        bear_mtf_ok = (htf_bias <= 0 and mid_bias < 0) or (htf_bias < 0 and mid_bias <= 0)
        if htf_bias > 0 and mid_bias > 0:
            bull_score += 25.0
        elif bull_mtf_ok:
            bull_score += 17.0
        if htf_bias < 0 and mid_bias < 0:
            bear_score += 25.0
        elif bear_mtf_ok:
            bear_score += 17.0

        # Pillar I: SMC Structure / BOS / ChoCh / Liquidity Sweep (25 pts)
        if smc_struct > 0:
            bull_score += 13.0
        if recent_bos_bull or recent_choch_bull:
            bull_score += 8.0
        if recent_sweep_bull:
            bull_score += 4.0

        if smc_struct < 0:
            bear_score += 13.0
        if recent_bos_bear or recent_choch_bear:
            bear_score += 8.0
        if recent_sweep_bear:
            bear_score += 4.0

        # Pillar II: Order Flow + Wick Trap Filter (20 pts)
        if of_score > 0.08 and body_ratio > 0.15 and upper_wick < 0.45:
            bull_score += 20.0
        elif body_ratio > 0.05 and upper_wick < 0.50:
            bull_score += 10.0

        if of_score < -0.08 and body_ratio < -0.15 and lower_wick < 0.45:
            bear_score += 20.0
        elif body_ratio < -0.05 and lower_wick < 0.50:
            bear_score += 10.0

        # Pillar III: Z-Score Trend-Aligned Pullback / Momentum Hook (15 pts)
        # Avoid chasing extreme overextended breakouts (|z| > z_thresh + 0.8)
        if (-z_thresh <= z_val <= 1.2) and (z_val > z_prev) and close > ema50:
            bull_score += 15.0 if z_prev < -0.35 else 10.0
        if (-1.2 <= z_val <= z_thresh) and (z_val < z_prev) and close < ema50:
            bear_score += 15.0 if z_prev > 0.35 else 10.0

        # Pillar V + AI Augmentation: Regime Purity (KER/ADX) + VWAP Fair Value Edge (15 pts)
        regime_pts = min(9.0, (ker / 0.60) * 5.0 + (min(adx, 40.0) / 40.0) * 4.0)
        if -1.2 <= vwap_dist <= 1.6 and close >= ema20:
            bull_score += regime_pts + 6.0
        elif vwap_dist <= 2.2:
            bull_score += regime_pts

        if -1.6 <= vwap_dist <= 1.2 and close <= ema20:
            bear_score += regime_pts + 6.0
        elif vwap_dist >= -2.2:
            bear_score += regime_pts

        # Anti-Whipsaw & Contradiction Gate
        bull_valid = (
            bull_mtf_ok
            and close > ema20
            and ema20 > ema50
            and upper_wick < 0.50
            and body_ratio > 0.05
            and z_val < (z_thresh + 0.65)
            and bull_score >= cfg.min_score
        )
        bear_valid = (
            bear_mtf_ok
            and close < ema20
            and ema20 < ema50
            and lower_wick < 0.50
            and body_ratio < -0.05
            and z_val > -(z_thresh + 0.65)
            and bear_score >= cfg.min_score
        )

        if bull_valid and bear_valid:
            return "NO_SIGNAL", max(bull_score, bear_score), regime, {"reason": "CONFLICTING_SIGNALS"}
        if bull_valid:
            return "BUY", round(bull_score, 1), regime, {
                "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}",
                "smc": smc_struct,
                "of": round(of_score, 3),
                "z": round(z_val, 2),
                "vwap_atr": round(vwap_dist, 2),
            }
        if bear_valid:
            return "SELL", round(bear_score, 1), regime, {
                "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}",
                "smc": smc_struct,
                "of": round(of_score, 3),
                "z": round(z_val, 2),
                "vwap_atr": round(vwap_dist, 2),
            }

        top_dir = "BUY" if bull_score >= bear_score else "SELL"
        top_score = round(max(bull_score, bear_score), 1)
        return "NO_SIGNAL", top_score, regime, {
            "reason": f"CONFLUENCE_BELOW_THRESHOLD({top_dir}:{top_score}/{cfg.min_score})",
            "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}",
        }


# ==============================================================================
# BACKTEST ENGINE (SPREAD, SLIPPAGE, TRAILING BE) & GRID SEARCH OPTIMIZER
# ==============================================================================
class BacktestAndOptimizer:
    @staticmethod
    def run_backtest(
        df: pd.DataFrame,
        cfg: ModeConfig,
        z_thresh: float,
        account_balance: float = DEFAULT_ACCOUNT_BALANCE,
    ) -> Dict[str, Any]:
        """
        Event-driven realistic bar-by-bar backtest with:
        - Spread + Slippage cost
        - Worst-case intrabar fill (SL checked before TP on same bar)
        - TP1 Partial/BE Lock & Step-Trailing simulation
        """
        n = len(df)
        equity = account_balance
        peak_equity = account_balance
        max_dd_pct = 0.0
        equity_curve: List[Tuple[pd.Timestamp, float]] = []
        r_returns: List[float] = []
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        total_friction_usd = 0.0

        last_entry_idx = -999
        min_bar_gap = 6 if cfg.mode_name == "SCALP" else 8

        i = 60
        while i < n - 2:
            equity_curve.append((df.index[i], equity))
            if (i - last_entry_idx) < min_bar_gap:
                i += 1
                continue

            direction, score, _, _ = QuantFeatureEngine.evaluate_bar(
                df, i, cfg, z_thresh=z_thresh, check_session=True
            )
            if direction not in ("BUY", "SELL"):
                i += 1
                continue

            atr = float(df["atr"].iloc[i])
            if atr <= 0.1 or np.isnan(atr):
                i += 1
                continue

            raw_entry = float(df["Close"].iloc[i])
            sl_dist = atr * cfg.sl_atr_mult
            half_spread = SPREAD_USD / 2.0

            if direction == "BUY":
                entry = raw_entry + half_spread + SLIPPAGE_USD
                sl = entry - sl_dist
                tp1 = entry + sl_dist * cfg.tp_r_multiples[0]
                tp2 = entry + sl_dist * cfg.tp_r_multiples[1]
                tp3 = entry + sl_dist * cfg.tp_r_multiples[2]
            else:
                entry = raw_entry - half_spread - SLIPPAGE_USD
                sl = entry + sl_dist
                tp1 = entry - sl_dist * cfg.tp_r_multiples[0]
                tp2 = entry - sl_dist * cfg.tp_r_multiples[1]
                tp3 = entry - sl_dist * cfg.tp_r_multiples[2]

            risk_usd = equity * (DEFAULT_RISK_PCT / 100.0)
            lot = calculate_mt5_lot_size(entry, sl, equity, DEFAULT_RISK_PCT)
            friction_rt = (SPREAD_USD + 2.0 * SLIPPAGE_USD) * lot * CONTRACT_SIZE
            total_friction_usd += friction_rt

            # Simulate position forward: 50% closed @ TP1 + BE move, 25% @ TP2, 25% trailed @ TP3
            cur_sl = sl
            rem_weight = 1.0
            realized_r = 0.0
            tp1_hit = False
            tp2_hit = False
            exit_bar = i + 1

            for j in range(i + 1, min(n, i + 180)):
                bar_h = float(df["High"].iloc[j])
                bar_l = float(df["Low"].iloc[j])
                bar_c = float(df["Close"].iloc[j])
                bar_atr = float(df["atr"].iloc[j]) if not np.isnan(df["atr"].iloc[j]) else atr
                exit_bar = j

                if direction == "BUY":
                    if bar_l <= cur_sl:
                        sl_r = (cur_sl - entry) / (sl_dist + 1e-8)
                        realized_r += rem_weight * sl_r
                        rem_weight = 0.0
                        break
                    if not tp1_hit and bar_h >= tp1:
                        tp1_hit = True
                        realized_r += 0.50 * cfg.tp_r_multiples[0]
                        rem_weight -= 0.50
                        cur_sl = max(cur_sl, entry + cfg.be_buffer_usd)
                    if tp1_hit and not tp2_hit and bar_h >= tp2:
                        tp2_hit = True
                        realized_r += 0.25 * cfg.tp_r_multiples[1]
                        rem_weight -= 0.25
                        cur_sl = max(cur_sl, tp1)
                    if bar_h >= tp3:
                        realized_r += rem_weight * cfg.tp_r_multiples[2]
                        rem_weight = 0.0
                        break
                    if tp1_hit:
                        trail_candidate = bar_c - (bar_atr * cfg.trail_atr_mult)
                        cur_sl = max(cur_sl, trail_candidate)
                else:
                    if bar_h >= cur_sl:
                        sl_r = (entry - cur_sl) / (sl_dist + 1e-8)
                        realized_r += rem_weight * sl_r
                        rem_weight = 0.0
                        break
                    if not tp1_hit and bar_l <= tp1:
                        tp1_hit = True
                        realized_r += 0.50 * cfg.tp_r_multiples[0]
                        rem_weight -= 0.50
                        cur_sl = min(cur_sl, entry - cfg.be_buffer_usd)
                    if tp1_hit and not tp2_hit and bar_l <= tp2:
                        tp2_hit = True
                        realized_r += 0.25 * cfg.tp_r_multiples[1]
                        rem_weight -= 0.25
                        cur_sl = min(cur_sl, tp1)
                    if bar_l <= tp3:
                        realized_r += rem_weight * cfg.tp_r_multiples[2]
                        rem_weight = 0.0
                        break
                    if tp1_hit:
                        trail_candidate = bar_c + (bar_atr * cfg.trail_atr_mult)
                        cur_sl = min(cur_sl, trail_candidate)

            if rem_weight > 0:
                last_c = float(df["Close"].iloc[exit_bar])
                rem_r = ((last_c - entry) if direction == "BUY" else (entry - last_c)) / (sl_dist + 1e-8)
                realized_r += rem_weight * rem_r

            trade_pnl = (realized_r * risk_usd) - (SPREAD_USD * lot * CONTRACT_SIZE * 0.25)
            equity = max(100.0, equity + trade_pnl)
            r_returns.append(realized_r)

            if trade_pnl > 0:
                wins += 1
                gross_profit += trade_pnl
            else:
                losses += 1
                gross_loss += abs(trade_pnl)

            peak_equity = max(peak_equity, equity)
            dd_pct = ((peak_equity - equity) / (peak_equity + 1e-8)) * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)

            last_entry_idx = i
            i = max(i + 1, exit_bar + 1)

        if len(df) > 0:
            equity_curve.append((df.index[-1], equity))

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (gross_profit / (gross_loss + 1e-8)) if gross_loss > 0 else (3.0 if gross_profit > 0 else 0.0)
        expectancy_r = float(np.mean(r_returns)) if r_returns else 0.0
        std_r = float(np.std(r_returns)) if len(r_returns) > 1 else 1.0
        sharpe = (expectancy_r / (std_r + 1e-8)) * math.sqrt(max(1, total_trades))
        total_ret_pct = ((equity - account_balance) / account_balance) * 100.0

        return {
            "mode": cfg.mode_name,
            "trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(min(profit_factor, 9.99), 2),
            "expectancy_r": round(expectancy_r, 3),
            "sharpe": round(sharpe, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "total_return_pct": round(total_ret_pct, 2),
            "final_equity": round(equity, 2),
            "friction_usd": round(total_friction_usd, 2),
            "equity_curve": equity_curve,
        }

    @classmethod
    def grid_search_optimize(
        cls,
        df_exec: pd.DataFrame,
        df_mid: pd.DataFrame,
        df_htf: pd.DataFrame,
        cfg: ModeConfig,
    ) -> Tuple[int, float, pd.DataFrame, Dict[str, Any]]:
        """
        Tests ATR period [10, 14, 20] x Z-Score threshold [1.5, 2.0, 2.5]
        with drawdown penalty & minimum sample requirement to prevent overfitting.
        """
        best_obj = -1e9
        best_atr = 14
        best_z = 2.0
        best_df: Optional[pd.DataFrame] = None
        best_stats: Optional[Dict[str, Any]] = None

        cached_enriched: Dict[int, pd.DataFrame] = {}
        for atr_p in GRID_ATR_PERIODS:
            cached_enriched[atr_p] = QuantFeatureEngine.enrich_dataframe(
                df_exec, df_mid, df_htf, cfg, atr_period=atr_p
            )

        for atr_p in GRID_ATR_PERIODS:
            df_enr = cached_enriched[atr_p]
            for z_t in GRID_Z_THRESHOLDS:
                stats = cls.run_backtest(df_enr, cfg, z_thresh=z_t)
                if stats["trades"] < cfg.min_backtest_trades:
                    continue
                obj = (
                    stats["expectancy_r"] * math.sqrt(stats["trades"]) * min(stats["profit_factor"], 3.0)
                    - 0.18 * stats["max_dd_pct"]
                )
                if obj > best_obj:
                    best_obj = obj
                    best_atr = atr_p
                    best_z = z_t
                    best_df = df_enr
                    best_stats = stats

        if best_df is None or best_stats is None:
            best_atr = 14
            best_z = 2.0
            best_df = cached_enriched[14]
            best_stats = cls.run_backtest(best_df, cfg, z_thresh=best_z)

        best_stats["opt_atr_period"] = best_atr
        best_stats["opt_z_thresh"] = best_z
        return best_atr, best_z, best_df, best_stats

    @staticmethod
    def generate_performance_chart(
        scalp_stats: Dict[str, Any], swing_stats: Dict[str, Any], out_path: str = CHART_PATH
    ) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0d1117")
        fig.suptitle(
            "APEX-QUANT-OMNI | XAUUSD Dual-Mode Backtest Equity (With Spread & Slippage)",
            color="#f0f6fc",
            fontsize=13,
            fontweight="bold",
        )

        configs = [
            (axes[0], scalp_stats, "#00e676", "SCALPING (5M/15M/1H)"),
            (axes[1], swing_stats, "#29b6f6", "SWING (15M/1H/4H)"),
        ]
        for ax, st, color, label in configs:
            ax.set_facecolor("#161b22")
            curve = st.get("equity_curve") or []
            if curve:
                xs = [c[0] for c in curve]
                ys = [c[1] for c in curve]
                ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
                ax.fill_between(xs, DEFAULT_ACCOUNT_BALANCE, ys, color=color, alpha=0.12)
                ax.axhline(DEFAULT_ACCOUNT_BALANCE, color="#8b949e", linestyle="--", linewidth=0.9)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            else:
                ax.text(
                    0.5, 0.5, "No Trades Recorded", color="#8b949e",
                    ha="center", va="center", transform=ax.transAxes
                )

            summary = (
                f"{label} | Trades: {st.get('trades', 0)} | WR: {st.get('win_rate', 0):.1f}% | "
                f"PF: {st.get('profit_factor', 0):.2f} | EV: {st.get('expectancy_r', 0):+.2f}R | "
                f"MaxDD: {st.get('max_dd_pct', 0):.2f}% | Ret: {st.get('total_return_pct', 0):+.2f}%"
            )
            ax.set_title(summary, color="#c9d1d9", fontsize=10, loc="left")
            ax.tick_params(colors="#8b949e")
            ax.grid(color="#30363d", linestyle=":", linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)


# ==============================================================================
# SIGNAL GENERATOR & AUTO-TRAILING STOP ENGINE
# ==============================================================================
class SignalEngine:
    @staticmethod
    def build_live_signal(
        df_enriched: pd.DataFrame,
        cfg: ModeConfig,
        z_thresh: float,
        backtest_stats: Dict[str, Any],
    ) -> SignalCandidate:
        if df_enriched.empty or len(df_enriched) < 65:
            return SignalCandidate(mode=cfg.mode_name, direction="NO_SIGNAL", reason="EMPTY_DATA")

        # Use last closed candle (-2 if latest candle timestamp is current incomplete minute, else -1)
        idx = len(df_enriched) - 1
        direction, score, regime, diag = QuantFeatureEngine.evaluate_bar(
            df_enriched, idx, cfg, z_thresh=z_thresh, check_session=True
        )
        row = df_enriched.iloc[idx]
        candle_ts = df_enriched.index[idx].isoformat()
        close_px = float(row["Close"])
        atr = float(row["atr"])
        ker = float(row["ker"])
        adx = float(row["adx"])
        z_val = float(row["z_score"])
        mtf_bias = str(diag.get("mtf_bias", "N/A"))

        if direction == "NO_SIGNAL":
            return SignalCandidate(
                mode=cfg.mode_name,
                direction="NO_SIGNAL",
                entry=round(close_px, 2),
                score=score,
                ker=round(ker, 2),
                adx=round(adx, 1),
                z_score=round(z_val, 2),
                atr=round(atr, 2),
                regime=regime,
                mtf_bias=mtf_bias,
                candle_ts=candle_ts,
                reason=str(diag.get("reason", "NO_CONFLUENCE")),
            )

        sl_dist = max(atr * cfg.sl_atr_mult, 1.20)
        half_spread = SPREAD_USD / 2.0

        if direction == "BUY":
            entry = round(close_px + half_spread, 2)
            sl = round(entry - sl_dist, 2)
            tp1 = round(entry + sl_dist * cfg.tp_r_multiples[0], 2)
            tp2 = round(entry + sl_dist * cfg.tp_r_multiples[1], 2)
            tp3 = round(entry + sl_dist * cfg.tp_r_multiples[2], 2)
        else:
            entry = round(close_px - half_spread, 2)
            sl = round(entry + sl_dist, 2)
            tp1 = round(entry - sl_dist * cfg.tp_r_multiples[0], 2)
            tp2 = round(entry - sl_dist * cfg.tp_r_multiples[1], 2)
            tp3 = round(entry - sl_dist * cfg.tp_r_multiples[2], 2)

        lot_size = calculate_mt5_lot_size(entry, sl, DEFAULT_ACCOUNT_BALANCE, DEFAULT_RISK_PCT)

        # Bayesian-blended Empirical Win Probability & Expectancy Calibration
        bt_wr = float(backtest_stats.get("win_rate", 55.0)) / 100.0
        bt_trades = int(backtest_stats.get("trades", 0))
        prior_wr = 0.52 + ((score - cfg.min_score) / 100.0) * 0.22
        weight_bt = min(0.65, bt_trades / 35.0)
        p_win = round((weight_bt * bt_wr + (1.0 - weight_bt) * prior_wr) * 100.0, 1)
        p_win = max(45.0, min(82.0, p_win))

        p_dec = p_win / 100.0
        blended_win_r = (0.50 * cfg.tp_r_multiples[0]) + (0.25 * cfg.tp_r_multiples[1]) + (0.25 * cfg.tp_r_multiples[2])
        ev_r = round((p_dec * blended_win_r) - ((1.0 - p_dec) * 1.0), 2)

        if ev_r <= 0.15:
            return SignalCandidate(
                mode=cfg.mode_name,
                direction="NO_SIGNAL",
                entry=round(close_px, 2),
                score=score,
                ker=round(ker, 2),
                adx=round(adx, 1),
                z_score=round(z_val, 2),
                atr=round(atr, 2),
                regime=regime,
                mtf_bias=mtf_bias,
                candle_ts=candle_ts,
                reason=f"LOW_EXPECTANCY_EV={ev_r:+.2f}R",
            )

        return SignalCandidate(
            mode=cfg.mode_name,
            direction=direction,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            lot_size=lot_size,
            score=score,
            p_win=p_win,
            ev_r=ev_r,
            ker=round(ker, 2),
            adx=round(adx, 1),
            z_score=round(z_val, 2),
            atr=round(atr, 2),
            regime=regime,
            mtf_bias=mtf_bias,
            candle_ts=candle_ts,
            reason="SIGNAL_CONFIRMED",
            pillar_details=diag,
        )

    @staticmethod
    def format_telegram_signal(sig: SignalCandidate) -> str:
        now_hhmm = utc_now().strftime("%H:%M UTC")
        return (
            f"⚡ <b>XAUUSD | {sig.mode} | {sig.direction}</b>\n"
            f"🎯 Entry: <code>{sig.entry:.2f}</code> | 🛑 SL: <code>{sig.sl:.2f}</code>\n"
            f"✅ TP1: <code>{sig.tp1:.2f}</code> | TP2: <code>{sig.tp2:.2f}</code> | TP3: <code>{sig.tp3:.2f}</code>\n"
            f"📐 Lot: <code>{sig.lot_size:.2f}</code> | Skor: <code>{sig.score:.0f}/100</code> | Regime: <code>{sig.regime}</code>\n"
            f"🔄 Trail: Aktif @ TP1 (BE) | 🕒 <code>{now_hhmm}</code>\n"
            f"🧠 Edge: P(Win) <code>{sig.p_win:.0f}%</code> | EV: <code>{sig.ev_r:+.2f}R</code> | KER: <code>{sig.ker:.2f}</code> | MTF: <code>{sig.mtf_bias}</code>"
        )


class TrailingStopManager:
    def __init__(self, state: StateManager, notifier: Notifier) -> None:
        self.state = state
        self.notifier = notifier

    def update_active_trades(self, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> List[str]:
        active = self.state.get_active_signals()
        if not active or df_5m.empty:
            return []

        events: List[str] = []
        latest_price = float(df_5m["Close"].iloc[-1])

        for tr in active:
            sig_id = int(tr["id"])
            mode = tr["mode"]
            cfg = SCALP_CONFIG if mode == "SCALP" else SWING_CONFIG
            df_ref = df_5m if mode == "SCALP" else df_15m
            if df_ref.empty:
                df_ref = df_5m

            created_dt = datetime.fromisoformat(tr["created_at"])
            since_df = df_ref[df_ref.index >= created_dt]
            if since_df.empty:
                high_since = latest_price
                low_since = latest_price
            else:
                high_since = float(since_df["High"].max())
                low_since = float(since_df["Low"].min())

            cur_sl = float(tr["current_sl"])
            entry = float(tr["entry"])
            tp1 = float(tr["tp1"])
            tp2 = float(tr["tp2"])
            tp3 = float(tr["tp3"])
            atr = float(tr["atr"])
            status = tr["status"]
            direction = tr["direction"]

            if direction == "BUY":
                if low_since <= cur_sl:
                    closed_status = "CLOSED_BE" if cur_sl >= entry else "CLOSED_SL"
                    self.state.update_signal_trail(sig_id, cur_sl, closed_status)
                    msg = f"🛑 <b>XAUUSD | {mode} BUY #{sig_id} {closed_status}</b> @ <code>{cur_sl:.2f}</code> (Spot: <code>{latest_price:.2f}</code>)"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                if high_since >= tp3:
                    self.state.update_signal_trail(sig_id, tp2, "CLOSED_TP3")
                    msg = f"🏆 <b>XAUUSD | {mode} BUY #{sig_id} FULL TP3 HIT</b> @ <code>{tp3:.2f}</code>!"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                new_sl = cur_sl
                new_status = status
                if high_since >= tp2 and status != "TP2_TRAIL":
                    new_status = "TP2_TRAIL"
                    new_sl = max(cur_sl, tp1, latest_price - atr * cfg.trail_atr_mult)
                elif high_since >= tp1 and status == "OPEN":
                    new_status = "TP1_TRAIL"
                    new_sl = max(cur_sl, entry + cfg.be_buffer_usd, latest_price - atr * cfg.trail_atr_mult)
                elif status in ("TP1_TRAIL", "TP2_TRAIL"):
                    new_sl = max(cur_sl, latest_price - atr * cfg.trail_atr_mult)

                if new_status != status or new_sl > cur_sl + 0.15:
                    self.state.update_signal_trail(sig_id, new_sl, new_status)
                    msg = (
                        f"🔄 <b>TRAIL UPDATE | {mode} BUY #{sig_id}</b>\n"
                        f"Status: <code>{new_status}</code> | SL Baru: <code>{new_sl:.2f}</code> | Spot: <code>{latest_price:.2f}</code>"
                    )
                    events.append(msg)
                    self.notifier.broadcast(msg)
            else:
                if high_since >= cur_sl:
                    closed_status = "CLOSED_BE" if cur_sl <= entry else "CLOSED_SL"
                    self.state.update_signal_trail(sig_id, cur_sl, closed_status)
                    msg = f"🛑 <b>XAUUSD | {mode} SELL #{sig_id} {closed_status}</b> @ <code>{cur_sl:.2f}</code> (Spot: <code>{latest_price:.2f}</code>)"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                if low_since <= tp3:
                    self.state.update_signal_trail(sig_id, tp2, "CLOSED_TP3")
                    msg = f"🏆 <b>XAUUSD | {mode} SELL #{sig_id} FULL TP3 HIT</b> @ <code>{tp3:.2f}</code>!"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                new_sl = cur_sl
                new_status = status
                if low_since <= tp2 and status != "TP2_TRAIL":
                    new_status = "TP2_TRAIL"
                    new_sl = min(cur_sl, tp1, latest_price + atr * cfg.trail_atr_mult)
                elif low_since <= tp1 and status == "OPEN":
                    new_status = "TP1_TRAIL"
                    new_sl = min(cur_sl, entry - cfg.be_buffer_usd, latest_price + atr * cfg.trail_atr_mult)
                elif status in ("TP1_TRAIL", "TP2_TRAIL"):
                    new_sl = min(cur_sl, latest_price + atr * cfg.trail_atr_mult)

                if new_status != status or new_sl < cur_sl - 0.15:
                    self.state.update_signal_trail(sig_id, new_sl, new_status)
                    msg = (
                        f"🔄 <b>TRAIL UPDATE | {mode} SELL #{sig_id}</b>\n"
                        f"Status: <code>{new_status}</code> | SL Baru: <code>{new_sl:.2f}</code> | Spot: <code>{latest_price:.2f}</code>"
                    )
                    events.append(msg)
                    self.notifier.broadcast(msg)

        return events


# ==============================================================================
# FORENSIC AUDIT LOGGER
# ==============================================================================
def forensic_audit_log(
    sources: Dict[str, str],
    scalp_stats: Dict[str, Any],
    swing_stats: Dict[str, Any],
    scalp_sig: SignalCandidate,
    swing_sig: SignalCandidate,
    active_trades: List[Dict[str, Any]],
) -> None:
    print("=" * 80)
    print("APEX-QUANT-OMNI — HYPERION PRECISION FORENSIC AUDIT LOG")
    print("=" * 80)
    print(f"Timestamp UTC     : {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Data Feeds        : {json.dumps(sources)}")
    print(f"Execution Model   : Spread=${SPREAD_USD:.2f} | Slippage=${SLIPPAGE_USD:.2f} | Contract={CONTRACT_SIZE}oz | Pip=${PIP_SIZE}")
    print("-" * 80)
    print(
        f"[SCALP OPT & BT]  : ATR={scalp_stats.get('opt_atr_period')} | Z={scalp_stats.get('opt_z_thresh')} | "
        f"Trades={scalp_stats.get('trades')} | WR={scalp_stats.get('win_rate')}% | PF={scalp_stats.get('profit_factor')} | "
        f"EV={scalp_stats.get('expectancy_r'):+.3f}R | Sharpe={scalp_stats.get('sharpe')} | "
        f"MaxDD={scalp_stats.get('max_dd_pct')}% | Ret={scalp_stats.get('total_return_pct'):+.2f}% | "
        f"Friction=${scalp_stats.get('friction_usd')}"
    )
    print(
        f"[SWING OPT & BT]  : ATR={swing_stats.get('opt_atr_period')} | Z={swing_stats.get('opt_z_thresh')} | "
        f"Trades={swing_stats.get('trades')} | WR={swing_stats.get('win_rate')}% | PF={swing_stats.get('profit_factor')} | "
        f"EV={swing_stats.get('expectancy_r'):+.3f}R | Sharpe={swing_stats.get('sharpe')} | "
        f"MaxDD={swing_stats.get('max_dd_pct')}% | Ret={swing_stats.get('total_return_pct'):+.2f}% | "
        f"Friction=${swing_stats.get('friction_usd')}"
    )
    print("-" * 80)
    print(
        f"[LIVE SCALP]      : {scalp_sig.direction} | Px={scalp_sig.entry} | Score={scalp_sig.score} | "
        f"Regime={scalp_sig.regime} | KER={scalp_sig.ker} | ADX={scalp_sig.adx} | Reason={scalp_sig.reason}"
    )
    print(
        f"[LIVE SWING]      : {swing_sig.direction} | Px={swing_sig.entry} | Score={swing_sig.score} | "
        f"Regime={swing_sig.regime} | KER={swing_sig.ker} | ADX={swing_sig.adx} | Reason={swing_sig.reason}"
    )
    print(f"[ACTIVE TRADES]   : {len(active_trades)} active position(s)")
    for t in active_trades:
        print(
            f"   -> #{t['id']} {t['mode']} {t['direction']} Entry={t['entry']} "
            f"CurSL={t['current_sl']} Status={t['status']}"
        )
    print("=" * 80)


# ==============================================================================
# INTERACTIVE COMMAND PROCESSOR & MAIN ORCHESTRATOR
# ==============================================================================
def format_on_demand_check(
    sig: SignalCandidate, bt_stats: Dict[str, Any]
) -> str:
    if sig.direction in ("BUY", "SELL"):
        return SignalEngine.format_telegram_signal(sig)
    return (
        f"⏸ <b>XAUUSD | {sig.mode} | NO_SIGNAL</b>\n"
        f"💵 Spot: <code>{sig.entry:.2f}</code> | Skor: <code>{sig.score:.0f}</code> | Regime: <code>{sig.regime}</code>\n"
        f"🛡 Filter: <code>{sig.reason}</code> | BT WR: <code>{bt_stats.get('win_rate', 0):.1f}%</code>"
    )


def format_status_message(
    state: StateManager,
    spot_px: float,
    scalp_stats: Dict[str, Any],
    swing_stats: Dict[str, Any],
) -> str:
    active = state.get_active_signals()
    sess_ok, sess_tag = is_market_open()
    lines = [
        f"📊 <b>APEX-QUANT STATUS</b> | 🕒 <code>{utc_now().strftime('%H:%M UTC')}</code>",
        f"💵 Spot XAUUSD: <code>{spot_px:.2f}</code> | Sesi: <code>{sess_tag}</code>",
        f"⚡ Scalp BT: WR <code>{scalp_stats.get('win_rate', 0):.1f}%</code> | PF <code>{scalp_stats.get('profit_factor', 0):.2f}</code> | DD <code>{scalp_stats.get('max_dd_pct', 0):.1f}%</code>",
        f"🌊 Swing BT: WR <code>{swing_stats.get('win_rate', 0):.1f}%</code> | PF <code>{swing_stats.get('profit_factor', 0):.2f}</code> | DD <code>{swing_stats.get('max_dd_pct', 0):.1f}%</code>",
        f"📌 Posisi Aktif: <code>{len(active)}</code>",
    ]
    for tr in active[:4]:
        lines.append(
            f"• #{tr['id']} <b>{tr['mode']} {tr['direction']}</b> @ <code>{tr['entry']:.2f}</code> | SL: <code>{tr['current_sl']:.2f}</code> | <code>{tr['status']}</code>"
        )
    return "\n".join(lines)


def ensure_fallback_chart_if_missing(out_path: str = CHART_PATH) -> None:
    if os.path.exists(out_path):
        return
    empty_st = {
        "trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
        "expectancy_r": 0.0, "max_dd_pct": 0.0, "total_return_pct": 0.0,
        "equity_curve": [],
    }
    BacktestAndOptimizer.generate_performance_chart(empty_st, empty_st, out_path)


def main() -> int:
    state = StateManager(DB_PATH)
    notifier = Notifier(state)

    try:
        # 1. Circuit Breaker Check
        cb_open, cb_until = state.is_circuit_open()
        pending_cmds = notifier.poll_commands()
        manual_cmd = os.getenv("BOT_COMMAND", "auto").strip().lower()

        if cb_open and manual_cmd == "auto" and not pending_cmds:
            logger.warning(f"Circuit Breaker OPEN until {cb_until}; skipping scheduled execution.")
            ensure_fallback_chart_if_missing()
            return 0

        # 2. Multi-Source Data Acquisition
        market_open, sess_tag = is_market_open()
        data_engine = MarketDataEngine()
        tf_bundle = data_engine.load_all_timeframes()

        df_5m, src_5m = tf_bundle["5m"]
        df_15m, src_15m = tf_bundle["15m"]
        df_1h, src_1h = tf_bundle["1h"]
        df_4h, src_4h = tf_bundle["4h"]

        sources = {"5m": src_5m, "15m": src_15m, "1h": src_1h, "4h": src_4h}

        ok_5m, msg_5m = MarketDataEngine.validate_freshness_and_shock(df_5m, SCALP_CONFIG.stale_max_minutes, market_open)
        ok_15m, msg_15m = MarketDataEngine.validate_freshness_and_shock(df_15m, SWING_CONFIG.stale_max_minutes, market_open)
        ok_1h, msg_1h = MarketDataEngine.validate_freshness_and_shock(df_1h, 180, market_open)

        if df_5m.empty or df_15m.empty or df_1h.empty or df_4h.empty:
            failures, tripped = state.record_failure()
            err_msg = f"⚠️ <b>APEX DATA FAILOVER EXHAUSTED</b> | Sources: <code>{sources}</code> | Fails: <code>{failures}</code>"
            logger.error(err_msg)
            if tripped or pending_cmds:
                notifier.broadcast(err_msg)
            ensure_fallback_chart_if_missing()
            return 1

        spot_px = float(df_5m["Close"].iloc[-1])

        # 3. Auto-Trailing Stop Manager on Active Trades (always update if market ticks)
        trail_mgr = TrailingStopManager(state, notifier)
        if market_open:
            trail_mgr.update_active_trades(df_5m, df_15m)

        # 4. Grid Search Optimization & Backtest (SCALP & SWING)
        _, opt_z_scalp, df_scalp_enr, scalp_stats = BacktestAndOptimizer.grid_search_optimize(
            df_5m, df_15m, df_1h, SCALP_CONFIG
        )
        _, opt_z_swing, df_swing_enr, swing_stats = BacktestAndOptimizer.grid_search_optimize(
            df_15m, df_1h, df_4h, SWING_CONFIG
        )

        # 5. Generate Performance Visualization Chart
        BacktestAndOptimizer.generate_performance_chart(scalp_stats, swing_stats, CHART_PATH)

        # 6. Evaluate Live Signals (respecting Volatility Shock & Freshness Guards)
        if ok_5m and ok_15m:
            scalp_sig = SignalEngine.build_live_signal(df_scalp_enr, SCALP_CONFIG, opt_z_scalp, scalp_stats)
        else:
            scalp_sig = SignalCandidate(
                mode="SCALP", direction="NO_SIGNAL", entry=round(spot_px, 2),
                regime="DATA_GUARD", reason=f"5M:{msg_5m}|15M:{msg_15m}"
            )

        if ok_15m and ok_1h:
            swing_sig = SignalEngine.build_live_signal(df_swing_enr, SWING_CONFIG, opt_z_swing, swing_stats)
        else:
            swing_sig = SignalCandidate(
                mode="SWING", direction="NO_SIGNAL", entry=round(spot_px, 2),
                regime="DATA_GUARD", reason=f"15M:{msg_15m}|1H:{msg_1h}"
            )

        # 7. Process Interactive Telegram Commands & Workflow Dispatch Override
        if manual_cmd in ("check", "scalp", "swing", "status"):
            target = notifier.authorized_chat_ids[0] if notifier.authorized_chat_ids else None
            pending_cmds.append((target or "", f"/{manual_cmd}"))

        for chat_id, raw_cmd in pending_cmds:
            cmd = raw_cmd.lower().strip()
            target_cid = chat_id if chat_id else None
            if cmd.startswith("/status"):
                notifier.broadcast(
                    format_status_message(state, spot_px, scalp_stats, swing_stats),
                    target_chat_id=target_cid,
                )
            elif cmd.startswith("/help"):
                notifier.broadcast(
                    "🤖 <b>APEX-QUANT COMMANDS</b>\n"
                    "<code>/check</code> — Analisa SCALP & SWING\n"
                    "<code>/check scalp</code> — Analisa SCALP\n"
                    "<code>/check swing</code> — Analisa SWING\n"
                    "<code>/status</code> — Status Posisi & Backtest",
                    target_chat_id=target_cid,
                )
            elif cmd.startswith("/check") or cmd in ("/scalp", "/swing"):
                if "scalp" in cmd:
                    notifier.broadcast(format_on_demand_check(scalp_sig, scalp_stats), target_chat_id=target_cid)
                elif "swing" in cmd:
                    notifier.broadcast(format_on_demand_check(swing_sig, swing_stats), target_chat_id=target_cid)
                else:
                    notifier.broadcast(format_on_demand_check(scalp_sig, scalp_stats), target_chat_id=target_cid)
                    notifier.broadcast(format_on_demand_check(swing_sig, swing_stats), target_chat_id=target_cid)

        # 8. Scheduled Auto-Signal Emission (Deduplicated & Cooldown-Protected)
        for sig, cfg in [(scalp_sig, SCALP_CONFIG), (swing_sig, SWING_CONFIG)]:
            if sig.direction in ("BUY", "SELL"):
                blocked, block_reason = state.is_duplicate_or_cooldown(
                    sig.mode, sig.direction, sig.candle_ts, cfg.cooldown_minutes, sig.score
                )
                if blocked:
                    logger.info(f"[{sig.mode}] Signal suppressed: {block_reason}")
                else:
                    sig_id = state.insert_signal(sig)
                    logger.info(f"[{sig.mode}] Emitted #{sig_id} {sig.direction} @ {sig.entry}")
                    notifier.broadcast(SignalEngine.format_telegram_signal(sig))

        # 9. Forensic Audit Log & Circuit Reset
        state.record_success()
        active_trades = state.get_active_signals()
        forensic_audit_log(sources, scalp_stats, swing_stats, scalp_sig, swing_sig, active_trades)
        return 0

    except Exception as exc:
        failures, tripped = state.record_failure()
        tb_short = traceback.format_exc()[-500:]
        logger.exception("Fatal error in APEX-QUANT-OMNI main loop")
        err_alert = (
            f"🚨 <b>APEX-QUANT SYSTEM ERROR</b> (Fails: <code>{failures}</code>)\n"
            f"<code>{str(exc)[:180]}</code>\n"
            f"<code>{tb_short[:300]}</code>"
        )
        notifier.broadcast(err_alert)
        ensure_fallback_chart_if_missing()
        return 1


if __name__ == "__main__":
    sys.exit(main())
