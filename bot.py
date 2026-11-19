#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APEX-QUANT-OMNI — HYPERION DUAL-TIER PRECISION EDITION
Supports BOTH Balanced Pro & Sniper Extreme with Intermarket DXY/US10Y Veto,
USD High-Impact News Blackout, FVG Confluence, +0.75R Early BE Shield,
and Walk-Forward L2 Logistic Probability Calibration.
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

try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
except Exception:
    NY_TZ = timezone(timedelta(hours=-5))

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
    stream=sys.stdout,
    force=True,
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger("APEX_QUANT_OMNI")

DB_PATH = os.getenv("DB_PATH", "trading_signals.db")
CHART_PATH = os.getenv("CHART_PATH", "backtest_performance.png")

CONTRACT_SIZE = 100.0
PIP_SIZE = 0.10
MIN_LOT = 0.01
LOT_STEP = 0.01
MAX_LOT = float(os.getenv("MAX_LOT", "10.0"))

SPREAD_USD = float(os.getenv("SPREAD_USD", "0.30"))
SLIPPAGE_USD = float(os.getenv("SLIPPAGE_USD", "0.10"))

DEFAULT_ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000.0"))
DEFAULT_RISK_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
SIGNAL_TIER_MODE = os.getenv("SIGNAL_TIER_MODE", "BOTH").strip().upper()

RETRY_BACKOFFS = [2, 4, 8]
CIRCUIT_MAX_FAILURES = 5
CIRCUIT_COOLDOWN_MINUTES = 60

GRID_ATR_PERIODS = [14, 20, 26]
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
    sniper_score: float
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
    use_swing_anchor_sl: bool


SCALP_CONFIG = ModeConfig(
    mode_name="SCALP",
    exec_tf="5m",
    mid_tf="15m",
    htf_tf="1h",
    sl_atr_mult=1.75,
    tp_r_multiples=(1.4, 2.2, 3.2),
    trail_atr_mult=1.05,
    min_score=74.0,
    sniper_score=82.0,
    min_ker=0.34,
    max_chop=55.0,
    min_adx=19.5,
    cooldown_minutes=30,
    min_backtest_trades=12,
    z_window=20,
    vwap_window=48,
    swing_lr=3,
    stale_max_minutes=35,
    be_buffer_usd=0.25,
    use_swing_anchor_sl=False,
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
    sniper_score=83.0,
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
    use_swing_anchor_sl=True,
)


@dataclass
class SignalCandidate:
    mode: str
    direction: str
    tier: str = "NONE"
    entry: float = 0.0
    sl: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    lot_size: float = 0.01
    risk_usd: float = 0.0
    score: float = 0.0
    p_win: float = 0.0
    ev_r: float = 0.0
    ker: float = 0.0
    adx: float = 0.0
    z_score: float = 0.0
    atr: float = 0.0
    regime: str = "UNKNOWN"
    mtf_bias: str = "NEUTRAL"
    macro_ctx: str = "NEUTRAL"
    candle_ts: str = ""
    reason: str = ""
    pillar_details: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_market_open(now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
    dt = now_utc or utc_now()
    wd, hr, mn = dt.weekday(), dt.hour, dt.minute
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


def is_peak_london_ny_session(dt: datetime) -> bool:
    return 7 <= dt.hour < 19


def calculate_mt5_lot_size(
    entry_price: float,
    sl_price: float,
    account_balance: float = DEFAULT_ACCOUNT_BALANCE,
    risk_pct: float = DEFAULT_RISK_PCT,
) -> float:
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 1e-6 or account_balance <= 0 or risk_pct <= 0:
        return MIN_LOT
    risk_usd = account_balance * (risk_pct / 100.0)
    raw_lot = risk_usd / ((sl_distance / PIP_SIZE) * (CONTRACT_SIZE * PIP_SIZE) + 1e-8)
    rounded_lot = math.floor(raw_lot / LOT_STEP) * LOT_STEP
    return max(MIN_LOT, min(MAX_LOT, round(rounded_lot, 2)))


# ==============================================================================
# SQLITE STATE MANAGER
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_mode_ts ON signals(mode, candle_ts)")
            conn.commit()

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        now_s = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, value, now_s),
            )
            conn.commit()

    def is_circuit_open(self) -> Tuple[bool, str]:
        open_until = self.get_state("circuit_open_until")
        if not open_until:
            return False, ""
        try:
            if utc_now() < datetime.fromisoformat(open_until):
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
            self.set_state("circuit_open_until", (utc_now() + timedelta(minutes=CIRCUIT_COOLDOWN_MINUTES)).isoformat())
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
                "SELECT id FROM signals WHERE mode = ? AND direction = ? AND candle_ts = ?",
                (mode, direction, candle_ts),
            ).fetchone()
            if dup:
                return True, f"DUPLICATE_CANDLE_{candle_ts}"

            last_sig = conn.execute(
                "SELECT direction, created_at FROM signals WHERE mode = ? ORDER BY id DESC LIMIT 1",
                (mode,),
            ).fetchone()
            if last_sig:
                elapsed_min = (utc_now() - datetime.fromisoformat(last_sig["created_at"])).total_seconds() / 60.0
                if elapsed_min < cooldown_minutes:
                    return True, f"COOLDOWN_ACTIVE_{elapsed_min:.1f}m/{cooldown_minutes}m"
                if last_sig["direction"] != direction and elapsed_min < (cooldown_minutes * 2.0) and score < 88.0:
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
                    sig.mode, sig.direction, sig.entry, sig.sl, sig.sl,
                    sig.tp1, sig.tp2, sig.tp3, sig.lot_size, sig.score,
                    sig.p_win, sig.ev_r, sig.atr, sig.regime, sig.candle_ts,
                    now_s, now_s,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_active_signals(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM signals WHERE status IN ('OPEN', 'BE_LOCKED', 'TP1_TRAIL', 'TP2_TRAIL') ORDER BY id DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_signals(self, limit: int = 5) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def update_signal_trail(self, sig_id: int, new_sl: float, new_status: str) -> None:
        now_s = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE signals SET current_sl = ?, status = ?, updated_at = ? WHERE id = ?",
                (round(new_sl, 2), new_status, now_s, sig_id),
            )
            conn.commit()


# ==============================================================================
# NOTIFICATION & INTERACTIVE TELEGRAM POLLING
# ==============================================================================
class Notifier:
    def __init__(self, state: StateManager) -> None:
        self.state = state
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        raw_chat_ids = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.authorized_chat_ids = [c.strip() for c in raw_chat_ids.split(",") if c.strip()]
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    def _request_with_backoff(self, method: str, url: str, **kwargs: Any) -> Optional[requests.Response]:
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
                                retry_after = int(resp.json().get("parameters", {}).get("retry_after", wait_s))
                                wait_s = min(max(wait_s, retry_after), 20)
                            except Exception:
                                pass
                        logger.warning(f"HTTP {resp.status_code}; retrying in {wait_s}s")
                        time.sleep(wait_s)
                        continue
                logger.error(f"HTTP {resp.status_code} error: {resp.text[:250]}")
                return resp
            except requests.RequestException as exc:
                if attempt < len(RETRY_BACKOFFS):
                    time.sleep(RETRY_BACKOFFS[attempt])
                else:
                    logger.error(f"Network delivery failed permanently: {exc}")
        return None

    def send_telegram(self, html_text: str, chat_id: Optional[str] = None) -> bool:
        if not self.tg_token or not self.authorized_chat_ids:
            logger.info("Telegram credentials absent; local message:\n" + html_text)
            return False
        targets = [chat_id] if chat_id else self.authorized_chat_ids
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        ok = True
        for cid in targets:
            resp = self._request_with_backoff(
                "POST",
                url,
                json={"chat_id": cid, "text": html_text, "parse_mode": "HTML", "disable_web_page_preview": True},
            )
            if resp is None or resp.status_code != 200:
                ok = False
        return ok

    def send_telegram_photo(self, image_path: str, caption: str = "", chat_id: Optional[str] = None) -> bool:
        if not self.tg_token or not self.authorized_chat_ids or not os.path.exists(image_path):
            return False
        targets = [chat_id] if chat_id else self.authorized_chat_ids
        url = f"https://api.telegram.org/bot{self.tg_token}/sendPhoto"
        ok = True
        for cid in targets:
            with open(image_path, "rb") as img_file:
                resp = self._request_with_backoff(
                    "POST",
                    url,
                    data={"chat_id": cid, "caption": caption[:1024], "parse_mode": "HTML"},
                    files={"photo": img_file},
                )
            if resp is None or resp.status_code != 200:
                ok = False
        return ok

    def send_discord(self, content: str) -> None:
        if not self.discord_webhook:
            return
        clean = content.replace("<b>", "**").replace("</b>", "**").replace("<code>", "`").replace("</code>", "`")
        self._request_with_backoff("POST", self.discord_webhook, json={"content": clean})

    def broadcast(self, html_text: str, target_chat_id: Optional[str] = None) -> None:
        self.send_telegram(html_text, chat_id=target_chat_id)
        if not target_chat_id:
            self.send_discord(html_text)

    def poll_commands(self) -> List[Tuple[str, str]]:
        if not self.tg_token or not self.authorized_chat_ids:
            return []
        offset = int(self.state.get_state("telegram_update_offset", "0") or "0")
        resp = self._request_with_backoff(
            "GET",
            f"https://api.telegram.org/bot{self.tg_token}/getUpdates",
            params={"offset": offset, "timeout": 3, "allowed_updates": ["message"]},
        )
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
            max_uid = max(max_uid, uid)
            msg = upd.get("message") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            text = (msg.get("text") or "").strip()
            if not text or not chat_id:
                continue
            if chat_id not in self.authorized_chat_ids:
                logger.warning(f"Unauthorized Telegram command from chat_id={chat_id}")
                continue
            cmd_norm = text.lower().split("@")[0].strip()
            if any(cmd_norm.startswith(p) for p in ("/check", "/status", "/chart", "/history", "/sniper", "/help")):
                commands.append((chat_id, text))

        if max_uid >= offset:
            self.state.set_state("telegram_update_offset", str(max_uid + 1))
        return commands


# ==============================================================================
# MACRO NEWS BLACKOUT & INTERMARKET VETO ENGINE
# ==============================================================================
class MacroAndNewsGuard:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
    }

    @classmethod
    def check_usd_news_blackout(cls, now_utc: Optional[datetime] = None) -> Tuple[bool, str]:
        dt = now_utc or utc_now()
        try:
            resp = requests.get(
                "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                headers=cls.HEADERS,
                timeout=6,
            )
            if resp.status_code == 200:
                events = resp.json()
                for ev in events:
                    if str(ev.get("country", "")).upper() != "USD":
                        continue
                    if str(ev.get("impact", "")).lower() != "high":
                        continue
                    raw_date = ev.get("date")
                    if not raw_date:
                        continue
                    ev_dt = datetime.fromisoformat(raw_date).astimezone(timezone.utc)
                    if abs((dt - ev_dt).total_seconds()) / 60.0 <= 30.0:
                        title = str(ev.get("title", "USD_HIGH_NEWS"))[:28]
                        return True, f"NEWS_BLACKOUT({title})"
                return False, "NEWS_CLEAR(LIVE)"
        except Exception:
            pass

        ny_dt = dt.astimezone(NY_TZ)
        if ny_dt.weekday() == 3 and ny_dt.hour == 8 and 20 <= ny_dt.minute <= 45:
            return True, "NEWS_BLACKOUT(US_0830ET_MACRO)"
        if ny_dt.weekday() == 4 and ny_dt.day <= 7 and ny_dt.hour == 8 and ny_dt.minute >= 15:
            return True, "NEWS_BLACKOUT(NFP_WINDOW)"
        return False, "NEWS_CLEAR(SCHED)"

    @classmethod
    def fetch_intermarket_bias(cls) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {"dxy_z": 0.0, "dxy_slope": 0.0, "tnx_z": 0.0, "dxy_dir": 0, "tag": "DXY:NEUTRAL"}
        try:
            dxy = yf.download("DX-Y.NYB", period="5d", interval="15m", progress=False, threads=False, timeout=10)
            if isinstance(dxy.columns, pd.MultiIndex):
                dxy.columns = dxy.columns.get_level_values(0)
            c = pd.to_numeric(dxy["Close"], errors="coerce").dropna() if not dxy.empty else pd.Series(dtype=float)
            if len(c) >= 30:
                z = float((c.iloc[-1] - c.rolling(20).mean().iloc[-1]) / (c.rolling(20).std(ddof=0).iloc[-1] + 1e-8))
                ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
                ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
                slope = (ema9 - ema21) / (float(c.diff().abs().rolling(14).mean().iloc[-1]) + 1e-8)
                d_dir = 1 if (z > 1.25 and slope > 0.25) else (-1 if (z < -1.25 and slope < -0.25) else 0)
                ctx["dxy_z"] = round(z, 2)
                ctx["dxy_slope"] = round(slope, 2)
                ctx["dxy_dir"] = d_dir
                ctx["tag"] = f"DXY:{'SURGE' if d_dir > 0 else ('DUMP' if d_dir < 0 else 'NEUT')}({z:+.1f})"
        except Exception:
            pass

        try:
            tnx = yf.download("^TNX", period="5d", interval="15m", progress=False, threads=False, timeout=8)
            if isinstance(tnx.columns, pd.MultiIndex):
                tnx.columns = tnx.columns.get_level_values(0)
            c_t = pd.to_numeric(tnx["Close"], errors="coerce").dropna() if not tnx.empty else pd.Series(dtype=float)
            if len(c_t) >= 25:
                tz = float((c_t.iloc[-1] - c_t.rolling(20).mean().iloc[-1]) / (c_t.rolling(20).std(ddof=0).iloc[-1] + 1e-8))
                ctx["tnx_z"] = round(tz, 2)
        except Exception:
            pass
        return ctx


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
        col_map = {str(c).lower(): c for c in df.columns}
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

        df = df[~df.index.isna()].sort_index()
        df = df[~df.index.duplicated(keep="last")]

        valid_mask = (
            (df["Open"] > 100.0)
            & (df["Close"] > 100.0)
            & (df["High"] >= df[["Open", "Close"]].max(axis=1))
            & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
            & (df["Low"] > 100.0)
        )
        df = df[valid_mask]
        pct_jump = df["Close"].pct_change().abs().fillna(0.0)
        bar_range_pct = ((df["High"] - df["Low"]) / (df["Close"] + 1e-8)).fillna(0.0)
        return df[(pct_jump < 0.045) & (bar_range_pct < 0.055)]

    def fetch_live_spot_quote(self) -> Dict[str, Any]:
        quotes: Dict[str, Dict[str, float]] = {}
        try:
            r = requests.get("https://biquote.io/api/XAUUSD", headers=self.HEADERS, timeout=8)
            if r.status_code == 200:
                q = r.json()
                bid, ask, mid = float(q.get("bid", 0) or 0), float(q.get("ask", 0) or 0), float(q.get("mid", 0) or 0)
                if mid > 100.0:
                    spread = abs(ask - bid) if (ask > bid > 100.0) else float(q.get("spread", SPREAD_USD) or SPREAD_USD)
                    quotes["biquote"] = {"mid": mid, "spread": spread}
        except Exception:
            pass

        try:
            r = requests.get(
                "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD",
                headers=self.HEADERS,
                timeout=8,
            )
            if r.status_code == 200:
                arr = r.json()
                if isinstance(arr, list) and arr:
                    prices = arr[0].get("spreadProfilePrices") or []
                    if prices:
                        bid, ask = float(prices[0].get("bid", 0) or 0), float(prices[0].get("ask", 0) or 0)
                        if bid > 100.0 and ask > 100.0:
                            quotes["swissquote"] = {"mid": (bid + ask) / 2.0, "spread": abs(ask - bid)}
        except Exception:
            pass

        chosen = quotes.get("biquote") or quotes.get("swissquote") or {}
        if chosen:
            spread_cal = max(0.12, min(1.20, float(chosen.get("spread", SPREAD_USD))))
            return {"mid": float(chosen["mid"]), "spread": round(spread_cal, 2), "sources": list(quotes.keys())}
        return {"mid": 0.0, "spread": SPREAD_USD, "sources": []}

    def _fetch_biquote_ohlc(self, interval: str, limit: int = 500) -> pd.DataFrame:
        try:
            resp = requests.get(
                "https://biquote.io/api/XAUUSD/ohlc",
                params={"interval": interval, "limit": limit},
                headers=self.HEADERS,
                timeout=12,
            )
            if resp.status_code != 200:
                return pd.DataFrame()
            bars = [b for b in (resp.json().get("bars") or []) if not b.get("isOpen", False)]
            if not bars:
                return pd.DataFrame()
            records = [
                {
                    "Time": b.get("openTime"),
                    "Open": b.get("open"), "High": b.get("high"),
                    "Low": b.get("low"), "Close": b.get("close"),
                    "Volume": float(b.get("volume") or b.get("tickVolume") or 0.0),
                }
                for b in bars
            ]
            df = pd.DataFrame(records)
            df.index = pd.to_datetime(df["Time"], utc=True, errors="coerce")
            return self._sanitize_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]])
        except Exception:
            return pd.DataFrame()

    def _fetch_kraken_paxg(self, interval: str) -> pd.DataFrame:
        tf_map = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
        try:
            resp = requests.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": "PAXGUSD", "interval": tf_map.get(interval, 15)},
                headers=self.HEADERS,
                timeout=12,
            )
            if resp.status_code != 200:
                return pd.DataFrame()
            data = resp.json()
            if data.get("error"):
                return pd.DataFrame()
            rows = next((v for k, v in (data.get("result") or {}).items() if k != "last" and isinstance(v, list)), None)
            if not rows or len(rows) < 2:
                return pd.DataFrame()
            df = pd.DataFrame(rows[:-1], columns=["Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"])
            df.index = pd.to_datetime(df["Time"].astype(float), unit="s", utc=True)
            return self._sanitize_ohlcv(df[["Open", "High", "Low", "Close", "Volume"]])
        except Exception:
            return pd.DataFrame()

    def _fetch_yfinance(self, symbol: str, interval: str, period: str) -> pd.DataFrame:
        try:
            raw = yf.download(
                tickers=symbol, period=period, interval=interval,
                auto_adjust=False, progress=False, threads=False, timeout=15,
            )
            return self._sanitize_ohlcv(raw)
        except Exception:
            return pd.DataFrame()

    def _fetch_yahoo_chart_api(self, symbol: str, interval: str, range_str: str) -> pd.DataFrame:
        for host in ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]:
            try:
                resp = requests.get(
                    f"https://{host}/v8/finance/chart/{symbol}",
                    params={"interval": interval, "range": range_str},
                    headers=self.HEADERS,
                    timeout=10,
                )
                if resp.status_code != 200:
                    continue
                res = (resp.json().get("chart") or {}).get("result") or []
                if not res:
                    continue
                ts = res[0].get("timestamp") or []
                quote = ((res[0].get("indicators") or {}).get("quote") or [{}])[0]
                if not ts or not quote:
                    continue
                df = pd.DataFrame(
                    {
                        "Open": quote.get("open"), "High": quote.get("high"),
                        "Low": quote.get("low"), "Close": quote.get("close"),
                        "Volume": quote.get("volume") or [0.0] * len(ts),
                    },
                    index=pd.to_datetime(ts, unit="s", utc=True),
                )
                clean = self._sanitize_ohlcv(df)
                if not clean.empty:
                    return clean
            except Exception:
                continue
        return pd.DataFrame()

    def fetch_timeframe(self, interval: str, period: str, min_rows: int = 120) -> Tuple[pd.DataFrame, str]:
        sources = [
            ("BIQUOTE:XAUUSD", lambda: self._fetch_biquote_ohlc(interval, limit=500)),
            ("KRAKEN:PAXGUSD", lambda: self._fetch_kraken_paxg(interval)),
            ("YF:PAXG-USD", lambda: self._fetch_yfinance("PAXG-USD", "1h" if interval == "4h" else interval, period)),
            ("YAPI:XAU=X", lambda: self._fetch_yahoo_chart_api("XAU=X", "1h" if interval == "4h" else interval, period)),
        ]
        for name, loader in sources:
            df = loader()
            if df.empty:
                continue
            if interval == "4h" and name in ("YF:PAXG-USD", "YAPI:XAU=X"):
                df = (
                    df.resample("4h")
                    .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                    .dropna(subset=["Open", "High", "Low", "Close"])
                )
            if len(df) >= min_rows:
                logger.info(f"Data [{interval}] loaded from {name} ({len(df)} bars, last={df.index[-1]})")
                return df, name
        return pd.DataFrame(), "NONE"

    def load_deep_calibration_history(self, spot_reference: float) -> Dict[str, Tuple[pd.DataFrame, str]]:
        df_5m = self._fetch_yfinance("PAXG-USD", "5m", "55d")
        df_15m = self._fetch_yfinance("PAXG-USD", "15m", "55d")
        df_1h = self._fetch_yfinance("PAXG-USD", "1h", "90d")

        out: Dict[str, Tuple[pd.DataFrame, str]] = {}
        for label, df_raw in [("5m", df_5m), ("15m", df_15m), ("1h", df_1h)]:
            if df_raw.empty or len(df_raw) < 250:
                out[label] = (pd.DataFrame(), "NONE")
                continue
            fx_mask = [is_market_open(ts.to_pydatetime())[0] for ts in df_raw.index]
            df_fx = df_raw.loc[fx_mask].copy()
            df_fx = df_fx[df_fx["High"] > df_fx["Low"]].copy()
            if len(df_fx) < 200:
                out[label] = (pd.DataFrame(), "NONE")
                continue
            if spot_reference > 100.0:
                ratio = spot_reference / float(df_fx["Close"].iloc[-1])
                if 0.90 <= ratio <= 1.10:
                    for col in ["Open", "High", "Low", "Close"]:
                        df_fx[col] = df_fx[col] * ratio
            out[label] = (df_fx, "YF:PAXG-USD(55D-FX)")

        df_1h_cal, src_1h = out.get("1h", (pd.DataFrame(), "NONE"))
        if not df_1h_cal.empty:
            df_4h_cal = (
                df_1h_cal.resample("4h")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna(subset=["Open", "High", "Low", "Close"])
            )
            out["4h"] = (df_4h_cal, f"{src_1h}->4h")
        else:
            out["4h"] = (pd.DataFrame(), "NONE")
        return out

    @staticmethod
    def validate_freshness_and_shock(
        df: pd.DataFrame, stale_max_minutes: int, market_open: bool
    ) -> Tuple[bool, str]:
        if df.empty or len(df) < 80:
            return False, "INSUFFICIENT_DATA"
        age_min = (utc_now() - df.index[-1]).total_seconds() / 60.0
        if market_open and age_min > stale_max_minutes:
            return False, f"STALE_DATA_{age_min:.1f}m"

        prev_c = df["Close"].shift(1)
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
        atr_fast, atr_base = tr.rolling(5).mean().iloc[-1], tr.rolling(50).mean().iloc[-1]
        if atr_base > 0 and ((atr_fast / (atr_base + 1e-8)) > 2.85 or (tr.iloc[-1] / (atr_base + 1e-8)) > 4.2):
            return False, f"VOLATILITY_SHOCK_ATR_RATIO={(atr_fast/(atr_base+1e-8)):.2f}"
        return True, "OK"

    def load_all_timeframes(self) -> Dict[str, Tuple[pd.DataFrame, str]]:
        df_5m, src_5m = self.fetch_timeframe("5m", "10d", min_rows=120)
        df_15m, src_15m = self.fetch_timeframe("15m", "30d", min_rows=120)
        df_1h, src_1h = self.fetch_timeframe("1h", "90d", min_rows=120)
        df_4h, src_4h = self.fetch_timeframe("4h", "180d", min_rows=100)
        if df_4h.empty and not df_1h.empty:
            df_4h = (
                df_1h.resample("4h")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna(subset=["Open", "High", "Low", "Close"])
            )
            src_4h = f"{src_1h}->4h"
        return {"5m": (df_5m, src_5m), "15m": (df_15m, src_15m), "1h": (df_1h, src_1h), "4h": (df_4h, src_4h)}


# ==============================================================================
# 5-PILLAR QUANTITATIVE FEATURE ENGINE + FVG + L2 LOGISTIC CALIBRATOR
# ==============================================================================
class LogisticCalibrator:
    def __init__(self) -> None:
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        if len(y) < 12 or len(np.unique(y)) < 2:
            return False
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0))
        Xs = (X - self.mean) / self.std
        n, d = Xs.shape
        w = np.zeros(d, dtype=float)
        b = 0.0
        lr, l2 = 0.15, 2.5
        for _ in range(180):
            z = np.clip(Xs @ w + b, -15.0, 15.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - y
            w -= lr * ((Xs.T @ err) / n + (l2 / n) * w)
            b -= lr * float(np.mean(err))
        self.weights, self.bias = w, b
        return True

    def predict_proba(self, x: np.ndarray) -> Optional[float]:
        if self.weights is None or self.mean is None or self.std is None:
            return None
        xs = (x - self.mean) / self.std
        z = float(np.clip(xs @ self.weights + self.bias, -10.0, 10.0))
        return float(1.0 / (1.0 + math.exp(-z)))


class QuantFeatureEngine:
    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        prev_c = df["Close"].shift(1)
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        up_m, dn_m = df["High"].diff(), -df["Low"].diff()
        plus_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
        minus_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
        atr = QuantFeatureEngine.compute_atr(df, period) + 1e-8
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        return dx.ewm(alpha=1.0 / period, adjust=False).mean()

    @staticmethod
    def compute_ker(close: pd.Series, window: int = 10) -> pd.Series:
        return (close - close.shift(window)).abs() / (close.diff().abs().rolling(window).sum() + 1e-8)

    @staticmethod
    def compute_choppiness(df: pd.DataFrame, window: int = 14) -> pd.Series:
        prev_c = df["Close"].shift(1)
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev_c).abs(), (df["Low"] - prev_c).abs()], axis=1).max(axis=1)
        hl_span = df["High"].rolling(window).max() - df["Low"].rolling(window).min()
        ratio = (tr.rolling(window).sum() / (hl_span + 1e-8)).clip(lower=1e-8)
        return 100.0 * np.log10(ratio) / np.log10(float(window))

    @staticmethod
    def compute_smc_and_fvg_causal(df: pd.DataFrame, lr: int = 3) -> pd.DataFrame:
        n = len(df)
        highs = df["High"].to_numpy(dtype=float)
        lows = df["Low"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)

        last_sh, last_sl = np.full(n, np.nan), np.full(n, np.nan)
        structure = np.zeros(n, dtype=int)
        bos_bull, bos_bear = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        choch_bull, choch_bear = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        sweep_bull, sweep_bear = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
        fvg_bull_zone, fvg_bear_zone = np.zeros(n, dtype=int), np.zeros(n, dtype=int)

        cur_sh, cur_sl = np.nan, np.nan
        sh_broken, sl_broken, cur_struct = False, False, 0
        bull_fvg: Optional[Tuple[float, float, int]] = None
        bear_fvg: Optional[Tuple[float, float, int]] = None

        for i in range(max(2, 2 * lr), n):
            if i >= 2 * lr:
                p = i - lr
                win_h, win_l = highs[p - lr : p + lr + 1], lows[p - lr : p + lr + 1]
                if highs[p] == np.max(win_h) and np.sum(win_h == highs[p]) == 1:
                    cur_sh, sh_broken = highs[p], False
                if lows[p] == np.min(win_l) and np.sum(win_l == lows[p]) == 1:
                    cur_sl, sl_broken = lows[p], False

            last_sh[i], last_sl[i] = cur_sh, cur_sl

            if lows[i] > highs[i - 2]:
                bull_fvg = (highs[i - 2], lows[i], i)
            if highs[i] < lows[i - 2]:
                bear_fvg = (highs[i], lows[i - 2], i)

            if bull_fvg is not None:
                fb, ft, fidx = bull_fvg
                if (i - fidx) <= 20 and lows[i] <= ft and closes[i] >= fb:
                    fvg_bull_zone[i] = 1
                elif closes[i] < fb or (i - fidx) > 20:
                    bull_fvg = None

            if bear_fvg is not None:
                fb, ft, fidx = bear_fvg
                if (i - fidx) <= 20 and highs[i] >= fb and closes[i] <= ft:
                    fvg_bear_zone[i] = 1
                elif closes[i] > ft or (i - fidx) > 20:
                    bear_fvg = None

            if not np.isnan(cur_sl) and lows[i] < cur_sl and closes[i] > cur_sl:
                sweep_bull[i] = 1
            if not np.isnan(cur_sh) and highs[i] > cur_sh and closes[i] < cur_sh:
                sweep_bear[i] = 1

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
                "last_sh": last_sh, "last_sl": last_sl, "smc_structure": structure,
                "bos_bull": bos_bull, "bos_bear": bos_bear,
                "choch_bull": choch_bull, "choch_bear": choch_bear,
                "sweep_bull": sweep_bull, "sweep_bear": sweep_bear,
                "fvg_bull": fvg_bull_zone, "fvg_bear": fvg_bear_zone,
            },
            index=df.index,
        )

    @staticmethod
    def compute_order_flow_and_vwap(df: pd.DataFrame, vwap_window: int = 32) -> pd.DataFrame:
        candle_range = (df["High"] - df["Low"]) + 1e-8
        body_ratio = (df["Close"] - df["Open"]) / candle_range
        upper_wick_ratio = (df["High"] - df[["Open", "Close"]].max(axis=1)) / candle_range
        lower_wick_ratio = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / candle_range

        if (df["Volume"] > 0).sum() > (len(df) * 0.25):
            vol_base = df["Volume"].replace(0.0, np.nan).ffill().fillna(1.0)
            rel_vol = (vol_base / (vol_base.rolling(20).mean() + 1e-8)).clip(0.25, 3.0)
        else:
            rel_vol = (candle_range / (candle_range.rolling(20).median() + 1e-8)).clip(0.35, 2.5)

        of_score = (body_ratio * rel_vol).rolling(3).mean().fillna(0.0)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        rolling_vwap = (typical_price * rel_vol).rolling(vwap_window).sum() / (rel_vol.rolling(vwap_window).sum() + 1e-8)

        return pd.DataFrame(
            {
                "of_score": of_score, "body_ratio": body_ratio,
                "upper_wick_ratio": upper_wick_ratio, "lower_wick_ratio": lower_wick_ratio,
                "rolling_vwap": rolling_vwap,
            },
            index=df.index,
        )

    @classmethod
    def compute_htf_bias(cls, df_htf: pd.DataFrame, tf_delta: pd.Timedelta) -> pd.DataFrame:
        if df_htf.empty or len(df_htf) < 55:
            return pd.DataFrame()
        ema_fast = df_htf["Close"].ewm(span=20, adjust=False).mean()
        ema_slow = df_htf["Close"].ewm(span=50, adjust=False).mean()
        smc = cls.compute_smc_and_fvg_causal(df_htf, lr=3)
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
        return pd.DataFrame(
            {"htf_bias": bias, "htf_adx": adx.to_numpy(), "htf_smc": smc["smc_structure"].to_numpy()},
            index=df_htf.index + tf_delta,
        )

    @classmethod
    def enrich_dataframe(
        cls, df_exec: pd.DataFrame, df_mid: pd.DataFrame, df_htf: pd.DataFrame, cfg: ModeConfig, atr_period: int = 14
    ) -> pd.DataFrame:
        df = df_exec.copy()
        df["atr"] = cls.compute_atr(df, atr_period)
        df["atr_base"] = cls.compute_atr(df, 50)
        df["adx"] = cls.compute_adx(df, 14)
        df["ker"] = cls.compute_ker(df["Close"], 10)
        df["chop"] = cls.compute_choppiness(df, 14)
        df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["Close"].ewm(span=50, adjust=False).mean()

        z_mean = df["Close"].rolling(cfg.z_window).mean()
        z_std = df["Close"].rolling(cfg.z_window).std(ddof=0)
        df["z_score"] = (df["Close"] - z_mean) / (z_std + 1e-8)
        df["z_prev"] = df["z_score"].shift(1)

        smc = cls.compute_smc_and_fvg_causal(df, lr=cfg.swing_lr)
        of_vwap = cls.compute_order_flow_and_vwap(df, vwap_window=cfg.vwap_window)
        df = pd.concat([df, smc, of_vwap], axis=1)
        df["vwap_dist_atr"] = (df["Close"] - df["rolling_vwap"]) / (df["atr"] + 1e-8)

        mid_delta = pd.Timedelta(minutes=15) if cfg.mid_tf == "15m" else pd.Timedelta(hours=1)
        htf_delta = pd.Timedelta(hours=1) if cfg.htf_tf == "1h" else pd.Timedelta(hours=4)

        df_reset = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        if "ts" not in df_reset.columns:
            df_reset = df_reset.rename(columns={df_reset.columns[0]: "ts"})

        mid_bias_df = cls.compute_htf_bias(df_mid, mid_delta)
        if not mid_bias_df.empty:
            mid_r = mid_bias_df.reset_index().rename(columns={mid_bias_df.index.name or "index": "ts"})
            if "ts" not in mid_r.columns:
                mid_r = mid_r.rename(columns={mid_r.columns[0]: "ts"})
            mid_r = mid_r.rename(columns={"htf_bias": "mid_bias", "htf_adx": "mid_adx", "htf_smc": "mid_smc"})
            df_reset = pd.merge_asof(df_reset.sort_values("ts"), mid_r.sort_values("ts"), on="ts", direction="backward")
        else:
            df_reset["mid_bias"], df_reset["mid_adx"], df_reset["mid_smc"] = 0, 20.0, 0

        htf_bias_df = cls.compute_htf_bias(df_htf, htf_delta)
        if not htf_bias_df.empty:
            htf_r = htf_bias_df.reset_index().rename(columns={htf_bias_df.index.name or "index": "ts"})
            if "ts" not in htf_r.columns:
                htf_r = htf_r.rename(columns={htf_r.columns[0]: "ts"})
            df_reset = pd.merge_asof(df_reset.sort_values("ts"), htf_r.sort_values("ts"), on="ts", direction="backward")
        else:
            df_reset["htf_bias"], df_reset["htf_adx"], df_reset["htf_smc"] = 0, 20.0, 0

        df_out = df_reset.set_index("ts")
        df_out["mid_bias"] = df_out["mid_bias"].fillna(0).astype(int)
        df_out["htf_bias"] = df_out["htf_bias"].fillna(0).astype(int)
        return df_out

    @staticmethod
    def classify_regime(row: pd.Series, cfg: ModeConfig, ts_dt: datetime) -> Tuple[str, bool, float]:
        atr, atr_base = float(row.get("atr", 0.0)), float(row.get("atr_base", 0.0))
        adx, ker, chop = float(row.get("adx", 0.0)), float(row.get("ker", 0.0)), float(row.get("chop", 50.0))
        ema20, ema50 = float(row.get("ema20", 0.0)), float(row.get("ema50", 0.0))
        bar_range = float(row.get("High", 0.0)) - float(row.get("Low", 0.0))

        if atr > 0 and (bar_range / (atr + 1e-8)) > 2.80:
            return "EXHAUSTION_SPIKE", False, cfg.min_score
        if atr_base > 0 and (atr / (atr_base + 1e-8)) > 2.6:
            return "VOLATILITY_SHOCK", False, cfg.min_score
        if atr_base > 0 and (atr / (atr_base + 1e-8)) < 0.42:
            return "LOW_LIQUIDITY_DEAD", False, cfg.min_score

        eff_ker, eff_adx, eff_score = cfg.min_ker, cfg.min_adx, cfg.min_score
        if cfg.mode_name == "SCALP" and not is_peak_london_ny_session(ts_dt):
            eff_ker = max(cfg.min_ker, 0.40)
            eff_adx = max(cfg.min_adx, 22.5)
            eff_score = max(cfg.min_score, 78.0)

        if chop > cfg.max_chop and ker < eff_ker:
            return "CHOPPY_SIDEWAYS", False, eff_score
        if adx < eff_adx and ker < eff_ker:
            return "RANGING_WEAK", False, eff_score
        if ema20 > ema50 and (adx >= eff_adx or ker >= eff_ker):
            return "TREND_BULL", True, eff_score
        if ema20 < ema50 and (adx >= eff_adx or ker >= eff_ker):
            return "TREND_BEAR", True, eff_score
        return "TRANSITIONAL", False, eff_score

    @classmethod
    def evaluate_bar(
        cls, df: pd.DataFrame, idx: int, cfg: ModeConfig, z_thresh: float,
        check_session: bool = True, dxy_dir: int = 0,
    ) -> Tuple[str, float, str, Dict[str, Any]]:
        if idx < 60 or idx >= len(df):
            return "NO_SIGNAL", 0.0, "WARMUP", {"reason": "WARMUP"}

        row = df.iloc[idx]
        ts_dt = df.index[idx].to_pydatetime()

        if check_session:
            sess_ok, sess_reason = is_market_open(ts_dt)
            if not sess_ok:
                return "NO_SIGNAL", 0.0, sess_reason, {"reason": sess_reason}

        regime, tradeable, eff_min_score = cls.classify_regime(row, cfg, ts_dt)
        if not tradeable:
            return "NO_SIGNAL", 0.0, regime, {"reason": f"REGIME_BLOCK:{regime}"}

        close, ema20, ema50 = float(row["Close"]), float(row["ema20"]), float(row["ema50"])
        z_val = float(row["z_score"])
        z_prev = float(row["z_prev"]) if not pd.isna(row["z_prev"]) else z_val
        of_score, body_ratio = float(row["of_score"]), float(row["body_ratio"])
        upper_wick, lower_wick = float(row["upper_wick_ratio"]), float(row["lower_wick_ratio"])
        vwap_dist, ker, adx = float(row["vwap_dist_atr"]), float(row["ker"]), float(row["adx"])
        mid_bias, htf_bias, smc_struct = int(row["mid_bias"]), int(row["htf_bias"]), int(row["smc_structure"])

        recent_bos_bull = int(df["bos_bull"].iloc[max(0, idx - 12) : idx + 1].sum() > 0)
        recent_choch_bull = int(df["choch_bull"].iloc[max(0, idx - 12) : idx + 1].sum() > 0)
        recent_bos_bear = int(df["bos_bear"].iloc[max(0, idx - 12) : idx + 1].sum() > 0)
        recent_choch_bear = int(df["choch_bear"].iloc[max(0, idx - 12) : idx + 1].sum() > 0)
        recent_sweep_bull = int(df["sweep_bull"].iloc[max(0, idx - 6) : idx + 1].sum() > 0)
        recent_sweep_bear = int(df["sweep_bear"].iloc[max(0, idx - 6) : idx + 1].sum() > 0)
        recent_fvg_bull = int(df["fvg_bull"].iloc[max(0, idx - 2) : idx + 1].sum() > 0)
        recent_fvg_bear = int(df["fvg_bear"].iloc[max(0, idx - 2) : idx + 1].sum() > 0)

        bull_score, bear_score = 0.0, 0.0

        bull_mtf_ok = (htf_bias >= 0 and mid_bias > 0) or (htf_bias > 0 and mid_bias >= 0)
        bear_mtf_ok = (htf_bias <= 0 and mid_bias < 0) or (htf_bias < 0 and mid_bias <= 0)
        if htf_bias > 0 and mid_bias > 0:
            bull_score += 23.0
        elif bull_mtf_ok:
            bull_score += 15.0
        if htf_bias < 0 and mid_bias < 0:
            bear_score += 23.0
        elif bear_mtf_ok:
            bear_score += 15.0

        if smc_struct > 0:
            bull_score += 12.0
        if recent_bos_bull or recent_choch_bull:
            bull_score += 7.0
        if recent_sweep_bull:
            bull_score += 4.0
        if recent_fvg_bull:
            bull_score += 5.0

        if smc_struct < 0:
            bear_score += 12.0
        if recent_bos_bear or recent_choch_bear:
            bear_score += 7.0
        if recent_sweep_bear:
            bear_score += 4.0
        if recent_fvg_bear:
            bear_score += 5.0

        if of_score > 0.08 and body_ratio > 0.15 and upper_wick < 0.45:
            bull_score += 19.0
        elif body_ratio > 0.05 and upper_wick < 0.50:
            bull_score += 10.0
        if of_score < -0.08 and body_ratio < -0.15 and lower_wick < 0.45:
            bear_score += 19.0
        elif body_ratio < -0.05 and lower_wick < 0.50:
            bear_score += 10.0

        if (-z_thresh <= z_val <= 1.0) and (z_val > z_prev) and close > ema50:
            bull_score += 15.0 if z_prev < -0.30 else 11.0
        elif (1.0 < z_val <= z_thresh) and (z_val >= z_prev) and ker >= 0.45 and close > ema50:
            bull_score += 10.0

        if (-1.0 <= z_val <= z_thresh) and (z_val < z_prev) and close < ema50:
            bear_score += 15.0 if z_prev > 0.30 else 11.0
        elif (-z_thresh <= z_val < -1.0) and (z_val <= z_prev) and ker >= 0.45 and close < ema50:
            bear_score += 10.0

        regime_pts = min(9.0, (ker / 0.60) * 5.0 + (min(adx, 40.0) / 40.0) * 4.0)
        if -1.2 <= vwap_dist <= 1.6 and close >= ema20:
            bull_score += regime_pts + 6.0
        elif vwap_dist <= 2.4:
            bull_score += regime_pts

        if -1.6 <= vwap_dist <= 1.2 and close <= ema20:
            bear_score += regime_pts + 6.0
        elif vwap_dist >= -2.4:
            bear_score += regime_pts

        if dxy_dir < 0:
            bull_score = min(100.0, bull_score + 5.0)
        elif dxy_dir > 0:
            bear_score = min(100.0, bear_score + 5.0)

        z_cap = z_thresh + 0.50
        bull_valid = (
            bull_mtf_ok and close > ema20 and ema20 > ema50
            and upper_wick < 0.50 and body_ratio > 0.05
            and z_val < z_cap and bull_score >= eff_min_score
        )
        bear_valid = (
            bear_mtf_ok and close < ema20 and ema20 < ema50
            and lower_wick < 0.50 and body_ratio < -0.05
            and z_val > -z_cap and bear_score >= eff_min_score
        )

        if bull_valid and dxy_dir > 0:
            return "NO_SIGNAL", round(bull_score, 1), regime, {"reason": "INTERMARKET_DXY_SURGE_VETO", "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}"}
        if bear_valid and dxy_dir < 0:
            return "NO_SIGNAL", round(bear_score, 1), regime, {"reason": "INTERMARKET_DXY_DUMP_VETO", "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}"}
        if bull_valid and bear_valid:
            return "NO_SIGNAL", max(bull_score, bear_score), regime, {"reason": "CONFLICTING_SIGNALS"}

        feat_vec = [
            round(max(bull_score, bear_score) / 100.0, 3), round(ker, 3),
            round(min(adx, 50.0) / 50.0, 3), round(min(abs(z_val), 3.0) / 3.0, 3),
            float(recent_fvg_bull if bull_valid else recent_fvg_bear),
            float((recent_bos_bull or recent_choch_bull) if bull_valid else (recent_bos_bear or recent_choch_bear)),
            float(1.0 if (htf_bias == mid_bias and htf_bias != 0) else 0.0),
        ]

        if bull_valid:
            return "BUY", round(bull_score, 1), regime, {"mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}", "fvg": recent_fvg_bull, "feat": feat_vec}
        if bear_valid:
            return "SELL", round(bear_score, 1), regime, {"mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}", "fvg": recent_fvg_bear, "feat": feat_vec}

        top_dir = "BUY" if bull_score >= bear_score else "SELL"
        top_score = round(max(bull_score, bear_score), 1)
        return "NO_SIGNAL", top_score, regime, {
            "reason": f"CONFLUENCE_BELOW_THRESHOLD({top_dir}:{top_score}/{eff_min_score})",
            "mtf_bias": f"H:{htf_bias:+d}/M:{mid_bias:+d}",
        }

    @staticmethod
    def compute_sl_and_tps(
        direction: str, entry: float, atr: float, last_sl_px: float, last_sh_px: float, cfg: ModeConfig
    ) -> Tuple[float, float, float, float]:
        min_floor = 2.00 if cfg.mode_name == "SCALP" else 1.50
        base_sl_dist = max(atr * cfg.sl_atr_mult, min_floor)
        if direction == "BUY":
            sl = entry - base_sl_dist
            if cfg.use_swing_anchor_sl and not np.isnan(last_sl_px) and last_sl_px < entry:
                swing_dist = entry - (last_sl_px - 0.25 * atr)
                if 0.85 * base_sl_dist <= swing_dist <= 1.45 * base_sl_dist:
                    sl = min(sl, last_sl_px - 0.25 * atr)
            sl_dist = max(entry - sl, min_floor)
            return (
                round(sl, 2), round(entry + sl_dist * cfg.tp_r_multiples[0], 2),
                round(entry + sl_dist * cfg.tp_r_multiples[1], 2), round(entry + sl_dist * cfg.tp_r_multiples[2], 2),
            )
        else:
            sl = entry + base_sl_dist
            if cfg.use_swing_anchor_sl and not np.isnan(last_sh_px) and last_sh_px > entry:
                swing_dist = (last_sh_px + 0.25 * atr) - entry
                if 0.85 * base_sl_dist <= swing_dist <= 1.45 * base_sl_dist:
                    sl = max(sl, last_sh_px + 0.25 * atr)
            sl_dist = max(sl - entry, min_floor)
            return (
                round(sl, 2), round(entry - sl_dist * cfg.tp_r_multiples[0], 2),
                round(entry - sl_dist * cfg.tp_r_multiples[1], 2), round(entry - sl_dist * cfg.tp_r_multiples[2], 2),
            )


# ==============================================================================
# WALK-FORWARD BACKTEST ENGINE & DUAL-TIER OPTIMIZER
# ==============================================================================
class BacktestAndOptimizer:
    @staticmethod
    def run_backtest(
        df: pd.DataFrame, cfg: ModeConfig, z_thresh: float,
        spread_usd: float = SPREAD_USD, account_balance: float = DEFAULT_ACCOUNT_BALANCE,
        start_idx: int = 60, end_idx: Optional[int] = None, min_score_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        n = min(len(df), end_idx) if end_idx is not None else len(df)
        start_i = max(60, start_idx)
        equity, peak_equity, max_dd_pct = account_balance, account_balance, 0.0
        equity_curve: List[Tuple[pd.Timestamp, float]] = []
        r_returns, feat_X, feat_y = [], [], []
        wins, be_exits, losses = 0, 0, 0
        gross_profit, gross_loss, total_friction_usd = 0.0, 0.0, 0.0
        last_entry_idx = -999
        min_bar_gap = 6 if cfg.mode_name == "SCALP" else 8

        i = start_i
        while i < n - 2:
            equity_curve.append((df.index[i], equity))
            if (i - last_entry_idx) < min_bar_gap:
                i += 1
                continue

            direction, score, _, diag = QuantFeatureEngine.evaluate_bar(df, i, cfg, z_thresh=z_thresh, check_session=True)
            if direction not in ("BUY", "SELL") or (min_score_override is not None and score < min_score_override):
                i += 1
                continue

            atr = float(df["atr"].iloc[i])
            if atr <= 0.1 or np.isnan(atr):
                i += 1
                continue

            entry = float(df["Close"].iloc[i])
            last_sl_px = float(df["last_sl"].iloc[i]) if not pd.isna(df["last_sl"].iloc[i]) else np.nan
            last_sh_px = float(df["last_sh"].iloc[i]) if not pd.isna(df["last_sh"].iloc[i]) else np.nan
            sl, tp1, tp2, tp3 = QuantFeatureEngine.compute_sl_and_tps(direction, entry, atr, last_sl_px, last_sh_px, cfg)
            sl_dist = abs(entry - sl)
            be_trigger_dist = 0.75 * sl_dist

            risk_usd = equity * (DEFAULT_RISK_PCT / 100.0)
            lot = calculate_mt5_lot_size(entry, sl, equity, DEFAULT_RISK_PCT)
            friction_rt = (spread_usd + 2.0 * SLIPPAGE_USD) * lot * CONTRACT_SIZE
            total_friction_usd += friction_rt

            cur_sl, rem_weight, realized_r = sl, 1.0, 0.0
            be_locked, tp1_hit, tp2_hit = False, False, False
            exit_bar = i + 1

            for j in range(i + 1, min(n, i + 180)):
                bar_h, bar_l, bar_c = float(df["High"].iloc[j]), float(df["Low"].iloc[j]), float(df["Close"].iloc[j])
                bar_atr = float(df["atr"].iloc[j]) if not np.isnan(df["atr"].iloc[j]) else atr
                exit_bar = j

                if direction == "BUY":
                    if bar_l <= cur_sl:
                        realized_r += rem_weight * ((cur_sl - entry) / (sl_dist + 1e-8))
                        rem_weight = 0.0
                        break
                    if not be_locked and bar_h >= (entry + be_trigger_dist):
                        be_locked = True
                        cur_sl = max(cur_sl, entry + cfg.be_buffer_usd)
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
                    if (cfg.mode_name == "SWING" and tp1_hit) or tp2_hit:
                        cur_sl = max(cur_sl, bar_c - (bar_atr * cfg.trail_atr_mult))
                else:
                    if bar_h >= cur_sl:
                        realized_r += rem_weight * ((entry - cur_sl) / (sl_dist + 1e-8))
                        rem_weight = 0.0
                        break
                    if not be_locked and bar_l <= (entry - be_trigger_dist):
                        be_locked = True
                        cur_sl = min(cur_sl, entry - cfg.be_buffer_usd)
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
                    if (cfg.mode_name == "SWING" and tp1_hit) or tp2_hit:
                        cur_sl = min(cur_sl, bar_c + (bar_atr * cfg.trail_atr_mult))

            if rem_weight > 0:
                last_c = float(df["Close"].iloc[exit_bar])
                realized_r += rem_weight * (((last_c - entry) if direction == "BUY" else (entry - last_c)) / (sl_dist + 1e-8))

            trade_pnl = (realized_r * risk_usd) - friction_rt
            net_r = trade_pnl / (risk_usd + 1e-8)
            equity = max(100.0, equity + trade_pnl)
            r_returns.append(net_r)

            if "feat" in diag:
                feat_X.append(diag["feat"])
                feat_y.append(1 if (tp1_hit or trade_pnl > 0) else 0)

            if trade_pnl > 0:
                wins += 1
                gross_profit += trade_pnl
            elif be_locked:
                be_exits += 1
                gross_loss += abs(trade_pnl)
            else:
                losses += 1
                gross_loss += abs(trade_pnl)

            peak_equity = max(peak_equity, equity)
            max_dd_pct = max(max_dd_pct, ((peak_equity - equity) / (peak_equity + 1e-8)) * 100.0)
            last_entry_idx = i
            i = max(i + 1, exit_bar + 1)

        if n > start_i:
            equity_curve.append((df.index[n - 1], equity))

        total_trades = wins + be_exits + losses
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        non_loss_rate = ((wins + be_exits) / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (gross_profit / (gross_loss + 1e-8)) if gross_loss > 0 else (3.0 if gross_profit > 0 else 0.0)
        expectancy_r = float(np.mean(r_returns)) if r_returns else 0.0
        std_r = float(np.std(r_returns, ddof=1)) if len(r_returns) >= 3 else 0.0
        sharpe = max(-10.0, min(10.0, (expectancy_r / std_r) * math.sqrt(total_trades))) if (len(r_returns) >= 3 and std_r > 1e-3) else 0.0

        return {
            "mode": cfg.mode_name, "trades": total_trades, "wins": wins, "be_exits": be_exits, "losses": losses,
            "win_rate": round(win_rate, 2), "non_loss_rate": round(non_loss_rate, 2),
            "profit_factor": round(min(profit_factor, 9.99), 2), "expectancy_r": round(expectancy_r, 3),
            "sharpe": round(sharpe, 2), "max_dd_pct": round(max_dd_pct, 2),
            "total_return_pct": round(((equity - account_balance) / account_balance) * 100.0, 2),
            "final_equity": round(equity, 2), "friction_usd": round(total_friction_usd, 2),
            "equity_curve": equity_curve, "feat_X": feat_X, "feat_y": feat_y,
        }

    @classmethod
    def grid_search_optimize(
        cls,
        df_exec_bt: pd.DataFrame, df_mid_bt: pd.DataFrame, df_htf_bt: pd.DataFrame,
        df_exec_live: pd.DataFrame, df_mid_live: pd.DataFrame, df_htf_live: pd.DataFrame,
        cfg: ModeConfig, spread_usd: float = SPREAD_USD,
    ) -> Tuple[int, float, pd.DataFrame, Dict[str, Any], LogisticCalibrator]:
        use_bt_exec = df_exec_bt if len(df_exec_bt) > len(df_exec_live) else df_exec_live
        use_bt_mid = df_mid_bt if len(df_mid_bt) > len(df_mid_live) else df_mid_live
        use_bt_htf = df_htf_bt if len(df_htf_bt) > len(df_htf_live) else df_htf_live

        n_bt = len(use_bt_exec)
        split_idx = int(n_bt * 0.70)
        has_oos = (split_idx >= 240) and ((n_bt - split_idx) >= 150)

        cached_bt = {
            atr_p: QuantFeatureEngine.enrich_dataframe(use_bt_exec, use_bt_mid, use_bt_htf, cfg, atr_period=atr_p)
            for atr_p in GRID_ATR_PERIODS
        }

        best_obj, best_atr, best_z, found = -1e9, 14, 2.0, False
        best_is_stats: Optional[Dict[str, Any]] = None
        is_end = split_idx if has_oos else n_bt

        for atr_p in GRID_ATR_PERIODS:
            for z_t in GRID_Z_THRESHOLDS:
                is_stats = cls.run_backtest(
                    cached_bt[atr_p], cfg, z_thresh=z_t, spread_usd=spread_usd, start_idx=60, end_idx=is_end
                )
                if is_stats["trades"] < cfg.min_backtest_trades:
                    continue
                obj = is_stats["expectancy_r"] * math.sqrt(is_stats["trades"]) * min(is_stats["profit_factor"], 3.0) - 0.18 * is_stats["max_dd_pct"]
                if obj > best_obj:
                    best_obj, best_atr, best_z, found, best_is_stats = obj, atr_p, z_t, True, is_stats

        if not found:
            best_atr, best_z = 20, 1.5
            best_is_stats = cls.run_backtest(cached_bt[best_atr], cfg, z_thresh=best_z, spread_usd=spread_usd, start_idx=60, end_idx=is_end)

        win_df = cached_bt[best_atr]
        calibrator = LogisticCalibrator()
        if best_is_stats and len(best_is_stats.get("feat_y", [])) >= 12:
            calibrator.fit(np.array(best_is_stats["feat_X"], dtype=float), np.array(best_is_stats["feat_y"], dtype=float))

        full_stats = cls.run_backtest(win_df, cfg, z_thresh=best_z, spread_usd=spread_usd, start_idx=60)
        sniper_stats = cls.run_backtest(
            win_df, cfg, z_thresh=best_z, spread_usd=spread_usd, start_idx=60, min_score_override=cfg.sniper_score
        )

        if has_oos:
            oos_stats = cls.run_backtest(win_df, cfg, z_thresh=best_z, spread_usd=spread_usd, start_idx=split_idx)
            full_stats["oos_split_ts"] = win_df.index[split_idx]
            full_stats["oos_trades"] = oos_stats["trades"]
            full_stats["oos_win_rate"] = oos_stats["win_rate"]
            full_stats["oos_non_loss_rate"] = oos_stats["non_loss_rate"]
            full_stats["oos_pf"] = oos_stats["profit_factor"]
            full_stats["oos_ev_r"] = oos_stats["expectancy_r"]
        else:
            full_stats["oos_split_ts"] = None
            full_stats["oos_trades"] = full_stats["trades"]
            full_stats["oos_win_rate"] = full_stats["win_rate"]
            full_stats["oos_non_loss_rate"] = full_stats["non_loss_rate"]
            full_stats["oos_pf"] = full_stats["profit_factor"]
            full_stats["oos_ev_r"] = full_stats["expectancy_r"]

        full_stats["opt_atr_period"] = best_atr
        full_stats["opt_z_thresh"] = best_z
        full_stats["bt_bars"] = n_bt
        full_stats["sniper_trades"] = sniper_stats["trades"]
        full_stats["sniper_wr"] = sniper_stats["win_rate"]
        full_stats["sniper_non_loss"] = sniper_stats["non_loss_rate"]
        full_stats["sniper_pf"] = sniper_stats["profit_factor"]
        full_stats["sniper_ret"] = sniper_stats["total_return_pct"]

        df_live_enr = QuantFeatureEngine.enrich_dataframe(df_exec_live, df_mid_live, df_htf_live, cfg, atr_period=best_atr)
        return best_atr, best_z, df_live_enr, full_stats, calibrator

    @staticmethod
    def generate_performance_chart(scalp_stats: Dict[str, Any], swing_stats: Dict[str, Any], out_path: str = CHART_PATH) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor="#0d1117")
        fig.suptitle(
            "APEX-QUANT-OMNI | Dual-Tier Walk-Forward Backtest (+0.75R BE Shield & Spread/Slippage)",
            color="#f0f6fc", fontsize=12.5, fontweight="bold",
        )
        for ax, st, color, label in [
            (axes[0], scalp_stats, "#00e676", "SCALP (5M/15M/1H)"),
            (axes[1], swing_stats, "#29b6f6", "SWING (15M/1H/4H)"),
        ]:
            ax.set_facecolor("#161b22")
            curve = st.get("equity_curve") or []
            if curve:
                xs, ys = [c[0] for c in curve], [c[1] for c in curve]
                ax.plot(xs, ys, color=color, linewidth=1.8)
                ax.fill_between(xs, DEFAULT_ACCOUNT_BALANCE, ys, color=color, alpha=0.12)
                ax.axhline(DEFAULT_ACCOUNT_BALANCE, color="#8b949e", linestyle="--", linewidth=0.9)
                if st.get("oos_split_ts") is not None:
                    ax.axvline(st["oos_split_ts"], color="#ffa726", linestyle="-.", linewidth=1.2)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            else:
                ax.text(0.5, 0.5, "No Trades Recorded", color="#8b949e", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(
                f"{label} | PRO: {st.get('trades',0)}T WR {st.get('win_rate',0):.1f}% (NonLoss {st.get('non_loss_rate',0):.1f}%) PF {st.get('profit_factor',0):.2f} Ret {st.get('total_return_pct',0):+.1f}% | "
                f"SNIPER: {st.get('sniper_trades',0)}T WR {st.get('sniper_wr',0):.1f}% NonLoss {st.get('sniper_non_loss',0):.1f}% PF {st.get('sniper_pf',0):.2f}",
                color="#c9d1d9", fontsize=9.5, loc="left",
            )
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
        df_enriched: pd.DataFrame, cfg: ModeConfig, z_thresh: float,
        backtest_stats: Dict[str, Any], calibrator: LogisticCalibrator,
        intermarket: Dict[str, Any], news_blackout: Tuple[bool, str],
        live_spread_usd: float = SPREAD_USD, live_mid: float = 0.0,
    ) -> SignalCandidate:
        if df_enriched.empty or len(df_enriched) < 65:
            return SignalCandidate(mode=cfg.mode_name, direction="NO_SIGNAL", reason="EMPTY_DATA")

        idx = len(df_enriched) - 1
        row = df_enriched.iloc[idx]
        candle_ts = df_enriched.index[idx].isoformat()
        close_px, atr = float(row["Close"]), float(row["atr"])
        ker, adx, z_val = float(row["ker"]), float(row["adx"]), float(row["z_score"])
        spot_ref = live_mid if live_mid > 100.0 else close_px
        macro_tag = f"{intermarket.get('tag', 'DXY:N/A')}|{news_blackout[1]}"

        if news_blackout[0]:
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2), atr=round(atr, 2),
                regime="NEWS_GUARD", macro_ctx=macro_tag, candle_ts=candle_ts, reason=news_blackout[1],
            )

        direction, score, regime, diag = QuantFeatureEngine.evaluate_bar(
            df_enriched, idx, cfg, z_thresh=z_thresh, check_session=True, dxy_dir=int(intermarket.get("dxy_dir", 0))
        )
        mtf_bias = str(diag.get("mtf_bias", "N/A"))

        if direction == "NO_SIGNAL":
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, macro_ctx=macro_tag,
                candle_ts=candle_ts, reason=str(diag.get("reason", "NO_CONFLUENCE")),
            )

        oos_trades = int(backtest_stats.get("oos_trades", 0))
        oos_ev_r = float(backtest_stats.get("oos_ev_r", 0.0))
        if oos_trades >= 6 and oos_ev_r < -0.25 and score < (cfg.min_score + 6.0):
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, macro_ctx=macro_tag,
                candle_ts=candle_ts, reason=f"OOS_DEGRADATION_GUARD(EV={oos_ev_r:+.2f}R)",
            )

        if live_mid > 100.0 and atr > 0 and abs(live_mid - close_px) > (0.65 * atr):
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(live_mid, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, macro_ctx=macro_tag,
                candle_ts=candle_ts, reason=f"PRICE_DRIFT_GUARD(diff={abs(live_mid-close_px):.2f})",
            )

        half_spread = live_spread_usd / 2.0
        entry = round(spot_ref + half_spread if direction == "BUY" else spot_ref - half_spread, 2)
        last_sl_px = float(row["last_sl"]) if not pd.isna(row["last_sl"]) else np.nan
        last_sh_px = float(row["last_sh"]) if not pd.isna(row["last_sh"]) else np.nan
        sl, tp1, tp2, tp3 = QuantFeatureEngine.compute_sl_and_tps(
            direction, entry, atr, last_sl_px, last_sh_px, cfg
        )
        lot_size = calculate_mt5_lot_size(entry, sl, DEFAULT_ACCOUNT_BALANCE, DEFAULT_RISK_PCT)
        risk_usd = round(abs(entry - sl) * lot_size * CONTRACT_SIZE, 2)

        oos_wr = float(backtest_stats.get("oos_win_rate", backtest_stats.get("win_rate", 55.0))) / 100.0
        prior_wr = 0.52 + ((score - cfg.min_score) / 100.0) * 0.22
        weight_oos = min(0.50, oos_trades / 25.0)
        bayesian_p = weight_oos * oos_wr + (1.0 - weight_oos) * prior_wr

        lr_p = calibrator.predict_proba(np.array(diag["feat"], dtype=float)) if "feat" in diag else None
        blended_p = (0.55 * bayesian_p + 0.45 * lr_p) if lr_p is not None else bayesian_p
        p_win = max(45.0, min(85.0, round(blended_p * 100.0, 1)))

        p_dec = p_win / 100.0
        blended_r = (0.50 * cfg.tp_r_multiples[0]) + (0.25 * cfg.tp_r_multiples[1]) + (0.25 * cfg.tp_r_multiples[2])
        ev_r = round((p_dec * blended_r) - ((1.0 - p_dec) * 1.0), 2)

        if ev_r <= 0.15:
            return SignalCandidate(
                mode=cfg.mode_name, direction="NO_SIGNAL", entry=round(spot_ref, 2),
                score=score, ker=round(ker, 2), adx=round(adx, 1), z_score=round(z_val, 2),
                atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias, macro_ctx=macro_tag,
                candle_ts=candle_ts, reason=f"LOW_EXPECTANCY_EV={ev_r:+.2f}R",
            )

        tier = "SNIPER" if (score >= cfg.sniper_score and p_win >= 64.0) else "PRO"
        return SignalCandidate(
            mode=cfg.mode_name, direction=direction, tier=tier, entry=entry, sl=sl,
            tp1=tp1, tp2=tp2, tp3=tp3, lot_size=lot_size, risk_usd=risk_usd,
            score=score, p_win=p_win, ev_r=ev_r, ker=round(ker, 2), adx=round(adx, 1),
            z_score=round(z_val, 2), atr=round(atr, 2), regime=regime, mtf_bias=mtf_bias,
            macro_ctx=macro_tag, candle_ts=candle_ts, reason=f"SIGNAL_{tier}_CONFIRMED", pillar_details=diag,
        )

    @staticmethod
    def format_telegram_signal(sig: SignalCandidate) -> str:
        badge = "💎 SNIPER A+" if sig.tier == "SNIPER" else "⚡ PRO"
        return (
            f"{badge} | <b>XAUUSD | {sig.mode} | {sig.direction}</b>\n"
            f"🎯 Entry: <code>{sig.entry:.2f}</code> | 🛑 SL: <code>{sig.sl:.2f}</code>\n"
            f"✅ TP1: <code>{sig.tp1:.2f}</code> | TP2: <code>{sig.tp2:.2f}</code> | TP3: <code>{sig.tp3:.2f}</code>\n"
            f"📐 Lot: <code>{sig.lot_size:.2f}</code> (<code>${sig.risk_usd:.0f}</code>) | Skor: <code>{sig.score:.0f}/100</code> | Regime: <code>{sig.regime}</code>\n"
            f"🛡 Shield: BE @ +0.75R & TP1 | 🕒 <code>{utc_now().strftime('%H:%M UTC')}</code>\n"
            f"🧠 Edge: P(Win) <code>{sig.p_win:.0f}%</code> | EV <code>{sig.ev_r:+.2f}R</code> | KER <code>{sig.ker:.2f}</code> | <code>{sig.macro_ctx}</code>"
        )


class TrailingStopManager:
    def __init__(self, state: StateManager, notifier: Notifier) -> None:
        self.state = state
        self.notifier = notifier

    def update_active_trades(
        self, df_5m: pd.DataFrame, df_15m: pd.DataFrame, live_mid: float = 0.0, news_active: bool = False
    ) -> List[str]:
        active = self.state.get_active_signals()
        if not active or df_5m.empty:
            return []
        events: List[str] = []
        latest_price = live_mid if live_mid > 100.0 else float(df_5m["Close"].iloc[-1])

        for tr in active:
            sig_id, mode, direction, status = int(tr["id"]), tr["mode"], tr["direction"], tr["status"]
            cfg = SCALP_CONFIG if mode == "SCALP" else SWING_CONFIG
            df_ref = df_5m if mode == "SCALP" else df_15m
            if df_ref.empty:
                df_ref = df_5m

            since_df = df_ref[df_ref.index >= datetime.fromisoformat(tr["created_at"])]
            high_since = max(float(since_df["High"].max()), latest_price) if not since_df.empty else latest_price
            low_since = min(float(since_df["Low"].min()), latest_price) if not since_df.empty else latest_price

            cur_sl, entry, init_sl = float(tr["current_sl"]), float(tr["entry"]), float(tr["initial_sl"])
            tp1, tp2, tp3, atr = float(tr["tp1"]), float(tr["tp2"]), float(tr["tp3"]), float(tr["atr"])
            sl_dist = abs(entry - init_sl)
            be_trigger = 0.75 * sl_dist

            if direction == "BUY":
                if low_since <= cur_sl:
                    cs = "CLOSED_BE" if cur_sl >= entry else "CLOSED_SL"
                    self.state.update_signal_trail(sig_id, cur_sl, cs)
                    msg = f"🛑 <b>XAUUSD | {mode} BUY #{sig_id} {cs}</b> @ <code>{cur_sl:.2f}</code> (Spot: <code>{latest_price:.2f}</code>)"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue
                if high_since >= tp3:
                    self.state.update_signal_trail(sig_id, tp2, "CLOSED_TP3")
                    msg = f"🏆 <b>XAUUSD | {mode} BUY #{sig_id} FULL TP3 HIT</b> @ <code>{tp3:.2f}</code>!"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                new_sl, new_status = cur_sl, status
                if high_since >= tp2 and status != "TP2_TRAIL":
                    new_status = "TP2_TRAIL"
                    new_sl = max(cur_sl, tp1, latest_price - atr * cfg.trail_atr_mult)
                elif high_since >= tp1 and status in ("OPEN", "BE_LOCKED"):
                    new_status = "TP1_TRAIL"
                    new_sl = max(cur_sl, entry + cfg.be_buffer_usd)
                elif (high_since >= (entry + be_trigger) or (news_active and latest_price >= entry + 0.40 * sl_dist)) and status == "OPEN":
                    new_status = "BE_LOCKED"
                    new_sl = max(cur_sl, entry + cfg.be_buffer_usd)
                elif status == "TP2_TRAIL" or (mode == "SWING" and status == "TP1_TRAIL"):
                    new_sl = max(cur_sl, latest_price - atr * cfg.trail_atr_mult)

                if new_status != status or new_sl > cur_sl + 0.15:
                    self.state.update_signal_trail(sig_id, new_sl, new_status)
                    msg = f"🛡 <b>SHIELD | {mode} BUY #{sig_id}</b> Status: <code>{new_status}</code> | SL: <code>{new_sl:.2f}</code> | Spot: <code>{latest_price:.2f}</code>"
                    events.append(msg)
                    self.notifier.broadcast(msg)
            else:
                if high_since >= cur_sl:
                    cs = "CLOSED_BE" if cur_sl <= entry else "CLOSED_SL"
                    self.state.update_signal_trail(sig_id, cur_sl, cs)
                    msg = f"🛑 <b>XAUUSD | {mode} SELL #{sig_id} {cs}</b> @ <code>{cur_sl:.2f}</code> (Spot: <code>{latest_price:.2f}</code>)"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue
                if low_since <= tp3:
                    self.state.update_signal_trail(sig_id, tp2, "CLOSED_TP3")
                    msg = f"🏆 <b>XAUUSD | {mode} SELL #{sig_id} FULL TP3 HIT</b> @ <code>{tp3:.2f}</code>!"
                    events.append(msg)
                    self.notifier.broadcast(msg)
                    continue

                new_sl, new_status = cur_sl, status
                if low_since <= tp2 and status != "TP2_TRAIL":
                    new_status = "TP2_TRAIL"
                    new_sl = min(cur_sl, tp1, latest_price + atr * cfg.trail_atr_mult)
                elif low_since <= tp1 and status in ("OPEN", "BE_LOCKED"):
                    new_status = "TP1_TRAIL"
                    new_sl = min(cur_sl, entry - cfg.be_buffer_usd)
                elif (low_since <= (entry - be_trigger) or (news_active and latest_price <= entry - 0.40 * sl_dist)) and status == "OPEN":
                    new_status = "BE_LOCKED"
                    new_sl = min(cur_sl, entry - cfg.be_buffer_usd)
                elif status == "TP2_TRAIL" or (mode == "SWING" and status == "TP1_TRAIL"):
                    new_sl = min(cur_sl, latest_price + atr * cfg.trail_atr_mult)

                if new_status != status or new_sl < cur_sl - 0.15:
                    self.state.update_signal_trail(sig_id, new_sl, new_status)
                    msg = f"🛡 <b>SHIELD | {mode} SELL #{sig_id}</b> Status: <code>{new_status}</code> | SL: <code>{new_sl:.2f}</code> | Spot: <code>{latest_price:.2f}</code>"
                    events.append(msg)
                    self.notifier.broadcast(msg)
        return events


# ==============================================================================
# FORENSIC AUDIT LOGGER & MAIN ORCHESTRATOR
# ==============================================================================
def forensic_audit_log(
    sources: Dict[str, str], live_quote: Dict[str, Any], intermarket: Dict[str, Any], news_info: Tuple[bool, str],
    scalp_stats: Dict[str, Any], swing_stats: Dict[str, Any],
    scalp_sig: SignalCandidate, swing_sig: SignalCandidate, active_trades: List[Dict[str, Any]],
) -> None:
    lines = [
        "=" * 86,
        "APEX-QUANT-OMNI — HYPERION DUAL-TIER PRECISION FORENSIC AUDIT LOG",
        "=" * 86,
        f"Timestamp UTC     : {utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')} | TierFilter={SIGNAL_TIER_MODE}",
        f"Data Feeds        : {json.dumps(sources)}",
        f"Live Spot Cross   : {json.dumps(live_quote)}",
        f"Macro & News      : {intermarket.get('tag')} | TNX_Z={intermarket.get('tnx_z')} | {news_info[1]}",
        "-" * 86,
        (
            f"[SCALP PRO]       : Bars={scalp_stats.get('bt_bars')} | ATR={scalp_stats.get('opt_atr_period')} | Z={scalp_stats.get('opt_z_thresh')} | "
            f"Trades={scalp_stats.get('trades')} | WR={scalp_stats.get('win_rate')}% | NonLoss={scalp_stats.get('non_loss_rate')}% | "
            f"PF={scalp_stats.get('profit_factor')} | OOS={scalp_stats.get('oos_win_rate')}%/{scalp_stats.get('oos_trades')}T (NonLoss={scalp_stats.get('oos_non_loss_rate')}%, EV={scalp_stats.get('oos_ev_r'):+.2f}R) | "
            f"MaxDD={scalp_stats.get('max_dd_pct')}% | Ret={scalp_stats.get('total_return_pct'):+.2f}%"
        ),
        (
            f"[SCALP SNIPER 82+]: Trades={scalp_stats.get('sniper_trades')} | WR={scalp_stats.get('sniper_wr')}% | "
            f"NonLoss={scalp_stats.get('sniper_non_loss')}% | PF={scalp_stats.get('sniper_pf')} | Ret={scalp_stats.get('sniper_ret'):+.2f}%"
        ),
        (
            f"[SWING PRO]       : Bars={swing_stats.get('bt_bars')} | ATR={swing_stats.get('opt_atr_period')} | Z={swing_stats.get('opt_z_thresh')} | "
            f"Trades={swing_stats.get('trades')} | WR={swing_stats.get('win_rate')}% | NonLoss={swing_stats.get('non_loss_rate')}% | "
            f"PF={swing_stats.get('profit_factor')} | OOS={swing_stats.get('oos_win_rate')}%/{swing_stats.get('oos_trades')}T (NonLoss={swing_stats.get('oos_non_loss_rate')}%, EV={swing_stats.get('oos_ev_r'):+.2f}R) | "
            f"MaxDD={swing_stats.get('max_dd_pct')}% | Ret={swing_stats.get('total_return_pct'):+.2f}%"
        ),
        (
            f"[SWING SNIPER 83+]: Trades={swing_stats.get('sniper_trades')} | WR={swing_stats.get('sniper_wr')}% | "
            f"NonLoss={swing_stats.get('sniper_non_loss')}% | PF={swing_stats.get('sniper_pf')} | Ret={swing_stats.get('sniper_ret'):+.2f}%"
        ),
        "-" * 86,
        f"[LIVE SCALP]      : {scalp_sig.direction} ({scalp_sig.tier}) | Px={scalp_sig.entry} | Score={scalp_sig.score} | Regime={scalp_sig.regime} | KER={scalp_sig.ker} | ADX={scalp_sig.adx} | Reason={scalp_sig.reason}",
        f"[LIVE SWING]      : {swing_sig.direction} ({swing_sig.tier}) | Px={swing_sig.entry} | Score={swing_sig.score} | Regime={swing_sig.regime} | KER={swing_sig.ker} | ADX={swing_sig.adx} | Reason={swing_sig.reason}",
        f"[ACTIVE TRADES]   : {len(active_trades)} active position(s)",
    ]
    for t in active_trades:
        lines.append(f"   -> #{t['id']} {t['mode']} {t['direction']} Entry={t['entry']} CurSL={t['current_sl']} Status={t['status']}")
    lines.append("=" * 86)
    print("\n".join(lines), flush=True)


def format_on_demand_check(sig: SignalCandidate, bt_stats: Dict[str, Any]) -> str:
    if sig.direction in ("BUY", "SELL"):
        return SignalEngine.format_telegram_signal(sig)
    return (
        f"⏸ <b>XAUUSD | {sig.mode} | NO_SIGNAL</b>\n"
        f"💵 Spot: <code>{sig.entry:.2f}</code> | Skor: <code>{sig.score:.0f}</code> | Regime: <code>{sig.regime}</code>\n"
        f"🛡 Filter: <code>{sig.reason}</code> | NonLoss OOS: <code>{bt_stats.get('oos_non_loss_rate', 0):.1f}%</code>"
    )


def format_status_message(
    state: StateManager, spot_px: float, scalp_stats: Dict[str, Any], swing_stats: Dict[str, Any],
    intermarket: Dict[str, Any], news_info: Tuple[bool, str],
) -> str:
    active = state.get_active_signals()
    _, sess_tag = is_market_open()
    peak_tag = "LONDON/NY" if is_peak_london_ny_session(utc_now()) else "OFF-PEAK"
    lines = [
        f"📊 <b>APEX-QUANT STATUS</b> | 🕒 <code>{utc_now().strftime('%H:%M UTC')}</code>",
        f"💵 Spot: <code>{spot_px:.2f}</code> | <code>{sess_tag} ({peak_tag})</code> | <code>{intermarket.get('tag')}</code>",
        f"📰 News: <code>{news_info[1]}</code> | Tier: <code>{SIGNAL_TIER_MODE}</code>",
        f"⚡ Scalp: OOS NonLoss <code>{scalp_stats.get('oos_non_loss_rate', 0):.1f}%</code> | Sniper PF <code>{scalp_stats.get('sniper_pf', 0):.2f}</code>",
        f"🌊 Swing: OOS NonLoss <code>{swing_stats.get('oos_non_loss_rate', 0):.1f}%</code> | Sniper PF <code>{swing_stats.get('sniper_pf', 0):.2f}</code>",
        f"📌 Posisi Aktif: <code>{len(active)}</code>",
    ]
    for tr in active[:4]:
        sl_dist = abs(float(tr["entry"]) - float(tr["initial_sl"])) + 1e-8
        float_r = ((spot_px - float(tr["entry"])) if tr["direction"] == "BUY" else (float(tr["entry"]) - spot_px)) / sl_dist
        lines.append(
            f"• #{tr['id']} <b>{tr['mode']} {tr['direction']}</b> @ <code>{tr['entry']:.2f}</code> | "
            f"SL: <code>{tr['current_sl']:.2f}</code> | PnL: <code>{float_r:+.2f}R</code> | <code>{tr['status']}</code>"
        )
    return "\n".join(lines)


def format_history_message(state: StateManager) -> str:
    recent = state.get_recent_signals(limit=5)
    if not recent:
        return "📜 <b>APEX HISTORY</b>\nBelum ada sinyal terekam di database."
    lines = ["📜 <b>APEX LAST 5 SIGNALS</b>"]
    for r in recent:
        lines.append(
            f"• #{r['id']} <b>{r['mode']} {r['direction']}</b> @ <code>{r['entry']:.2f}</code> | SL <code>{r['current_sl']:.2f}</code> | <code>{r['status']}</code>"
        )
    return "\n".join(lines)


def ensure_fallback_chart_if_missing(out_path: str = CHART_PATH) -> None:
    if os.path.exists(out_path):
        return
    empty_st = {
        "trades": 0, "win_rate": 0.0, "non_loss_rate": 0.0, "profit_factor": 0.0,
        "oos_win_rate": 0.0, "oos_non_loss_rate": 0.0, "oos_trades": 0,
        "sniper_trades": 0, "sniper_wr": 0.0, "sniper_non_loss": 0.0, "sniper_pf": 0.0,
        "total_return_pct": 0.0, "equity_curve": [],
    }
    BacktestAndOptimizer.generate_performance_chart(empty_st, empty_st, out_path)


def should_emit_tier(sig: SignalCandidate, tier_mode: str) -> bool:
    if sig.direction not in ("BUY", "SELL"):
        return False
    if tier_mode == "SNIPER_ONLY":
        return sig.tier == "SNIPER"
    if tier_mode == "PRO_ONLY":
        return sig.tier in ("PRO", "SNIPER")
    return True


def main() -> int:
    state = StateManager(DB_PATH)
    notifier = Notifier(state)

    try:
        cb_open, cb_until = state.is_circuit_open()
        pending_cmds = notifier.poll_commands()
        manual_cmd = os.getenv("BOT_COMMAND", "auto").strip().lower()

        if cb_open and manual_cmd == "auto" and not pending_cmds:
            logger.warning(f"Circuit Breaker OPEN until {cb_until}; skipping.")
            ensure_fallback_chart_if_missing()
            return 0

        market_open, _ = is_market_open()
        news_info = MacroAndNewsGuard.check_usd_news_blackout()
        intermarket = MacroAndNewsGuard.fetch_intermarket_bias()

        data_engine = MarketDataEngine()
        live_quote = data_engine.fetch_live_spot_quote()
        live_spread = float(live_quote.get("spread", SPREAD_USD) or SPREAD_USD)
        live_mid = float(live_quote.get("mid", 0.0) or 0.0)

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

        spot_px = live_mid if live_mid > 100.0 else float(df_5m["Close"].iloc[-1])

        trail_mgr = TrailingStopManager(state, notifier)
        if market_open:
            trail_mgr.update_active_trades(df_5m, df_15m, live_mid=live_mid, news_active=news_info[0])

        deep_hist = data_engine.load_deep_calibration_history(spot_reference=spot_px)
        bt_5m, _ = deep_hist.get("5m", (pd.DataFrame(), "NONE"))
        bt_15m, _ = deep_hist.get("15m", (pd.DataFrame(), "NONE"))
        bt_1h, _ = deep_hist.get("1h", (pd.DataFrame(), "NONE"))
        bt_4h, _ = deep_hist.get("4h", (pd.DataFrame(), "NONE"))

        _, opt_z_scalp, df_scalp_enr, scalp_stats, scalp_cal = BacktestAndOptimizer.grid_search_optimize(
            bt_5m, bt_15m, bt_1h, df_5m, df_15m, df_1h, SCALP_CONFIG, spread_usd=live_spread
        )
        _, opt_z_swing, df_swing_enr, swing_stats, swing_cal = BacktestAndOptimizer.grid_search_optimize(
            bt_15m, bt_1h, bt_4h, df_15m, df_1h, df_4h, SWING_CONFIG, spread_usd=live_spread
        )

        BacktestAndOptimizer.generate_performance_chart(scalp_stats, swing_stats, CHART_PATH)

        if ok_5m and ok_15m:
            scalp_sig = SignalEngine.build_live_signal(
                df_scalp_enr, SCALP_CONFIG, opt_z_scalp, scalp_stats, scalp_cal,
                intermarket, news_info, live_spread_usd=live_spread, live_mid=live_mid,
            )
        else:
            scalp_sig = SignalCandidate(
                mode="SCALP", direction="NO_SIGNAL", entry=round(spot_px, 2),
                regime="DATA_GUARD", reason=f"5M:{msg_5m}|15M:{msg_15m}",
            )

        if ok_15m and ok_1h:
            swing_sig = SignalEngine.build_live_signal(
                df_swing_enr, SWING_CONFIG, opt_z_swing, swing_stats, swing_cal,
                intermarket, news_info, live_spread_usd=live_spread, live_mid=live_mid,
            )
        else:
            swing_sig = SignalCandidate(
                mode="SWING", direction="NO_SIGNAL", entry=round(spot_px, 2),
                regime="DATA_GUARD", reason=f"15M:{msg_15m}|1H:{msg_1h}",
            )

        if manual_cmd in ("check", "scalp", "swing", "status", "chart", "history", "sniper"):
            target = notifier.authorized_chat_ids[0] if notifier.authorized_chat_ids else None
            pending_cmds.append((target or "", f"/{manual_cmd}"))

        for chat_id, raw_cmd in pending_cmds:
            cmd = raw_cmd.lower().strip()
            target_cid = chat_id if chat_id else None
            if cmd.startswith("/status"):
                notifier.broadcast(
                    format_status_message(state, spot_px, scalp_stats, swing_stats, intermarket, news_info),
                    target_chat_id=target_cid,
                )
            elif cmd.startswith("/chart"):
                caption = (
                    f"📈 <b>APEX Dual-Tier Walk-Forward</b>\n"
                    f"SCALP: Pro <code>{scalp_stats.get('total_return_pct',0):+.1f}%</code> | Sniper <code>{scalp_stats.get('sniper_ret',0):+.1f}%</code>\n"
                    f"SWING: Pro <code>{swing_stats.get('total_return_pct',0):+.1f}%</code> | Sniper <code>{swing_stats.get('sniper_ret',0):+.1f}%</code>"
                )
                notifier.send_telegram_photo(CHART_PATH, caption=caption, chat_id=target_cid)
            elif cmd.startswith("/history"):
                notifier.broadcast(format_history_message(state), target_chat_id=target_cid)
            elif cmd.startswith("/sniper"):
                s_msg = (
                    f"💎 <b>SNIPER MODE SNAPSHOT</b> (<code>{intermarket.get('tag')}</code>)\n"
                    f"• SCALP: <code>{scalp_sig.direction} ({scalp_sig.tier})</code> Skor <code>{scalp_sig.score:.0f}/82</code> | {scalp_sig.reason}\n"
                    f"• SWING: <code>{swing_sig.direction} ({swing_sig.tier})</code> Skor <code>{swing_sig.score:.0f}/83</code> | {swing_sig.reason}"
                )
                notifier.broadcast(s_msg, target_chat_id=target_cid)
            elif cmd.startswith("/help"):
                notifier.broadcast(
                    "🤖 <b>APEX-QUANT COMMANDS</b>\n"
                    "<code>/check</code> — Analisa SCALP & SWING\n"
                    "<code>/sniper</code> — Cek Peluang Sniper A+\n"
                    "<code>/status</code> — Status Posisi, DXY & News\n"
                    "<code>/chart</code> — Grafik Dual-Tier OOS\n"
                    "<code>/history</code> — 5 Sinyal Terakhir",
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

        for sig, cfg in [(scalp_sig, SCALP_CONFIG), (swing_sig, SWING_CONFIG)]:
            if should_emit_tier(sig, SIGNAL_TIER_MODE):
                blocked, block_reason = state.is_duplicate_or_cooldown(
                    sig.mode, sig.direction, sig.candle_ts, cfg.cooldown_minutes, sig.score
                )
                if blocked:
                    logger.info(f"[{sig.mode}] Signal suppressed: {block_reason}")
                else:
                    sig_id = state.insert_signal(sig)
                    logger.info(f"[{sig.mode}] Emitted #{sig_id} [{sig.tier}] {sig.direction} @ {sig.entry}")
                    notifier.broadcast(SignalEngine.format_telegram_signal(sig))
            elif sig.direction in ("BUY", "SELL"):
                logger.info(f"[{sig.mode}] Tier {sig.tier} skipped by SIGNAL_TIER_MODE={SIGNAL_TIER_MODE}")

        state.record_success()
        forensic_audit_log(
            sources, live_quote, intermarket, news_info,
            scalp_stats, swing_stats, scalp_sig, swing_sig, state.get_active_signals()
        )
        return 0

    except Exception as exc:
        failures, _ = state.record_failure()
        tb_short = traceback.format_exc()[-500:]
        logger.exception("Fatal error in APEX-QUANT-OMNI main loop")
        notifier.broadcast(
            f"🚨 <b>APEX-QUANT SYSTEM ERROR</b> (Fails: <code>{failures}</code>)\n"
            f"<code>{str(exc)[:180]}</code>\n<code>{tb_short[:300]}</code>"
        )
        ensure_fallback_chart_if_missing()
        return 1


if __name__ == "__main__":
    sys.exit(main())
