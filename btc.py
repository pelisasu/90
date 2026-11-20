import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

EPS = 1e-8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("APEX-HYPERION")


@dataclass(frozen=True)
class ModeSpec:
    name: str
    trigger_tf: str
    confirm_tf: str
    anchor_tf: str
    sl_atr_mult: float
    tp1_r: float
    tp2_r: float
    tp3_r: float
    min_score: float
    min_confluence: int
    cooldown_sec: int
    trail_rule: str


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    yf_ticker: str
    coinbase_product: str
    kraken_pair: str
    kucoin_symbol: str
    contract_size: float
    pip_size: float
    price_offset: float
    default_spread: float


@dataclass
class SignalDecision:
    symbol: str
    mode: str
    action: str  # BUY, SELL, NO_SIGNAL
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    lot: float
    score: float
    regime: str
    rel_strength: str
    trail_desc: str
    reasons: List[str]
    timestamp_utc: str


@dataclass
class BacktestSummary:
    symbol: str
    mode: str
    best_atr: int
    best_z: float
    trades: int
    win_rate: float
    profit_factor: float
    max_dd_pct: float
    net_return_pct: float
    equity_curve: List[float]


ASSETS: Dict[str, AssetSpec] = {
    "BTCUSD": AssetSpec(
        symbol="BTCUSD",
        yf_ticker="BTC-USD",
        coinbase_product="BTC-USD",
        kraken_pair="XXBTZUSD",
        kucoin_symbol="BTC-USDT",
        contract_size=float(os.getenv("BTC_CONTRACT_SIZE", "1.0")),
        pip_size=float(os.getenv("BTC_PIP_SIZE", "0.01")),
        price_offset=float(os.getenv("BTC_PRICE_OFFSET", "0.0")),
        default_spread=float(os.getenv("BTC_DEFAULT_SPREAD", "15.0")),
    ),
    "ETHUSD": AssetSpec(
        symbol="ETHUSD",
        yf_ticker="ETH-USD",
        coinbase_product="ETH-USD",
        kraken_pair="XETHZUSD",
        kucoin_symbol="ETH-USDT",
        contract_size=float(os.getenv("ETH_CONTRACT_SIZE", "1.0")),
        pip_size=float(os.getenv("ETH_PIP_SIZE", "0.01")),
        price_offset=float(os.getenv("ETH_PRICE_OFFSET", "0.0")),
        default_spread=float(os.getenv("ETH_DEFAULT_SPREAD", "0.50")),
    ),
}

MODES: Dict[str, ModeSpec] = {
    "SCALP": ModeSpec(
        name="SCALP",
        trigger_tf="5m",
        confirm_tf="15m",
        anchor_tf="1h",
        sl_atr_mult=1.25,
        tp1_r=1.2,
        tp2_r=2.0,
        tp3_r=3.0,
        min_score=82.0,
        min_confluence=4,
        cooldown_sec=1800,
        trail_rule="Aktif @ TP1 (BE)",
    ),
    "SWING": ModeSpec(
        name="SWING",
        trigger_tf="1h",
        confirm_tf="4h",
        anchor_tf="1d",
        sl_atr_mult=1.80,
        tp1_r=1.5,
        tp2_r=2.5,
        tp3_r=4.0,
        min_score=80.0,
        min_confluence=4,
        cooldown_sec=7200,
        trail_rule="Aktif @ TP1 (BE+0.5ATR)",
    ),
}

TF_SECONDS: Dict[str, int] = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class CircuitBreaker:
    def __init__(self, max_failures: int = 4, reset_after_sec: int = 600):
        self.max_failures = max_failures
        self.reset_after_sec = reset_after_sec
        self.failures: Dict[str, int] = {}
        self.opened_at: Dict[str, float] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        opened = self.opened_at.get(key)
        if opened is not None:
            if now - opened >= self.reset_after_sec:
                self.failures[key] = 0
                del self.opened_at[key]
                return True
            return False
        return True

    def record_success(self, key: str) -> None:
        self.failures[key] = 0
        if key in self.opened_at:
            del self.opened_at[key]

    def record_failure(self, key: str) -> None:
        count = self.failures.get(key, 0) + 1
        self.failures[key] = count
        if count >= self.max_failures and key not in self.opened_at:
            self.opened_at[key] = time.monotonic()
            logger.warning("CircuitBreaker OPEN untuk '%s' setelah %d kegagalan.", key, count)


class ResilientHttpClient:
    def __init__(self, breaker: CircuitBreaker, timeout: int = 12):
        self.breaker = breaker
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "APEX-QUANT-OMNI-HYPERION/1.0"})

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None, service_key: str = "http") -> Optional[Any]:
        if not self.breaker.allow(service_key):
            return None
        backoffs = [2.0, 4.0, 8.0]
        for attempt, wait_s in enumerate(backoffs, start=1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    logger.warning("HTTP 429 pada %s (percobaan %d), backoff %.1fs", service_key, attempt, wait_s)
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                self.breaker.record_success(service_key)
                return resp.json()
            except Exception as exc:
                logger.warning("Permintaan gagal %s (%s), percobaan %d: %s", service_key, url, attempt, exc)
                if attempt < len(backoffs):
                    time.sleep(wait_s)
        self.breaker.record_failure(service_key)
        return None

    def post_json(self, url: str, payload: Dict[str, Any], service_key: str = "http_post") -> Optional[Dict[str, Any]]:
        if not self.breaker.allow(service_key):
            return None
        backoffs = [2.0, 4.0, 8.0]
        for attempt, wait_s in enumerate(backoffs, start=1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 429:
                    logger.warning("HTTP 429 POST %s (percobaan %d), tidur %.1fs", service_key, attempt, wait_s)
                    time.sleep(wait_s)
                    continue
                resp.raise_for_status()
                self.breaker.record_success(service_key)
                data = resp.json()
                return data if isinstance(data, dict) else {"ok": True}
            except Exception as exc:
                logger.warning("POST gagal %s (percobaan %d): %s", service_key, attempt, exc)
                if attempt < len(backoffs):
                    time.sleep(wait_s)
        self.breaker.record_failure(service_key)
        return None


def flatten_and_clean_ohlcv(df: pd.DataFrame, tf: str, offset: float = 0.0) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = work.columns.get_level_values(0)
    work.columns = [str(c).strip().lower() for c in work.columns]

    rename_map = {}
    if "datetime" in work.columns and "timestamp" not in work.columns:
        rename_map["datetime"] = "timestamp"
    elif "date" in work.columns and "timestamp" not in work.columns:
        rename_map["date"] = "timestamp"
    if rename_map:
        work = work.rename(columns=rename_map)

    if "timestamp" not in work.columns:
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.reset_index().rename(columns={work.index.name or "index": "timestamp"})
        else:
            return None

    required = ["timestamp", "open", "high", "low", "close"]
    if not all(col in work.columns for col in required):
        return None
    if "volume" not in work.columns:
        work["volume"] = 0.0

    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["timestamp", "open", "high", "low", "close"])
    work["volume"] = work["volume"].fillna(0.0)
    work = work[work["close"] > 0.0]
    work = work[work["high"] >= work["low"]]
    work = work.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    if len(work) < 60:
        return None

    # Buang candle yang masih terbuka (belum selesai membentuk periode timeframe)
    tf_sec = TF_SECONDS.get(tf, 300)
    now_utc = datetime.now(timezone.utc)
    last_open = work["timestamp"].iloc[-1].to_pydatetime()
    if last_open.tzinfo is None:
        last_open = last_open.replace(tzinfo=timezone.utc)
    if (now_utc - last_open).total_seconds() < tf_sec:
        work = work.iloc[:-1].copy().reset_index(drop=True)

    if len(work) < 60:
        return None

    # Deteksi stale data (baris terakhir terlalu tua dari waktu saat ini)
    latest_bar = work["timestamp"].iloc[-1].to_pydatetime()
    if latest_bar.tzinfo is None:
        latest_bar = latest_bar.replace(tzinfo=timezone.utc)
    max_stale_sec = max(tf_sec * 4, 900)
    if (now_utc - latest_bar).total_seconds() > max_stale_sec:
        logger.warning("Data stale terdeteksi untuk tf=%s: last_bar=%s now=%s", tf, latest_bar, now_utc)
        return None

    # Sanity & spike filter: blokir data dengan lonjakan ekstrem tak wajar (>35% dalam 1 bar)
    ret = work["close"].pct_change().abs().fillna(0.0)
    if float(ret.tail(30).max()) > 0.35:
        logger.warning("Spike anomali ekstrem (>35%%) terdeteksi pada tf=%s.", tf)
        return None

    if abs(offset) > 0.0:
        for col in ["open", "high", "low", "close"]:
            work[col] = work[col] + offset

    return work[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


class MarketDataProvider:
    def __init__(self, http_client: ResilientHttpClient):
        self.http = http_client

    def _from_coinbase(self, asset: AssetSpec, tf: str, limit: int) -> Optional[pd.DataFrame]:
        gran_map = {"5m": 300, "15m": 900, "1h": 3600, "4h": 21600, "1d": 86400}
        gran = gran_map.get(tf)
        if not gran:
            return None
        url = f"https://api.exchange.coinbase.com/products/{asset.coinbase_product}/candles"
        raw = self.http.get_json(url, params={"granularity": gran}, service_key="coinbase")
        if not isinstance(raw, list) or not raw:
            return None
        rows = []
        for item in raw[:limit]:
            if isinstance(item, (list, tuple)) and len(item) >= 6:
                rows.append(
                    {
                        "timestamp": int(item[0]),
                        "low": float(item[1]),
                        "high": float(item[2]),
                        "open": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                    }
                )
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return flatten_and_clean_ohlcv(df, tf, asset.price_offset)

    def _from_kraken(self, asset: AssetSpec, tf: str, limit: int) -> Optional[pd.DataFrame]:
        interval_map = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        interval = interval_map.get(tf)
        if not interval:
            return None
        url = "https://api.kraken.com/0/public/OHLC"
        data = self.http.get_json(url, params={"pair": asset.kraken_pair, "interval": interval}, service_key="kraken")
        if not isinstance(data, dict) or data.get("error"):
            return None
        result = data.get("result") or {}
        series_key = next((k for k in result.keys() if k != "last"), None)
        if not series_key or not isinstance(result.get(series_key), list):
            return None
        rows = []
        for item in result[series_key][-limit:]:
            if isinstance(item, (list, tuple)) and len(item) >= 7:
                rows.append(
                    {
                        "timestamp": int(item[0]),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[6]),
                    }
                )
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return flatten_and_clean_ohlcv(df, tf, asset.price_offset)

    def _from_kucoin(self, asset: AssetSpec, tf: str, limit: int) -> Optional[pd.DataFrame]:
        tf_map = {"5m": "5min", "15m": "15min", "1h": "1hour", "4h": "4hour", "1d": "1day"}
        k_tf = tf_map.get(tf)
        if not k_tf:
            return None
        url = "https://api.kucoin.com/api/v1/market/candles"
        data = self.http.get_json(url, params={"symbol": asset.kucoin_symbol, "type": k_tf}, service_key="kucoin")
        if not isinstance(data, dict) or data.get("code") != "200000" or not isinstance(data.get("data"), list):
            return None
        rows = []
        for item in data["data"][:limit]:
            if isinstance(item, (list, tuple)) and len(item) >= 6:
                rows.append(
                    {
                        "timestamp": int(float(item[0])),
                        "open": float(item[1]),
                        "close": float(item[2]),
                        "high": float(item[3]),
                        "low": float(item[4]),
                        "volume": float(item[5]),
                    }
                )
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return flatten_and_clean_ohlcv(df, tf, asset.price_offset)

    def _from_yfinance(self, asset: AssetSpec, tf: str, limit: int) -> Optional[pd.DataFrame]:
        yf_map = {
            "5m": ("5m", "15d"),
            "15m": ("15m", "30d"),
            "1h": ("60m", "90d"),
            "4h": ("1h", "120d"),
            "1d": ("1d", "365d"),
        }
        interval, period = yf_map.get(tf, ("1h", "90d"))
        try:
            raw = yf.download(
                tickers=asset.yf_ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if raw is None or raw.empty:
                return None
            df = flatten_and_clean_ohlcv(raw, tf, asset.price_offset)
            if df is not None and tf == "4h":
                df = (
                    df.set_index("timestamp")
                    .resample("4h")
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                    .dropna(subset=["open", "high", "low", "close"])
                    .reset_index()
                )
                df = flatten_and_clean_ohlcv(df, "4h", 0.0)
            if df is not None and len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
            return df
        except Exception as exc:
            logger.warning("yfinance gagal untuk %s (%s): %s", asset.symbol, tf, exc)
            return None

    def fetch(self, symbol: str, tf: str, limit: int = 250) -> Tuple[Optional[pd.DataFrame], str]:
        asset = ASSETS[symbol]
        providers = [
            ("Coinbase", lambda: self._from_coinbase(asset, tf, limit)),
            ("Kraken", lambda: self._from_kraken(asset, tf, limit)),
            ("KuCoin", lambda: self._from_kucoin(asset, tf, limit)),
            ("yfinance", lambda: self._from_yfinance(asset, tf, limit)),
        ]
        for name, fn in providers:
            df = fn()
            if df is not None and len(df) >= 60:
                return df.tail(limit).reset_index(drop=True), name
        return None, "NONE"


class QuantFeatureEngine:
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1.0 / max(period, 1), adjust=False).mean()

    @staticmethod
    def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
        mean = series.rolling(window=window, min_periods=window).mean()
        std = series.rolling(window=window, min_periods=window).std().fillna(0.0)
        return (series - mean) / (std + EPS)

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        up = df["high"].diff()
        down = -df["low"].diff()
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
        atr_s = QuantFeatureEngine.atr(df, period=period)
        plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr_s + EPS)
        minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr_s + EPS)
        dx = 100.0 * (plus_di - minus_di).abs() / ((plus_di + minus_di).abs() + EPS)
        return dx.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0.0)

    @staticmethod
    def detect_smc_structure(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
        """Pilar I: Swing High/Low, Break of Structure (BOS), Change of Character (ChoCh)."""
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        n = len(df)

        swing_highs: List[Tuple[int, float]] = []
        swing_lows: List[Tuple[int, float]] = []

        for i in range(lookback, n - lookback):
            win_h = highs[i - lookback : i + lookback + 1]
            win_l = lows[i - lookback : i + lookback + 1]
            if highs[i] == np.max(win_h):
                swing_highs.append((i, float(highs[i])))
            if lows[i] == np.min(win_l):
                swing_lows.append((i, float(lows[i])))

        last_close = float(closes[-1])
        prev_close = float(closes[-2])
        last_sh = swing_highs[-1][1] if swing_highs else float(np.max(highs[-20:]))
        prev_sh = swing_highs[-2][1] if len(swing_highs) >= 2 else last_sh
        last_sl = swing_lows[-1][1] if swing_lows else float(np.min(lows[-20:]))
        prev_sl = swing_lows[-2][1] if len(swing_lows) >= 2 else last_sl

        # Tentukan bias struktur mayor berdasarkan urutan swing terakhir
        if last_sh >= prev_sh and last_sl >= prev_sl:
            major_bias = "BULLISH"
        elif last_sh <= prev_sh and last_sl <= prev_sl:
            major_bias = "BEARISH"
        else:
            major_bias = "NEUTRAL"

        bos_bull = last_close > last_sh and prev_close <= last_sh
        bos_bear = last_close < last_sl and prev_close >= last_sl
        choch_bull = major_bias == "BEARISH" and last_close > last_sh
        choch_bear = major_bias == "BULLISH" and last_close < last_sl

        return {
            "major_bias": major_bias,
            "last_sh": last_sh,
            "last_sl": last_sl,
            "bos_bull": bos_bull,
            "bos_bear": bos_bear,
            "choch_bull": choch_bull,
            "choch_bear": choch_bear,
        }

    @staticmethod
    def order_flow_microstructure(df: pd.DataFrame) -> Dict[str, float]:
        """Pilar II: Rasio body-to-wick berbobot volume dan deteksi wick trap / stop hunt."""
        last = df.iloc[-1]
        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        c = float(last["close"])
        v = float(last["volume"])

        vol_ma = float(df["volume"].tail(20).mean())
        vol_ratio = v / (vol_ma + EPS) if vol_ma > 0 else 1.0

        total_range = max(h - l, EPS)
        body = abs(c - o)
        upper_wick = max(0.0, h - max(o, c))
        lower_wick = max(0.0, min(o, c) - l)

        body_ratio = body / total_range
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        vol_weighted_body = body_ratio * min(max(vol_ratio, 0.5), 2.5)

        # Wick trap / stop hunt: ekor sangat panjang (>58% range) dibanding body
        wick_trap_bear = upper_wick_ratio > 0.58 and body_ratio < 0.25 and c <= o
        wick_trap_bull = lower_wick_ratio > 0.58 and body_ratio < 0.25 and c >= o

        return {
            "body_ratio": body_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "vol_ratio": vol_ratio,
            "vol_weighted_body": vol_weighted_body,
            "wick_trap_bear": 1.0 if wick_trap_bear else 0.0,
            "wick_trap_bull": 1.0 if wick_trap_bull else 0.0,
        }

    @staticmethod
    def regime_and_shock_filter(df: pd.DataFrame, atr_period: int = 14) -> Tuple[str, bool, float, float]:
        """Pilar V + 24/7 Regime & Volatility Shock Guard."""
        atr_s = QuantFeatureEngine.atr(df, period=atr_period)
        adx_s = QuantFeatureEngine.adx(df, period=14)
        ema_fast = df["close"].ewm(span=20, adjust=False).mean()
        ema_slow = df["close"].ewm(span=50, adjust=False).mean()

        last_tr = float((df["high"].iloc[-1] - df["low"].iloc[-1]))
        med_atr = float(atr_s.rolling(30, min_periods=10).median().iloc[-1])
        shock_ratio = last_tr / (med_atr + EPS)

        # Volatility shock mendadak: true range > 2.6x median ATR
        volatility_shock = shock_ratio > 2.60

        adx_val = float(adx_s.iloc[-1])
        fast_val = float(ema_fast.iloc[-1])
        slow_val = float(ema_slow.iloc[-1])

        if volatility_shock:
            regime = "VOLATILITY_SHOCK"
        elif adx_val >= 22.0 and fast_val > slow_val:
            regime = "TREND_BULL"
        elif adx_val >= 22.0 and fast_val < slow_val:
            regime = "TREND_BEAR"
        else:
            regime = "CHOP_RANGE"

        return regime, volatility_shock, adx_val, shock_ratio


class GridSearchOptimizer:
    """Grid search tervalidasi dengan penalti drawdown & syarat jumlah sampel minimum."""

    def __init__(self, atr_candidates: Optional[List[int]] = None, z_candidates: Optional[List[float]] = None):
        self.atr_candidates = atr_candidates or [10, 14, 20]
        self.z_candidates = z_candidates or [1.5, 2.0, 2.5]

    def optimize_and_backtest(
        self,
        symbol: str,
        mode: str,
        df: pd.DataFrame,
        spread_cost: float,
        min_trades: int = 6,
    ) -> BacktestSummary:
        closes = df["close"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        n = len(df)

        best_obj = -1e9
        best_result: Optional[BacktestSummary] = None

        for atr_p in self.atr_candidates:
            atr_arr = QuantFeatureEngine.atr(df, period=atr_p).to_numpy(dtype=float)
            ema20 = df["close"].ewm(span=20, adjust=False).mean().to_numpy(dtype=float)
            ema50 = df["close"].ewm(span=50, adjust=False).mean().to_numpy(dtype=float)

            for z_th in self.z_candidates:
                z_arr = QuantFeatureEngine.rolling_zscore(df["close"], window=20).to_numpy(dtype=float)
                trades_pnl: List[float] = []
                equity = 10000.0
                eq_curve = [equity]

                i = max(atr_p + 20, 52)
                while i < n - 3:
                    trend_up = ema20[i] > ema50[i] and closes[i] > ema20[i]
                    trend_dn = ema20[i] < ema50[i] and closes[i] < ema20[i]
                    atr_i = max(atr_arr[i], EPS)

                    # Mean-reversion pullback searah tren
                    long_cond = trend_up and z_arr[i] <= -z_th
                    short_cond = trend_dn and z_arr[i] >= z_th

                    if long_cond or short_cond:
                        direction = 1.0 if long_cond else -1.0
                        entry = closes[i] + (direction * spread_cost * 0.5)
                        sl_dist = atr_i * (1.25 if mode == "SCALP" else 1.80)
                        tp_dist = sl_dist * (1.5 if mode == "SCALP" else 2.2)

                        exit_px = closes[min(i + 3, n - 1)]
                        for j in range(i + 1, min(i + 4, n)):
                            if direction > 0:
                                if lows[j] <= entry - sl_dist:
                                    exit_px = entry - sl_dist
                                    break
                                if highs[j] >= entry + tp_dist:
                                    exit_px = entry + tp_dist
                                    break
                            else:
                                if highs[j] >= entry + sl_dist:
                                    exit_px = entry + sl_dist
                                    break
                                if lows[j] <= entry - tp_dist:
                                    exit_px = entry - tp_dist
                                    break

                        raw_pnl = direction * (exit_px - entry) - spread_cost
                        # Ukuran posisi konservatif (1% risiko per trade)
                        risk_units = min(max((equity * 0.01) / (sl_dist + EPS), 0.01), 5.0)
                        pnl_usd = raw_pnl * risk_units
                        trades_pnl.append(pnl_usd)
                        equity = max(equity + pnl_usd, 100.0)
                        eq_curve.append(equity)
                        i += 3
                    else:
                        i += 1

                t_count = len(trades_pnl)
                wins = [p for p in trades_pnl if p > 0]
                losses = [abs(p) for p in trades_pnl if p < 0]
                gross_win = float(sum(wins))
                gross_loss = float(sum(losses))
                pf = gross_win / (gross_loss + EPS) if gross_loss > 0 else (2.5 if gross_win > 0 else 0.0)
                win_rate = (len(wins) / t_count * 100.0) if t_count > 0 else 0.0

                peak = np.maximum.accumulate(np.array(eq_curve, dtype=float))
                dd_pct_series = ((peak - np.array(eq_curve, dtype=float)) / (peak + EPS)) * 100.0
                max_dd = float(np.max(dd_pct_series)) if len(dd_pct_series) > 0 else 0.0
                net_ret_pct = ((eq_curve[-1] - 10000.0) / 10000.0) * 100.0

                # Penalti sampel kecil dan drawdown
                sample_penalty = 0.0 if t_count >= min_trades else (min_trades - t_count) * 1.5
                dd_penalty = max_dd * 0.8
                obj = (net_ret_pct * 0.6) + (min(pf, 4.0) * 4.0) - dd_penalty - sample_penalty

                if obj > best_obj:
                    best_obj = obj
                    best_result = BacktestSummary(
                        symbol=symbol,
                        mode=mode,
                        best_atr=atr_p,
                        best_z=z_th,
                        trades=t_count,
                        win_rate=round(win_rate, 2),
                        profit_factor=round(min(pf, 99.99), 2),
                        max_dd_pct=round(max_dd, 2),
                        net_return_pct=round(net_ret_pct, 2),
                        equity_curve=eq_curve,
                    )

        assert best_result is not None
        return best_result


class RiskAndPositionSizer:
    @staticmethod
    def compute_lot(
        balance: float,
        risk_pct: float,
        entry: float,
        sl: float,
        asset: AssetSpec,
    ) -> float:
        stop_dist = abs(entry - sl)
        if stop_dist <= EPS or balance <= 0.0 or risk_pct <= 0.0:
            return 0.01
        risk_amount = balance * (risk_pct / 100.0)
        # Nilai kerugian per 1.00 lot untuk pergerakan stop_dist pada instrumen kripto/CFD
        value_per_lot = stop_dist * asset.contract_size
        raw_lot = risk_amount / (value_per_lot + EPS)
        stepped_lot = math.floor(raw_lot / 0.01) * 0.01
        return round(max(0.01, min(stepped_lot, 100.0)), 2)


class HyperionSignalEngine:
    def __init__(
        self,
        provider: MarketDataProvider,
        balance: float = 10000.0,
        risk_pct: float = 1.0,
    ):
        self.provider = provider
        self.balance = balance
        self.risk_pct = risk_pct

    @staticmethod
    def _relative_strength_status(
        symbol: str,
        btc_df: Optional[pd.DataFrame],
        eth_df: Optional[pd.DataFrame],
    ) -> Tuple[str, int]:
        """Augmentasi analitis: mengukur kekuatan relatif ETH vs BTC dan konsensus arah."""
        if btc_df is None or eth_df is None or len(btc_df) < 25 or len(eth_df) < 25:
            return "NEUTRAL_SOLO", 0
        btc_ret = (float(btc_df["close"].iloc[-1]) - float(btc_df["close"].iloc[-21])) / (
            float(btc_df["close"].iloc[-21]) + EPS
        )
        eth_ret = (float(eth_df["close"].iloc[-1]) - float(eth_df["close"].iloc[-21])) / (
            float(eth_df["close"].iloc[-21]) + EPS
        )
        spread_rs = (eth_ret - btc_ret) * 100.0

        if symbol == "BTCUSD":
            if btc_ret > 0.005 and spread_rs < 2.5:
                return f"BTC_LEAD_BULL({spread_rs:+.2f}%)", 1
            if btc_ret < -0.005 and spread_rs > -2.5:
                return f"BTC_LEAD_BEAR({spread_rs:+.2f}%)", 1
            if abs(spread_rs) > 4.0:
                return f"DIVERGENCE_CAUTION({spread_rs:+.2f}%)", -1
            return f"ALIGNED_NEUTRAL({spread_rs:+.2f}%)", 0

        # ETHUSD
        if eth_ret > 0.005 and btc_ret > -0.002 and spread_rs > -3.5:
            return f"ETH_CONFIRM_BULL({spread_rs:+.2f}%)", 1
        if eth_ret < -0.005 and btc_ret < 0.002 and spread_rs < 3.5:
            return f"ETH_CONFIRM_BEAR({spread_rs:+.2f}%)", 1
        if abs(spread_rs) > 4.0:
            return f"ETH_BTC_DIVERGENCE({spread_rs:+.2f}%)", -1
        return f"ETH_BALANCED({spread_rs:+.2f}%)", 0

    def evaluate(
        self,
        symbol: str,
        mode_spec: ModeSpec,
        opt_summary: BacktestSummary,
        data_bundle: Dict[str, Dict[str, Optional[pd.DataFrame]]],
        now_utc_str: str,
    ) -> SignalDecision:
        asset = ASSETS[symbol]
        df_trig = data_bundle.get(symbol, {}).get(mode_spec.trigger_tf)
        df_conf = data_bundle.get(symbol, {}).get(mode_spec.confirm_tf)
        df_anch = data_bundle.get(symbol, {}).get(mode_spec.anchor_tf)

        def no_signal(reasons: List[str], regime: str = "FILTERED", rel_str: str = "N/A", score: float = 0.0) -> SignalDecision:
            return SignalDecision(
                symbol=symbol,
                mode=mode_spec.name,
                action="NO_SIGNAL",
                entry=0.0,
                sl=0.0,
                tp1=0.0,
                tp2=0.0,
                tp3=0.0,
                lot=0.0,
                score=round(score, 1),
                regime=regime,
                rel_strength=rel_str,
                trail_desc=mode_spec.trail_rule,
                reasons=reasons,
                timestamp_utc=now_utc_str,
            )

        if df_trig is None or df_conf is None or df_anch is None:
            return no_signal(["DATA_MTF_INCOMPLETE"])

        if len(df_trig) < 60 or len(df_conf) < 60 or len(df_anch) < 60:
            return no_signal(["INSUFFICIENT_BARS"])

        # Validasi kualitas parameter backtest (anti-overfit & syarat sampel)
        if opt_summary.trades < 6 or opt_summary.profit_factor < 1.05 or opt_summary.max_dd_pct > 35.0:
            return no_signal(
                [
                    f"OPT_QUALITY_FAIL(T={opt_summary.trades},PF={opt_summary.profit_factor},DD={opt_summary.max_dd_pct}%)"
                ]
            )

        regime, shock, adx_val, shock_ratio = QuantFeatureEngine.regime_and_shock_filter(
            df_trig, atr_period=opt_summary.best_atr
        )
        if shock or regime == "VOLATILITY_SHOCK":
            return no_signal([f"VOLATILITY_SHOCK({shock_ratio:.2f}xATR)"], regime="VOLATILITY_SHOCK")

        if regime == "CHOP_RANGE":
            return no_signal([f"CHOP_ANTI_WHIPSAW(ADX={adx_val:.1f})"], regime="CHOP_RANGE")

        rel_str, rs_bonus = self._relative_strength_status(
            symbol,
            data_bundle.get("BTCUSD", {}).get(mode_spec.confirm_tf),
            data_bundle.get("ETHUSD", {}).get(mode_spec.confirm_tf),
        )
        if rs_bonus < 0:
            return no_signal([f"REL_STRENGTH_DIVERGENCE({rel_str})"], regime=regime, rel_str=rel_str)

        # Pilar I: SMC Struktur pada timeframe pemicu & konfirmasi
        smc_trig = QuantFeatureEngine.detect_smc_structure(df_trig, lookback=4)
        smc_conf = QuantFeatureEngine.detect_smc_structure(df_conf, lookback=4)

        # Pilar II: Order Flow Mikrostruktur
        of_trig = QuantFeatureEngine.order_flow_microstructure(df_trig)

        # Pilar III: Z-Score Mean Reversion terkalibrasi
        z_series = QuantFeatureEngine.rolling_zscore(df_trig["close"], window=20)
        z_now = float(z_series.iloc[-1])

        # Pilar IV: Multi-Timeframe Fractal Harmony
        anch_ema20 = float(df_anch["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        anch_ema50 = float(df_anch["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        conf_ema20 = float(df_conf["close"].ewm(span=20, adjust=False).mean().iloc[-1])
        conf_ema50 = float(df_conf["close"].ewm(span=50, adjust=False).mean().iloc[-1])

        anch_bull = float(df_anch["close"].iloc[-1]) > anch_ema20 and anch_ema20 >= anch_ema50
        anch_bear = float(df_anch["close"].iloc[-1]) < anch_ema20 and anch_ema20 <= anch_ema50
        conf_bull = float(df_conf["close"].iloc[-1]) > conf_ema20 and conf_ema20 >= conf_ema50
        conf_bear = float(df_conf["close"].iloc[-1]) < conf_ema20 and conf_ema20 <= conf_ema50

        # Pilar V: ATR teroptimasi
        atr_series = QuantFeatureEngine.atr(df_trig, period=opt_summary.best_atr)
        atr_now = float(atr_series.iloc[-1])
        if atr_now <= EPS:
            return no_signal(["ATR_ZERO"], regime=regime, rel_str=rel_str)

        close_now = float(df_trig["close"].iloc[-1])
        reasons: List[str] = []
        confluence = 0
        score = 35.0

        # Evaluasi arah BUY
        buy_bias_ok = regime == "TREND_BULL" and conf_bull and anch_bull
        sell_bias_ok = regime == "TREND_BEAR" and conf_bear and anch_bear

        if buy_bias_ok and not sell_bias_ok:
            if of_trig["wick_trap_bull"] > 0.5 or of_trig["body_ratio"] < 0.22:
                return no_signal(["BUY_WICK_TRAP_OR_WEAK_BODY"], regime=regime, rel_str=rel_str, score=score)

            confluence += 1  # MTF Harmony (trigger + confirm + anchor)
            score += 18.0
            reasons.append("MTF_BULL_ALIGNED")

            if smc_trig["bos_bull"] or smc_trig["choch_bull"] or smc_conf["major_bias"] == "BULLISH":
                confluence += 1
                score += 14.0
                reasons.append("SMC_BULL_VALID")

            if of_trig["vol_weighted_body"] >= 0.30 and float(df_trig["close"].iloc[-1]) >= float(df_trig["open"].iloc[-1]):
                confluence += 1
                score += 12.0
                reasons.append("ORDERFLOW_BULL_IMPULSE")

            if z_now <= -opt_summary.best_z or (-0.8 <= z_now <= 0.8 and adx_val >= 28.0):
                confluence += 1
                score += 12.0
                reasons.append(f"ZSCORE_OK({z_now:+.2f})")

            if rs_bonus > 0:
                confluence += 1
                score += 8.0
                reasons.append("RS_CONFIRMED")
            elif opt_summary.profit_factor >= 1.35:
                score += 4.0

            if confluence < mode_spec.min_confluence or score < mode_spec.min_score:
                return no_signal(
                    [f"CONFLUENCE_OR_SCORE_LOW(C={confluence}/{mode_spec.min_confluence},S={score:.1f})"],
                    regime=regime,
                    rel_str=rel_str,
                    score=score,
                )

            spread = asset.default_spread
            entry = round(close_now + (spread * 0.5), 2)
            sl_dist = atr_now * mode_spec.sl_atr_mult
            sl = round(entry - sl_dist, 2)
            r_unit = max(entry - sl, EPS)
            tp1 = round(entry + (r_unit * mode_spec.tp1_r), 2)
            tp2 = round(entry + (r_unit * mode_spec.tp2_r), 2)
            tp3 = round(entry + (r_unit * mode_spec.tp3_r), 2)
            lot = RiskAndPositionSizer.compute_lot(self.balance, self.risk_pct, entry, sl, asset)

            return SignalDecision(
                symbol=symbol,
                mode=mode_spec.name,
                action="BUY",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                lot=lot,
                score=round(min(score, 99.0), 1),
                regime=regime,
                rel_strength=rel_str,
                trail_desc=mode_spec.trail_rule,
                reasons=reasons,
                timestamp_utc=now_utc_str,
            )

        if sell_bias_ok and not buy_bias_ok:
            if of_trig["wick_trap_bear"] > 0.5 or of_trig["body_ratio"] < 0.22:
                return no_signal(["SELL_WICK_TRAP_OR_WEAK_BODY"], regime=regime, rel_str=rel_str, score=score)

            confluence += 1
            score += 18.0
            reasons.append("MTF_BEAR_ALIGNED")

            if smc_trig["bos_bear"] or smc_trig["choch_bear"] or smc_conf["major_bias"] == "BEARISH":
                confluence += 1
                score += 14.0
                reasons.append("SMC_BEAR_VALID")

            if of_trig["vol_weighted_body"] >= 0.30 and float(df_trig["close"].iloc[-1]) <= float(df_trig["open"].iloc[-1]):
                confluence += 1
                score += 12.0
                reasons.append("ORDERFLOW_BEAR_IMPULSE")

            if z_now >= opt_summary.best_z or (-0.8 <= z_now <= 0.8 and adx_val >= 28.0):
                confluence += 1
                score += 12.0
                reasons.append(f"ZSCORE_OK({z_now:+.2f})")

            if rs_bonus > 0:
                confluence += 1
                score += 8.0
                reasons.append("RS_CONFIRMED")
            elif opt_summary.profit_factor >= 1.35:
                score += 4.0

            if confluence < mode_spec.min_confluence or score < mode_spec.min_score:
                return no_signal(
                    [f"CONFLUENCE_OR_SCORE_LOW(C={confluence}/{mode_spec.min_confluence},S={score:.1f})"],
                    regime=regime,
                    rel_str=rel_str,
                    score=score,
                )

            spread = asset.default_spread
            entry = round(close_now - (spread * 0.5), 2)
            sl_dist = atr_now * mode_spec.sl_atr_mult
            sl = round(entry + sl_dist, 2)
            r_unit = max(sl - entry, EPS)
            tp1 = round(entry - (r_unit * mode_spec.tp1_r), 2)
            tp2 = round(entry - (r_unit * mode_spec.tp2_r), 2)
            tp3 = round(entry - (r_unit * mode_spec.tp3_r), 2)
            lot = RiskAndPositionSizer.compute_lot(self.balance, self.risk_pct, entry, sl, asset)

            return SignalDecision(
                symbol=symbol,
                mode=mode_spec.name,
                action="SELL",
                entry=entry,
                sl=sl,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                lot=lot,
                score=round(min(score, 99.0), 1),
                regime=regime,
                rel_strength=rel_str,
                trail_desc=mode_spec.trail_rule,
                reasons=reasons,
                timestamp_utc=now_utc_str,
            )

        return no_signal(["CONFLICT_OR_UNALIGNED_MTF"], regime=regime, rel_str=rel_str, score=score)


class StateRepository:
    def __init__(self, db_path: str = "trading_signals.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sig_hash TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry REAL NOT NULL,
                    sl REAL NOT NULL,
                    tp1 REAL NOT NULL,
                    tp2 REAL NOT NULL,
                    tp3 REAL NOT NULL,
                    lot REAL NOT NULL,
                    score REAL NOT NULL,
                    regime TEXT NOT NULL,
                    created_at_epoch REAL NOT NULL,
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT v FROM bot_state WHERE k = ?", (key,)).fetchone()
            return str(row[0]) if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bot_state(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                (key, value),
            )
            conn.commit()

    def is_in_cooldown(self, symbol: str, mode: str, action: str, cooldown_sec: int, now_epoch: float) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT created_at_epoch FROM signals
                WHERE symbol = ? AND mode = ? AND action = ?
                ORDER BY created_at_epoch DESC LIMIT 1
                """,
                (symbol, mode, action),
            ).fetchone()
            if not row:
                return False
            return (now_epoch - float(row[0])) < float(cooldown_sec)

    def save_signal(self, sig: SignalDecision, now_epoch: float) -> None:
        raw_key = f"{sig.symbol}|{sig.mode}|{sig.action}|{sig.entry}|{sig.sl}|{sig.timestamp_utc}"
        sig_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals(
                    sig_hash, symbol, mode, action, entry, sl, tp1, tp2, tp3, lot, score, regime,
                    created_at_epoch, created_at_utc
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sig_hash,
                    sig.symbol,
                    sig.mode,
                    sig.action,
                    sig.entry,
                    sig.sl,
                    sig.tp1,
                    sig.tp2,
                    sig.tp3,
                    sig.lot,
                    sig.score,
                    sig.regime,
                    now_epoch,
                    sig.timestamp_utc,
                ),
            )
            conn.commit()

    def recent_signals(self, limit: int = 8) -> List[Tuple[Any, ...]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT symbol, mode, action, entry, sl, tp1, lot, score, created_at_utc
                FROM signals ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return list(rows)


def generate_backtest_chart(summaries: List[BacktestSummary], out_path: str = "backtest_performance.png") -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("#0f172a")
    ax_list = axes.flatten()

    for idx, ax in enumerate(ax_list):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#e2e8f0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        if idx < len(summaries):
            s = summaries[idx]
            curve = s.equity_curve if s.equity_curve else [10000.0]
            ax.plot(curve, color="#38bdf8", linewidth=1.8)
            ax.set_title(
                f"{s.symbol} {s.mode} (ATR={s.best_atr}, Z={s.best_z}) | PF={s.profit_factor} | DD={s.max_dd_pct}%",
                color="#f8fafc",
                fontsize=9,
                fontweight="bold",
            )
            ax.set_xlabel("Trade Step", color="#94a3b8", fontsize=8)
            ax.set_ylabel("Equity ($)", color="#94a3b8", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
        else:
            ax.axis("off")

    fig.suptitle("APEX-QUANT-OMNI HYPERION — Backtest & Grid Search Equity", color="#f8fafc", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


class AlertNotifier:
    def __init__(self, http_client: ResilientHttpClient):
        self.http = http_client
        self.tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.discord_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    @staticmethod
    def format_signal_message(sig: SignalDecision) -> str:
        hh_mm = sig.timestamp_utc[11:16] if len(sig.timestamp_utc) >= 16 else sig.timestamp_utc
        return (
            f"⚡ <b>{sig.symbol} | {sig.mode} | {sig.action}</b>\n"
            f"🎯 Entry: <code>{sig.entry:.2f}</code> | 🛑 SL: <code>{sig.sl:.2f}</code>\n"
            f"✅ TP1: <code>{sig.tp1:.2f}</code> | TP2: <code>{sig.tp2:.2f}</code> | TP3: <code>{sig.tp3:.2f}</code>\n"
            f"📐 Lot: <code>{sig.lot:.2f}</code> | Skor: <code>{sig.score:.0f}/100</code> | Regime: <code>{sig.regime}</code>\n"
            f"🔄 Trail: {sig.trail_desc} | 🕒 <code>{hh_mm} UTC</code>\n"
            f"📊 RelStr: <code>{sig.rel_strength}</code>"
        )

    def send_text_telegram(self, text: str, chat_id: Optional[str] = None) -> bool:
        target_chat = (chat_id or self.tg_chat_id).strip()
        if not self.tg_token or not target_chat:
            logger.info("Telegram tidak dikonfigurasi; pesan dilewati.")
            return False
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        res = self.http.post_json(url, payload, service_key="telegram_send")
        return bool(res and res.get("ok"))

    def send_signal(self, sig: SignalDecision) -> None:
        msg = self.format_signal_message(sig)
        self.send_text_telegram(msg)
        if self.discord_url:
            plain_msg = (
                msg.replace("<b>", "**")
                .replace("</b>", "**")
                .replace("<code>", "`")
                .replace("</code>", "`")
            )
            self.http.post_json(self.discord_url, {"content": plain_msg}, service_key="discord_webhook")

    def send_error(self, context_msg: str) -> None:
        hh_mm = datetime.now(timezone.utc).strftime("%H:%M UTC")
        err_text = f"🚨 <b>APEX-HYPERION ERROR</b> | 🕒 <code>{hh_mm}</code>\n<code>{context_msg[:900]}</code>"
        self.send_text_telegram(err_text)


class TelegramCommandHandler:
    """2-way command handler tervalidasi chat_id dan pelacakan update_id."""

    def __init__(self, http_client: ResilientHttpClient, repo: StateRepository, notifier: AlertNotifier):
        self.http = http_client
        self.repo = repo
        self.notifier = notifier
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.allowed_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    def poll_and_handle(self, latest_decisions: List[SignalDecision]) -> int:
        if not self.token or not self.allowed_chat_id:
            return 0
        last_update_raw = self.repo.get_meta("tg_last_update_id", "0")
        try:
            offset_val = int(last_update_raw) + 1
        except ValueError:
            offset_val = 0

        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        data = self.http.get_json(url, params={"offset": offset_val, "timeout": 2}, service_key="telegram_poll")
        if not isinstance(data, dict) or not data.get("ok"):
            return 0

        updates = data.get("result") or []
        handled = 0
        max_upd = int(last_update_raw) if last_update_raw.isdigit() else 0

        for upd in updates:
            if not isinstance(upd, dict):
                continue
            upd_id = int(upd.get("update_id", 0))
            if upd_id > max_upd:
                max_upd = upd_id

            message = upd.get("message") or upd.get("edited_message") or {}
            chat = message.get("chat") or {}
            sender_chat_id = str(chat.get("id", "")).strip()
            text = str(message.get("text", "")).strip()

            if not text.startswith("/check"):
                continue

            if sender_chat_id != self.allowed_chat_id:
                logger.warning("Akses /check ditolak untuk chat_id tidak sah: %s", sender_chat_id)
                continue

            parts = [p.strip().lower() for p in text.split()[1:]]
            sym_filter = "BTCUSD" if "btc" in parts else ("ETHUSD" if "eth" in parts else None)
            mode_filter = "SCALP" if "scalp" in parts else ("SWING" if "swing" in parts else None)
            is_status = "status" in parts

            if is_status:
                hist = self.repo.recent_signals(limit=5)
                if not hist:
                    reply = "ℹ️ <b>STATUS</b>: Belum ada sinyal tersimpan di SQLite."
                else:
                    lines = ["📜 <b>5 SINYAL TERAKHIR</b>"]
                    for r in hist:
                        lines.append(f"• <code>{r[0]} {r[1]} {r[2]}</code> @ <code>{r[3]:.2f}</code> (Skor {r[7]:.0f})")
                    reply = "\n".join(lines)
                self.notifier.send_text_telegram(reply, chat_id=sender_chat_id)
                handled += 1
                continue

            matched = [
                d
                for d in latest_decisions
                if (sym_filter is None or d.symbol == sym_filter)
                and (mode_filter is None or d.mode == mode_filter)
            ]
            if not matched:
                self.notifier.send_text_telegram("ℹ️ Tidak ada data evaluasi untuk filter tersebut.", chat_id=sender_chat_id)
                handled += 1
                continue

            blocks = ["🔎 <b>HASIL /CHECK (LIVE SNAPSHOT)</b>"]
            for d in matched:
                if d.action == "NO_SIGNAL":
                    r_txt = ",".join(d.reasons[:2])
                    blocks.append(
                        f"• <code>{d.symbol} {d.mode}</code>: <b>NO_SIGNAL</b> | {d.regime} | <code>{r_txt}</code>"
                    )
                else:
                    blocks.append(
                        f"• <code>{d.symbol} {d.mode} {d.action}</code> @ <code>{d.entry:.2f}</code> "
                        f"SL <code>{d.sl:.2f}</code> TP1 <code>{d.tp1:.2f}</code> Lot <code>{d.lot:.2f}</code>"
                    )
            self.notifier.send_text_telegram("\n".join(blocks), chat_id=sender_chat_id)
            handled += 1

        if max_upd > 0:
            self.repo.set_meta("tg_last_update_id", str(max_upd))
        return handled


def forensic_audit_log(
    now_utc_str: str,
    sources_used: Dict[str, Dict[str, str]],
    summaries: List[BacktestSummary],
    decisions: List[SignalDecision],
    dispatched_count: int,
) -> None:
    sep = "=" * 96
    dash = "-" * 96
    print(sep)
    print(f"APEX-QUANT-OMNI — HYPERION FORENSIC AUDIT LOG | {now_utc_str}")
    print(sep)
    for sym in ["BTCUSD", "ETHUSD"]:
        src_map = sources_used.get(sym, {})
        src_str = ", ".join(f"{k}:{v}" for k, v in src_map.items())
        print(f"[DATA FEED] {sym:<6} -> {src_str}")
    print(dash)
    for s in summaries:
        print(
            f"[GRID/BT  ] {s.symbol:<6} {s.mode:<5} | ATR={s.best_atr:<2} Z={s.best_z:<3} | "
            f"T={s.trades:<3} WR={s.win_rate:>6.2f}% PF={s.profit_factor:>5.2f} "
            f"DD={s.max_dd_pct:>5.2f}% Ret={s.net_return_pct:>+7.2f}%"
        )
    print(dash)
    for d in decisions:
        reasons_txt = ";".join(d.reasons) if d.reasons else "CONFLUENCE_MET"
        print(
            f"[SIGNAL   ] {d.symbol:<6} {d.mode:<5} -> {d.action:<9} | "
            f"Entry={d.entry:>9.2f} SL={d.sl:>9.2f} TP1={d.tp1:>9.2f} Lot={d.lot:>5.2f} "
            f"Skor={d.score:>5.1f} Regime={d.regime:<15} | {reasons_txt}"
        )
    print(dash)
    print(f"[DISPATCH ] Sinyal Terkirim={dispatched_count}")
    print(sep)


def run_pipeline() -> None:
    now_dt = datetime.now(timezone.utc)
    now_epoch = now_dt.timestamp()
    now_utc_str = now_dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    balance = float(os.getenv("ACCOUNT_BALANCE", "10000.0"))
    risk_pct = float(os.getenv("RISK_PERCENT", "1.0"))
    db_path = os.getenv("SQLITE_DB_PATH", "trading_signals.db")
    chart_path = os.getenv("BACKTEST_CHART_PATH", "backtest_performance.png")

    breaker = CircuitBreaker(max_failures=4, reset_after_sec=600)
    http_client = ResilientHttpClient(breaker=breaker, timeout=12)
    provider = MarketDataProvider(http_client=http_client)
    optimizer = GridSearchOptimizer(atr_candidates=[10, 14, 20], z_candidates=[1.5, 2.0, 2.5])
    engine = HyperionSignalEngine(provider=provider, balance=balance, risk_pct=risk_pct)
    repo = StateRepository(db_path=db_path)
    notifier = AlertNotifier(http_client=http_client)
    cmd_handler = TelegramCommandHandler(http_client=http_client, repo=repo, notifier=notifier)

    needed_tfs = {"5m", "15m", "1h", "4h", "1d"}
    data_bundle: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {"BTCUSD": {}, "ETHUSD": {}}
    sources_used: Dict[str, Dict[str, str]] = {"BTCUSD": {}, "ETHUSD": {}}

    for sym in ["BTCUSD", "ETHUSD"]:
        for tf in sorted(needed_tfs, key=lambda x: TF_SECONDS[x]):
            df, src = provider.fetch(sym, tf, limit=250)
            data_bundle[sym][tf] = df
            sources_used[sym][tf] = src

    summaries: List[BacktestSummary] = []
    summary_map: Dict[Tuple[str, str], BacktestSummary] = {}

    for sym in ["BTCUSD", "ETHUSD"]:
        for mode_name, mode_spec in MODES.items():
            bt_df = data_bundle[sym].get(mode_spec.confirm_tf)
            if bt_df is None or len(bt_df) < 60:
                bt_df = data_bundle[sym].get(mode_spec.trigger_tf)
            if bt_df is not None and len(bt_df) >= 60:
                summ = optimizer.optimize_and_backtest(
                    symbol=sym,
                    mode=mode_name,
                    df=bt_df,
                    spread_cost=ASSETS[sym].default_spread,
                    min_trades=6,
                )
            else:
                summ = BacktestSummary(
                    symbol=sym,
                    mode=mode_name,
                    best_atr=14,
                    best_z=2.0,
                    trades=0,
                    win_rate=0.0,
                    profit_factor=0.0,
                    max_dd_pct=0.0,
                    net_return_pct=0.0,
                    equity_curve=[balance],
                )
            summaries.append(summ)
            summary_map[(sym, mode_name)] = summ

    generate_backtest_chart(summaries, out_path=chart_path)

    decisions: List[SignalDecision] = []
    dispatched = 0

    for sym in ["BTCUSD", "ETHUSD"]:
        for mode_name, mode_spec in MODES.items():
            dec = engine.evaluate(
                symbol=sym,
                mode_spec=mode_spec,
                opt_summary=summary_map[(sym, mode_name)],
                data_bundle=data_bundle,
                now_utc_str=now_utc_str,
            )
            decisions.append(dec)

            if dec.action in ("BUY", "SELL"):
                in_cd = repo.is_in_cooldown(
                    symbol=dec.symbol,
                    mode=dec.mode,
                    action=dec.action,
                    cooldown_sec=mode_spec.cooldown_sec,
                    now_epoch=now_epoch,
                )
                if not in_cd:
                    repo.save_signal(dec, now_epoch=now_epoch)
                    notifier.send_signal(dec)
                    dispatched += 1
                else:
                    dec.action = "NO_SIGNAL"
                    dec.reasons.append("COOLDOWN_DEDUP_ACTIVE")

    cmd_handler.poll_and_handle(decisions)
    forensic_audit_log(
        now_utc_str=now_utc_str,
        sources_used=sources_used,
        summaries=summaries,
        decisions=decisions,
        dispatched_count=dispatched,
    )


def main() -> None:
    notifier: Optional[AlertNotifier] = None
    try:
        breaker = CircuitBreaker(max_failures=4, reset_after_sec=600)
        notifier = AlertNotifier(http_client=ResilientHttpClient(breaker=breaker, timeout=10))
        run_pipeline()
    except Exception as exc:
        logger.exception("Kesalahan fatal pada eksekusi APEX-QUANT-OMNI: %s", exc)
        if notifier is not None:
            notifier.send_error(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
