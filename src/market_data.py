"""Thin wrappers around yfinance for historical bars, live prices and FX conversion."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
OHLC = ["Open", "High", "Low", "Close"]


def yf_symbol(pair: str) -> str:
    return f"{pair}=X"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLC, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty()
    df = df[OHLC].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[(df > 0).all(axis=1)].dropna()


def download_ohlcv(pair: str, period: str = "730d", interval: str = "1h") -> pd.DataFrame:
    """UTC-indexed OHLC bars. Index = bar *open* time. FX volume from Yahoo is always 0, so it is dropped."""
    return _clean(yf.Ticker(yf_symbol(pair)).history(period=period, interval=interval, auto_adjust=False))


def minute_bars(pair: str, start: pd.Timestamp) -> pd.DataFrame:
    """1-minute bars since `start` (Yahoo only serves ~7 days of 1m data per request)."""
    now = pd.Timestamp.now(tz="UTC")
    earliest = now - pd.Timedelta(days=6, hours=23)
    if start < earliest:
        log.warning("%s: requested 1m bars from %s; Yahoo limit clamps to %s", pair, start, earliest)
        start = earliest
    if start.floor("min") >= now.floor("min"):
        # no completed minute yet (e.g. a trade opened seconds ago); Yahoo would log "possibly delisted"
        return _empty()
    df = yf.Ticker(yf_symbol(pair)).history(start=start.floor("min"), end=now + pd.Timedelta(minutes=1),
                                            interval="1m", auto_adjust=False)
    return _clean(df)


def latest_price(pair: str) -> float:
    df = _clean(yf.Ticker(yf_symbol(pair)).history(period="1d", interval="1m", auto_adjust=False))
    if df.empty:
        df = _clean(yf.Ticker(yf_symbol(pair)).history(period="5d", interval="1h", auto_adjust=False))
    if df.empty:
        raise RuntimeError(f"No price data for {pair}")
    return float(df["Close"].iloc[-1])


def quote_to_usd(pair: str, pair_price: Optional[float] = None) -> float:
    """USD value of one unit of the pair's quote currency (needed for position sizing and P&L)."""
    base, quote = pair[:3], pair[3:]
    if quote == "USD":
        return 1.0
    if base == "USD":
        return 1.0 / (pair_price or latest_price(pair))
    try:
        return latest_price(f"{quote}USD")
    except Exception:
        return 1.0 / latest_price(f"USD{quote}")
