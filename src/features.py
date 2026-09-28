"""Feature engineering shared by training AND live inference (one code path = no train/serve skew).

Timing convention: a bar indexed at `bar_time` (open) closes at `close_time = bar_time + 1h`.
Every feature for that row uses only information available at `close_time`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import BAR_HOURS, HORIZON_BARS

TECH_COLS = [
    "rsi_14", "atr_pct", "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "ret_1", "ret_4", "ret_24", "vol_24", "range_pct", "dist_ema50_atr",
    "hour_sin", "hour_cos", "dow",
]
SENT_COLS = ["pair_sentiment", "base_sentiment", "quote_sentiment", "news_count_24h"]
FEATURE_COLS = TECH_COLS + SENT_COLS


# --------------------------------------------------------------------------- indicators
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift()
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def technical_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    a = atr(df)
    m_line, m_sig, m_hist = macd(c)
    ret_1 = c.pct_change()
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    out["atr"] = a
    out["rsi_14"] = rsi(c)
    out["atr_pct"] = a / c
    # MACD is in price units -> normalise by ATR so one model works across all pairs
    out["macd_norm"] = m_line / a
    out["macd_signal_norm"] = m_sig / a
    out["macd_hist_norm"] = m_hist / a
    out["ret_1"] = ret_1
    out["ret_4"] = c.pct_change(4)
    out["ret_24"] = c.pct_change(24)
    out["vol_24"] = ret_1.rolling(24).std()
    out["range_pct"] = (df["High"] - df["Low"]) / c
    out["dist_ema50_atr"] = (c - c.ewm(span=50, adjust=False).mean()) / a
    return out


# --------------------------------------------------------------------------- sentiment join
def attach_sentiment(feat: pd.DataFrame, sentiment: Optional[pd.DataFrame], base: str, quote: str) -> pd.DataFrame:
    """As-of join currency sentiment (indexed by `known_at`) onto bars by `close_time` -> no look-ahead."""
    feat = feat.copy()
    if sentiment is None or sentiment.empty:
        for col in SENT_COLS:
            feat[col] = 0.0
        return feat

    def col(name: str) -> pd.Series:
        return sentiment[name] if name in sentiment else pd.Series(0.0, index=sentiment.index)

    s = pd.DataFrame({
        "base_sentiment": col(f"{base}_sent"),
        "quote_sentiment": col(f"{quote}_sent"),
        "news_count_24h": col(f"{base}_cnt24") + col(f"{quote}_cnt24"),
    }, index=sentiment.index).rename_axis("known_at").reset_index()

    merged = pd.merge_asof(
        feat.reset_index().sort_values("close_time"),
        s.sort_values("known_at"),
        left_on="close_time", right_on="known_at", direction="backward",
    ).set_index("bar_time").drop(columns="known_at")
    for c in ["base_sentiment", "quote_sentiment", "news_count_24h"]:
        merged[c] = merged[c].fillna(0.0)
    merged["pair_sentiment"] = merged["base_sentiment"] - merged["quote_sentiment"]
    return merged


# --------------------------------------------------------------------------- public entry point
def build_feature_frame(
    ohlcv: pd.DataFrame,
    pair: str,
    sentiment: Optional[pd.DataFrame] = None,
    with_label: bool = True,
    horizon: int = HORIZON_BARS,
) -> pd.DataFrame:
    df = ohlcv.sort_index()
    feat = technical_features(df)
    feat.index.name = "bar_time"
    feat["close_time"] = feat.index + pd.Timedelta(hours=BAR_HOURS)
    hour = feat["close_time"].dt.hour
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["dow"] = feat["close_time"].dt.dayofweek
    feat = attach_sentiment(feat, sentiment, pair[:3], pair[3:])
    feat["pair"] = pair

    if with_label:
        fut = feat["close"].shift(-horizon)
        feat["fwd_ret"] = fut / feat["close"] - 1
        # when the label becomes known -> used to purge overlapping samples in walk-forward CV
        feat["label_time"] = feat["close_time"].shift(-horizon)
        feat = feat[fut.notna() & (fut != feat["close"])].copy()  # drop unknown / exactly-flat outcomes
        feat["target"] = (feat["fwd_ret"] > 0).astype(int)

    feat[FEATURE_COLS] = feat[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return feat.dropna(subset=FEATURE_COLS)
