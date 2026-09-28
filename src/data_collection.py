"""Download 1h OHLC for the candidate FX universe, keep the N most volatile pairs, merge technicals with
FinBERT currency sentiment, label 4h-ahead direction and write the training set to data/features.parquet.

Usage:
  python -m src.data_collection                 # download + build
  python -m src.data_collection --skip-download # rebuild features from data/raw cache
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from . import db
from .config import (CANDIDATE_PAIRS, FEATURES_PATH, HISTORY_PERIOD, INTERVAL, N_PAIRS, RAW_DIR,
                     UNIVERSE_PATH)
from .features import atr, build_feature_frame
from .market_data import download_ohlcv
from .sentiment import hourly_currency_sentiment

log = logging.getLogger(__name__)
MIN_BARS = 1000


def download_universe(pairs: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for pair in pairs:
        try:
            df = download_ohlcv(pair, period=HISTORY_PERIOD, interval=INTERVAL)
        except Exception as e:
            log.warning("%s: download failed (%s)", pair, e)
            continue
        if len(df) < MIN_BARS:
            log.warning("%s: only %d bars - skipped", pair, len(df))
            continue
        df.to_parquet(RAW_DIR / f"{pair}.parquet")
        frames[pair] = df
        log.info("%s: %d bars %s -> %s", pair, len(df), df.index[0], df.index[-1])
        time.sleep(0.3)  # be polite to Yahoo
    return frames


def load_cached(pairs: list[str]) -> dict[str, pd.DataFrame]:
    return {p: pd.read_parquet(RAW_DIR / f"{p}.parquet") for p in pairs if (RAW_DIR / f"{p}.parquet").exists()}


def rank_by_volatility(frames: dict[str, pd.DataFrame], n: int, lookback: int = 24 * 90) -> dict[str, float]:
    """Median ATR(14) as % of price over the last ~90 days; returns the top-n pairs -> score."""
    scores = {p: float((atr(df) / df["Close"]).iloc[-lookback:].median()) for p, df in frames.items()}
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    db.init_db()
    frames = load_cached(CANDIDATE_PAIRS) if args.skip_download else download_universe(CANDIDATE_PAIRS)
    if not frames:
        raise SystemExit("No price data available.")

    universe = rank_by_volatility(frames, args.n_pairs)
    UNIVERSE_PATH.write_text(json.dumps({"pairs": list(universe), "atr_pct": universe}, indent=2))
    log.info("universe (%d): %s", len(universe), ", ".join(universe))

    sentiment = hourly_currency_sentiment()
    if sentiment.empty:
        log.warning("news table is empty - sentiment features will be 0. "
                    "Backfill with `python -m src.sentiment import-csv` for a meaningful NLP signal.")

    parts = [build_feature_frame(frames[p], p, sentiment) for p in universe]
    data = pd.concat(parts).reset_index().sort_values(["close_time", "pair"], ignore_index=True)
    data.to_parquet(FEATURES_PATH, index=False)

    log.info("wrote %s: %d rows, %s -> %s, up-rate %.3f, rows with news %.1f%%",
             FEATURES_PATH, len(data), data["close_time"].min(), data["close_time"].max(),
             data["target"].mean(), 100 * (data["news_count_24h"] > 0).mean())


if __name__ == "__main__":
    main()
