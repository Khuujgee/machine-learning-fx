"""Historical news tone per currency from GDELT (exported with sql/gdelt_currency_tone.sql).

Put the BigQuery CSV export(s) in data/gdelt/ (any file names). Columns: hour, currency, events, articles,
tone, goldstein.

Features (per FX row, joined as-of the bar close; an hour's news is only "known" when that hour ends):
  news_tone_diff     : base currency news tone minus quote currency tone (each as a z-score vs its last 30 days)
  news_tone_base / _quote : the two z-scores
  news_volume        : how unusual today's coverage volume is for the two currencies (z-score, summed)
  news_goldstein_diff: base minus quote "cooperation vs conflict" score over the last 24h
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DATA_DIR

log = logging.getLogger(__name__)
GDELT_DIR = DATA_DIR / "gdelt"
NEWS_COLS = ["news_tone_diff", "news_tone_base", "news_tone_quote", "news_volume", "news_goldstein_diff"]


def load_gdelt() -> pd.DataFrame:
    files = sorted(GDELT_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["hour"] = pd.to_datetime(raw["hour"], utc=True)
    raw = raw.drop_duplicates(["hour", "currency"], keep="last")
    log.info("GDELT: %d rows, %d currencies, %s -> %s", len(raw), raw.currency.nunique(), raw.hour.min(), raw.hour.max())
    return raw


def currency_news_frame(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Wide frame indexed by known_at (hour end) with <CCY>_tone_z, <CCY>_vol_z, <CCY>_gold columns."""
    raw = load_gdelt() if raw is None else raw
    if raw.empty:
        return pd.DataFrame()
    idx = pd.date_range(raw.hour.min(), raw.hour.max(), freq="h")
    cols = {}
    for ccy, g in raw.groupby("currency"):
        g = g.set_index("hour").reindex(idx)
        art = g["articles"].fillna(0)
        tone_x_art = (g["tone"] * g["articles"]).fillna(0)
        art24, tone24 = art.rolling(24).sum(), tone_x_art.rolling(24).sum()
        tone = tone24 / art24.replace(0, np.nan)              # article-weighted tone over the last 24h
        base_mean = tone.rolling(24 * 30, min_periods=24 * 7).mean()
        base_std = tone.rolling(24 * 30, min_periods=24 * 7).std()
        cols[f"{ccy}_tone_z"] = ((tone - base_mean) / base_std).clip(-6, 6)
        lv = np.log1p(art24)
        cols[f"{ccy}_vol_z"] = ((lv - lv.rolling(24 * 30, min_periods=24 * 7).mean())
                                / lv.rolling(24 * 30, min_periods=24 * 7).std()).clip(-6, 6)
        cols[f"{ccy}_gold"] = g["goldstein"].rolling(24, min_periods=1).mean()
    out = pd.DataFrame(cols)
    # stamp each hour's value at the END of that hour, when it is actually known. (Shift the index itself:
    # passing a new index to the constructor would re-align by label and silently peek 1 hour ahead.)
    out.index = out.index + pd.Timedelta(hours=1)
    out.index.name = "known_at"
    return out.replace([np.inf, -np.inf], np.nan)


def add_news_features(df: pd.DataFrame, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join news features onto FX rows (needs `pair`, `close_time`); same index/order as `df`. NaN if no data."""
    frame = currency_news_frame() if frame is None else frame
    orig = df.index
    d = df.drop(columns=[c for c in NEWS_COLS if c in df], errors="ignore").copy()
    for c in NEWS_COLS:
        d[c] = np.nan
    if frame.empty:
        return d
    d["_row"] = np.arange(len(d))
    d = d.reset_index(drop=True).sort_values("close_time", kind="stable")
    f = frame.reset_index().sort_values("known_at")
    j = pd.merge_asof(d[["close_time"]], f, left_on="close_time", right_on="known_at", direction="backward",
                      tolerance=pd.Timedelta(hours=6))
    base, quote = d["pair"].str[:3].values, d["pair"].str[3:].values

    def pick(suffix: str, ccys: np.ndarray) -> np.ndarray:
        vals = np.full(len(d), np.nan)
        for c in np.unique(ccys):
            col = f"{c}_{suffix}"
            if col in j:
                m = ccys == c
                vals[m] = j[col].values[m]
        return vals

    d["news_tone_base"], d["news_tone_quote"] = pick("tone_z", base), pick("tone_z", quote)
    d["news_tone_diff"] = d["news_tone_base"] - d["news_tone_quote"]
    d["news_volume"] = pick("vol_z", base) + pick("vol_z", quote)
    d["news_goldstein_diff"] = pick("gold", base) - pick("gold", quote)
    d = d.sort_values("_row").drop(columns="_row")
    d.index = orig
    return d
