"""Keep the GDELT news-tone archive up to date from GDELT's public 15-minute export files.

Produces exactly what sql/gdelt_currency_tone.sql produces (same columns, same country -> currency mapping),
appended to data/gdelt/live.csv, so news_history.py sees one continuous history: 3 years from BigQuery +
whatever this adds. No API, no account, no rate limit (each file is ~100 KB).

    python -m src.gdelt_live            # fill in every missing hour up to the last complete one (max 72h back)
"""
from __future__ import annotations

import io
import logging
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from .news_history import GDELT_DIR, load_gdelt

log = logging.getLogger(__name__)
LIVE_PATH = GDELT_DIR / "live.csv"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) research"}
# English feed + translated (non-English) feed: the BigQuery `events` table contains both
FEEDS = ["http://data.gdeltproject.org/gdeltv2/{stamp}.export.CSV.zip",
         "http://data.gdeltproject.org/gdeltv2/{stamp}.translation.export.CSV.zip"]
MAX_BACKFILL_HOURS = 72

EURO = ["DEU", "FRA", "ITA", "ESP", "NLD", "BEL", "AUT", "IRL", "PRT", "FIN", "GRC", "SVK", "SVN", "LUX", "EST",
        "LVA", "LTU", "HRV", "CYP", "MLT", "EUR"]
COUNTRY_TO_CCY = {"USA": "USD", **{c: "EUR" for c in EURO}, "GBR": "GBP", "JPN": "JPY", "CHE": "CHF", "AUS": "AUD",
                  "CAN": "CAD", "NZL": "NZD", "SWE": "SEK", "NOR": "NOK", "DNK": "DKK", "POL": "PLN", "HUN": "HUF",
                  "CZE": "CZK", "TUR": "TRY", "ZAF": "ZAR", "MEX": "MXN", "BRA": "BRL", "CHN": "CNH", "IND": "INR",
                  "THA": "THB", "ISR": "ILS", "SGP": "SGD", "HKG": "HKD"}
# GDELT 2.0 events export: column positions
A1_COUNTRY, A2_COUNTRY, GOLDSTEIN, NUM_ARTICLES, AVG_TONE = 7, 17, 30, 33, 34


def _file(url: str) -> pd.DataFrame | None:
    stamp = url.rsplit("/", 1)[-1]
    try:
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code == 404:
            return None                       # GDELT occasionally skips a 15-minute slot
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0])
        return pd.read_csv(io.BytesIO(raw), sep="\t", header=None, low_memory=False,
                           usecols=[A1_COUNTRY, A2_COUNTRY, GOLDSTEIN, NUM_ARTICLES, AVG_TONE])
    except Exception as e:
        log.warning("GDELT %s: %s", stamp, type(e).__name__)
        return None


def aggregate_hour(hour: pd.Timestamp) -> pd.DataFrame:
    """One hour of events -> rows of hour, currency, events, articles, tone, goldstein (like the SQL)."""
    frames = [f for m in (0, 15, 30, 45) for feed in FEEDS
              if (f := _file(feed.format(stamp=f"{hour:%Y%m%d%H}{m:02d}00"))) is not None]
    if not frames:
        return pd.DataFrame()
    ev = pd.concat(frames, ignore_index=True)
    long = pd.concat([ev[[A1_COUNTRY, GOLDSTEIN, NUM_ARTICLES, AVG_TONE]].rename(columns={A1_COUNTRY: "cc"}),
                      ev[[A2_COUNTRY, GOLDSTEIN, NUM_ARTICLES, AVG_TONE]].rename(columns={A2_COUNTRY: "cc"})])
    long["currency"] = long["cc"].map(COUNTRY_TO_CCY)
    long = long.dropna(subset=["currency"])
    long["tone_x"] = long[AVG_TONE] * long[NUM_ARTICLES]
    g = long.groupby("currency").agg(events=("cc", "size"), articles=(NUM_ARTICLES, "sum"),
                                     tone_x=("tone_x", "sum"), goldstein=(GOLDSTEIN, "mean"))
    g["tone"] = (g["tone_x"] / g["articles"].replace(0, np.nan)).round(3)
    g["goldstein"] = g["goldstein"].round(3)
    g = g.drop(columns="tone_x").reset_index()
    g.insert(0, "hour", hour.strftime("%Y-%m-%d %H:00:00 UTC"))
    return g[["hour", "currency", "events", "articles", "tone", "goldstein"]]


def update(max_hours: int = MAX_BACKFILL_HOURS) -> int:
    """Append every missing complete hour (GDELT publishes ~15 min after each slot). Returns hours added."""
    GDELT_DIR.mkdir(parents=True, exist_ok=True)
    have = load_gdelt()
    last_complete = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20)).floor("h") - pd.Timedelta(hours=1)
    start = last_complete - pd.Timedelta(hours=max_hours - 1)
    if not have.empty:
        start = max(start, have["hour"].max() + pd.Timedelta(hours=1))
    added = 0
    for hour in pd.date_range(start, last_complete, freq="h"):
        rows = aggregate_hour(hour)
        if rows.empty:
            continue
        rows.to_csv(LIVE_PATH, mode="a", header=not LIVE_PATH.exists(), index=False)
        added += 1
        time.sleep(0.2)
    if added:
        log.info("GDELT archive: +%d hours (now up to %s)", added, last_complete)
    return added


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print("hours added:", update())
