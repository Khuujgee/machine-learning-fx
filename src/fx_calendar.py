"""Economic calendar from Forex Factory's public weekly JSON feed (this week's events).

The feed is fetched at most every CAL_MAX_AGE_HOURS and each week's copy is kept in data/calendar/, which also
builds a calendar archive over time.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Iterable, Optional

import pandas as pd
import requests

from .config import DATA_DIR

log = logging.getLogger(__name__)
FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CAL_DIR = DATA_DIR / "calendar"
CAL_MAX_AGE_HOURS = 2
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) research"}
G10 = ("USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK")


def _week_file(now: pd.Timestamp):
    monday = (now.tz_convert("America/New_York") - pd.Timedelta(days=now.tz_convert("America/New_York").weekday()))
    return CAL_DIR / f"ff_{monday:%Y-%m-%d}.json"


def fetch(now: Optional[pd.Timestamp] = None, max_age_hours: float = CAL_MAX_AGE_HOURS) -> pd.DataFrame:
    """This week's events with UTC times. Uses the cached copy if it is fresh or the feed is down."""
    now = now or pd.Timestamp.now(tz="UTC")
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    path = _week_file(now)
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600
    if not fresh:
        try:
            r = requests.get(FEED, headers=UA, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                path.write_text(json.dumps(data))
        except Exception as e:
            log.warning("calendar fetch failed (%s) - using cached copy", type(e).__name__)
    if not path.exists():
        return pd.DataFrame(columns=["title", "currency", "time", "impact", "forecast", "previous"])
    df = pd.DataFrame(json.loads(path.read_text()))
    df = df.rename(columns={"country": "currency"})
    df["time"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    return df.dropna(subset=["time"])[["title", "currency", "time", "impact", "forecast", "previous"]]


def window(start: pd.Timestamp, end: pd.Timestamp, impacts: Iterable[str] = ("High",),
           currencies: Iterable[str] = G10, events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    ev = fetch(start) if events is None else events
    m = ev["time"].between(start, end) & ev["impact"].isin(list(impacts)) & ev["currency"].isin(list(currencies))
    return ev[m].sort_values("time")
