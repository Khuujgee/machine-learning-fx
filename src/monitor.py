"""FX Risk & News Monitor: a Discord feed for traders (G10 currencies). Information, not trading advice.

  * Morning briefing (daily, MONITOR_BRIEFING_HOUR_UTC:MINUTE, Sunday-Thursday = before each FX day):
    high-impact events, volatility outlook per pair (HAR model), news tone vs normal per currency,
    short-term interest rates, notable headlines.
  * Alerts ("important only"): high-impact event reminders, unusually large hourly moves,
    unusual news-tone swings, overnight-rate changes (USD/EUR/GBP, daily data).

Runs inside the scanner service (src.scanner calls tick() every minute). Manual use:
    python -m src.monitor briefing --dry-run     # print the briefing instead of posting it
    python -m src.monitor briefing               # post it now
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import db, fx_calendar, gdelt_live, notify, vol_model
from .config import (MONITOR_BRIEFING_HOUR_UTC, MONITOR_BRIEFING_MINUTE, MONITOR_EVENT_LEAD_MIN,
                     MONITOR_NEWS_Z, MONITOR_VOL_SPIKE_X, MONITOR_WEBHOOK_URL)
from .macro_data import RATES_PATH, download_rates
from .market_data import download_ohlcv
from .news_history import currency_news_frame

log = logging.getLogger(__name__)
G10 = list(fx_calendar.G10)
NAME = "FX Risk & News Monitor"
FOOTER = ("For information only - not financial advice. Sources: Forex Factory calendar, GDELT, FRED, "
          "Yahoo Finance, investingLive headlines scored by FinBERT.")
LEVEL_ICON = {"calm": "🟢", "normal": "⚪", "elevated": "🟠", "high": "🔴"}
DAILY_RATE_SERIES = {"USD": "DFF", "EUR": "ECBDFR", "GBP": "IUDSOIA"}


def _post(title: str, text: str = "", color: int = notify.BLUE, fields: Optional[list] = None,
          inline: bool = False, dry_run: bool = False) -> None:
    if dry_run:
        print(f"\n### {title}\n{text}")
        for n, v in fields or []:
            print(f"\n**{n}**\n{v}")
        print(f"\n_{FOOTER}_")
        return
    notify._send(title, text, color, fields, url=MONITOR_WEBHOOK_URL, username=NAME, footer=FOOTER, inline=inline)


def _ts(t: pd.Timestamp, style: str = "t") -> str:
    """Discord timestamp: shows in each reader's own time zone."""
    return f"<t:{int(t.timestamp())}:{style}>"


# --------------------------------------------------------------------------- building blocks
def events_text(start: pd.Timestamp, hours: int = 24) -> str:
    ev = fx_calendar.window(start, start + pd.Timedelta(hours=hours))
    if ev.empty:
        return "No high-impact G10 releases scheduled."
    lines = []
    for e in ev.itertuples():
        extra = ", ".join(x for x in (f"forecast {e.forecast}" if e.forecast else "",
                                      f"previous {e.previous}" if e.previous else "") if x)
        lines.append(f"{_ts(e.time)} **{e.currency}** {e.title}" + (f" ({extra})" if extra else ""))
    return "\n".join(lines[:15])


def vol_text() -> str:
    o = vol_model.outlook()
    if o.empty:
        return "Unavailable (price data)."
    o = o.sort_values("percentile", ascending=False)
    def nth(x: float) -> str:
        n = int(round(x))
        return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
    return "\n".join(f"{LEVEL_ICON.get(str(r.level), '⚪')} `{r.pair}` {r.forecast_vol * 100:4.1f}% - {r.level} "
                     f"({nth(r.percentile)} percentile of the past year)" for r in o.itertuples())


def news_z(now: Optional[pd.Timestamp] = None) -> pd.Series:
    """Latest 24h news-tone z-score per G10 currency (vs its own last 30 days)."""
    f = currency_news_frame()
    if f.empty:
        return pd.Series(dtype=float)
    now = now or pd.Timestamp.now(tz="UTC")
    f = f[f.index <= now]
    last = f.iloc[-1]
    return pd.Series({c: last.get(f"{c}_tone_z", np.nan) for c in G10}).dropna()


def news_text(z: pd.Series) -> str:
    if z.empty:
        return "Unavailable."
    def arrow(v):
        return "▲" if v > 0.5 else "▼" if v < -0.5 else "▬"
    z = z.reindex(z.abs().sort_values(ascending=False).index)
    return " · ".join(f"**{c}** {arrow(v)} {v:+.1f}σ" for c, v in z.items()) + \
        "\n(24h tone of global coverage vs its own last 30 days; ▲ more positive than usual)"


def rates_now() -> dict:
    r = pd.read_parquet(RATES_PATH)
    r = r[r.index <= pd.Timestamp.now(tz="UTC")].ffill()
    return r.iloc[-1].dropna().to_dict()


def rates_text() -> str:
    r = rates_now()
    return " · ".join(f"**{c}** {r[c]:.2f}%" for c in G10 if c in r) + \
        "\n(overnight rates for USD/EUR/GBP; 3-month or call rates for others, which update monthly)"


def headlines(currency: Optional[str] = None, hours: int = 24, n: int = 3) -> list[dict]:
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)).isoformat()
    sql = "SELECT headline, url, currencies, score FROM news WHERE published_at >= ?"
    with db.connect() as c:
        rows = [dict(r) for r in c.execute(sql, (since,))]
    if currency:
        rows = [r for r in rows if currency in (r["currencies"] or "").split(",")]
    rows = [r for r in rows if any(x in G10 for x in (r["currencies"] or "").split(","))]
    return sorted(rows, key=lambda r: abs(r["score"] or 0), reverse=True)[:n]


def headlines_text(rows: list[dict]) -> str:
    if not rows:
        return "No scored headlines in the last 24h."
    out = []
    for r in rows:
        tone = "positive" if r["score"] > 0.2 else "negative" if r["score"] < -0.2 else "neutral"
        title = r["headline"].replace("[", "(").replace("]", ")")
        title = title if len(title) <= 95 else title[:92].rsplit(" ", 1)[0] + "…"
        link = f"[{title}]({r['url']})" if r.get("url") else title
        out.append(f"• {link} - {tone} ({r['currencies']})")
    return "\n".join(out)


# --------------------------------------------------------------------------- messages
def briefing(dry_run: bool = False, now: Optional[pd.Timestamp] = None) -> None:
    now = now or pd.Timestamp.now(tz="UTC")
    try:
        download_rates()
    except Exception as e:
        log.warning("rate refresh failed (%s)", type(e).__name__)
    check_rate_changes(dry_run=dry_run)
    z = news_z(now)
    fields = [("📅 High-impact events, next 24h", events_text(now)),
              ("🌡️ Volatility outlook, next FX day (model forecast)", vol_text()),
              ("📰 News tone vs normal", news_text(z)),
              ("🏦 Short-term interest rates", rates_text()),
              ("🗞️ Notable headlines (FinBERT)", headlines_text(headlines()))]
    _post(f"☀️ FX briefing - {now + pd.Timedelta(hours=2):%a %d %b %Y}",
          "What's scheduled, how volatile it's likely to be, and what the news is saying for the G10 currencies.",
          notify.BLUE, fields, dry_run=dry_run)


def event_reminders(now: pd.Timestamp, dry_run: bool = False) -> int:
    ev = fx_calendar.window(now, now + pd.Timedelta(minutes=MONITOR_EVENT_LEAD_MIN))
    sent = 0
    for e in ev.itertuples():
        key = f"mon_event:{e.currency}:{e.title}:{e.time.isoformat()}"
        if notify.kv_get(key):
            continue
        notify.kv_set(key, now.isoformat())
        extra = " · ".join(x for x in (f"forecast **{e.forecast}**" if e.forecast else "",
                                       f"previous **{e.previous}**" if e.previous else "") if x)
        _post(f"⏰ {e.currency}: {e.title} {_ts(e.time, 'R')}", f"{_ts(e.time, 'f')}\n{extra}\n"
              "High-impact release: spreads often widen and prices can jump around it.",
              notify.ORANGE, dry_run=dry_run)
        sent += 1
    return sent


def vol_spikes(now: pd.Timestamp, dry_run: bool = False) -> int:
    sent = 0
    for pair in vol_model.PAIRS:
        try:
            c = download_ohlcv(pair, period="30d", interval="1h")["Close"]
        except Exception:
            continue
        c = c[c.index + pd.Timedelta(hours=1) <= now]      # completed bars only
        r = np.log(c).diff().dropna()
        if len(r) < 200 or now - (r.index[-1] + pd.Timedelta(hours=1)) > pd.Timedelta(hours=2):
            continue
        normal = r.iloc[:-1].abs().median()
        x = abs(r.iloc[-1]) / normal if normal > 0 else 0
        key = f"mon_spike:{pair}"
        last = notify.kv_get(key)
        if x >= MONITOR_VOL_SPIKE_X and (not last or now - pd.Timestamp(last) > pd.Timedelta(hours=4)):
            notify.kv_set(key, now.isoformat())
            _post(f"⚡ {pair} moved {r.iloc[-1] * 100:+.2f}% in the last hour",
                  f"That's **{x:.1f}×** its typical hourly move over the past month "
                  f"(hour ending {_ts(r.index[-1] + pd.Timedelta(hours=1))}).", notify.ORANGE, dry_run=dry_run)
            sent += 1
    return sent


def news_shifts(now: pd.Timestamp, dry_run: bool = False) -> int:
    z = news_z(now)
    sent = 0
    for ccy, v in z.items():
        key = f"mon_news:{ccy}"
        last = notify.kv_get(key)
        if abs(v) >= MONITOR_NEWS_Z and (not last or now - pd.Timestamp(last) > pd.Timedelta(hours=12)):
            notify.kv_set(key, now.isoformat())
            word = "more positive" if v > 0 else "more negative"
            _post(f"📰 {ccy} news coverage is unusually {'positive' if v > 0 else 'negative'} ({v:+.1f}σ)",
                  f"The tone of global news involving {ccy}'s economy over the last 24h is {abs(v):.1f} standard "
                  f"deviations {word} than its 30-day norm.\n\n**Recent headlines**\n"
                  + headlines_text(headlines(ccy)), notify.GREEN if v > 0 else notify.RED, dry_run=dry_run)
            sent += 1
    return sent


def check_rate_changes(dry_run: bool = False) -> int:
    r = rates_now()
    sent = 0
    for ccy in DAILY_RATE_SERIES:
        if ccy not in r:
            continue
        key = f"mon_rate:{ccy}"
        prev = notify.kv_get(key)
        notify.kv_set(key, str(r[ccy]))
        if prev is not None and abs(float(prev) - r[ccy]) >= 0.15:   # DFF wiggles a few bp day to day
            _post(f"🏦 {ccy} overnight rate moved: {float(prev):.2f}% → {r[ccy]:.2f}%",
                  "Central-bank policy change (daily FRED data).", notify.ORANGE, dry_run=dry_run)
            sent += 1
    return sent


# --------------------------------------------------------------------------- schedule
def _due_once(key: str, period_key: str) -> bool:
    if notify.kv_get(key) == period_key:
        return False
    notify.kv_set(key, period_key)
    return True


def tick(now: Optional[pd.Timestamp] = None) -> None:
    """Called every minute by the scanner service."""
    now = now or pd.Timestamp.now(tz="UTC")
    fx_open = not (now.weekday() == 5 or (now.weekday() == 6 and now.hour < 21) or (now.weekday() == 4 and now.hour >= 22))
    event_reminders(now)
    # hourly, 20 minutes past (GDELT files and the last hourly bar are published by then)
    if now.minute >= 20 and _due_once("mon_hourly", now.strftime("%Y-%m-%d %H")):
        try:
            gdelt_live.update()
        except Exception as e:
            log.warning("GDELT update failed (%s)", type(e).__name__)
        news_shifts(now)
        if fx_open:
            vol_spikes(now)
    # morning briefing before each FX day: Sunday-Thursday evenings UTC
    due = now.hour > MONITOR_BRIEFING_HOUR_UTC or (now.hour == MONITOR_BRIEFING_HOUR_UTC and now.minute >= MONITOR_BRIEFING_MINUTE)
    if due and now.weekday() in (6, 0, 1, 2, 3) and _due_once("mon_briefing", now.strftime("%Y-%m-%d")):
        briefing(now=now)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["briefing", "alerts"])
    ap.add_argument("--dry-run", action="store_true", help="print instead of posting to Discord")
    args = ap.parse_args()
    db.init_db()
    now = pd.Timestamp.now(tz="UTC")
    if args.cmd == "briefing":
        briefing(dry_run=args.dry_run)
    else:
        print("reminders:", event_reminders(now, args.dry_run), "| spikes:", vol_spikes(now, args.dry_run),
              "| news:", news_shifts(now, args.dry_run))
    notify.flush()


if __name__ == "__main__":
    main()
