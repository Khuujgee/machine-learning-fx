"""Discord alerts (optional): trades opened/closed, scanner problems, daily summary.

Enabled by setting DISCORD_WEBHOOK_URL in .env. With no URL every function here is a silent no-op, so the
rest of the code can call them unconditionally.

- Sending never blocks or breaks trading: messages go through one background worker thread and every
  failure is swallowed (and logged without the URL, which is a secret).
- Test your setup:  python -m src.notify test
"""
from __future__ import annotations

import functools
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd
import requests

from . import db
from .config import DISCORD_WEBHOOK_URL

log = logging.getLogger(__name__)

GREEN, RED, BLUE, ORANGE, GREY = 0x2ECC71, 0xE74C3C, 0x3498DB, 0xF39C12, 0x95A5A6
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="discord")  # one worker keeps messages in order
_URL_RE = re.compile(r"^https://(?:(?:ptb|canary)\.)?(?:discord|discordapp)\.com/api/webhooks/\d+/[\w-]+$")


def _valid(url: str) -> bool:
    if _URL_RE.match(url):
        return True
    p = urlparse(url)  # allow a local server so tests never need the real webhook
    return p.scheme == "http" and p.hostname in ("127.0.0.1", "localhost")


def enabled() -> bool:
    return bool(DISCORD_WEBHOOK_URL) and _valid(DISCORD_WEBHOOK_URL)


def _post(payload: dict) -> bool:
    for attempt in range(3):
        try:
            r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            if r.status_code == 429:  # rate limited: wait as long as Discord asks
                time.sleep(min(float(r.json().get("retry_after", 2)), 30))
                continue
            if r.ok:
                return True
            log.warning("discord: HTTP %s", r.status_code)
            return False
        except Exception as e:  # never include the exception text: requests puts the URL in it
            log.warning("discord: send failed (%s)", type(e).__name__)
            time.sleep(2)
    return False


def _send(title: str, description: str = "", color: int = BLUE, fields: Optional[list[tuple]] = None) -> None:
    if not enabled():
        return
    embed: dict[str, Any] = {"title": title[:250], "description": description[:3500], "color": color,
                             "timestamp": pd.Timestamp.now(tz="UTC").isoformat()}
    if fields:
        embed["fields"] = [{"name": n, "value": str(v)[:1000] or "-", "inline": True} for n, v in fields][:25]
    _pool.submit(_post, {"username": "FX Paper Trader", "embeds": [embed], "allowed_mentions": {"parse": []}})


def _safe(fn):
    """A broken alert must never break trading: log the error type only and carry on."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            log.warning("discord: %s failed (%s)", fn.__name__, type(e).__name__)
    return wrapper


def _px(x: float) -> str:
    return f"{x:,.2f}" if abs(x) >= 100 else f"{x:.5f}"


# --------------------------------------------------------------------------- trade events
@_safe
def trade_opened(t: dict, pred: dict) -> None:
    arrow = "🟢 LONG" if t["direction"] == "long" else "🔴 SHORT"
    risk = abs(t["entry_price"] - t["stop_loss"]) * t["units"] * t["quote_usd"]
    _send(f"{arrow} {t['pair']}  (paper)", "", GREEN if t["direction"] == "long" else RED, [
        ("Entry", _px(t["entry_price"])), ("Stop-loss", _px(t["stop_loss"])), ("Take-profit", _px(t["take_profit"])),
        ("Size", f"{t['units']:,.0f} units"), ("Risk", f"${risk:,.0f}"),
        ("Model P(up)", f"{pred['prob_up']:.1%}"),
    ])


@_safe
def trade_closed(t: dict, exit_price: float, reason: str, pnl: float) -> None:
    equity = db.account_summary()["equity"]
    won = pnl > 0
    label = {"take_profit": "take-profit hit", "stop_loss": "stop-loss hit", "time_exit": "time limit reached",
             "manual": "closed manually"}.get(reason, reason)
    _send(f"{'✅' if won else '❌'} {t['pair']} {t['direction']} closed: {label}", "", GREEN if won else RED, [
        ("P&L", f"${pnl:+,.2f}"), ("Entry → exit", f"{_px(t['entry_price'])} → {_px(exit_price)}"),
        ("Equity", f"${equity:,.2f}"),
    ])


# --------------------------------------------------------------------------- scanner / system events
@_safe
def info(title: str, text: str = "") -> None:
    _send(title, text, BLUE)


@_safe
def warning(title: str, text: str = "") -> None:
    _send(f"⚠️ {title}", text, ORANGE)


@_safe
def recovered(title: str, text: str = "") -> None:
    _send(f"✅ {title}", text, GREEN)


# --------------------------------------------------------------------------- daily summary
@_safe
def daily_summary() -> None:
    """Account snapshot + last 24h activity. Fetches live prices, so call it from a worker, not a hot path."""
    from .market_data import latest_price  # local import: keeps `python -m src.notify test` fast

    acct = db.account_summary()
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)).isoformat()
    with db.connect() as c:
        opened = c.execute("SELECT COUNT(*) FROM trades WHERE entry_time >= ?", (since,)).fetchone()[0]
        closed = c.execute("SELECT COUNT(*), COALESCE(SUM(pnl_usd),0), COALESCE(SUM(pnl_usd>0),0) FROM trades "
                           "WHERE status='closed' AND exit_time >= ?", (since,)).fetchone()
    unreal, lines = 0.0, []
    for t in db.get_open_trades():
        try:
            px = latest_price(t["pair"])
            pnl = (1 if t["direction"] == "long" else -1) * (px - t["entry_price"]) * t["units"] * t["quote_usd"]
            unreal += pnl
            lines.append(f"`{t['pair']}` {t['direction']:5} {pnl:+9,.0f}")
        except Exception:
            lines.append(f"`{t['pair']}` {t['direction']:5}       n/a")
    news = db.news_summary()
    win = f"{acct['win_rate']:.0%}" if acct["win_rate"] is not None else "-"
    _send("📊 Daily summary (paper account)", "\n".join(lines) or "No open positions.", BLUE, [
        ("Equity (realized)", f"${acct['equity']:,.2f}"),
        ("Unrealized", f"${unreal:+,.0f}"),
        ("Equity incl. open", f"${acct['equity'] + unreal:,.2f}"),
        ("Last 24h", f"{opened} opened, {closed[0]} closed (${closed[1]:+,.0f})"),
        ("All-time", f"{acct['closed_trades']} closed, win rate {win}"),
        ("News archive", f"{news['total']:,} headlines"),
    ])


# --------------------------------------------------------------------------- tiny key/value store
def kv_get(key: str) -> Optional[str]:
    with db.connect() as c:
        r = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return r["value"] if r else None


def kv_set(key: str, value: str) -> None:
    with db.connect() as c:
        c.execute("INSERT INTO kv(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, value))


def flush(timeout: float = 20.0) -> None:
    """Wait for queued messages (used by short-lived commands)."""
    done = _pool.submit(lambda: None)
    try:
        done.result(timeout=timeout)
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if not DISCORD_WEBHOOK_URL:
            raise SystemExit("DISCORD_WEBHOOK_URL is not set. Add it to .env first (see README).")
        if not enabled():
            raise SystemExit("DISCORD_WEBHOOK_URL doesn't look like a Discord webhook "
                             "(expected https://discord.com/api/webhooks/<id>/<token>).")
        ok = _post({"username": "FX Paper Trader", "embeds": [{
            "title": "✅ Discord alerts are connected", "color": GREEN,
            "description": "You'll get trade opens/closes, scanner problems and a daily summary here."}]})
        raise SystemExit(0 if ok else "Send failed (check the webhook URL is still valid).")
    print("usage: python -m src.notify test")
