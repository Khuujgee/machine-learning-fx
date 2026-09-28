"""Built-in signal scanner - a free replacement for TradingView webhook alerts.

At every hourly bar close it scores every pair in the trained universe, ranks the signals by model
confidence and opens paper trades for the strongest ones (respecting MAX_OPEN_TRADES and one position
per pair). Every decision is logged to the `webhook_alerts` table with an alert_id like
`scan:2026-09-28T14:00:00+00:00:EURUSD`, so it shows up in GET /alerts next to real webhook alerts.

Usage:
  python -m src.scanner              # run forever: scan each hour + check SL/TP every minute
  python -m src.scanner --once       # one scan right now on the latest completed bars, then exit
  python -m src.scanner --no-monitor # scan only (use when the FastAPI server is running its own monitor)

Or set SCANNER_ENABLED=true to run it inside the FastAPI server (uvicorn src.webhook_server:app).
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import pandas as pd

from . import db
from .config import (MAX_BAR_STALENESS_HOURS, MODEL_PATH, MONITOR_INTERVAL_SEC, SCANNER_DELAY_SEC,
                     SCANNER_REFRESH_NEWS, SCANNER_WORKERS)
from .paper_engine import NoTrade, PaperEngine

log = logging.getLogger("scanner")

FRESH_BAR_HOURS = 0.75  # scheduled scans only trade on the bar that just closed
RETRY_WAIT_SEC = 60


# --------------------------------------------------------------------------- schedule
def fx_market_open(ts: pd.Timestamp) -> bool:
    """Spot FX trades from Sunday ~21:00 UTC to Friday ~21:00 UTC (DST shifts it by an hour; the
    staleness check in PaperEngine.predict catches the edge cases)."""
    wd, h = ts.weekday(), ts.hour
    if wd == 5:  # Saturday
        return False
    if wd == 4 and h >= 22:  # Friday late
        return False
    if wd == 6 and h < 21:  # Sunday before open
        return False
    return True


def next_scan_time(now: pd.Timestamp) -> pd.Timestamp:
    t = now.floor("h") + pd.Timedelta(seconds=SCANNER_DELAY_SEC)
    return t if t > now else t + pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- one scan
def _refresh_news() -> None:
    try:
        from .sentiment import ingest, scrape_headlines

        log.info("news refresh: %d new headlines scored", ingest(scrape_headlines()))
    except ImportError as e:
        log.warning("news refresh skipped (%s) - install transformers + torch for live sentiment", e)
    except Exception:
        log.exception("news refresh failed - scanning with existing sentiment")


def _predict_all(engine: PaperEngine, pairs: list[str], max_staleness: float) -> dict[str, Any]:
    """pair -> prediction dict, or the NoTrade / Exception that stopped it."""
    def one(pair: str):
        try:
            return pair, engine.predict(pair, max_staleness_hours=max_staleness)
        except Exception as e:  # NoTrade or data error; handled per pair
            return pair, e

    with ThreadPoolExecutor(max_workers=SCANNER_WORKERS) as pool:
        return dict(pool.map(one, pairs))


def run_scan(engine: PaperEngine, scheduled: bool = False, refresh_news: bool = SCANNER_REFRESH_NEWS) -> dict:
    """Score every pair, then open trades from most to least confident until limits are hit."""
    started = pd.Timestamp.now(tz="UTC")
    if refresh_news:
        _refresh_news()

    pairs = sorted(engine.pairs)
    max_staleness = FRESH_BAR_HOURS if scheduled else MAX_BAR_STALENESS_HOURS
    results = _predict_all(engine, pairs, max_staleness)

    # Yahoo sometimes publishes the just-closed bar a little late: retry those pairs once.
    late = [p for p, r in results.items() if isinstance(r, NoTrade) and "not published" in str(r)]
    if scheduled and late and len(late) < len(pairs):
        log.info("%d pairs missing the latest bar, retrying in %ds", len(late), RETRY_WAIT_SEC)
        time.sleep(RETRY_WAIT_SEC)
        results.update(_predict_all(engine, late, max_staleness))

    candidates, decisions = [], {}
    for pair, r in results.items():
        if isinstance(r, NoTrade):
            decisions[pair] = {"action": "no_trade", "reason": str(r), "prediction": None}
        elif isinstance(r, Exception):
            decisions[pair] = {"action": "error", "error": repr(r)}
        else:
            try:
                candidates.append((abs(r["prob_up"] - 0.5), pair, engine.direction_for(r["prob_up"]), r))
            except NoTrade as e:
                decisions[pair] = {"action": "no_trade", "reason": str(e), "prediction": r}

    # strongest conviction first, so MAX_OPEN_TRADES keeps the best signals
    opened = []
    for _, pair, direction, pred in sorted(candidates, key=lambda c: c[0], reverse=True):
        alert_id = f"scan:{pred['bar_close_time']}:{pair}"
        if db.alert_exists(alert_id):  # another scanner instance already handled this bar
            continue
        db.log_alert(alert_id, pair, {"source": "scanner", "bar_close_time": pred["bar_close_time"]})
        d = engine.try_open(pair, direction, pred, alert_id)
        db.update_alert(alert_id, "traded" if d["action"] == "opened" else "no_trade", d)
        decisions[pair] = d
        if d["action"] == "opened":
            opened.append({k: d["trade"][k] for k in ("id", "pair", "direction", "units", "entry_price",
                                                      "stop_loss", "take_profit", "prob_up")})

    # log the non-candidates too, so model calibration can be audited later from /alerts
    for pair, d in decisions.items():
        pred = d.get("prediction")
        if pred and not any(c[1] == pair for c in candidates):
            alert_id = f"scan:{pred['bar_close_time']}:{pair}"
            if not db.alert_exists(alert_id):
                db.log_alert(alert_id, pair, {"source": "scanner", "bar_close_time": pred["bar_close_time"]})
                db.update_alert(alert_id, "no_trade", d)

    probs = {p: round(r["prob_up"], 3) for p, r in results.items() if isinstance(r, dict)}
    summary = {
        "started": started.isoformat(timespec="seconds"),
        "seconds": round((pd.Timestamp.now(tz="UTC") - started).total_seconds(), 1),
        "pairs": len(pairs),
        "scored": len(probs),
        "signals": len(candidates),
        "opened": opened,
        "skipped": {p: d.get("reason") or d.get("error") for p, d in decisions.items()
                    if d["action"] != "opened" and p not in probs},
        "strongest": dict(sorted(probs.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)[:5]),
    }
    log.info("scan done in %.0fs: %d/%d scored, %d signals, %d opened %s", summary["seconds"],
             summary["scored"], summary["pairs"], summary["signals"], len(opened),
             [f"{o['direction']} {o['pair']}" for o in opened])
    return summary


# --------------------------------------------------------------------------- loops
def run_forever(engine: PaperEngine, monitor: bool = True, refresh_news: bool = SCANNER_REFRESH_NEWS) -> None:
    """Blocking loop: scan at each hourly close (market hours only); check SL/TP every minute."""
    next_scan = next_scan_time(pd.Timestamp.now(tz="UTC"))
    log.info("scanner started - %d pairs, next scan %s, monitor=%s", len(engine.pairs), next_scan, monitor)
    while True:
        now = pd.Timestamp.now(tz="UTC")
        if now >= next_scan:
            if fx_market_open(now - pd.Timedelta(hours=1)):
                try:
                    run_scan(engine, scheduled=True, refresh_news=refresh_news)
                except Exception:
                    log.exception("scan failed")
            else:
                log.info("FX market closed - skipping scan")
            next_scan = next_scan_time(pd.Timestamp.now(tz="UTC"))
        if monitor:
            try:
                closed = engine.check_open_trades()
                if closed:
                    log.info("closed %s", closed)
            except Exception:
                log.exception("monitor cycle failed")
        wait = (next_scan - pd.Timestamp.now(tz="UTC")).total_seconds()
        time.sleep(max(1.0, min(MONITOR_INTERVAL_SEC if monitor else wait, wait)))


async def scanner_task(engine: PaperEngine, stop) -> None:
    """Async variant used inside the FastAPI server (which already runs the SL/TP monitor)."""
    import asyncio

    while not stop.is_set():
        wait = (next_scan_time(pd.Timestamp.now(tz="UTC")) - pd.Timestamp.now(tz="UTC")).total_seconds()
        try:
            await asyncio.wait_for(stop.wait(), timeout=wait)
            return
        except asyncio.TimeoutError:
            pass
        if fx_market_open(pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)):
            try:
                await asyncio.to_thread(run_scan, engine, True)
            except Exception:
                log.exception("scan failed")


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="run a single scan now and exit")
    ap.add_argument("--no-monitor", action="store_true", help="don't check SL/TP (server does it)")
    ap.add_argument("--no-news", action="store_true", help="skip the news refresh before each scan")
    args = ap.parse_args(argv)

    if not MODEL_PATH.exists():
        raise SystemExit(f"No model at {MODEL_PATH} - run `python -m src.train` first.")
    db.init_db()
    engine = PaperEngine(MODEL_PATH)
    refresh_news = SCANNER_REFRESH_NEWS and not args.no_news
    if args.once:
        import json

        print(json.dumps(run_scan(engine, scheduled=False, refresh_news=refresh_news), indent=2, default=str))
        return
    run_forever(engine, monitor=not args.no_monitor, refresh_news=refresh_news)


if __name__ == "__main__":
    main()
