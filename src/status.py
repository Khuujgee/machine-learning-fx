"""Print the paper-trading account, open positions (with unrealized P&L) and recent closed trades.

Usage: ./status.sh   (works from any folder)   or   python -m src.status [--closed 10]
"""
from __future__ import annotations

import argparse
import logging
import subprocess

import pandas as pd

from . import carry, db, notify
from .config import HOURLY_ML_ENABLED
from .market_data import latest_price


def scanner_status() -> str:
    """Is a scanner process alive, and how did the last scan go?"""
    try:
        pids = subprocess.run(["pgrep", "-f", "src.scanner"], capture_output=True, text=True).stdout.split()
    except FileNotFoundError:
        pids = []
    line = f"Scanner: RUNNING (pid {', '.join(pids)})" if pids else "Scanner: NOT RUNNING"
    last = db.last_scan()
    if last:
        started = pd.Timestamp(last["started_at"])
        mins = (pd.Timestamp.now(tz="UTC") - started).total_seconds() / 60
        line += (f"\nLast scan: {mins:.0f} min ago ({started:%m-%d %H:%M} UTC), took {last['seconds']:.0f}s, "
                 f"{last['scored']}/{last['pairs']} pairs scored, {last['signals']} signals, {last['opened']} opened")
        if last["errors"]:
            line += f"\n  !! {last['errors']} pairs failed, e.g. {last['first_error']}  (network down / Mac asleep?)"
        if not HOURLY_ML_ENABLED:
            line += "\n  Hourly ML is PAUSED (HOURLY_ML_ENABLED=false): no new hourly trades; news still refreshes hourly"
        elif mins > 75 and started.weekday() < 5:
            line += "\n  !! no scan for over an hour - is the Mac asleep?"
    return line


def news_status() -> str:
    n = db.news_summary()
    if not n["total"]:
        return "News: no headlines stored yet (is FinBERT installed? see README)"
    now = pd.Timestamp.now(tz="UTC")
    oldest, newest = pd.Timestamp(n["oldest"]), pd.Timestamp(n["newest"])
    span_days = (now - oldest).total_seconds() / 86400
    age_h = (now - newest).total_seconds() / 3600
    line = (f"News: {n['total']:,} headlines scored, {n['last_24h']} in the last 24h, "
            f"archive spans {span_days:.1f} days (since {oldest:%m-%d}), newest {age_h:.0f}h old")
    if age_h > 12 and now.weekday() < 5:
        line += "\n  !! no new headlines for 12h+ - is the scanner running / FinBERT installed?"
    return line


def main() -> None:
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed", type=int, default=10, help="how many recent closed trades to show")
    args = ap.parse_args()

    print(scanner_status())
    print(news_status())
    print("Alerts: Discord ON" if notify.enabled() else "Alerts: Discord OFF (set DISCORD_WEBHOOK_URL in .env)")
    print()
    print(carry.report())
    print()
    print("HOURLY ML (paper):")
    acct = db.account_summary()
    win = f"{acct['win_rate']:.0%}" if acct["win_rate"] is not None else "-"
    print(f"Equity ${acct['equity']:,.2f}   realized P&L ${acct['realized_pnl']:+,.2f}   "
          f"closed {acct['closed_trades']} (win rate {win})   open {acct['open_trades']}")

    open_trades = db.list_trades("open", 100)
    if open_trades:
        print("\nOPEN")
        print(f"{'#':>3} {'pair':7} {'dir':5} {'units':>10} {'entry':>11} {'now':>11} {'SL':>11} {'TP':>11} {'unreal $':>10}  opened (UTC)")
        total = 0.0
        for t in reversed(open_trades):
            try:
                px = latest_price(t["pair"])
                sign = 1 if t["direction"] == "long" else -1
                pnl = sign * (px - t["entry_price"]) * t["units"] * t["quote_usd"]
                total += pnl
                px_s, pnl_s = f"{px:11.5f}", f"{pnl:+10,.0f}"
            except Exception:
                px_s, pnl_s = f"{'n/a':>11}", f"{'n/a':>10}"
            print(f"{t['id']:>3} {t['pair']:7} {t['direction']:5} {t['units']:>10,.0f} {t['entry_price']:11.5f} {px_s} "
                  f"{t['stop_loss']:11.5f} {t['take_profit']:11.5f} {pnl_s}  {t['entry_time'][:16].replace('T', ' ')}")
        print(f"{'unrealized total':>84} {total:+10,.0f}  (before costs)")

    closed = db.list_trades("closed", args.closed)
    if closed:
        print(f"\nLAST {len(closed)} CLOSED")
        print(f"{'#':>3} {'pair':7} {'dir':5} {'entry':>11} {'exit':>11} {'reason':12} {'P&L $':>10}  closed (UTC)")
        for t in closed:
            print(f"{t['id']:>3} {t['pair']:7} {t['direction']:5} {t['entry_price']:11.5f} {t['exit_price']:11.5f} "
                  f"{t['exit_reason']:12} {t['pnl_usd']:+10,.2f}  {str(t['exit_time'])[:16].replace('T', ' ')}")


if __name__ == "__main__":
    main()
