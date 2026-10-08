"""FX carry strategy - paper trading (its own account, separate from the hourly ML trades).

Rule (what src/research_daily.py tested): for every pair where both short-term interest rates are known,
hold the higher-yielding currency against the lower-yielding one. Every pair gets the same risk budget
(position size ~ 1 / its recent volatility). Rebalanced once a week; a position only changes when its
direction flips or its size drifts more than CARRY_RESIZE_TOLERANCE from target. No stop-losses.

P&L of a position = spot move + interest earned/paid (rate difference, daily accrual)
                    - broker overnight-interest mark-up (CARRY_SWAP_MARKUP_PCT per year on the notional)
                    - half the pair's assumed spread on entry and again on exit.

    python -m src.carry status          # account + positions (also shown by ./status.sh)
    python -m src.carry rebalance       # rebalance now (normally done by the scanner service weekly)
    python -m src.carry close-all       # close every open carry position at current prices (e.g. switching off)
"""
from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

from . import db, notify
from . import config as _cfg
from .backtest import spread_table
from .config import (CANDIDATE_PAIRS, CARRY_MAX_GROSS_LEVERAGE, CARRY_MAX_PAIR_PCT, CARRY_MIN_RATE_DIFF, CARRY_PAIR_VOL,
                     CARRY_REBALANCE_HOUR_UTC, CARRY_REBALANCE_WEEKDAY, CARRY_RESIZE_TOLERANCE,
                     CARRY_STARTING_EQUITY, CARRY_SWAP_MARKUP_PCT, CARRY_TARGET_VOL)
from .macro_data import RATES_PATH, _history, download_rates
from .market_data import latest_price, quote_to_usd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- inputs
def latest_rates(refresh: bool = True) -> dict[str, float]:
    """Latest knowable short rate per currency (FRED). Uses the cached file if FRED is down."""
    if refresh:
        try:
            download_rates()
        except Exception as e:
            log.warning("rate refresh failed (%s) - using cached rates", type(e).__name__)
    r = pd.read_parquet(RATES_PATH)
    r = r[r.index <= pd.Timestamp.now(tz="UTC")].ffill()
    return r.iloc[-1].dropna().to_dict()


def daily_returns(pair: str) -> pd.Series:
    """~3 months of daily log returns (Yahoo)."""
    d = _history(f"{pair}=X", period="3mo", interval="1d", auto_adjust=False)["Close"]
    d.index = pd.to_datetime(d.index).tz_localize(None).normalize() if d.index.tz is not None else d.index.normalize()
    d = d[(d > 0) & ~d.index.duplicated(keep="last")]
    return np.log(d).diff().dropna()


def eligible_pairs(rates: dict) -> list[str]:
    return [p for p in CANDIDATE_PAIRS if p[:3] in rates and p[3:] in rates]


# --------------------------------------------------------------------------- accounting
def position_pnl(p: dict, price: float, when: pd.Timestamp, q_usd: Optional[float] = None) -> dict:
    """Spot + carry - mark-up - trading costs, for an open position valued at `price` at time `when`."""
    q_usd = q_usd if q_usd is not None else p["quote_usd"]
    days = max((when - pd.Timestamp(p["entry_time"])).total_seconds() / 86400, 0.0)
    spot = p["side"] * (price - p["entry_price"]) * p["units"] * q_usd
    carry = p["side"] * (p["rate_base"] - p["rate_quote"]) / 100 * p["notional_usd"] * days / 365
    markup = CARRY_SWAP_MARKUP_PCT / 100 * p["notional_usd"] * days / 365
    trading = p["spread_bps"] / 2 / 1e4 * p["notional_usd"] * 2   # half spread in + half spread out
    return {"spot": spot, "carry": carry, "markup": markup, "trading": trading,
            "pnl": spot + carry - markup - trading, "days": days}


def equity() -> float:
    with db.connect() as c:
        realized = c.execute("SELECT COALESCE(SUM(pnl_usd),0) FROM carry_positions WHERE status='closed'").fetchone()[0]
    return CARRY_STARTING_EQUITY + realized


def open_positions() -> dict[str, dict]:
    with db.connect() as c:
        return {r["pair"]: dict(r) for r in c.execute("SELECT * FROM carry_positions WHERE status='open'")}


def _open(pair: str, side: int, units: float, price: float, q_usd: float, rates: dict, spread: float) -> dict:
    p = dict(pair=pair, side=side, units=units, entry_price=price, entry_time=db.utcnow_iso(), quote_usd=q_usd,
             notional_usd=units * price * q_usd, rate_base=rates[pair[:3]], rate_quote=rates[pair[3:]],
             spread_bps=spread)
    cols = list(p)
    with db.connect() as c:
        p["id"] = c.execute(f"INSERT INTO carry_positions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                            [p[k] for k in cols]).lastrowid
    return p


def _close(p: dict, price: float, reason: str) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    r = position_pnl(p, price, now)
    with db.connect() as c:
        c.execute("""UPDATE carry_positions SET status='closed', exit_price=?, exit_time=?, exit_reason=?,
                     spot_pnl=?, carry_pnl=?, markup_cost=?, trading_cost=?, pnl_usd=? WHERE id=? AND status='open'""",
                  (price, now.isoformat(timespec="seconds"), reason, r["spot"], r["carry"], r["markup"],
                   r["trading"], round(r["pnl"], 2), p["id"]))
    return r


# --------------------------------------------------------------------------- strategy
def rebalance(refresh_rates: bool = True) -> dict:
    rates = latest_rates(refresh_rates)
    pairs = eligible_pairs(rates)
    spreads = spread_table(pairs)
    held = open_positions()
    eq = equity()
    n = len(pairs)
    # 1) direction and relative weight per pair (equal risk: weight ~ 1/volatility), as in the backtest
    plan, rets, skipped = {}, {}, {}
    for pair in pairs:
        diff = rates[pair[:3]] - rates[pair[3:]]
        side = 0 if abs(diff) < CARRY_MIN_RATE_DIFF else (1 if diff > 0 else -1)
        try:
            r = daily_returns(pair)
            vol = float(r.tail(20).std() * np.sqrt(252))
            price = latest_price(pair)
            q_usd = quote_to_usd(pair, price)
        except Exception as e:
            skipped[pair] = type(e).__name__
            continue
        if not np.isfinite(vol) or vol <= 0:
            skipped[pair] = "no volatility data"
            continue
        plan[pair] = dict(side=side, w=side * min(CARRY_PAIR_VOL / vol, 5.0) / n, price=price, q_usd=q_usd, diff=diff)
        rets[pair] = r
    # 2) scale the whole basket to CARRY_TARGET_VOL using the last ~60 days of the pairs' actual co-movement
    w = pd.Series({p: v["w"] for p, v in plan.items()})
    R = pd.DataFrame(rets).tail(60).fillna(0.0)
    port_vol = float((R[w.index] @ w).std() * np.sqrt(252)) if len(R) > 20 and w.abs().sum() > 0 else np.nan
    gross = w.abs().sum()
    scale = CARRY_TARGET_VOL / port_vol if np.isfinite(port_vol) and port_vol > 0 else 1.0
    scale = min(scale, CARRY_MAX_GROSS_LEVERAGE / gross) if gross > 0 else 0.0
    # 3) trade only what changed
    opened, closed, kept = [], [], []
    for pair, v in plan.items():
        notional = min(eq * abs(v["w"]) * scale, eq * CARRY_MAX_PAIR_PCT / 100)
        units = round(notional / (v["price"] * v["q_usd"]))
        cur = held.pop(pair, None)
        if cur and v["side"] != 0 and cur["side"] == v["side"] and units > 0 \
                and abs(units / cur["units"] - 1) <= CARRY_RESIZE_TOLERANCE:
            kept.append(pair)
            continue
        if cur:
            r = _close(cur, v["price"], "rebalance" if v["side"] == cur["side"] else "direction flip")
            closed.append((pair, cur["side"], r["pnl"]))
        if v["side"] != 0 and units > 0:
            _open(pair, v["side"], units, v["price"], v["q_usd"], rates, spreads[pair])
            opened.append((pair, v["side"], round(v["diff"], 2)))
    for pair, cur in held.items():  # pairs that lost rate or price data
        try:
            r = _close(cur, latest_price(pair), "no longer eligible")
            closed.append((pair, cur["side"], r["pnl"]))
        except Exception as e:
            skipped[pair] = type(e).__name__
    summary = {"pairs": n, "opened": opened, "closed": closed, "kept": len(kept), "skipped": skipped, "equity": eq,
               "basket_vol_unscaled": port_vol, "scale": scale, "gross_leverage": gross * scale}
    log.info("carry rebalance: %d pairs, %d opened, %d closed, %d kept, %d skipped; basket vol %.1f%% -> x%.2f, "
             "gross %.2fx equity", n, len(opened), len(closed), len(kept), len(skipped), port_vol * 100, scale,
             gross * scale)
    fmt = lambda xs: ", ".join(f"{'+' if s > 0 else '-'}{p}" for p, s, *_ in xs) or "none"
    notify.info("Carry rebalance (paper)",
                f"Opened: {fmt(opened)}\nClosed: {fmt(closed)}\nKept: {len(kept)}   Skipped: {len(skipped)}\n"
                f"Target vol {CARRY_TARGET_VOL:.0%}, gross exposure {gross * scale:.2f}x equity. "
                f"Realized equity: ${eq:,.2f}")
    return summary


def close_all(reason: str = "strategy switched off") -> list[tuple]:
    closed = []
    for pair, p in open_positions().items():
        try:
            r = _close(p, latest_price(pair), reason)
            closed.append((pair, p["side"], r["pnl"]))
        except Exception as e:
            log.warning("%s: could not close (%s)", pair, type(e).__name__)
    total = sum(c[2] for c in closed)
    log.info("closed %d carry positions, P&L $%+.2f", len(closed), total)
    notify.info("Carry positions closed (paper)", f"{len(closed)} positions closed ({reason}), P&L ${total:+,.2f}. "
                                                  f"Carry equity: ${equity():,.2f}")
    return closed


def next_rebalance_due(now: pd.Timestamp) -> pd.Timestamp:
    """This week's rebalance time (Monday 08:00 UTC by default)."""
    start = (now - pd.Timedelta(days=now.weekday())).normalize()
    return start + pd.Timedelta(days=CARRY_REBALANCE_WEEKDAY, hours=CARRY_REBALANCE_HOUR_UTC)


def maybe_rebalance(now: Optional[pd.Timestamp] = None) -> Optional[dict]:
    """Called every minute by the scanner loop: rebalance once per week (or right away if never done)."""
    now = now or pd.Timestamp.now(tz="UTC")
    due = next_rebalance_due(now)
    last = notify.kv_get("carry_last_rebalance")
    never = last is None
    if not never and (now < due or pd.Timestamp(last) >= due):
        return None
    if now.weekday() == 5 or (now.weekday() == 6 and now.hour < 22) or (now.weekday() == 4 and now.hour >= 21):
        return None  # FX market closed / about to close: wait for Monday
    summary = rebalance()
    notify.kv_set("carry_last_rebalance", now.isoformat())
    return summary


# --------------------------------------------------------------------------- reporting
def report(live_prices: bool = True) -> str:
    pos = open_positions()
    now = pd.Timestamp.now(tz="UTC")
    lines, tot = [], {"spot": 0.0, "carry": 0.0, "markup": 0.0, "trading": 0.0, "pnl": 0.0}
    for pair, p in sorted(pos.items()):
        try:
            px = latest_price(pair) if live_prices else p["entry_price"]
        except Exception:
            px = p["entry_price"]
        r = position_pnl(p, px, now)
        for k in tot:
            tot[k] += r[k]
        lines.append(f"  {pair:7} {'long ' if p['side'] > 0 else 'short'} rate diff {p['side'] * (p['rate_base'] - p['rate_quote']):+5.2f}%"
                     f"  ${p['notional_usd']:>9,.0f}  spot {r['spot']:+8,.0f}  carry {r['carry']:+7,.0f}  "
                     f"total {r['pnl']:+8,.0f}  ({r['days']:.0f}d)")
    eq = equity()
    with db.connect() as c:
        closed = c.execute("SELECT COUNT(*), COALESCE(SUM(pnl_usd),0), COALESCE(SUM(carry_pnl),0) FROM carry_positions "
                           "WHERE status='closed'").fetchone()
    last = notify.kv_get("carry_last_rebalance")
    if not _cfg.CARRY_ENABLED and not pos:
        return (f"CARRY (paper): switched off (CARRY_ENABLED=false). {closed[0]} closed positions, "
                f"realized ${closed[1]:+,.2f}, carry equity ${eq:,.2f}")
    head = (f"CARRY (paper): equity ${eq + tot['pnl']:,.2f} incl. open (start ${CARRY_STARTING_EQUITY:,.0f}), "
            f"{len(pos)} open, {closed[0]} closed (${closed[1]:+,.0f}).  Open P&L: spot {tot['spot']:+,.0f}, "
            f"carry {tot['carry']:+,.0f}, mark-up -{tot['markup']:,.0f}, spreads -{tot['trading']:,.0f}.  "
            f"Last rebalance: {pd.Timestamp(last):%m-%d %H:%M} UTC" if last else
            f"CARRY (paper): not started yet (first rebalance runs when the scanner service starts on a weekday)")
    return head + ("\n" + "\n".join(lines) if lines else "")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "rebalance", "close-all"])
    args = ap.parse_args()
    db.init_db()
    if args.cmd == "close-all":
        c = close_all()
        notify.flush()
        print(f"closed {len(c)} positions, P&L ${sum(x[2] for x in c):+,.2f}")
    if args.cmd == "rebalance":
        s = rebalance()
        notify.kv_set("carry_last_rebalance", pd.Timestamp.now(tz="UTC").isoformat())
        notify.flush()
        print(f"opened {len(s['opened'])}, closed {len(s['closed'])}, kept {s['kept']}, skipped {s['skipped'] or 'none'}")
    print(report())


if __name__ == "__main__":
    main()
