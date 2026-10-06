"""Inference + simulated execution.

PaperEngine.handle_signal()   : live features -> XGBoost probability -> (maybe) open a paper trade
PaperEngine.check_open_trades(): replay 1m bars since the last check and close trades that hit SL/TP
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from typing import Any, Optional

import joblib
import pandas as pd

from . import db, notify
from .config import (ALLOW_UNTRAINED_PAIRS, BAR_HOURS, COST_BPS, LOT_STEP, MAX_BAR_STALENESS_HOURS,
                     MAX_HOLD_HOURS, MAX_LEVERAGE, MAX_OPEN_TRADES, MAX_TRADES_PER_CURRENCY, MODEL_PATH, ONE_POSITION_PER_PAIR, REQUIRE_TV_AGREEMENT,
                     RISK_PER_TRADE, SL_ATR_MULT, TP_ATR_MULT, UNIVERSE_PATH)
from .features import build_feature_frame
from .market_data import download_ohlcv, latest_price, minute_bars, quote_to_usd
from .sentiment import hourly_currency_sentiment

log = logging.getLogger(__name__)

_SIGNAL_MAP = {"buy": "long", "long": "long", "sell": "short", "short": "short"}


def currency_exposure(pair: str, direction: str) -> dict[str, int]:
    """Which way a trade bets on each currency: long EURMXN = {EUR: +1, MXN: -1}; short flips the signs."""
    sign = 1 if direction == "long" else -1
    return {pair[:3]: sign, pair[3:]: -sign}


FX_CLOSE_WEEKDAY, FX_CLOSE_HOUR_UTC = 4, 22  # spot FX closes Friday ~22:00 UTC


def hold_crosses_weekend(now: pd.Timestamp, hours: float) -> bool:
    """True if a trade opened now with a `hours` limit could still be open when the market shuts for the weekend."""
    if hours <= 0:
        return False
    return now.weekday() == FX_CLOSE_WEEKDAY and now.hour + now.minute / 60 + hours > FX_CLOSE_HOUR_UTC


class NoTrade(Exception):
    """Raised when a signal is rejected for a normal business reason (not an error)."""


def normalize_ticker(raw: str) -> Optional[str]:
    """'FX:EURUSD', 'OANDA:EUR_USD', 'EUR/USD', 'EURUSD=X' -> 'EURUSD'."""
    s = raw.upper().split(":")[-1].replace("=X", "")
    s = re.sub(r"[^A-Z]", "", s)
    return s if len(s) == 6 else None


class PaperEngine:
    def __init__(self, model_path=MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features: list[str] = bundle["features"]
        self.threshold: float = bundle["threshold"]
        self.meta = {k: v for k, v in bundle.items() if k not in ("model", "cv")}
        self.pairs = set(bundle.get("pairs", []))
        if UNIVERSE_PATH.exists():
            self.pairs |= set(json.loads(UNIVERSE_PATH.read_text())["pairs"])
        self._trade_lock = threading.Lock()
        log.info("model loaded: %s", self.meta)

    # ------------------------------------------------------------------ inference
    def predict(self, pair: str, max_staleness_hours: float = MAX_BAR_STALENESS_HOURS) -> dict[str, Any]:
        if not ALLOW_UNTRAINED_PAIRS and self.pairs and pair not in self.pairs:
            raise NoTrade(f"{pair} is not in the trained universe")
        now = pd.Timestamp.now(tz="UTC")
        bars = download_ohlcv(pair, period="60d", interval="1h")
        bars = bars[bars.index + pd.Timedelta(hours=BAR_HOURS) <= now]  # completed bars only (as in training)
        if len(bars) < 100:
            raise NoTrade(f"not enough completed bars for {pair} ({len(bars)})")

        sentiment = hourly_currency_sentiment(since=now - pd.Timedelta(days=3), until=now)
        feat = build_feature_frame(bars, pair, sentiment, with_label=False)
        row = feat.iloc[-1]
        age_h = (now - row["close_time"]).total_seconds() / 3600
        if age_h > max_staleness_hours:
            raise NoTrade(f"latest bar closed {age_h:.1f}h ago (market closed or data not published yet)")

        X = row[self.features].astype(float).to_frame().T
        prob_up = float(self.model.predict_proba(X)[:, 1][0])
        return {
            "prob_up": prob_up,
            "atr": float(row["atr"]),
            "bar_close_time": row["close_time"].isoformat(),
            "features": {k: round(float(v), 6) for k, v in row[self.features].items()},
        }

    def direction_for(self, prob_up: float, external_signal: Optional[str] = None) -> str:
        """Model direction, or NoTrade if inside the neutral band / contradicting an external signal."""
        if prob_up >= self.threshold:
            direction = "long"
        elif prob_up <= 1 - self.threshold:
            direction = "short"
        else:
            raise NoTrade(f"prob_up={prob_up:.3f} inside neutral band "
                          f"[{1 - self.threshold:.2f}, {self.threshold:.2f}]")
        ext = _SIGNAL_MAP.get(str(external_signal or "").lower())
        if REQUIRE_TV_AGREEMENT and ext and ext != direction:
            raise NoTrade(f"model says {direction} (p={prob_up:.3f}) but the alert signal is {ext}")
        return direction

    # ------------------------------------------------------------------ signal handling
    def handle_signal(self, pair: str, alert: dict[str, Any], alert_id: Optional[str] = None) -> dict[str, Any]:
        """Webhook path: predict one pair, then maybe trade it."""
        pred: Optional[dict] = None
        try:
            pred = self.predict(pair)
            direction = self.direction_for(pred["prob_up"], alert.get("signal"))
            return self.try_open(pair, direction, pred, alert_id)
        except NoTrade as e:
            return {"action": "no_trade", "reason": str(e), "prediction": pred}

    def try_open(self, pair: str, direction: str, pred: dict, alert_id: Optional[str] = None) -> dict[str, Any]:
        """Apply portfolio limits and open the trade. Serialised so concurrent signals can't overshoot limits."""
        try:
            with self._trade_lock:
                open_trades = db.get_open_trades()
                if ONE_POSITION_PER_PAIR and any(t["pair"] == pair for t in open_trades):
                    raise NoTrade(f"already holding an open {pair} position")
                if len(open_trades) >= MAX_OPEN_TRADES:
                    raise NoTrade(f"max open trades reached ({MAX_OPEN_TRADES})")
                if hold_crosses_weekend(pd.Timestamp.now(tz="UTC"), MAX_HOLD_HOURS):
                    raise NoTrade(f"a {MAX_HOLD_HOURS:g}h hold would run into the weekend close")
                if MAX_TRADES_PER_CURRENCY > 0:
                    for ccy, side in currency_exposure(pair, direction).items():
                        same = [t["pair"] for t in open_trades
                                if currency_exposure(t["pair"], t["direction"]).get(ccy) == side]
                        if len(same) >= MAX_TRADES_PER_CURRENCY:
                            raise NoTrade(f"already {len(same)} trades {'long' if side > 0 else 'short'} {ccy} "
                                          f"({', '.join(same)}); limit {MAX_TRADES_PER_CURRENCY}")
                trade = self._open_trade(pair, direction, pred, alert_id)
            return {"action": "opened", "trade": trade, "prediction": pred}
        except NoTrade as e:
            return {"action": "no_trade", "reason": str(e), "prediction": pred}

    def equity(self) -> float:
        return db.account_summary()["equity"]

    def _open_trade(self, pair: str, direction: str, pred: dict, alert_id: Optional[str]) -> dict:
        entry = latest_price(pair)
        sign = 1 if direction == "long" else -1
        sl_dist, tp_dist = SL_ATR_MULT * pred["atr"], TP_ATR_MULT * pred["atr"]
        q_usd = quote_to_usd(pair, entry)

        equity = self.equity()
        units_by_risk = equity * RISK_PER_TRADE / (sl_dist * q_usd)
        units_by_leverage = equity * MAX_LEVERAGE / (entry * q_usd)
        units = math.floor(min(units_by_risk, units_by_leverage) / LOT_STEP) * LOT_STEP
        if units <= 0:
            raise NoTrade("position size rounds to zero")

        trade = {
            "alert_id": alert_id,
            "pair": pair,
            "direction": direction,
            "units": units,
            "entry_price": entry,
            "entry_time": db.utcnow_iso(),
            "stop_loss": entry - sign * sl_dist,
            "take_profit": entry + sign * tp_dist,
            "quote_usd": q_usd,
            "prob_up": pred["prob_up"],
            "atr": pred["atr"],
            "max_hold_hours": MAX_HOLD_HOURS or None,
        }
        trade["id"] = db.insert_trade(trade)
        notify.trade_opened(trade, pred)
        log.info("OPEN #%s %s %s %s @ %.5f SL %.5f TP %.5f (p_up=%.3f)", trade["id"], direction, units, pair,
                 entry, trade["stop_loss"], trade["take_profit"], pred["prob_up"])
        return trade

    # ------------------------------------------------------------------ position monitoring
    @staticmethod
    def _pnl(t: dict, exit_price: float) -> float:
        sign = 1 if t["direction"] == "long" else -1
        gross = sign * (exit_price - t["entry_price"]) * t["units"] * t["quote_usd"]
        costs = COST_BPS / 1e4 * t["units"] * t["quote_usd"] * (t["entry_price"] + exit_price)
        return round(gross - costs, 2)

    def check_open_trades(self) -> list[dict]:
        """Walk every 1m bar since the last check. If a bar touches both SL and TP we assume the
        stop filled first (conservative - intrabar order is unknowable from OHLC). Trades with a
        max_hold_hours limit are closed at the first price on/after their deadline (after SL/TP are checked
        for every earlier minute, so a stop that fired before the deadline always wins)."""
        closed = []
        open_trades = db.get_open_trades()
        by_pair: dict[str, list[dict]] = {}
        for t in open_trades:
            by_pair.setdefault(t["pair"], []).append(t)

        for pair, trades in by_pair.items():
            starts = [pd.Timestamp(t["last_checked"]) if t["last_checked"]
                      else pd.Timestamp(t["entry_time"]).ceil("min") for t in trades]
            try:
                bars = minute_bars(pair, min(starts))
            except Exception as e:
                log.warning("%s: price fetch failed (%s)", pair, e)
                continue
            # Only judge finished 1m bars: Yahoo's still-forming bar can carry a provisional Open/High/Low
            # (seen live: a phantom "gap" open below the stop that the final bar never had).
            bars = bars[bars.index + pd.Timedelta(minutes=1) <= pd.Timestamp.now(tz="UTC")]
            if bars.empty:
                continue

            for t, start in zip(trades, starts):
                long_ = t["direction"] == "long"
                exit_px = reason = exit_ts = None
                deadline = (pd.Timestamp(t["entry_time"]) + pd.Timedelta(hours=t["max_hold_hours"])
                            if t.get("max_hold_hours") else None)
                for ts, bar in bars[bars.index >= start].iterrows():
                    if deadline is not None and ts >= deadline:
                        # time is up and neither SL nor TP was hit: close at the first price on/after the deadline
                        reason, exit_px, exit_ts = "time_exit", float(bar["Open"]), ts
                        break
                    hit_sl = bar["Low"] <= t["stop_loss"] if long_ else bar["High"] >= t["stop_loss"]
                    hit_tp = bar["High"] >= t["take_profit"] if long_ else bar["Low"] <= t["take_profit"]
                    if hit_sl or hit_tp:
                        reason = "stop_loss" if hit_sl else "take_profit"
                        level = t["stop_loss"] if hit_sl else t["take_profit"]
                        # bar opened beyond the level (gap) -> fill at the open, not the level.
                        # The level sits below price for long-SL / short-TP, above it otherwise.
                        level_below = long_ == hit_sl
                        gapped = bar["Open"] < level if level_below else bar["Open"] > level
                        exit_px, exit_ts = (bar["Open"] if gapped else level), ts
                        break
                if reason:
                    pnl = self._pnl(t, float(exit_px))
                    db.close_trade(t["id"], float(exit_px), exit_ts.isoformat(), reason, pnl)
                    notify.trade_closed(t, float(exit_px), reason, pnl)
                    closed.append({"id": t["id"], "pair": pair, "reason": reason,
                                   "exit_price": float(exit_px), "pnl_usd": pnl})
                    log.info("CLOSE #%s %s %s @ %.5f pnl $%.2f", t["id"], pair, reason, exit_px, pnl)
                else:
                    db.touch_trade(t["id"], bars.index[-1].isoformat())
        return closed

    def close_trade_manually(self, trade_id: int) -> dict:
        t = next((x for x in db.get_open_trades() if x["id"] == trade_id), None)
        if t is None:
            raise KeyError(trade_id)
        px = latest_price(t["pair"])
        pnl = self._pnl(t, px)
        db.close_trade(trade_id, px, db.utcnow_iso(), "manual", pnl)
        notify.trade_closed(t, px, "manual", pnl)
        return {"id": trade_id, "exit_price": px, "pnl_usd": pnl}
