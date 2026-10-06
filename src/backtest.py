"""Out-of-sample backtest: does the model's signal survive realistic spreads with a 4-hour hold?

Why this exists: the live paper trades use ATR stops/targets and hold ~9h, while the model only predicts the
4h direction. Here every signal is held for exactly HORIZON_BARS (4h) and charged a spread, which is the
trade the model was actually trained to make.

How it avoids look-ahead: predictions come from the walk-forward folds in train.py (each fold's model has
never seen that fold's period, and rows whose label overlaps the test window are purged).

Usage:
  python -m src.backtest                    # default 5 folds, threshold from .env
  python -m src.backtest --spread-scale 2   # stress test: double all spreads
  python -m src.backtest --refresh          # recompute predictions (after new data / retraining)

Spreads are ASSUMPTIONS (typical retail spreads in basis points of price, by pair tier). Replace them with
your broker's real numbers in data/spreads.csv (columns: pair,spread_bps). Entry uses the signal bar's close,
which is a little optimistic: live fills happen ~90 seconds later.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import (DATA_DIR, FEATURES_PATH, HORIZON_BARS, MAX_LEVERAGE, MAX_OPEN_TRADES,
                     MAX_TRADES_PER_CURRENCY, PROB_THRESHOLD, RISK_PER_TRADE, SL_ATR_MULT)
from .features import FEATURE_COLS
from .train import XGB_PARAMS, walk_forward_splits

log = logging.getLogger(__name__)
SPREADS_CSV = DATA_DIR / "spreads.csv"
TRADES_OUT = DATA_DIR / "backtest_trades.parquet"

G10 = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}
MAJOR_PAIRS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY", "EURGBP"}
EXOTIC_HIGH = {"TRY", "ZAR", "MXN", "BRL"}
# Round-trip cost in bps of price (one full spread: buy at the ask, sell at the bid). Rough retail levels.
TIER_SPREAD_BPS = {"major": 1.5, "g10 cross": 3.0, "exotic-mid": 8.0, "exotic-high": 20.0}
TIERS = list(TIER_SPREAD_BPS)


def tier(pair: str) -> str:
    a, b = pair[:3], pair[3:]
    if pair in MAJOR_PAIRS:
        return "major"
    if a in G10 and b in G10:
        return "g10 cross"
    return "exotic-high" if (a in EXOTIC_HIGH or b in EXOTIC_HIGH) else "exotic-mid"


def spread_table(pairs) -> dict[str, float]:
    spreads = {p: TIER_SPREAD_BPS[tier(p)] for p in pairs}
    if SPREADS_CSV.exists():
        spreads.update(pd.read_csv(SPREADS_CSV).set_index("pair")["spread_bps"].to_dict())
    return spreads


# --------------------------------------------------------------------------- out-of-sample predictions
def oos_predictions(df: pd.DataFrame, splits: int, refresh: bool) -> pd.DataFrame:
    path = DATA_DIR / f"oos_predictions_{splits}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    X, y = df[FEATURE_COLS].astype(float), df["target"].values
    keep = ["pair", "close_time", "label_time", "close", "atr_pct", "ret_1", "fwd_ret"]
    parts = []
    for k, (tr, te) in enumerate(walk_forward_splits(df, splits, HORIZON_BARS), 1):
        model = XGBClassifier(**XGB_PARAMS).fit(X.iloc[tr], y[tr])
        part = df.iloc[te][keep].copy()
        part["p"] = model.predict_proba(X.iloc[te])[:, 1]
        parts.append(part)
        log.info("fold %d/%d: train %d rows, predicted %d", k, splits, len(tr), len(te))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path)
    return out


# --------------------------------------------------------------------------- trades
def make_trades(oos: pd.DataFrame, thr: float, spreads: dict, scale: float) -> pd.DataFrame:
    s = oos[(oos.p >= thr) | (oos.p <= 1 - thr)].copy()
    # a 4-bar label that spans a weekend gap isn't a real 4-hour trade
    s = s[(s.label_time - s.close_time) <= pd.Timedelta(hours=HORIZON_BARS + 0.5)]
    s["side"] = np.where(s.p >= 0.5, 1, -1)
    s["tier"] = s.pair.map(tier)
    s["gross_bps"] = s.side * s.fwd_ret * 1e4
    s["spread_bps"] = s.pair.map(spreads) * scale
    s["net_bps"] = s.gross_bps - s.spread_bps
    s["conf"] = (s.p - 0.5).abs()
    s["exit_time"] = s.label_time
    # same sizing idea as live: risk RISK_PER_TRADE of equity to a SL_ATR_MULT*ATR move (capped by leverage)
    s["weight"] = np.minimum(RISK_PER_TRADE / (SL_ATR_MULT * s.atr_pct), MAX_LEVERAGE)
    s["pnl_pct"] = s.weight * s.net_bps / 1e4 * 100
    return s.reset_index(drop=True)


def non_overlapping(s: pd.DataFrame) -> pd.DataFrame:
    """One position per pair at a time: skip signals while that pair's previous trade is still open."""
    keep = []
    for _, g in s.sort_values("close_time").groupby("pair"):
        free = None
        for idx, ct, ex in zip(g.index, g.close_time, g.exit_time):
            if free is None or ct >= free:
                keep.append(idx)
                free = ex
    return s.loc[keep].sort_values("close_time")


def exposure(pair: str, side: int) -> dict[str, int]:
    return {pair[:3]: side, pair[3:]: -side}


def portfolio(s: pd.DataFrame, max_pos: int, per_ccy: int) -> pd.DataFrame:
    """Replay signals under the live rules: strongest first, max open trades, one per pair, per-currency cap."""
    s = s.sort_values(["close_time", "conf"], ascending=[True, False])
    open_: list[tuple] = []
    taken = []
    for ct, g in s.groupby("close_time", sort=True):
        open_ = [o for o in open_ if o[0] > ct]
        for row in g.itertuples():
            if len(open_) >= max_pos:
                break
            if any(o[1] == row.pair for o in open_):
                continue
            ex = exposure(row.pair, row.side)
            if per_ccy > 0 and any(sum(1 for o in open_ if o[2].get(c) == sd) >= per_ccy for c, sd in ex.items()):
                continue
            open_.append((row.exit_time, row.pair, ex))
            taken.append(row.Index)
    return s.loc[taken]


# --------------------------------------------------------------------------- reporting
def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return dict(n=0)
    eq = pd.concat([pd.Series([0.0]), t.sort_values("exit_time").pnl_pct.cumsum().reset_index(drop=True)])
    return dict(n=len(t), hit=(t.gross_bps > 0).mean() * 100, gross_bps=t.gross_bps.mean(),
                spread_bps=t.spread_bps.mean(), net_bps=t.net_bps.mean(), net_win=(t.net_bps > 0).mean() * 100,
                total_pct=t.pnl_pct.sum(), max_dd_pct=(eq.cummax() - eq).max())


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.round(2).to_string())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=PROB_THRESHOLD)
    ap.add_argument("--spread-scale", type=float, default=1.0, help="multiply every spread (stress test)")
    ap.add_argument("--refresh", action="store_true", help="recompute out-of-sample predictions")
    args = ap.parse_args()

    df = pd.read_parquet(FEATURES_PATH).sort_values(["close_time", "pair"], ignore_index=True)
    oos = oos_predictions(df, args.splits, args.refresh)
    spreads = spread_table(oos.pair.unique())
    trades = make_trades(oos, args.threshold, spreads, args.spread_scale)
    raw = non_overlapping(trades)

    print(f"Out-of-sample: {oos.close_time.min():%Y-%m-%d} to {oos.close_time.max():%Y-%m-%d}, "
          f"{len(oos):,} predictions, {oos.pair.nunique()} pairs. Hold {HORIZON_BARS}h, "
          f"signal if p>={args.threshold} or <={1 - args.threshold:.2f}, spreads x{args.spread_scale}.")
    print("Assumed round-trip spread (bps): " + ", ".join(f"{k} {v}" for k, v in TIER_SPREAD_BPS.items())
          + ("  [+ overrides from data/spreads.csv]" if SPREADS_CSV.exists() else ""))

    rows = {t: stats(raw[raw.tier == t]) for t in TIERS if (raw.tier == t).any()}
    rows["ALL"] = stats(raw)
    tab = pd.DataFrame(rows).T
    tab["breakeven_spread_bps"] = tab["gross_bps"]
    show("1) EVERY SIGNAL, one position per pair at a time (no portfolio limits). hit = % right direction,\n"
         "   gross = avg move in our favour, net = after spread. breakeven_spread = the most spread you could pay.",
         tab[["n", "hit", "gross_bps", "spread_bps", "net_bps", "net_win", "breakeven_spread_bps"]])

    sens = {}
    for sc in (0.5, 1.0, 2.0, 3.0):
        t = make_trades(oos, args.threshold, spreads, sc)
        t = non_overlapping(t)
        sens[f"spreads x{sc}"] = {**{k: t[t.tier == k].net_bps.mean() for k in TIERS if (t.tier == k).any()},
                                  "ALL": t.net_bps.mean()}
    show("2) NET bps per trade if real spreads are a multiple of the assumed ones", pd.DataFrame(sens).T)

    port = portfolio(trades, MAX_OPEN_TRADES, MAX_TRADES_PER_CURRENCY)
    ps = pd.DataFrame({"ALL": stats(port), **{t: stats(port[port.tier == t]) for t in TIERS if (port.tier == t).any()}}).T
    show(f"3) UNDER THE LIVE RULES (max {MAX_OPEN_TRADES} open, {MAX_TRADES_PER_CURRENCY} per currency, strongest first). "
         "total_pct = sum of trade P&L as % of equity,\n   sized to risk "
         f"{RISK_PER_TRADE:.0%} per {SL_ATR_MULT} ATR (no compounding); max_dd_pct = worst peak-to-trough",
         ps[["n", "hit", "gross_bps", "net_bps", "net_win", "total_pct", "max_dd_pct"]])
    yr = port.assign(year=port.exit_time.dt.year).groupby("year").agg(
        trades=("pnl_pct", "size"), net_bps=("net_bps", "mean"), return_pct=("pnl_pct", "sum"))
    show("   by year", yr)

    h = raw.assign(hours=pd.cut(raw.close_time.dt.hour, [-1, 6, 12, 16, 20, 23]))
    show("4) BY BAR-CLOSE HOUR (UTC), same trades as table 1",
         h.groupby("hours", observed=True).agg(n=("net_bps", "size"), hit=("gross_bps", lambda x: (x > 0).mean() * 100),
                                              gross_bps=("gross_bps", "mean"), net_bps=("net_bps", "mean")))

    # Is the model just fading the last hour? Compare against that trivial rule with the same number of signals.
    base = oos[(oos.label_time - oos.close_time) <= pd.Timedelta(hours=HORIZON_BARS + 0.5)].copy()
    base["tier"] = base.pair.map(tier)
    base["strength"] = base.ret_1.abs() / base.atr_pct
    comp = {}
    for t in TIERS:
        m, b = trades[trades.tier == t], base[base.tier == t]
        if m.empty or b.empty:
            continue
        top = b.nlargest(len(m), "strength")
        fade = -np.sign(top.ret_1) * top.fwd_ret * 1e4
        comp[t] = {"model_gross_bps": m.gross_bps.mean(), "fade_last_hour_gross_bps": fade.mean(),
                   "model_hit": (m.gross_bps > 0).mean() * 100, "fade_hit": (fade > 0).mean() * 100}
    show("5) DOES THE MODEL ADD ANYTHING BEYOND 'BET AGAINST THE LAST HOUR'? (same number of signals per tier)",
         pd.DataFrame(comp).T)

    sw = {}
    for thr in (0.55, 0.58, 0.62, 0.66, 0.70):
        t = non_overlapping(make_trades(oos, thr, spreads, args.spread_scale))
        sw[f"p>={thr}"] = {"n": len(t), "hit": (t.gross_bps > 0).mean() * 100, "gross_bps": t.gross_bps.mean(),
                           "net_bps": t.net_bps.mean()}
    show("6) STRICTER CONFIDENCE THRESHOLDS (table 1 'ALL' row at each threshold)", pd.DataFrame(sw).T)

    trades.to_parquet(TRADES_OUT)
    print(f"\nSaved every simulated trade to {TRADES_OUT.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
