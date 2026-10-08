"""Research: do non-FX inputs (markets, interest rates, news) improve the FX model out-of-sample?

Same targets, same period, same cost-aware trading rule as src/research_targets.py; only the inputs change:
  A  price only          : the live model's features + longer-horizon price features
  B  + markets & rates   : futures/indices/VIX/dollar index moves, pair-specific signals (oil for CAD/NOK...,
                           risk-on/off for JPY/CHF...), interest-rate difference (carry), currency identity
  C  + news history      : GDELT news tone per currency (only if data/gdelt/*.csv exists)

The period starts when the futures data starts (~mid 2024), so every feature set is judged on identical rows.

Usage:  python -m src.macro_data   (first, to download)   then   python -m src.research_features
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from . import research_targets as R
from .backtest import spread_table
from .config import DATA_DIR
from .macro_data import MACRO_COLS, add_macro_features
from .news_history import NEWS_COLS, add_news_features, load_gdelt
from .train import XGB_PARAMS

log = logging.getLogger(__name__)
OUT = DATA_DIR / "research_features.txt"
START = pd.Timestamp("2024-06-15", tz="UTC")
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK", "DKK", "PLN", "HUF", "CZK",
              "TRY", "ZAR", "MXN", "BRL", "CNH", "INR", "THB", "ILS", "SGD", "HKD"]
CCY_COLS = [f"ccy_{c}" for c in CURRENCIES]


def add_currency_identity(df: pd.DataFrame) -> pd.DataFrame:
    base, quote = df["pair"].str[:3], df["pair"].str[3:]
    for c in CURRENCIES:
        df[f"ccy_{c}"] = (base == c).astype(np.int8) - (quote == c).astype(np.int8)
    return df


def run_targets(df: pd.DataFrame, cols: list[str], tag: str, rows: list) -> None:
    span = (df.close_time.max() - df.close_time.min()).days / 365.25
    n_pairs = df.pair.nunique()

    def report(name: str, cand: pd.DataFrame) -> None:
        t = R.non_overlapping(cand.dropna(subset=["gross_bps", "exit_time"]))
        g10 = t[t.tier.isin(["major", "g10 cross"])]
        row = R.summarize(f"{name} | {tag}", t, span, n_pairs)
        row["net_bps_majors+G10"] = g10.net_bps.mean() if len(g10) else np.nan
        row["trades_majors+G10"] = len(g10)
        rows.append(row)
        log.info("%-34s %-22s trades %6d  net %+.2f  majors+G10 %+.2f", name, tag, len(t), row.get("net_bps", np.nan),
                 row["net_bps_majors+G10"])

    for h in (4, 12):
        y = f"y_dir_{h}"
        df[y] = np.where(df[f"ret_{h}h"].isna(), np.nan, (df[f"ret_{h}h"] > 0).astype(float))
        m = df.assign(p=R.oos_proba(df, y, f"res_dir_{h}", h, cols)).dropna(subset=["p", f"ret_{h}h"])
        if h == 4:
            m = m[(pd.to_datetime(m["res_dir_4"], utc=True) - m.close_time) <= pd.Timedelta(hours=4.5)]
        typical = m.groupby("pair")[f"ret_{h}h"].transform(lambda r: r.abs().mean()) * 1e4
        side = np.where(m.p >= 0.5, 1, -1)
        m = m.assign(ev_bps=(2 * m.p - 1).abs() * typical, gross_bps=side * m[f"ret_{h}h"] * 1e4,
                     exit_time=pd.to_datetime(m[f"res_dir_{h}"], utc=True))
        m["net_bps"] = m.gross_bps - m.spread_bps
        report(f"direction {h}h, cost-aware", m[m.ev_bps > m.spread_bps])

    up, dn, n = 1.5, 1.5, 24
    name = f"hit_{up:g}_{dn:g}_{n}"
    pL = R.oos_proba(df, f"{name}_L_y", f"{name}_purge", n, cols)
    pS = R.oos_proba(df, f"{name}_S_y", f"{name}_purge", n, cols)
    m = df.assign(pL=pL, pS=pS).dropna(subset=["pL", "pS"])
    evL = (m.pL * up - (1 - m.pL) * dn) * m.atr_bps
    evS = (m.pS * up - (1 - m.pS) * dn) * m.atr_bps
    long_ = evL >= evS
    m = m.assign(ev_bps=np.where(long_, evL, evS),
                 gross_bps=np.where(long_, m[f"{name}_L_pnl"], m[f"{name}_S_pnl"]),
                 exit_time=pd.to_datetime(np.where(long_, m[f"{name}_L_res"], m[f"{name}_S_res"]), utc=True))
    m["net_bps"] = m.gross_bps - m.spread_bps
    report("first touch +/-1.5 ATR 24h, cost-aware", m[m.ev_bps > m.spread_bps])


def importance(df: pd.DataFrame, cols: list[str], top: int = 20) -> pd.Series:
    """Which inputs does a 4h-direction model actually use? (gain importance, fit on the whole period)"""
    d = df.dropna(subset=["y_dir_4"])
    m = XGBClassifier(**XGB_PARAMS).fit(d[cols].astype(float), d["y_dir_4"])
    return pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False).head(top)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    df = R.build()
    df = df[df.close_time >= START].reset_index(drop=True)
    df["spread_bps"] = df.pair.map(spread_table(df.pair.unique()))
    df = add_currency_identity(add_macro_features(df))
    sets = {"A price only": R.FEATURE_COLS + R.EXTRA_COLS}
    sets["B + markets & rates"] = sets["A price only"] + MACRO_COLS + CCY_COLS
    have_news = not load_gdelt().empty
    if have_news:
        df = add_news_features(df)
        sets["C + news"] = sets["B + markets & rates"] + NEWS_COLS
    cover = {c: df[c].notna().mean() for c in MACRO_COLS + (NEWS_COLS if have_news else [])}
    log.info("%d rows, %d pairs, %s -> %s (%.0fs)", len(df), df.pair.nunique(), df.close_time.min().date(),
             df.close_time.max().date(), time.time() - t0)

    rows: list = []
    for tag, cols in sets.items():
        run_targets(df, cols, tag, rows)

    res = pd.DataFrame(rows).set_index("experiment")
    keep = ["trades", "win%", "gross_bps", "net_bps", "net_bps_per_pair_yr", "trades_majors+G10", "net_bps_majors+G10"]
    imp = importance(df, sets[list(sets)[-1]])
    low_cover = ", ".join(f"{k} {v:.0%}" for k, v in cover.items() if v < 0.9) or "none"
    text = (f"Period {df.close_time.min():%Y-%m-%d} -> {df.close_time.max():%Y-%m-%d}, {df.pair.nunique()} pairs, "
            f"{R.SPLITS}-fold walk-forward, cost-aware trading, spreads from src/backtest.py.\n"
            f"News history: {'included' if have_news else 'NOT available yet (no data/gdelt/*.csv)'}\n"
            f"Inputs with <90% coverage (missing = NaN, XGBoost handles it): {low_cover}\n\n"
            + res[keep].round(2).to_string()
            + "\n\nnet bps per trade by year:\n" + res["net_by_year"].to_string()
            + f"\n\nTop inputs of a 4h model using set '{list(sets)[-1]}' (gain importance):\n" + imp.round(4).to_string())
    print(text)
    OUT.write_text(text)
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60, OUT)


if __name__ == "__main__":
    main()
