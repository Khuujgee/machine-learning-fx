"""Research: which prediction target gives an edge that survives spreads?

Compares, out-of-sample (walk-forward, purged), two kinds of target:
  dir_H        : will the price be higher in H hours?  (H = 4, 12, 24, 48;  4 = what the live model does)
  hit_UP_DN_N  : "which level is hit first": +UP*ATR (take-profit) before -DN*ATR (stop) within N hours,
                 trained separately for longs and shorts. This matches how a trade with SL/TP actually exits.

Every model only trades when its *expected* move is bigger than the pair's spread (cost-aware), holds until
the label resolves, one position per pair at a time, and is charged the spread from src/backtest.py
(assumptions; override in data/spreads.csv).

Usage:  python -m src.research_targets            (~10 min; writes data/research_targets.txt)
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .backtest import TIERS, spread_table, tier
from .config import DATA_DIR, FEATURES_PATH, RAW_DIR
from .features import FEATURE_COLS, atr
from .train import XGB_PARAMS, walk_forward_splits

log = logging.getLogger(__name__)
OUT = DATA_DIR / "research_targets.txt"
SPLITS = 3
EXTRA_COLS = ["ret_72", "ret_240", "dist_ema200_atr", "range_pos_120", "vol_ratio_24_240"]
DIR_HORIZONS = [4, 12, 24, 48]
BARRIERS = [(2.0, 1.0, 24), (1.5, 1.5, 24), (3.0, 1.5, 48)]  # (take-profit ATRs, stop ATRs, max hours)


# --------------------------------------------------------------------------- data + labels
def first_touch(H, L, C, A, pos, up, dn, n, side):
    """For entries at raw positions `pos`: +1 if the TP level is hit first, -1 if the stop, 0 on timeout.
    A bar touching both counts as the stop (conservative). Returns outcome, P&L in bps, bars to resolve."""
    c, a = C[pos], A[pos]
    tp, sl = (c + up * a, c - dn * a) if side == 1 else (c - up * a, c + dn * a)
    out = np.zeros(len(pos), dtype=np.int8)
    k_res = np.full(len(pos), n)
    done = np.zeros(len(pos), dtype=bool)
    last = len(C) - 1
    for k in range(1, n + 1):
        j = np.minimum(pos + k, last)
        valid = pos + k <= last
        hi, lo = H[j], L[j]
        hit_tp = (hi >= tp) if side == 1 else (lo <= tp)
        hit_sl = (lo <= sl) if side == 1 else (hi >= sl)
        new_sl = ~done & valid & hit_sl
        new_tp = ~done & valid & hit_tp & ~hit_sl
        out[new_sl], out[new_tp] = -1, 1
        k_res[new_sl | new_tp] = k
        done |= new_sl | new_tp
    end = np.minimum(pos + n, last)
    timeout = side * (C[end] / c - 1)
    pnl = np.where(out == 1, up * a / c, np.where(out == -1, -dn * a / c, timeout)) * 1e4
    resolvable = done | (pos + n <= last)
    return out, np.where(resolvable, pnl, np.nan), k_res


def build() -> pd.DataFrame:
    feat = pd.read_parquet(FEATURES_PATH)
    parts = []
    for pair, g in feat.groupby("pair"):
        raw = pd.read_parquet(RAW_DIR / f"{pair}.parquet").sort_index()
        c, a = raw["Close"], atr(raw)
        ret_1 = c.pct_change()
        ext = pd.DataFrame({
            "ret_72": c.pct_change(72), "ret_240": c.pct_change(240),
            "dist_ema200_atr": (c - c.ewm(span=200, adjust=False).mean()) / a,
            "range_pos_120": (c - raw["Low"].rolling(120).min())
                             / (raw["High"].rolling(120).max() - raw["Low"].rolling(120).min()),
            "vol_ratio_24_240": ret_1.rolling(24).std() / ret_1.rolling(240).std(),
        }, index=raw.index)
        g = g.set_index("bar_time").sort_index()
        pos = raw.index.get_indexer(g.index)
        g = g[pos >= 0].copy()
        pos = pos[pos >= 0]
        g[EXTRA_COLS] = ext.iloc[pos].values
        H, L, C, A = raw["High"].values, raw["Low"].values, raw["Close"].values, a.values
        close_times = raw.index + pd.Timedelta(hours=1)
        last = len(C) - 1
        for h in DIR_HORIZONS:
            j = pos + h
            ok = j <= last
            g[f"ret_{h}h"] = np.where(ok, C[np.minimum(j, last)] / C[pos] - 1, np.nan)
            g[f"res_dir_{h}"] = np.where(ok, close_times[np.minimum(j, last)], pd.NaT)
        for up, dn, n in BARRIERS:
            name = f"hit_{up:g}_{dn:g}_{n}"
            for side, s in ((1, "L"), (-1, "S")):
                o, pnl, k = first_touch(H, L, C, A, pos, up, dn, n, side)
                g[f"{name}_{s}_y"] = np.where(np.isnan(pnl), np.nan, (o == 1).astype(float))
                g[f"{name}_{s}_pnl"] = pnl
                g[f"{name}_{s}_res"] = close_times[np.minimum(pos + k, last)]
            g[f"{name}_purge"] = close_times[np.minimum(pos + n, last)]
        parts.append(g.reset_index())
    df = pd.concat(parts, ignore_index=True).sort_values(["close_time", "pair"], ignore_index=True)
    df["tier"] = df["pair"].map(tier)
    df["atr_bps"] = df["atr"] / df["close"] * 1e4
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=EXTRA_COLS)


# --------------------------------------------------------------------------- walk-forward fit
def oos_proba(df: pd.DataFrame, y: str, purge_col: str, gap: int, cols: list[str]) -> pd.Series:
    d = df.dropna(subset=[y]).assign(label_time=lambda x: pd.to_datetime(x[purge_col], utc=True))
    out = pd.Series(np.nan, index=df.index)
    for tr, te in walk_forward_splits(d, SPLITS, gap):
        m = XGBClassifier(**XGB_PARAMS).fit(d[cols].iloc[tr].astype(float), d[y].iloc[tr])
        out.loc[d.index[te]] = m.predict_proba(d[cols].iloc[te].astype(float))[:, 1]
    return out


def non_overlapping(t: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, g in t.sort_values("close_time").groupby("pair"):
        free = None
        for idx, ct, ex in zip(g.index, g.close_time, g.exit_time):
            if free is None or ct >= free:
                keep.append(idx)
                free = ex
    return t.loc[keep]


def summarize(name: str, t: pd.DataFrame, years: float, n_pairs: int) -> dict:
    if t.empty:
        return {"experiment": name, "trades": 0}
    hours = (t.exit_time - t.close_time).dt.total_seconds() / 3600
    row = {"experiment": name, "trades": len(t), "trades_per_pair_yr": len(t) / n_pairs / years,
           "avg_hold_h": hours.mean(), "win%": (t.net_bps > 0).mean() * 100,
           "gross_bps": t.gross_bps.mean(), "net_bps": t.net_bps.mean(),
           "net_bps_per_pair_yr": t.net_bps.sum() / n_pairs / years}
    for k in TIERS:
        sub = t[t.tier == k]
        row[f"net[{k}]"] = sub.net_bps.mean() if len(sub) else np.nan
    yrs = t.groupby(t.exit_time.dt.year).net_bps.mean()
    row["net_by_year"] = " ".join(f"{y}:{v:+.1f}" for y, v in yrs.items())
    return row


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    df = build()
    spreads = spread_table(df.pair.unique())
    df["spread_bps"] = df.pair.map(spreads)
    log.info("built %d rows x %d pairs in %.0fs", len(df), df.pair.nunique(), time.time() - t0)
    ext_cols = FEATURE_COLS + EXTRA_COLS
    rows = []

    def evaluate(name, cand):
        cand = cand.dropna(subset=["gross_bps", "exit_time"])
        span = (cand.close_time.max() - cand.close_time.min()).days / 365.25 if len(cand) else 1
        t = non_overlapping(cand)
        rows.append(summarize(name, t, max(span, 0.1), df.pair.nunique()))
        log.info("%s: %d trades, net %.2f bps (%.0fs)", name, len(t), t.net_bps.mean() if len(t) else 0, time.time() - t0)

    # ---- direction targets: trade only if (2p-1) * typical |move| for that pair beats the spread
    for h in DIR_HORIZONS:
        for cols, tag in ([(FEATURE_COLS, "base feats")] if h == 4 else []) + [(ext_cols, "ext feats")]:
            y = f"y_dir_{h}"
            df[y] = np.where(df[f"ret_{h}h"].isna(), np.nan, (df[f"ret_{h}h"] > 0).astype(float))
            p = oos_proba(df, y, f"res_dir_{h}", h, cols)
            m = df.assign(p=p).dropna(subset=["p", f"ret_{h}h"])
            if h == 4:  # a 4-bar label across a weekend isn't a 4h trade
                m = m[(pd.to_datetime(m[f"res_dir_{h}"], utc=True) - m.close_time) <= pd.Timedelta(hours=4.5)]
            typical = m.groupby("pair")[f"ret_{h}h"].transform(lambda r: r.abs().mean()) * 1e4
            side = np.where(m.p >= 0.5, 1, -1)
            m = m.assign(side=side, ev_bps=(2 * m.p - 1).abs() * typical,
                         gross_bps=side * m[f"ret_{h}h"] * 1e4,
                         exit_time=pd.to_datetime(m[f"res_dir_{h}"], utc=True))
            m["net_bps"] = m.gross_bps - m.spread_bps
            evaluate(f"dir_{h}h  p>=0.58 ({tag})", m[(m.p >= 0.58) | (m.p <= 0.42)])
            evaluate(f"dir_{h}h  cost-aware ({tag})", m[m.ev_bps > m.spread_bps])

    # ---- first-touch targets: EV = p*TP - (1-p)*SL (in bps), take the better side if it beats the spread
    for up, dn, n in BARRIERS:
        name = f"hit_{up:g}_{dn:g}_{n}"
        pL = oos_proba(df, f"{name}_L_y", f"{name}_purge", n, ext_cols)
        pS = oos_proba(df, f"{name}_S_y", f"{name}_purge", n, ext_cols)
        m = df.assign(pL=pL, pS=pS).dropna(subset=["pL", "pS"])
        evL = (m.pL * up - (1 - m.pL) * dn) * m.atr_bps
        evS = (m.pS * up - (1 - m.pS) * dn) * m.atr_bps
        long_ = evL >= evS
        m = m.assign(ev_bps=np.where(long_, evL, evS),
                     gross_bps=np.where(long_, m[f"{name}_L_pnl"], m[f"{name}_S_pnl"]),
                     exit_time=pd.to_datetime(np.where(long_, m[f"{name}_L_res"], m[f"{name}_S_res"]), utc=True))
        m["net_bps"] = m.gross_bps - m.spread_bps
        label = f"hit +{up:g}/-{dn:g} ATR in {n}h"
        evaluate(f"{label}  cost-aware", m[m.ev_bps > m.spread_bps])
        evaluate(f"{label}  cost-aware, majors+G10", m[(m.ev_bps > m.spread_bps) & m.tier.isin(["major", "g10 cross"])])

    res = pd.DataFrame(rows).set_index("experiment")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    text = (f"Out-of-sample {SPLITS}-fold walk-forward, {df.pair.nunique()} pairs, spreads from src/backtest.py.\n"
            "net_bps = average result per trade after spread; net_bps_per_pair_yr = total edge per pair per year.\n\n"
            + res.drop(columns=["net_by_year"]).round(2).to_string()
            + "\n\nnet bps per trade by year:\n" + res["net_by_year"].to_string())
    print(text)
    OUT.write_text(text)
    log.info("done in %.0f min, saved %s", (time.time() - t0) / 60, OUT)


if __name__ == "__main__":
    main()
