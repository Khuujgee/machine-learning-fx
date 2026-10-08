"""Research: daily-horizon FX (days-to-weeks) - trend, carry, markets, news - vs simple benchmarks.

Daily-rebalanced portfolio test, out-of-sample:
  * position for each pair decided at day t's close from data known by then, held over day t+1
  * every pair is sized to the same risk (inverse of its recent volatility); the portfolio is reported at a
    10% annual volatility target so strategies are comparable
  * costs: half the assumed spread (src/backtest.py tiers) on every change in position size
  * carry: the interest-rate difference is earned (long the higher-yielding currency) or paid every day held,
    from FRED short rates (retail broker swap mark-ups are NOT included)

Strategies:
  benchmarks : MOM-3M (sign of 3-month return), MOM-12M, CARRY (long the higher rate), TREND+CARRY (average)
  ML         : XGBoost predicting the 5-day and 20-day direction with growing sets of inputs:
               price/momentum -> + rates/carry -> + markets (futures, indices, VIX, dollar index)
  news       : separately, on the ~3 years GDELT covers: ML 5-day with vs without news tone

Usage:  python -m src.research_daily            (downloads ~20y of daily data on first run, then ~5-10 min)
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier

from .backtest import TIER_SPREAD_BPS, spread_table, tier
from .config import CANDIDATE_PAIRS, DATA_DIR
from .macro_data import COPPER_BETA, GOLD_BETA, MARKETS, OIL_BETA, RATES_PATH, RISK_BETA, download_rates
from .news_history import currency_news_frame, load_gdelt
from .train import XGB_PARAMS, walk_forward_splits

log = logging.getLogger(__name__)
DAILY_DIR = DATA_DIR / "daily"
OUT = DATA_DIR / "research_daily.txt"
TARGET_VOL = 0.10
FOLDS = 6

PRICE_COLS = [f"mom_{h}" for h in (1, 5, 20, 60, 120, 250)] + ["ma50_dist", "ma200_dist", "vol20", "vol_ratio",
                                                               "rsi14", "dow"]
RATE_COLS = ["rate_diff", "base_rate", "quote_rate", "rate_diff_chg63"]
MKT_COLS = [f"{m}_z{h}" for m in MARKETS for h in (5, 20)] + ["vix_level", "risk_signal20", "oil_signal20",
                                                               "gold_signal20", "copper_signal20", "usd_signal20"]
NEWS_COLS = ["news_tone_diff_1d", "news_tone_diff_5d", "news_volume_5d"]


# --------------------------------------------------------------------------- data
def _yf_daily(symbol: str) -> pd.Series:
    for attempt in range(3):
        try:
            d = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=False)
            break
        except Exception:
            if attempt == 2:
                return pd.Series(dtype=float)
            time.sleep(2 * (attempt + 1))
    if d is None or d.empty:
        return pd.Series(dtype=float)
    c = d["Close"]
    c.index = pd.to_datetime(c.index.tz_localize(None) if c.index.tz is not None else c.index).normalize()
    return c[~c.index.duplicated(keep="last")].sort_index()


def clean_spikes(c: pd.Series) -> pd.Series:
    """Remove Yahoo data errors while keeping real shocks (CHF 2015, TRY 2018/2021, ZAR 2008 stay in):
    1. stale data: a run of >= 15 identical closes -> drop all history up to the end of the last such run
    2. bad prints: a > 10% move that comes back to within 5% of the old level within 5 days -> masked
    3. impossible jumps that don't come back (> 30% in a day: TRYJPY's x10 scale change in 2007, USDBRL's
       stale pre-2006 prices) -> drop the history before them
    """
    c = c[c > 0].copy()
    same = c.diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    stale_end = run[run >= 15].index
    if len(stale_end):
        c = c[c.index > stale_end[-1]]
    vals = c.values.copy()
    for i in range(1, len(vals) - 1):
        if abs(np.log(vals[i] / vals[i - 1])) > 0.10:
            for j in range(i + 1, min(i + 6, len(vals))):
                if abs(np.log(vals[j] / vals[i - 1])) < 0.05:   # came back: everything in between was bogus
                    vals[i:j] = np.nan
                    break
    c = pd.Series(vals, index=c.index).ffill()
    lr = np.log(c).diff()
    huge = lr.index[lr.abs() > 0.30]      # real shocks in this data top out ~21% (TRY 2018/2021, CHF 2015)
    if len(huge):
        c = c[c.index >= huge[-1]].iloc[1:]
    return c


def load_daily(refresh: bool = False) -> tuple[dict, dict]:
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    fx, mk = {}, {}
    for pair in CANDIDATE_PAIRS:
        path = DAILY_DIR / f"{pair}.parquet"
        if refresh or not path.exists():
            s = _yf_daily(f"{pair}=X")
            if len(s) > 500:
                s.to_frame("Close").to_parquet(path)
            time.sleep(0.2)
        if path.exists():
            s = clean_spikes(pd.read_parquet(path)["Close"])
            s = s[s > 0]
            if len(s) > 750:
                fx[pair] = s
    for name, sym in MARKETS.items():
        path = DAILY_DIR / f"mkt_{name}.parquet"
        if refresh or not path.exists():
            s = _yf_daily(sym)
            if len(s):
                s.to_frame("Close").to_parquet(path)
            time.sleep(0.2)
        if path.exists():
            mk[name] = pd.read_parquet(path)["Close"]
    log.info("daily data: %d FX pairs, %d markets", len(fx), len(mk))
    return fx, mk


def rsi(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def market_frame(mk: dict) -> pd.DataFrame:
    """Daily market z-scores, lagged ONE day: an FX daily close can come before the US equity close, so only
    the previous day's market data is safely known."""
    cols = {}
    for name, c in mk.items():
        lr = np.log(c).diff()
        vol = lr.rolling(60, min_periods=40).std()
        for h in (5, 20):
            cols[f"{name}_z{h}"] = (np.log(c).diff(h) / (vol * np.sqrt(h))).clip(-6, 6)
        if name == "vix":
            cols["vix_level"] = c
    f = pd.DataFrame(cols).sort_index().ffill(limit=5).shift(1)
    return f


def build(fx: dict, mk: dict) -> pd.DataFrame:
    rates = pd.read_parquet(RATES_PATH)
    rates.index = rates.index.tz_convert(None).normalize()
    rates = rates[~rates.index.duplicated(keep="last")].sort_index()
    mkt = market_frame(mk)
    parts = []
    for pair, c in fx.items():
        lr = np.log(c).diff()
        vol60 = lr.rolling(60, min_periods=40).std()
        vol20 = lr.rolling(20, min_periods=15).std()
        d = pd.DataFrame(index=c.index)
        d["pair"], d["close"] = pair, c
        for h in (1, 5, 20, 60, 120, 250):
            d[f"mom_{h}"] = (np.log(c).diff(h) / (vol60 * np.sqrt(h))).clip(-8, 8)
        d["ma50_dist"] = np.log(c / c.rolling(50).mean()) / vol60
        d["ma200_dist"] = np.log(c / c.rolling(200).mean()) / vol60
        d["vol20"], d["vol_ratio"] = vol20 * np.sqrt(252), vol20 / vol60
        d["rsi14"], d["dow"] = rsi(c), d.index.dayofweek
        d["ret_next"] = lr.shift(-1)                       # return earned by a position held over the next day
        d["days_next"] = (d.index.to_series().shift(-1) - d.index.to_series()).dt.days
        d["vol20_daily"] = vol20
        for h in (5, 20):
            d[f"fwd_{h}"] = np.log(c.shift(-h) / c)
            d[f"label_time_{h}"] = d.index.to_series().shift(-h)
        base, quote = pair[:3], pair[3:]
        r = rates.reindex(rates.index.union(d.index)).ffill().reindex(d.index)
        d["base_rate"] = r[base] if base in r else np.nan
        d["quote_rate"] = r[quote] if quote in r else np.nan
        d["rate_diff"] = d["base_rate"] - d["quote_rate"]
        d["rate_diff_chg63"] = d["rate_diff"] - d["rate_diff"].shift(63)
        m = mkt.reindex(mkt.index.union(d.index)).ffill(limit=5).reindex(d.index)
        for col in mkt.columns:
            d[col] = m[col]
        diff = lambda tbl: tbl.get(base, 0) - tbl.get(quote, 0)
        d["risk_signal20"] = d.get("spx_z20") * diff(RISK_BETA)
        d["oil_signal20"] = d.get("oil_z20") * diff(OIL_BETA)
        d["gold_signal20"] = d.get("gold_z20") * diff(GOLD_BETA)
        d["copper_signal20"] = d.get("copper_z20") * diff(COPPER_BETA)
        d["usd_signal20"] = d.get("dxy_z20") * (1 if base == "USD" else -1 if quote == "USD" else 0)
        parts.append(d)
    df = pd.concat(parts).rename_axis("date").reset_index()
    df["tier"] = df["pair"].map(tier)
    df["close_time"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    return df.replace([np.inf, -np.inf], np.nan)


def add_news(df: pd.DataFrame) -> pd.DataFrame:
    """News known at the START of each FX day (00:00 UTC), so it can't include the day being traded."""
    frame = currency_news_frame()
    if frame.empty:
        for c in NEWS_COLS:
            df[c] = np.nan
        return df
    daily = frame.copy()
    daily.index = daily.index.tz_convert(None)
    at_midnight = daily[daily.index.hour == 0]
    at_midnight.index = at_midnight.index.normalize()
    out = {c: np.full(len(df), np.nan) for c in NEWS_COLS}
    for pair, idx in df.groupby("pair").groups.items():
        b, q = pair[:3], pair[3:]
        if f"{b}_tone_z" not in at_midnight or f"{q}_tone_z" not in at_midnight:
            continue
        tone = (at_midnight[f"{b}_tone_z"] - at_midnight[f"{q}_tone_z"])
        vol = at_midnight[f"{b}_vol_z"] + at_midnight[f"{q}_vol_z"]
        dates = pd.DatetimeIndex(df.loc[idx, "date"])
        out["news_tone_diff_1d"][idx] = tone.reindex(dates).values
        out["news_tone_diff_5d"][idx] = tone.rolling(5, min_periods=3).mean().reindex(dates).values
        out["news_volume_5d"][idx] = vol.rolling(5, min_periods=3).mean().reindex(dates).values
    for c, v in out.items():
        df[c] = v
    return df


# --------------------------------------------------------------------------- strategies
def ml_positions(df: pd.DataFrame, h: int, cols: list[str], folds: int = FOLDS) -> pd.Series:
    """Out-of-sample walk-forward predictions -> position -1/0/+1 (trade only if the expected h-day move
    beats one round-trip spread)."""
    d = df.dropna(subset=[f"fwd_{h}", f"label_time_{h}"]).copy()
    d = d[d[f"fwd_{h}"] != 0]
    d["y"] = (d[f"fwd_{h}"] > 0).astype(int)
    d["label_time"] = pd.to_datetime(d[f"label_time_{h}"]).dt.tz_localize("UTC")
    d = d.sort_values(["close_time", "pair"])
    p = pd.Series(np.nan, index=df.index)
    for tr, te in walk_forward_splits(d, folds, h):
        model = XGBClassifier(**XGB_PARAMS).fit(d[cols].iloc[tr].astype(float), d["y"].iloc[tr])
        p.loc[d.index[te]] = model.predict_proba(d[cols].iloc[te].astype(float))[:, 1]
    typical = df.groupby("pair")[f"fwd_{h}"].transform(lambda r: r.abs().mean()) * 1e4
    ev = (2 * p - 1).abs() * typical
    pos = np.sign(p - 0.5).where(ev > df["spread_bps"], 0.0)
    return pos.where(p.notna())


def benchmark_positions(df: pd.DataFrame) -> dict[str, pd.Series]:
    carry = np.sign(df["rate_diff"]).fillna(0)
    m3, m12 = np.sign(df["mom_60"]).fillna(0), np.sign(df["mom_250"]).fillna(0)
    return {"MOM-3M": m3, "MOM-12M": m12, "CARRY": carry, "TREND+CARRY": (m3 + m12 + carry) / 3}


def daily_pnl(df: pd.DataFrame, pos: pd.Series, mask: pd.Series | None = None) -> pd.DataFrame:
    """Per pair-day P&L: position (vol-scaled) x next-day return + carry - costs."""
    d = df[["date", "pair", "tier", "ret_next", "days_next", "rate_diff", "vol20_daily", "spread_bps"]].copy()
    d["pos"] = pos.values if len(pos) == len(d) else pos
    if mask is not None:
        d = d[mask.values]
    d = d.dropna(subset=["pos", "ret_next", "vol20_daily"]).sort_values(["pair", "date"])
    d["held"] = d["pos"] * (TARGET_VOL / np.sqrt(252)) / d["vol20_daily"].clip(lower=1e-4)
    d["held"] = d["held"].clip(-5, 5)                     # cap leverage on ultra-low-vol pegged pairs
    d["gross"] = d["held"] * d["ret_next"]
    d["carry"] = d["held"] * d["rate_diff"].fillna(0) / 100 * d["days_next"].fillna(1) / 365
    d["cost"] = d.groupby("pair")["held"].diff().abs().fillna(d["held"].abs()) * d["spread_bps"] / 2 / 1e4
    d["pnl"] = d["gross"] + d["carry"] - d["cost"]
    return d


def portfolio_curve(df: pd.DataFrame, pos: pd.Series, mask: pd.Series | None = None) -> pd.Series:
    """Cumulative portfolio return (%, not compounded), scaled to TARGET_VOL - for charts."""
    daily = daily_pnl(df, pos, mask).groupby("date").pnl.mean()
    scale = TARGET_VOL / (daily.std() * np.sqrt(252))
    return (daily * scale).cumsum() * 100


def evaluate(df: pd.DataFrame, pos: pd.Series, name: str, mask: pd.Series | None = None) -> dict:
    """Portfolio = average of the pairs' daily P&L (see daily_pnl)."""
    d = daily_pnl(df, pos, mask)
    out = {"strategy": name}

    def stats(x: pd.DataFrame, prefix: str = "") -> dict:
        daily = x.groupby("date")[["pnl", "gross", "carry", "cost"]].mean()
        if len(daily) < 50 or daily.pnl.std() == 0:
            return {f"{prefix}sharpe": np.nan}
        sharpe = daily.pnl.mean() / daily.pnl.std() * np.sqrt(252)
        scale = TARGET_VOL / (daily.pnl.std() * np.sqrt(252))   # report at a 10% vol target
        eq = (daily.pnl * scale).cumsum()
        res = {f"{prefix}sharpe": sharpe, f"{prefix}ann_ret%": daily.pnl.mean() * 252 * scale * 100}
        if not prefix:
            res.update({"max_dd%": (eq.cummax() - eq).max() * 100,
                        "carry_share%": daily.carry.sum() / daily.pnl.sum() * 100 if daily.pnl.sum() else np.nan,
                        "cost_drag%": daily.cost.mean() * 252 * scale * 100,
                        "from": daily.index.min().year, "to": daily.index.max().year})
        return res

    out.update(stats(d))
    for t in ("major", "g10 cross"):
        out.update(stats(d[d.tier == t], f"{t}:"))
    em = d[d.tier.str.startswith("exotic")]
    out.update(stats(em, "exotics:"))
    yearly = d.groupby(d["date"].dt.year).apply(
        lambda x: (lambda s: s.mean() / s.std() * np.sqrt(252) if s.std() else np.nan)(x.groupby("date").pnl.mean()))
    out["sharpe_by_year"] = " ".join(f"{y}:{v:+.1f}" for y, v in yearly.items())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.time()
    if not RATES_PATH.exists():
        download_rates()
    fx, mk = load_daily()
    df = build(fx, mk)
    df["spread_bps"] = df["pair"].map(spread_table(df["pair"].unique()))
    df = add_news(df)
    log.info("%d pair-days, %d pairs, %s -> %s (%.0fs)", len(df), df.pair.nunique(), df.date.min().date(),
             df.date.max().date(), time.time() - t0)

    rows = []
    sets = {"price": PRICE_COLS, "price+rates": PRICE_COLS + RATE_COLS,
            "price+rates+markets": PRICE_COLS + RATE_COLS + MKT_COLS}
    ml = {}
    for h in (5, 20):
        for tag, cols in sets.items():
            ml[f"ML {h}d | {tag}"] = ml_positions(df, h, cols)
            log.info("ML %dd %s done (%.0fs)", h, tag, time.time() - t0)
    oos = pd.concat([p.notna() for p in ml.values()], axis=1).all(axis=1)   # dates every ML model covers
    for name, pos in benchmark_positions(df).items():
        rows.append(evaluate(df, pos, f"{name} (same period as ML)", oos))
        rows.append(evaluate(df, pos, f"{name} (full history)"))
    for name, pos in ml.items():
        rows.append(evaluate(df, pos, name, oos))

    # ---- news: only ~3 years, so a short separate test (fewer folds), with vs without news
    news_rows = []
    news_mask = df["news_tone_diff_5d"].notna()
    if news_mask.sum() > 5000:
        sub = df[df.date >= df.loc[news_mask, "date"].min()].reset_index(drop=True)
        base_cols = PRICE_COLS + RATE_COLS + MKT_COLS
        p0 = ml_positions(sub, 5, base_cols, folds=3)
        p1 = ml_positions(sub, 5, base_cols + NEWS_COLS, folds=3)
        both = p0.notna() & p1.notna()
        news_rows.append(evaluate(sub, p0, "ML 5d | price+rates+markets", both))
        news_rows.append(evaluate(sub, p1, "ML 5d | + news (GDELT)", both))
        for name, pos in benchmark_positions(sub).items():
            news_rows.append(evaluate(sub, pos, name, both))

    def table(rs: list) -> str:
        t = pd.DataFrame(rs).set_index("strategy")
        main_cols = ["sharpe", "ann_ret%", "max_dd%", "carry_share%", "cost_drag%",
                     "major:sharpe", "g10 cross:sharpe", "exotics:sharpe", "from", "to"]
        return (t[[c for c in main_cols if c in t]].round(2).to_string()
                + "\n\nSharpe by year:\n" + t["sharpe_by_year"].to_string())

    text = (f"Daily FX research: {df.pair.nunique()} pairs, {df.date.min():%Y} -> {df.date.max():%Y}. Daily rebalanced, "
            f"each pair risk-weighted, portfolio shown at {TARGET_VOL:.0%} annual vol.\n"
            f"Costs: half spread per change in position (spreads: {TIER_SPREAD_BPS}); carry from FRED short rates; "
            "broker swap mark-ups not included.\nSharpe: return / volatility per year (0.5 decent, 1.0 very good "
            "for a simple strategy). ML = walk-forward out-of-sample.\n\n"
            + table(rows)
            + ("\n\n=== NEWS TEST (GDELT period only, ~3 years, 3 folds - low statistical power) ===\n" + table(news_rows)
               if news_rows else "\n\n(no news history found in data/gdelt/)"))
    print(text)
    OUT.write_text(text)
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60, OUT)


if __name__ == "__main__":
    main()
