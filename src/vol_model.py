"""Next-day FX volatility forecast (HAR model) - the part of this project where prediction *does* work.

Realized volatility per FX trading day (22:00 -> 22:00 UTC, the New York close) from hourly returns.
HAR model (Corsi 2009): log RV(next day) = a + b_d log RV(day) + b_w log RV(last 5 days) + b_m log RV(last 22 days)
+ day-of-week effects, fitted on all monitored pairs together (in log space, so pairs are comparable).

    python -m src.vol_model validate      # walk-forward test vs naive forecasts (for RESEARCH.md)
    python -m src.vol_model fit           # fit on all history -> models/vol_har.json
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from .config import MODELS_DIR, RAW_DIR
from .market_data import download_ohlcv

log = logging.getLogger(__name__)
MODEL_PATH = MODELS_DIR / "vol_har.json"
PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "USDSEK", "USDNOK"]
FEATURES = ["lrv_d", "lrv_w", "lrv_m", "mon", "fri"]


def daily_rv(hourly_close: pd.Series) -> pd.Series:
    """Annualised realized vol per FX day (22:00 UTC roll). Days with < 18 hourly bars are dropped."""
    r = np.log(hourly_close).diff().dropna()
    day = (r.index + pd.Timedelta(hours=2)).normalize()      # 22:00 UTC belongs to the next FX day
    g = r.groupby(day)
    rv = np.sqrt((r ** 2).groupby(day).sum()) * np.sqrt(252)
    rv = rv[g.size() >= 18]
    rv.index = rv.index.tz_localize(None) if rv.index.tz is not None else rv.index
    return rv[rv > 0]


def har_frame(rv: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"rv": rv})
    d["lrv_d"] = np.log(rv)
    d["lrv_w"] = np.log(rv.rolling(5).mean())
    d["lrv_m"] = np.log(rv.rolling(22).mean())
    nxt = d.index.to_series().shift(-1)
    d["mon"] = (nxt.dt.dayofweek == 0).astype(float)          # the day being forecast is a Monday / Friday
    d["fri"] = (nxt.dt.dayofweek == 4).astype(float)
    d["target"] = np.log(rv.shift(-1))
    return d


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def _predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return coef[0] + X @ coef[1:]


def load_hourly(pair: str) -> pd.Series:
    path = RAW_DIR / f"{pair}.parquet"
    c = pd.read_parquet(path)["Close"] if path.exists() else pd.Series(dtype=float)
    try:  # top up with the latest bars
        new = download_ohlcv(pair, period="60d", interval="1h")["Close"]
        c = pd.concat([c, new])
        c = c[~c.index.duplicated(keep="last")].sort_index()
    except Exception as e:
        log.warning("%s: price refresh failed (%s)", pair, type(e).__name__)
    return c[c > 0]


def panel(pairs: list[str] = PAIRS) -> pd.DataFrame:
    parts = []
    for p in pairs:
        f = har_frame(daily_rv(load_hourly(p)))
        f["pair"] = p
        parts.append(f)
    return pd.concat(parts).rename_axis("date").reset_index()


def validate(min_train_days: int = 250, refit_every: int = 20) -> pd.DataFrame:
    """Walk-forward: refit every 20 days on all earlier days, forecast the next ones. Compare with naive."""
    d = panel().dropna(subset=FEATURES + ["target"]).sort_values("date")
    dates = np.sort(d["date"].unique())
    preds = []
    for i in range(min_train_days, len(dates), refit_every):
        train = d[d["date"] < dates[i]]
        test = d[(d["date"] >= dates[i]) & (d["date"] < dates[min(i + refit_every, len(dates) - 1)])]
        if test.empty:
            continue
        coef = _ols(train[FEATURES].values, train["target"].values)
        preds.append(test.assign(har=_predict(coef, test[FEATURES].values)))
    p = pd.concat(preds)
    p["naive_yesterday"] = p["lrv_d"]
    p["naive_month"] = p["lrv_m"]
    rows = {}
    for m in ("naive_yesterday", "naive_month", "har"):
        err = p["target"] - p[m]
        r2 = 1 - (err ** 2).sum() / ((p["target"] - p["target"].mean()) ** 2).sum()
        rows[m] = {"R2 (log vol)": r2, "median abs error (% of vol)": (np.exp(err.abs()) - 1).median() * 100,
                   "corr": np.corrcoef(p["target"], p[m])[0, 1]}
    res = pd.DataFrame(rows).T
    # does the forecast separate calm from wild days? actual vol by forecast quintile
    p["q"] = pd.qcut(p["har"], 5, labels=["lowest 20%", "2", "3", "4", "highest 20%"])
    by_q = p.groupby("q", observed=True).apply(lambda x: np.exp(x["target"]).mean())
    log.info("tested %d pair-days, %s -> %s", len(p), p["date"].min().date(), p["date"].max().date())
    return res, by_q, p


def fit() -> dict:
    d = panel().dropna(subset=FEATURES + ["target"])
    coef = _ols(d[FEATURES].values, d["target"].values)
    model = {"coef": coef.tolist(), "features": FEATURES, "fitted_at": pd.Timestamp.now(tz="UTC").isoformat(),
             "rows": len(d)}
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    return model


def load_model(max_age_days: int = 7) -> dict:
    if MODEL_PATH.exists():
        m = json.loads(MODEL_PATH.read_text())
        if pd.Timestamp.now(tz="UTC") - pd.Timestamp(m["fitted_at"]) < pd.Timedelta(days=max_age_days):
            return m
    return fit()


def outlook(pairs: list[str] = PAIRS) -> pd.DataFrame:
    """Forecast for the next FX day per pair, with its percentile vs that pair's last year of daily vol."""
    m = load_model()
    coef = np.array(m["coef"])
    rows = []
    for p in pairs:
        try:
            rv = daily_rv(load_hourly(p))
        except Exception as e:
            log.warning("%s: %s", p, type(e).__name__)
            continue
        f = har_frame(rv).dropna(subset=["lrv_d", "lrv_w", "lrv_m"])
        if f.empty:
            continue
        last = f.iloc[-1].copy()
        nxt = f.index[-1] + pd.offsets.BDay(1)
        last["mon"], last["fri"] = float(nxt.dayofweek == 0), float(nxt.dayofweek == 4)
        fc = float(np.exp(_predict(coef, last[FEATURES].values.astype(float).reshape(1, -1))[0]))
        hist = rv[rv.index >= rv.index[-1] - pd.Timedelta(days=365)]
        pct = float((hist < fc).mean() * 100)
        rows.append({"pair": p, "forecast_vol": fc, "percentile": pct, "last_day_vol": float(rv.iloc[-1]),
                     "for_day": nxt.date()})
    out = pd.DataFrame(rows)
    out["level"] = pd.cut(out["percentile"], [-1, 25, 75, 90, 101], labels=["calm", "normal", "elevated", "high"])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "fit", "outlook"])
    args = ap.parse_args()
    if args.cmd == "validate":
        res, by_q, _ = validate()
        print("Next-day realized volatility, walk-forward out-of-sample (9 G10-vs-USD pairs):")
        print(res.round(3).to_string())
        print("\nAverage actual next-day vol (annualised) by forecast quintile:")
        print((by_q * 100).round(1).astype(str).add("%").to_string())
    elif args.cmd == "fit":
        print(fit())
    else:
        print(outlook().round(3).to_string(index=False))


if __name__ == "__main__":
    main()
