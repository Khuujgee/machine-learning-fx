"""Non-FX inputs for the FX model: futures/indices (Yahoo, hourly) and interest rates (FRED, no API key).

    python -m src.macro_data            # download/refresh everything into data/markets and data/macro

Everything is stamped with the time it was actually *known* (an hourly bar is known at its close; a monthly
interest rate only ~2 months after the month starts) and joined "as of" each FX bar's close: no look-ahead.

Features added per FX row (see macro_feature_frame / pair_macro_features):
  global : each market's 1h/4h/24h return in volatility units (z), VIX level and 24h change
  pair   : signed signals that only make sense for a given pair, e.g. oil move x (oil sensitivity of base
           - quote currency), risk-on move x (risk beta of base - quote), USD move x USD side, and
           base/quote short rates + their difference (carry) and its 3-month change.
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR
import yfinance as yf

log = logging.getLogger(__name__)
MARKETS_DIR = DATA_DIR / "markets"
MACRO_DIR = DATA_DIR / "macro"
RATES_PATH = MACRO_DIR / "rates.parquet"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh) research"}

# name -> Yahoo symbol (hourly history ~730 days)
MARKETS = {
    "spx": "ES=F",     # S&P 500 futures  (risk appetite)
    "ndx": "NQ=F",     # Nasdaq-100 futures
    "oil": "CL=F",     # WTI crude        (CAD, NOK, MXN ...)
    "gold": "GC=F",    # gold             (AUD, ZAR, CHF ...)
    "copper": "HG=F",  # copper           (AUD, CLP, ZAR ...)
    "ust10": "ZN=F",   # 10y US Treasury note futures (price up = yields down)
    "vix": "^VIX",     # equity volatility / fear
    "dxy": "DX-Y.NYB", # US dollar index
    "nikkei": "^N225", # Japan equities
}

# currency -> FRED series to try in order (first one that's still being updated wins)
RATE_SERIES = {
    "USD": ["DFF"], "EUR": ["ECBDFR"], "GBP": ["IUDSOIA", "IR3TIB01GBM156N"],
    "JPY": ["IRSTCI01JPM156N", "IR3TIB01JPM156N"], "CHF": ["IR3TIB01CHM156N"],
    "AUD": ["IR3TIB01AUM156N"], "CAD": ["IR3TIB01CAM156N"], "NZD": ["IR3TIB01NZM156N"],
    "SEK": ["IR3TIB01SEM156N"], "NOK": ["IR3TIB01NOM156N"], "DKK": ["IR3TIB01DKM156N"],
    "PLN": ["IR3TIB01PLM156N"], "HUF": ["IR3TIB01HUM156N"], "CZK": ["IR3TIB01CZM156N"],
    "MXN": ["IR3TIB01MXM156N"], "ZAR": ["IR3TIB01ZAM156N"], "ILS": ["IR3TIB01ILM156N"],
    "CNH": ["IR3TIB01CNM156N"], "BRL": ["IRSTCI01BRM156N"],
    "TRY": ["IRSTCI01TRM156N", "IR3TIB01TRM156N"], "INR": ["IRSTCI01INM156N"], "THB": ["IRSTCI01THM156N"],
}

# rough, well-known sensitivities (+ = currency tends to rise when the driver rises)
RISK_BETA = {"JPY": -1, "CHF": -1, "USD": -0.5, "EUR": 0, "GBP": 0.3, "DKK": 0, "AUD": 1, "NZD": 1, "CAD": 0.5,
             "NOK": 0.7, "SEK": 0.7, "MXN": 1, "ZAR": 1, "TRY": 0.7, "BRL": 1, "PLN": 0.7, "HUF": 0.7, "CZK": 0.5,
             "ILS": 0.3, "CNH": 0.3, "INR": 0.3, "THB": 0.3, "SGD": 0.3, "HKD": 0}
OIL_BETA = {"CAD": 1, "NOK": 1, "MXN": 0.7, "BRL": 0.5, "JPY": -0.3, "INR": -0.5, "TRY": -0.5, "ZAR": 0.2}
GOLD_BETA = {"AUD": 0.7, "ZAR": 1, "CHF": 0.5, "JPY": 0.3}
COPPER_BETA = {"AUD": 1, "CNH": 0.5, "ZAR": 0.5, "BRL": 0.5, "NZD": 0.5}

MACRO_GLOBAL_COLS = [f"{m}_z{h}" for m in MARKETS for h in (1, 4, 24)] + ["vix_level", "vix_chg24"]
MACRO_PAIR_COLS = ["risk_signal", "oil_signal", "gold_signal", "copper_signal", "usd_signal", "rates_signal",
                   "base_rate", "quote_rate", "rate_diff", "rate_diff_chg90d"]
MACRO_COLS = MACRO_GLOBAL_COLS + MACRO_PAIR_COLS


# --------------------------------------------------------------------------- download
def _history(symbol: str, retries: int = 3, **kwargs) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            return yf.Ticker(symbol).history(**kwargs)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def download_markets(period: str = "730d") -> None:
    MARKETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, sym in MARKETS.items():
        try:
            df = _history(sym, period=period, interval="1h", auto_adjust=False)
            df = df[["Close"]].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_parquet(MARKETS_DIR / f"{name}.parquet")
            log.info("%s (%s): %d hourly bars %s -> %s", name, sym, len(df), df.index[0], df.index[-1])
        except Exception as e:
            log.warning("%s (%s) failed: %s", name, sym, type(e).__name__)
        time.sleep(0.3)


def _fred(series_id: str, retries: int = 4) -> pd.Series:
    """FRED CSV (no API key). The endpoint is flaky - same request answers in 0.1s or hangs - so retry."""
    for attempt in range(retries):
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", headers=UA, timeout=25)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    d = pd.read_csv(io.StringIO(r.text))
    s = pd.to_numeric(d.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(d.iloc[:, 0], utc=True)
    return s.dropna()


def download_rates() -> None:
    """Short-term rate per currency, stamped at the date it was knowable (daily: +1 day, monthly: +2 months)."""
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    cols, now = {}, pd.Timestamp.now(tz="UTC")
    for ccy, ids in RATE_SERIES.items():
        for sid in ids:
            try:
                s = _fred(sid)
            except Exception as e:
                log.warning("%s: %s download failed (%s)", ccy, sid, type(e).__name__)
                continue
            if s.empty or (now - s.index[-1]).days > 200:  # discontinued series (e.g. TRY 3m stops in 2008)
                continue
            monthly = sid.endswith("M156N") or sid.endswith("M193N")
            s.index = s.index + (pd.DateOffset(months=2) if monthly else pd.Timedelta(days=1))
            cols[ccy] = s
            log.info("%s: %s (%s), %d obs, last %.2f", ccy, sid, "monthly" if monthly else "daily", len(s), s.iloc[-1])
            break
        else:
            log.info("%s: no current FRED series - left missing", ccy)
        time.sleep(2)
    rates = pd.DataFrame(cols).sort_index()
    rates = rates[~rates.index.duplicated(keep="last")].ffill()
    rates.index.name = "known_at"
    rates.to_parquet(RATES_PATH)


# --------------------------------------------------------------------------- features
def macro_feature_frame() -> pd.DataFrame:
    """Global hourly market features indexed by `known_at` (= bar close)."""
    cols = {}
    for name in MARKETS:
        path = MARKETS_DIR / f"{name}.parquet"
        if not path.exists():
            continue
        c = pd.read_parquet(path)["Close"]
        c.index = c.index + pd.Timedelta(hours=1)  # an hourly bar is known when it closes
        r1 = np.log(c).diff()
        vol = r1.rolling(24 * 20, min_periods=100).std()  # ~1 month of hourly vol
        for h in (1, 4, 24):
            cols[f"{name}_z{h}"] = (np.log(c).diff(h) / (vol * np.sqrt(h))).clip(-6, 6)
        if name == "vix":
            cols["vix_level"] = c
            cols["vix_chg24"] = c.diff(24)
    out = pd.DataFrame(cols).sort_index()
    # markets trade different hours: carry each one's last known value forward across the shared timeline
    # (up to ~4 days, i.e. over a weekend), otherwise VIX/Nikkei look "missing" whenever another market ticks
    out = out[~out.index.duplicated(keep="last")].ffill(limit=96)
    out.index.name = "known_at"
    return out.replace([np.inf, -np.inf], np.nan)


def _beta(table: dict, ccy: pd.Series) -> np.ndarray:
    return ccy.map(table).fillna(0).values


def add_macro_features(df: pd.DataFrame, global_frame: pd.DataFrame | None = None,
                       rates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join macro features onto FX rows (needs columns `pair`, `close_time`). Missing data stays NaN.
    Returns a frame with the same index and row order as `df`."""
    if global_frame is None:
        global_frame = macro_feature_frame()
    if rates is None and RATES_PATH.exists():
        rates = pd.read_parquet(RATES_PATH)
    orig_index = df.index
    d = df.drop(columns=[c for c in MACRO_COLS if c in df], errors="ignore").copy()
    d["_row"] = np.arange(len(d))
    d = d.reset_index(drop=True).sort_values("close_time", kind="stable")
    if not global_frame.empty:
        g = global_frame.reset_index().sort_values("known_at")
        d = pd.merge_asof(d, g, left_on="close_time", right_on="known_at", direction="backward",
                          tolerance=pd.Timedelta(days=4)).drop(columns="known_at")
    for c in MACRO_GLOBAL_COLS:
        if c not in d:
            d[c] = np.nan
    base, quote = d["pair"].str[:3], d["pair"].str[3:]
    d["risk_signal"] = d["spx_z24"] * (_beta(RISK_BETA, base) - _beta(RISK_BETA, quote))
    d["oil_signal"] = d["oil_z24"] * (_beta(OIL_BETA, base) - _beta(OIL_BETA, quote))
    d["gold_signal"] = d["gold_z24"] * (_beta(GOLD_BETA, base) - _beta(GOLD_BETA, quote))
    d["copper_signal"] = d["copper_z24"] * (_beta(COPPER_BETA, base) - _beta(COPPER_BETA, quote))
    usd_side = np.where(base == "USD", 1, np.where(quote == "USD", -1, 0))
    d["usd_signal"] = d["dxy_z24"] * usd_side
    d["rates_signal"] = -d["ust10_z24"] * usd_side  # T-note futures up = US yields down = usually USD down
    for c in ("base_rate", "quote_rate", "rate_diff", "rate_diff_chg90d"):
        d[c] = np.nan
    if rates is not None and not rates.empty:
        r = rates.reset_index().sort_values("known_at")
        now = pd.merge_asof(d[["close_time"]], r, left_on="close_time", right_on="known_at", direction="backward")
        ago = pd.merge_asof(d[["close_time"]].assign(t=d["close_time"] - pd.Timedelta(days=90))[["t"]], r,
                            left_on="t", right_on="known_at", direction="backward")

        def lookup(frame: pd.DataFrame, ccys: pd.Series) -> np.ndarray:
            vals = np.full(len(d), np.nan)
            for c in ccys.unique():
                if c in frame:
                    m = (ccys == c).values
                    vals[m] = frame[c].values[m]
            return vals

        d["base_rate"], d["quote_rate"] = lookup(now, base), lookup(now, quote)
        d["rate_diff"] = d["base_rate"] - d["quote_rate"]
        d["rate_diff_chg90d"] = d["rate_diff"] - (lookup(ago, base) - lookup(ago, quote))
    d = d.sort_values("_row").drop(columns="_row")
    d.index = orig_index
    return d


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    download_markets()
    download_rates()


if __name__ == "__main__":
    main()
