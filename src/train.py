"""Train an XGBoost direction classifier with purged walk-forward (time-series) cross-validation.

Usage: python -m src.train [--splits 5] [--threshold 0.58]
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from .config import FEATURES_PATH, HORIZON_BARS, INTERVAL, MODEL_PATH, PROB_THRESHOLD
from .features import FEATURE_COLS

log = logging.getLogger(__name__)

XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


def walk_forward_splits(df: pd.DataFrame, n_splits: int, horizon: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window splits over *timestamps* (all pairs share the time axis).

    - sklearn TimeSeriesSplit on the sorted unique close_times, with `gap=horizon` bars.
    - Purge: additionally drop any training row whose label (close at t+horizon) is only known
      at/after the test window starts - these overlap the test period and would leak.
    """
    times = pd.DatetimeIndex(np.sort(df["close_time"].unique()))
    close_time, label_time = df["close_time"].values, df["label_time"].values
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=horizon).split(times):
        test_start, test_end = times[te[0]], times[te[-1]]
        train_mask = (close_time <= times[tr[-1]].to_datetime64()) & (label_time < test_start.to_datetime64())
        test_mask = (close_time >= test_start.to_datetime64()) & (close_time <= test_end.to_datetime64())
        yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


def evaluate(y: np.ndarray, p: np.ndarray, fwd_ret: np.ndarray, threshold: float) -> dict:
    long_, short = p >= threshold, p <= 1 - threshold
    signal = np.where(long_, 1, np.where(short, -1, 0))
    taken = signal != 0
    signed = signal[taken] * fwd_ret[taken]
    return {
        "n": len(y),
        "base_rate_up": y.mean(),
        "accuracy": accuracy_score(y, p >= 0.5),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "logloss": log_loss(y, p, labels=[0, 1]),
        "signals": int(taken.sum()),
        "signal_hit_rate": (signed > 0).mean() if taken.any() else np.nan,
        "avg_signal_ret_bps": signed.mean() * 1e4 if taken.any() else np.nan,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=PROB_THRESHOLD)
    args = ap.parse_args()

    df = pd.read_parquet(FEATURES_PATH).sort_values(["close_time", "pair"], ignore_index=True)
    X, y, fwd = df[FEATURE_COLS].astype(float), df["target"].values, df["fwd_ret"].values
    log.info("%d rows, %d pairs, %d features, up-rate %.3f", len(df), df["pair"].nunique(), len(FEATURE_COLS), y.mean())

    folds = []
    for k, (tr, te) in enumerate(walk_forward_splits(df, args.splits, HORIZON_BARS), 1):
        model = XGBClassifier(**XGB_PARAMS).fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        m = evaluate(y[te], p, fwd[te], args.threshold)
        m.update(fold=k, train_rows=len(tr),
                 test_from=df["close_time"].iloc[te[0]], test_to=df["close_time"].iloc[te[-1]])
        folds.append(m)
        log.info("fold %d | train %d | test %d | acc %.4f | auc %.4f | hit %.4f on %d signals",
                 k, len(tr), m["n"], m["accuracy"], m["auc"], m["signal_hit_rate"], m["signals"])

    cv = pd.DataFrame(folds).set_index("fold")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nWalk-forward CV\n", cv.round(4).to_string())
        print("\nMean:\n", cv.drop(columns=["test_from", "test_to"]).mean().round(4).to_string())

    final = XGBClassifier(**XGB_PARAMS).fit(X, y)
    imp = pd.Series(final.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance (gain-based)\n", imp.round(4).to_string())

    joblib.dump({
        "model": final,
        "features": FEATURE_COLS,
        "threshold": args.threshold,
        "horizon_bars": HORIZON_BARS,
        "interval": INTERVAL,
        "pairs": sorted(df["pair"].unique()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_range": (str(df["close_time"].min()), str(df["close_time"].max())),
        "cv": cv.reset_index().to_dict("records"),
    }, MODEL_PATH)
    log.info("saved model bundle -> %s", MODEL_PATH)


if __name__ == "__main__":
    main()
