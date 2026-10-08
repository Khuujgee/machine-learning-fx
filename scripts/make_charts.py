"""Rebuild the figures in docs/img/ used by RESEARCH.md.

    .venv/bin/python scripts/make_charts.py     (needs data from src.backtest and src.research_daily runs)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import research_daily as D  # noqa: E402
from src.backtest import TRADES_OUT, non_overlapping, spread_table  # noqa: E402

OUT = ROOT / "docs" / "img"
# reference palette (validated: categorical slots 1-3 + blue/red diverging pair, light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def titled(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=10.5, color=INK2, va="top")


def edge_vs_cost() -> None:
    t = non_overlapping(pd.read_parquet(TRADES_OUT))
    order = ["major", "g10 cross", "exotic-mid", "exotic-high"]
    names = ["Majors", "G10 crosses", "Exotics (mid)", "Exotics (TRY, ZAR, MXN, BRL)"]
    g = t.groupby("tier").agg(gross=("gross_bps", "mean"), spread=("spread_bps", "mean")).reindex(order)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y, h = np.arange(len(order)), 0.36
    ax.barh(y - h / 2 - 0.01, g["gross"], height=h, color=BLUE, label="Model's average gain, before costs")
    ax.barh(y + h / 2 + 0.01, g["spread"], height=h, color=ORANGE, label="Assumed spread: the cost")
    for yi, (gv, sv) in enumerate(zip(g["gross"], g["spread"])):
        ax.text(gv + 0.3, yi - h / 2, f"{gv:.1f}", va="center", color=INK, fontsize=10)
        ax.text(sv + 0.3, yi + h / 2, f"{sv:.1f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("basis points per trade (1 bp = 0.01% of price)")
    ax.set_xlim(0, max(g["gross"].max(), g["spread"].max()) * 1.12)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncol=2, fontsize=10, handlelength=1.2)
    titled(fig, "The model's edge is smaller than the cost of trading it",
           "Hourly XGBoost model, 4-hour hold, out-of-sample Jun 2024 - Oct 2026, 55k trades")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.13)
    fig.savefig(OUT / "edge_vs_cost.png", dpi=160)
    plt.close(fig)


def daily_charts() -> None:
    fx, mk = D.load_daily()
    df = D.build(fx, mk)
    df["spread_bps"] = df.pair.map(spread_table(df.pair.unique()))
    ml = {"ML, 20-day target (price + rates)": D.ml_positions(df, 20, D.PRICE_COLS + D.RATE_COLS),
          "ML, 5-day target (price + rates + markets)": D.ml_positions(df, 5, D.PRICE_COLS + D.RATE_COLS + D.MKT_COLS)}
    mask = pd.concat([p.notna() for p in ml.values()], axis=1).all(axis=1)
    bench = D.benchmark_positions(df)
    strategies = {"Carry": bench["CARRY"], "Trend + carry": bench["TREND+CARRY"],
                  "Trend, 12-month": bench["MOM-12M"], "Trend, 3-month": bench["MOM-3M"],
                  "ML, 20-day (price + rates)": ml["ML, 20-day target (price + rates)"],
                  "ML, 5-day (+ markets)": ml["ML, 5-day target (price + rates + markets)"]}
    sharpe = {k: D.evaluate(df, p, k, mask)["sharpe"] for k, p in strategies.items()}
    s = pd.Series(sharpe).sort_values()

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [BLUE if v > 0 else RED for v in s.values]
    ax.barh(range(len(s)), s.values, height=0.6, color=colors)
    for i, v in enumerate(s.values):
        ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                color=INK, fontsize=10)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(s)), s.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(s.min() * 1.35, -0.1), max(s.max() * 1.6, 0.2))
    ax.set_xlabel("Sharpe ratio after spreads and carry (return per unit of risk, per year)")
    titled(fig, "Over 25 years, only carry made money after costs",
           f"Daily FX strategies, 56 pairs, out-of-sample {df.loc[mask, 'date'].min():%Y} - {df.loc[mask, 'date'].max():%Y}. "
           "Blue = made money, red = lost money")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.82, bottom=0.14)
    fig.savefig(OUT / "daily_sharpe.png", dpi=160)
    plt.close(fig)

    curves = {"Carry": (bench["CARRY"], BLUE), "ML, 20-day target": (ml["ML, 20-day target (price + rates)"], ORANGE),
              "Trend, 3-month": (bench["MOM-3M"], AQUA)}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name, (pos, color) in curves.items():
        c = D.portfolio_curve(df, pos, mask)
        ax.plot(c.index, c.values, color=color, linewidth=2, label=name)
        ax.annotate(f"{name} {c.iloc[-1]:+.0f}%", (c.index[-1], c.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", color=INK, va="center", fontsize=10, annotation_clip=False)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_ylabel("cumulative return, %")
    ax.legend(loc="lower left", fontsize=10)
    titled(fig, "Carry slowly earns; the ML and trend strategies bleed",
           "Cumulative return at 10% annual volatility, out-of-sample, after costs, not compounded")
    fig.subplots_adjust(left=0.08, right=0.78, top=0.82, bottom=0.08)
    fig.savefig(OUT / "equity_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    OUT.mkdir(parents=True, exist_ok=True)
    edge_vs_cost()
    daily_charts()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.png"))))
