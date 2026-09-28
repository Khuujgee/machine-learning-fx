"""SQLite persistence for news headlines, webhook alerts and paper trades."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional

import pandas as pd

from .config import DB_PATH, STARTING_EQUITY

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    published_at  TEXT NOT NULL,             -- ISO-8601 UTC
    headline      TEXT NOT NULL UNIQUE,
    source        TEXT,
    url           TEXT,
    currencies    TEXT NOT NULL DEFAULT '',  -- comma-separated ISO codes the headline mentions
    positive      REAL,
    negative      REAL,
    neutral       REAL,
    score         REAL,                      -- positive - negative, in [-1, 1]
    label         TEXT
);
CREATE INDEX IF NOT EXISTS ix_news_published ON news(published_at);

CREATE TABLE IF NOT EXISTS webhook_alerts (
    alert_id      TEXT PRIMARY KEY,          -- sha256 of the raw body (dedupes retries)
    received_at   TEXT NOT NULL,
    pair          TEXT,
    payload       TEXT NOT NULL,             -- JSON with the secret stripped
    status        TEXT NOT NULL,             -- queued | traded | no_trade | error
    decision      TEXT                       -- JSON
);

CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id      TEXT,
    pair          TEXT NOT NULL,
    direction     TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    units         REAL NOT NULL,
    entry_price   REAL NOT NULL,
    entry_time    TEXT NOT NULL,
    stop_loss     REAL NOT NULL,
    take_profit   REAL NOT NULL,
    quote_usd     REAL NOT NULL,             -- USD value of 1 unit of quote currency at entry
    prob_up       REAL NOT NULL,
    atr           REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    last_checked  TEXT,
    exit_price    REAL,
    exit_time     TEXT,
    exit_reason   TEXT,                      -- stop_loss | take_profit | manual
    pnl_usd       REAL
);
CREATE INDEX IF NOT EXISTS ix_trades_status ON trades(status, pair);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        # WAL lets the scanner threads, the monitor and the API read while one writer commits.
        # It is persisted in the file, so setting it once here is enough.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- news
def filter_new_headlines(items: Iterable[dict]) -> list[dict]:
    items = list(items)
    if not items:
        return []
    with connect() as conn:
        seen = {
            r["headline"]
            for r in conn.execute(
                f"SELECT headline FROM news WHERE headline IN ({','.join('?' * len(items))})",
                [i["headline"] for i in items],
            )
        }
    return [i for i in items if i["headline"] not in seen]


def insert_news(rows: list[dict]) -> int:
    cols = ["published_at", "headline", "source", "url", "currencies",
            "positive", "negative", "neutral", "score", "label"]
    with connect() as conn:
        cur = conn.executemany(
            f"INSERT OR IGNORE INTO news ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [[r.get(c) for c in cols] for r in rows],
        )
        return cur.rowcount


def load_news(since: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    sql, params = "SELECT published_at, currencies, score FROM news", []
    if since is not None:
        sql += " WHERE published_at >= ?"
        params.append(since.isoformat())
    with connect() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# --------------------------------------------------------------------------- webhook alerts
def alert_exists(alert_id: str) -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 FROM webhook_alerts WHERE alert_id=?", (alert_id,)).fetchone() is not None


def log_alert(alert_id: str, pair: Optional[str], payload: dict, status: str = "queued") -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO webhook_alerts (alert_id, received_at, pair, payload, status) VALUES (?,?,?,?,?)",
            (alert_id, utcnow_iso(), pair, json.dumps(payload, default=str), status),
        )


def update_alert(alert_id: str, status: str, decision: dict) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE webhook_alerts SET status=?, decision=? WHERE alert_id=?",
            (status, json.dumps(decision, default=str), alert_id),
        )


def list_alerts(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM webhook_alerts ORDER BY received_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d["payload"])
        d["decision"] = json.loads(d["decision"]) if d["decision"] else None
        out.append(d)
    return out


# --------------------------------------------------------------------------- trades
def insert_trade(trade: dict[str, Any]) -> int:
    cols = list(trade)
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [trade[c] for c in cols],
        )
        return int(cur.lastrowid)


def get_open_trades(pair: Optional[str] = None) -> list[dict]:
    sql, params = "SELECT * FROM trades WHERE status='open'", []
    if pair:
        sql += " AND pair=?"
        params.append(pair)
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def touch_trade(trade_id: int, last_checked: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE trades SET last_checked=? WHERE id=?", (last_checked, trade_id))


def close_trade(trade_id: int, exit_price: float, exit_time: str, reason: str, pnl_usd: float) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE trades SET status='closed', exit_price=?, exit_time=?, exit_reason=?,
                   pnl_usd=?, last_checked=? WHERE id=? AND status='open'""",
            (exit_price, exit_time, reason, pnl_usd, exit_time, trade_id),
        )


def list_trades(status: Optional[str] = None, limit: int = 100) -> list[dict]:
    sql, params = "SELECT * FROM trades", []
    if status:
        sql += " WHERE status=?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params)]


def account_summary() -> dict:
    with connect() as conn:
        r = conn.execute(
            """SELECT COALESCE(SUM(pnl_usd),0)                             AS realized,
                      COALESCE(SUM(status='closed'),0)                     AS closed,
                      COALESCE(SUM(status='open'),0)                       AS open,
                      COALESCE(SUM(status='closed' AND pnl_usd>0),0)       AS wins
               FROM trades"""
        ).fetchone()
    closed = r["closed"] or 0
    return {
        "starting_equity": STARTING_EQUITY,
        "realized_pnl": round(r["realized"], 2),
        "equity": round(STARTING_EQUITY + r["realized"], 2),
        "open_trades": r["open"],
        "closed_trades": closed,
        "win_rate": round(r["wins"] / closed, 4) if closed else None,
    }
