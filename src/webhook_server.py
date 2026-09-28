"""FastAPI webhook listener for TradingView alerts + background paper-trade monitor.

Run:  uvicorn src.webhook_server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import db
from .config import (ENFORCE_IP_ALLOWLIST, MAX_ALERT_AGE_SEC, MAX_BODY_BYTES, MODEL_PATH,
                     MONITOR_INTERVAL_SEC, SCANNER_ENABLED, TRADINGVIEW_IPS, TRUST_PROXY_HEADERS,
                     WEBHOOK_SECRET)
from .paper_engine import PaperEngine, normalize_ticker
from .scanner import run_scan, scanner_task

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("webhook")


class TradingViewAlert(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    secret: str = Field(min_length=1)
    ticker: str = Field(min_length=3, max_length=40)
    exchange: Optional[str] = None
    timeframe: str = Field(max_length=10)
    price: Optional[float] = None
    time: Optional[str] = None                     # {{timenow}}, used for replay protection
    signal: Optional[str] = Field(default=None, max_length=20)  # buy | sell | long | short | any
    triggers: Dict[str, Union[float, int, str, bool, None]] = Field(default_factory=dict)


class State:
    def __init__(self) -> None:
        self.engine: Optional[PaperEngine] = None
        self.tasks: set = set()  # strong refs so background tasks aren't garbage-collected
        self.stop: Optional[asyncio.Event] = None


state = State()


async def monitor_loop() -> None:
    """Periodically check open paper trades against live prices for SL/TP hits."""
    while not state.stop.is_set():
        if state.engine is not None:
            try:
                closed = await asyncio.to_thread(state.engine.check_open_trades)
                if closed:
                    log.info("monitor closed %d trade(s)", len(closed))
            except Exception:
                log.exception("monitor cycle failed")
        try:
            await asyncio.wait_for(state.stop.wait(), timeout=MONITOR_INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    if len(WEBHOOK_SECRET) < 16 and not SCANNER_ENABLED:
        raise RuntimeError("Set WEBHOOK_SECRET (>=16 chars) in .env, or SCANNER_ENABLED=true for scanner-only mode.")
    db.init_db()
    if MODEL_PATH.exists():
        state.engine = PaperEngine(MODEL_PATH)
    else:
        log.error("model not found at %s - run `python -m src.train` first; webhooks will return 503", MODEL_PATH)
    state.stop = asyncio.Event()
    tasks = [asyncio.create_task(monitor_loop())]
    if SCANNER_ENABLED and state.engine is not None:
        log.info("built-in scanner enabled: scoring %d pairs at every hourly close", len(state.engine.pairs))
        tasks.append(asyncio.create_task(scanner_task(state.engine, state.stop)))
    yield
    state.stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="TradingView -> XGBoost paper trader", lifespan=lifespan)


def client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS and request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[0].strip()
    return request.client.host if request.client else ""


async def process_alert(alert_id: str, pair: str, alert: dict) -> None:
    try:
        decision = await asyncio.to_thread(state.engine.handle_signal, pair, alert, alert_id)
        status = "traded" if decision["action"] == "opened" else "no_trade"
        log.info("alert %s %s -> %s %s", alert_id, pair, decision["action"], decision.get("reason", ""))
    except Exception as e:
        log.exception("alert %s failed", alert_id)
        decision, status = {"action": "error", "error": repr(e)}, "error"
    db.update_alert(alert_id, status, decision)


@app.post("/webhook/tradingview", status_code=202)
async def tradingview_webhook(request: Request) -> dict[str, Any]:
    # 1. source IP (TradingView publishes a fixed set of sender IPs)
    ip = client_ip(request)
    if ENFORCE_IP_ALLOWLIST and ip not in TRADINGVIEW_IPS:
        log.warning("rejected webhook from %s", ip)
        raise HTTPException(403, "forbidden")

    # 2. size-limited parse + schema validation
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(413, "payload too large")
    try:
        alert = TradingViewAlert.model_validate_json(raw)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False, include_input=False))

    if len(WEBHOOK_SECRET) < 16:
        raise HTTPException(503, "webhook disabled: WEBHOOK_SECRET not configured")

    # 3. shared secret (TradingView can't send custom headers, so it lives in the body); constant-time compare
    if not hmac.compare_digest(alert.secret.encode(), WEBHOOK_SECRET.encode()):
        log.warning("bad secret from %s", ip)
        raise HTTPException(401, "unauthorized")

    # 4. replay protection: reject old alerts and exact duplicates
    if alert.time:
        try:
            sent = pd.Timestamp(alert.time)
            sent = sent.tz_localize("UTC") if sent.tzinfo is None else sent
            if abs((pd.Timestamp.now(tz="UTC") - sent).total_seconds()) > MAX_ALERT_AGE_SEC:
                raise HTTPException(400, "stale alert")
        except (ValueError, TypeError):
            raise HTTPException(422, "unparseable 'time'")
    alert_id = hashlib.sha256(raw).hexdigest()[:24]
    if db.alert_exists(alert_id):
        return {"status": "duplicate", "alert_id": alert_id}

    pair = normalize_ticker(alert.ticker)
    if pair is None:
        raise HTTPException(422, f"unsupported ticker {alert.ticker!r}")
    if state.engine is None:
        raise HTTPException(503, "model not loaded")

    payload = alert.model_dump(exclude={"secret"})
    db.log_alert(alert_id, pair, payload)

    # 5. TradingView times out after ~3s, so ack now and run inference in the background
    task = asyncio.create_task(process_alert(alert_id, pair, payload))
    state.tasks.add(task)
    task.add_done_callback(state.tasks.discard)
    return {"status": "accepted", "alert_id": alert_id, "pair": pair}


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "model_loaded": state.engine is not None,
            "model": state.engine.meta if state.engine else None}


@app.get("/account")
async def account() -> dict:
    return db.account_summary()


@app.get("/trades")
async def trades(status: Optional[str] = None, limit: int = 100) -> list:
    if status not in (None, "open", "closed"):
        raise HTTPException(422, "status must be open|closed")
    return db.list_trades(status, min(limit, 1000))


@app.get("/alerts")
async def alerts(limit: int = 50) -> list:
    return db.list_alerts(min(limit, 500))


@app.post("/trades/check")
async def check_now() -> list:
    """Force an immediate SL/TP sweep (the monitor also runs every MONITOR_INTERVAL_SEC)."""
    if state.engine is None:
        raise HTTPException(503, "model not loaded")
    return await asyncio.to_thread(state.engine.check_open_trades)


@app.post("/scan")
async def scan_now() -> dict:
    """Run the built-in scanner once, right now, on the latest completed bars."""
    if state.engine is None:
        raise HTTPException(503, "model not loaded")
    return await asyncio.to_thread(run_scan, state.engine, False)
