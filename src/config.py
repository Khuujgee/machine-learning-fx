"""Central configuration. Every setting can be overridden via environment variables or a .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


# --------------------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT / "models"
for _d in (DATA_DIR, RAW_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = DATA_DIR / "features.parquet"
UNIVERSE_PATH = DATA_DIR / "universe.json"
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "paper_trading.db")))
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "xgb_direction.joblib")))

# --------------------------------------------------------------------------- market data
INTERVAL = "1h"
BAR_HOURS = 1
HISTORY_PERIOD = "730d"  # Yahoo's maximum lookback for 1h bars
HORIZON_BARS = 4  # label = direction of close[t+4] vs close[t]  (4 hours on 1h bars)
N_PAIRS = int(os.getenv("N_PAIRS", "50"))

# Candidate universe; data_collection ranks these by ATR% and keeps the N_PAIRS most volatile.
CANDIDATE_PAIRS = [
    # majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
    # crosses
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD", "CADJPY", "CADCHF", "CHFJPY",
    "NZDJPY", "NZDCHF", "NZDCAD",
    # exotics / EM
    "USDTRY", "USDZAR", "USDMXN", "USDSEK", "USDNOK", "USDDKK", "USDPLN", "USDHUF",
    "USDCZK", "USDSGD", "USDHKD", "USDCNH", "USDINR", "USDTHB", "USDILS", "USDBRL",
    "EURTRY", "EURPLN", "EURHUF", "EURSEK", "EURNOK", "EURZAR", "EURMXN",
    "GBPZAR", "GBPSEK", "GBPNOK", "ZARJPY", "MXNJPY", "TRYJPY",
]

# --------------------------------------------------------------------------- news / NLP
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
# RSS feeds (preferred) or HTML pages. forexfactory.com/news is Cloudflare-protected and returns 403
# to scripts, and FXStreet's robots.txt disallows its RSS, so we use investingLive (ex-ForexLive) feeds.
NEWS_URLS = _env_list("NEWS_URLS", "https://investinglive.com/feed/news,https://investinglive.com/feed/forex,https://investinglive.com/feed/centralbank")
# CSS selector for headline elements on HTML pages (ignored for RSS feeds).
NEWS_HEADLINE_SELECTOR = os.getenv(
    "NEWS_HEADLINE_SELECTOR", "article h1, article h2, article h3, h2 a, h3 a"
)
NEWS_USER_AGENT = os.getenv("NEWS_USER_AGENT", "Mozilla/5.0 (research paper-trading bot)")
NEWS_REFRESH_SEC = int(os.getenv("NEWS_REFRESH_SEC", "900"))
SENTIMENT_HALFLIFE_HOURS = float(os.getenv("SENTIMENT_HALFLIFE_HOURS", "12"))

# --------------------------------------------------------------------------- webhook security
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
# TradingView's published webhook source IPs.
TRADINGVIEW_IPS = set(
    _env_list("TRADINGVIEW_IPS", "52.89.214.238,34.212.75.30,54.218.53.128,52.32.178.7")
)
ENFORCE_IP_ALLOWLIST = _env_bool("ENFORCE_IP_ALLOWLIST", False)
TRUST_PROXY_HEADERS = _env_bool("TRUST_PROXY_HEADERS", False)  # true behind ngrok/nginx/cloudflared
MAX_ALERT_AGE_SEC = int(os.getenv("MAX_ALERT_AGE_SEC", "300"))
MAX_BODY_BYTES = 10_000
ALLOW_UNTRAINED_PAIRS = _env_bool("ALLOW_UNTRAINED_PAIRS", False)

# --------------------------------------------------------------------------- paper trading
STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", "100000"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1% of equity risked to the stop
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))
PROB_THRESHOLD = float(os.getenv("PROB_THRESHOLD", "0.58"))  # long if p>=thr, short if p<=1-thr
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "20"))
LOT_STEP = int(os.getenv("LOT_STEP", "1000"))  # round position size down to micro lots
COST_BPS = float(os.getenv("COST_BPS", "1.0"))  # simulated spread+slippage per side, in bps of notional
# Close every new trade after this many hours if neither stop-loss nor take-profit was hit. The model predicts
# the 4h direction, so holding longer than that trades a prediction that has gone stale. 0 = no time limit.
MAX_HOLD_HOURS = float(os.getenv("MAX_HOLD_HOURS", "4"))
ONE_POSITION_PER_PAIR = _env_bool("ONE_POSITION_PER_PAIR", True)
MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "8"))  # caps total risk at MAX_OPEN_TRADES x RISK_PER_TRADE
# Max open trades betting the same way on one currency (e.g. EURMXN short + USDMXN short = 2x long MXN).
# Trades on opposite sides of a currency don't count against each other. 0 disables the limit.
MAX_TRADES_PER_CURRENCY = int(os.getenv("MAX_TRADES_PER_CURRENCY", "2"))
REQUIRE_TV_AGREEMENT = _env_bool("REQUIRE_TV_AGREEMENT", True)  # TV "buy"/"sell" must match model
MONITOR_INTERVAL_SEC = int(os.getenv("MONITOR_INTERVAL_SEC", "60"))
MAX_BAR_STALENESS_HOURS = float(os.getenv("MAX_BAR_STALENESS_HOURS", "3"))

# --------------------------------------------------------------------------- built-in scanner
# Replaces TradingView alerts: at every hourly bar close, score every pair and trade the strongest signals.
SCANNER_ENABLED = _env_bool("SCANNER_ENABLED", False)  # also run the scanner inside the FastAPI server
SCANNER_DELAY_SEC = int(os.getenv("SCANNER_DELAY_SEC", "90"))  # wait after the hour for Yahoo to publish the bar
SCANNER_WORKERS = int(os.getenv("SCANNER_WORKERS", "4"))  # parallel Yahoo downloads
SCANNER_REFRESH_NEWS = _env_bool("SCANNER_REFRESH_NEWS", True)  # scrape + FinBERT before each scan

# --------------------------------------------------------------------------- Discord alerts (optional)
# Create one in Discord: channel settings -> Integrations -> Webhooks. Keep it secret (lives in .env only).
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
DAILY_SUMMARY_HOUR_UTC = int(os.getenv("DAILY_SUMMARY_HOUR_UTC", "23"))  # 23:00 UTC = 08:00 in Tokyo/Seoul
