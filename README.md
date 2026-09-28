# Forex ML Paper Trader

TradingView alert **or the built-in hourly scanner** → live features (technicals + FinBERT news sentiment) → XGBoost
4-hour direction model → simulated trade in SQLite → background monitor closes trades on SL/TP.

**Paper trading only. Nothing here places real orders.**

## Layout

```
forex-ml-paper-trader/
├── .env.example                  # copy to .env, set WEBHOOK_SECRET
├── requirements.txt
├── tradingview_alert_template.json
├── data/
│   ├── raw/<PAIR>.parquet        # cached 1h OHLC from Yahoo
│   ├── features.parquet          # labelled training set
│   ├── universe.json             # the N most volatile pairs + their ATR%
│   └── paper_trading.db          # SQLite: news, webhook_alerts, trades
├── models/
│   └── xgb_direction.joblib      # model + feature list + threshold + CV report
└── src/
    ├── config.py                 # every tunable, overridable via .env
    ├── market_data.py            # yfinance: history, 1m bars, live price, quote→USD
    ├── features.py               # RSI/ATR/MACD + sentiment join + labels (shared by train & live)
    ├── sentiment.py              # scraper, FinBERT scoring, per-currency hourly sentiment
    ├── data_collection.py        # download → rank by volatility → build training set
    ├── train.py                  # purged walk-forward CV + final fit + joblib save
    ├── paper_engine.py           # inference, sizing, SL/TP, trade monitor
    ├── scanner.py                # free built-in signal source (replaces TradingView alerts)
    ├── db.py                     # SQLite schema + queries
    └── webhook_server.py         # FastAPI app
```

## Setup and run order

```bash
./setup.sh                               # venv + deps + .env  (add --nlp for FinBERT/torch)
                                         # picks Python 3.9-3.13; pre-releases like 3.15 lack package wheels
source .venv/bin/activate                # after this, `python` works (macOS itself only has `python3`)

# 1. News → FinBERT → SQLite. Backfill history first; the scraper only gets recent headlines.
python -m src.sentiment import-csv my_historical_headlines.csv   # published_at,headline[,source,url]
python -m src.sentiment scrape

# 2. Prices + features + labels
python -m src.data_collection

# 3. Train (prints per-fold walk-forward metrics and feature importance)
python -m src.train

# 4a. FREE: built-in scanner - scores all pairs at every hourly close, trades the strongest,
#     checks SL/TP every minute, refreshes news before each scan. No TradingView needed.
python -m src.scanner --once --no-news   # try one scan right now
python -m src.scanner                    # run continuously

# 4b. OR the webhook server (TradingView paid plan). Add SCANNER_ENABLED=true to run the scanner inside it too.
uvicorn src.webhook_server:app --host 0.0.0.0 --port 8000
python -m src.sentiment scrape --loop 900
```

TradingView only posts to ports **80/443** over public HTTPS. Put the server behind a tunnel or reverse proxy
(`cloudflared tunnel --url http://localhost:8000` or `ngrok http 8000`), then set `TRUST_PROXY_HEADERS=true`
and `ENFORCE_IP_ALLOWLIST=true` so only TradingView's IPs are accepted.

Webhook URL: `https://<your-host>/webhook/tradingview`

## Built-in scanner

TradingView webhooks need a paid plan. The model doesn't need TradingView at all: it builds every
feature from Yahoo prices and news. TradingView only decides *when* to evaluate. The scanner does
that itself:

1. At `HH:00 + SCANNER_DELAY_SEC` (default 90 s) during FX hours (Sunday 21:00 to Friday 22:00 UTC), it
   refreshes news, then scores every pair in `data/universe.json` in parallel.
2. In scheduled runs it only uses the bar that just closed. Pairs whose bar Yahoo hasn't published yet are
   retried once after 60 s, then skipped.
3. It ranks signals by confidence `|p_up − 0.5|` and opens trades from the strongest down, until
   `MAX_OPEN_TRADES` is reached (one position per pair).
4. Every decision, including no-trades with their probability, goes to the `webhook_alerts` table with an id
   like `scan:<bar close>:<PAIR>`. It's visible at `GET /alerts`, and running the scan twice for the same
   bar never double-trades.

| Command | What it does |
|---|---|
| `python -m src.scanner` | Scan every hour and monitor SL/TP every minute (standalone, no server needed) |
| `python -m src.scanner --once` | One scan now on the latest completed bars, printing a JSON summary |
| `--no-monitor` / `--no-news` | Skip the SL/TP monitor (when the server runs it) / skip the news refresh |
| `SCANNER_ENABLED=true uvicorn src.webhook_server:app` | Scanner + monitor + API in one process; `POST /scan` runs a scan now |

To watch results, run `uvicorn src.webhook_server:app` (no `WEBHOOK_SECRET` needed with `SCANNER_ENABLED=true`)
and open `http://localhost:8000/docs` for `/account`, `/trades` and `/alerts`.

## TradingView alert message

Paste this into the alert's **Message** box and turn on **Webhook URL**:

```json
{
  "secret": "PASTE_YOUR_WEBHOOK_SECRET_HERE",
  "ticker": "{{ticker}}",
  "exchange": "{{exchange}}",
  "timeframe": "{{interval}}",
  "price": {{close}},
  "time": "{{timenow}}",
  "signal": "buy",
  "triggers": {
    "rsi": {{plot_0}},
    "macd_hist": {{plot_1}},
    "condition": "rsi_cross_above_30"
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `secret` | yes | Must equal `WEBHOOK_SECRET`. TradingView can't send custom headers, so the secret goes in the body. |
| `ticker` | yes | `EURUSD`, `FX:EURUSD`, `OANDA:EUR_USD` are all accepted. |
| `timeframe` | yes | Logged only; the model always computes features on 1h bars. |
| `price` | no | Leave `{{close}}` **unquoted** (it becomes a number). Logged only; the fill uses the live Yahoo price so it matches the monitor. |
| `time` | recommended | `{{timenow}}` enables replay protection (alerts older than `MAX_ALERT_AGE_SEC` are rejected). |
| `signal` | no | `buy`/`sell`/`long`/`short`/`any`. With `REQUIRE_TV_AGREEMENT=true`, a trade opens only when your chart signal and the model agree. For strategy alerts use `"{{strategy.order.action}}"`. |
| `triggers` | no | Any flat key→value map (`{{plot_N}}` = the Nth plot of the indicator). Stored for analysis, not fed to the model, because there is no history of them to train on. |

After the placeholders are filled in, the body TradingView sends looks like this:

```json
{"secret":"…","ticker":"EURUSD","exchange":"FX","timeframe":"60","price":1.08412,"time":"2026-09-28T14:00:00Z","signal":"buy","triggers":{"rsi":31.4,"macd_hist":0.00012,"condition":"rsi_cross_above_30"}}
```

Local test:

```bash
curl -X POST localhost:8000/webhook/tradingview -H 'Content-Type: application/json' \
  -d '{"secret":"<your secret>","ticker":"FX:EURUSD","timeframe":"60","signal":"any"}'
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/webhook/tradingview` | Returns 202 right away; inference runs in the background (TradingView times out after ~3 s) |
| GET | `/alerts` | Every alert with the model decision (prob_up, features, reason for no-trade) |
| GET | `/trades?status=open\|closed` | Paper trades |
| GET | `/account` | Equity, realized P&L, win rate |
| POST | `/trades/check` | Force an SL/TP sweep now (it also runs every `MONITOR_INTERVAL_SEC`) |
| POST | `/scan` | Run the built-in scanner once now |
| GET | `/health` | Model metadata |

## How it works

- **Label:** `1` if `close[t+4h] > close[t]`, `0` if it's lower; exactly flat bars are dropped.
- **Features:** RSI(14), ATR% and ATR-normalised MACD/signal/histogram (scale-free, so one model covers every
  pair), returns over 1/4/24 bars, 24-bar volatility, distance from EMA50, hour/day of week, and
  sentiment: `base − quote` FinBERT score, each currency's own score, and the 24h headline count.
- **Sentiment series:** decayed sum of `P(pos) − P(neg)` ÷ (decayed headline count + 1), with a 12h half-life,
  so it fades back to 0 when there's no news. News from hour *h* is stamped at *h+1* and joined to bars
  as-of the bar **close**, so there's no look-ahead.
- **Walk-forward CV:** sklearn `TimeSeriesSplit` over shared timestamps (expanding window, `gap=4`), plus
  a purge of any training row whose label resolves inside the test window.
- **Sizing:** risks `RISK_PER_TRADE` of equity to a stop at `SL_ATR_MULT × ATR`, capped at `MAX_LEVERAGE`,
  rounded down to 1k units. TP is `TP_ATR_MULT × ATR`. Long if `p ≥ threshold`, short if `p ≤ 1 − threshold`.
- **Monitor:** replays every 1m bar since the last check. If a bar touches both SL and TP, the stop is
  assumed to fill first. Gaps fill at the bar open. `COST_BPS` per side is deducted.

## Caveats

- **Historical sentiment:** a scraper only sees current headlines. Without a historical headline archive
  (`import-csv`), the sentiment features are 0 for most of the training data and the model can't learn from them.
- **News sources:** `NEWS_URLS` defaults to three investingLive (formerly ForexLive) RSS feeds: news, forex
  and central bank. forexfactory.com/news is behind a Cloudflare bot challenge (HTTP 403 to scripts), and
  FXStreet's robots.txt disallows its RSS feeds, so neither is scraped.
  RSS feeds only hold the latest ~25–30 items, so keep `scrape --loop` running to build up history.
- **Yahoo data:** unofficial, can be delayed a few minutes, 1h history is capped at 730 days, and 1m bars
  only go back 7 days. If the server is down longer than that, open trades can't be fully replayed.
- **Yahoo reversal artifact:** on real Yahoo data (Dec 2023 to Sep 2026) walk-forward CV showed a 63% signal
  hit rate but only about 3–6 bps per trade, and most signals simply fade the previous hourly bar. That's
  typical of noise in free mid-price quotes and usually disappears after real spreads (exotics like
  GBPZAR or USDTRY cost tens of bps). The default `COST_BPS=1.0` is far too low for exotics.
- **Edge:** 4-hour FX direction is close to a coin flip. Judge the model on the walk-forward
  `signal_hit_rate` / `avg_signal_ret_bps` after costs, not on accuracy, and run it on paper for a while
  before trusting it.
