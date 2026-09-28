"""News scraping + FinBERT sentiment scoring + per-currency hourly sentiment series.

CLI:
  python -m src.sentiment scrape                 # one scrape + score pass
  python -m src.sentiment scrape --loop 900      # keep refreshing every 15 min (run alongside the server)
  python -m src.sentiment import-csv news.csv    # backfill historical headlines for training
                                                 # (columns: published_at, headline[, source, url])
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from typing import Optional
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import db
from .config import (FINBERT_MODEL, NEWS_HEADLINE_SELECTOR, NEWS_REFRESH_SEC, NEWS_URLS,
                     NEWS_USER_AGENT, SENTIMENT_HALFLIFE_HOURS)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- currency tagging
CURRENCY_KEYWORDS: dict[str, list[str]] = {
    "USD": [r"u\.?s\.? dollar", r"greenback", r"\bfed\b", r"\bfomc\b", r"\bpowell\b", r"non-?farm",
            r"\bnfp\b", r"\bdxy\b", r"\bu\.?s\.? (cpi|inflation|economy|jobs|gdp|yields?|treasur)"],
    "EUR": [r"\beuro\b", r"\becb\b", r"\blagarde\b", r"euro ?zone", r"euro area", r"\bgerman"],
    "GBP": [r"\bsterling\b", r"\bpound\b", r"\bcable\b", r"\bboe\b", r"bank of england", r"\buk\b", r"\bbrit"],
    "JPY": [r"\byen\b", r"\bboj\b", r"bank of japan", r"\bueda\b", r"\bjapan"],
    "CHF": [r"\bfranc\b", r"\bsnb\b", r"\bswiss\b"],
    "AUD": [r"\baussie\b", r"australian dollar", r"\brba\b", r"\baustralia"],
    "CAD": [r"\bloonie\b", r"canadian dollar", r"bank of canada", r"\bcanad", r"\bcrude\b", r"oil price"],
    "NZD": [r"\bkiwi\b", r"new zealand", r"\brbnz\b"],
    "SEK": [r"\bkrona\b", r"riksbank", r"\bswed"],
    "NOK": [r"norges", r"\bnorw"],
    "DKK": [r"\bdanish\b", r"\bdenmark\b"],
    "PLN": [r"\bzloty\b", r"\bpoland\b", r"\bpolish\b"],
    "HUF": [r"\bforint\b", r"\bhungar"],
    "CZK": [r"\bkoruna\b", r"\bczech\b"],
    "TRY": [r"\blira\b", r"\bturk", r"\bcbrt\b"],
    "ZAR": [r"\brand\b", r"south africa", r"\bsarb\b"],
    "MXN": [r"\bmexic", r"\bbanxico\b"],
    "SGD": [r"\bsingapore\b"],
    "HKD": [r"hong kong", r"\bhkma\b"],
    "CNH": [r"\byuan\b", r"\brenminbi\b", r"\bpboc\b", r"\bchina\b", r"\bchinese\b"],
    "INR": [r"\brupee\b", r"\brbi\b", r"\bindia"],
    "THB": [r"\bbaht\b", r"\bthai"],
    "ILS": [r"\bshekel\b", r"\bisrael"],
    "BRL": [r"\bbrazil"],
}
CURRENCIES = set(CURRENCY_KEYWORDS)
_KW_RE = {c: re.compile("|".join(p), re.I) for c, p in CURRENCY_KEYWORDS.items()}
_PAIR_RE = re.compile(r"\b([A-Z]{3})\s?/?\s?([A-Z]{3})\b")
_CODE_RE = re.compile(r"\b([A-Z]{3})\b")
# case-sensitive so the pronoun "us" doesn't match
_USD_CASED_RE = re.compile(r"\bU\.?S\.?\b|\bUSA?\b|\bTrump\b|\bWhite House\b|\bBessent\b")


def tag_currencies(text: str) -> list[str]:
    found = {c for c, rx in _KW_RE.items() if rx.search(text)}
    if _USD_CASED_RE.search(text):
        found.add("USD")
    for a, b in _PAIR_RE.findall(text):  # "EUR/USD", "EURUSD"
        if a in CURRENCIES and b in CURRENCIES:
            found.update((a, b))
    found.update(c for c in _CODE_RE.findall(text) if c in CURRENCIES)
    return sorted(found)


# --------------------------------------------------------------------------- scraping
_robots: dict[str, Optional[robotparser.RobotFileParser]] = {}


def allowed_by_robots(url: str) -> bool:
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    if base not in _robots:
        rp = robotparser.RobotFileParser(base + "/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        _robots[base] = rp
    rp = _robots[base]
    return True if rp is None else rp.can_fetch(NEWS_USER_AGENT, url)


def _to_utc(value) -> Optional[pd.Timestamp]:
    try:
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return None


def _parse_feed(content: bytes, source: str) -> list[dict]:
    import feedparser

    out = []
    for e in feedparser.parse(content).entries:
        ts = None
        if getattr(e, "published_parsed", None):
            ts = pd.Timestamp(time.mktime(e.published_parsed), unit="s", tz="UTC")
        out.append({"headline": e.get("title", "").strip(), "url": e.get("link"),
                    "published_at": ts, "source": source})
    return out


def _parse_html(html: str, page_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for el in soup.select(NEWS_HEADLINE_SELECTOR):
        text = el.get_text(" ", strip=True)
        if not 25 <= len(text) <= 300:
            continue
        a = el if el.name == "a" else (el.find("a") or el.find_parent("a"))
        container = el.find_parent(["article", "li"]) or el.parent
        t = container.find("time") if container else None
        ts = _to_utc(t.get("datetime") or t.get_text(strip=True)) if t else None
        out.append({
            "headline": text,
            "url": urljoin(page_url, a["href"]) if a is not None and a.get("href") else page_url,
            "published_at": ts or now,  # fall back to scrape time for live use
            "source": urlparse(page_url).netloc,
        })
    return out


def scrape_headlines(urls: list[str] = NEWS_URLS) -> list[dict]:
    items: list[dict] = []
    for url in urls:
        if not allowed_by_robots(url):
            log.warning("robots.txt disallows %s - skipping", url)
            continue
        try:
            resp = requests.get(url, headers={"User-Agent": NEWS_USER_AGENT}, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("fetch failed for %s: %s", url, e)
            continue
        ctype = resp.headers.get("content-type", "")
        is_feed = "xml" in ctype or "rss" in ctype or resp.text.lstrip().startswith("<?xml")
        parsed = _parse_feed(resp.content, url) if is_feed else _parse_html(resp.text, url)
        log.info("%s -> %d headlines", url, len(parsed))
        items.extend(parsed)
    now = pd.Timestamp.now(tz="UTC")
    unique = {}
    for i in items:
        if i["headline"]:
            i["published_at"] = i["published_at"] or now
            unique.setdefault(i["headline"], i)
    return list(unique.values())


# --------------------------------------------------------------------------- FinBERT
class FinBertScorer:
    def __init__(self, model_name: str = FINBERT_MODEL):
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("text-classification", model=model_name, tokenizer=model_name,
                             top_k=None, truncation=True, max_length=128, device=device)

    def score(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for out in self.pipe(texts, batch_size=batch_size):
            p = {d["label"].lower(): float(d["score"]) for d in out}
            pos, neg, neu = p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)
            results.append({"positive": pos, "negative": neg, "neutral": neu, "score": pos - neg,
                            "label": max(p, key=p.get)})
        return results


_scorer: Optional[FinBertScorer] = None


def get_scorer() -> FinBertScorer:
    global _scorer
    if _scorer is None:
        log.info("loading %s ...", FINBERT_MODEL)
        _scorer = FinBertScorer()
    return _scorer


def ingest(items: list[dict], batch: int = 256) -> int:
    """Score unseen headlines with FinBERT and store them."""
    new = db.filter_new_headlines(items)
    if not new:
        return 0
    scorer, inserted = get_scorer(), 0
    for i in range(0, len(new), batch):
        chunk = new[i:i + batch]
        scores = scorer.score([x["headline"] for x in chunk])
        rows = [{
            **s,
            "headline": x["headline"],
            "published_at": pd.Timestamp(x["published_at"]).isoformat(),
            "source": x.get("source"),
            "url": x.get("url"),
            "currencies": ",".join(tag_currencies(x["headline"])),
        } for x, s in zip(chunk, scores)]
        inserted += db.insert_news(rows)
        log.info("scored %d/%d headlines", min(i + batch, len(new)), len(new))
    return inserted


# --------------------------------------------------------------------------- hourly currency sentiment
def _decayed_sum(x: pd.Series, halflife: float) -> pd.Series:
    """S_t = x_t + d * S_{t-1}, with d = 0.5 ** (1 / halflife). Exact, vectorised via adjusted EWM."""
    d = 0.5 ** (1 / halflife)
    n = np.arange(1, len(x) + 1)
    return x.ewm(alpha=1 - d, adjust=True).mean() * (1 - d ** n) / (1 - d)


def hourly_currency_sentiment(
    since: Optional[pd.Timestamp] = None,
    until: Optional[pd.Timestamp] = None,
    halflife: float = SENTIMENT_HALFLIFE_HOURS,
) -> pd.DataFrame:
    """Wide frame indexed by `known_at` (UTC, hourly) with `<CCY>_sent` and `<CCY>_cnt24` columns.

    `<CCY>_sent` = decayed sum of FinBERT scores / (decayed headline count + 1). The +1 prior makes the
    signal fade back to 0 during quiet periods instead of freezing at the last value.
    Values aggregated over [h, h+1) are stamped at h+1 - the moment they are actually known.
    """
    warmup = pd.Timedelta(hours=halflife * 8)
    news = db.load_news(None if since is None else since - warmup)
    if news.empty:
        return pd.DataFrame()
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, format="ISO8601")
    news["currency"] = news["currencies"].fillna("").str.split(",")
    ex = news.explode("currency")
    ex = ex[ex["currency"].isin(CURRENCIES)]
    if ex.empty:
        return pd.DataFrame()
    ex["hour"] = ex["published_at"].dt.floor("h")
    grouped = ex.groupby(["currency", "hour"])["score"].agg(["sum", "count"])

    until = (until or pd.Timestamp.now(tz="UTC")).floor("h")
    idx = pd.date_range(ex["hour"].min(), max(until, ex["hour"].max()), freq="h")
    cols = {}
    for ccy, sub in grouped.groupby(level="currency"):
        sub = sub.droplevel("currency").reindex(idx, fill_value=0)
        s, c = _decayed_sum(sub["sum"], halflife), _decayed_sum(sub["count"].astype(float), halflife)
        cols[f"{ccy}_sent"] = s / (c + 1.0)
        cols[f"{ccy}_cnt24"] = sub["count"].rolling(24, min_periods=1).sum()
    out = pd.DataFrame(cols, index=idx)
    out.index = out.index + pd.Timedelta(hours=1)
    out.index.name = "known_at"
    return out


# --------------------------------------------------------------------------- CLI
def import_csv(path: str) -> int:
    df = pd.read_csv(path)
    missing = {"published_at", "headline"} - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["source"] = df.get("source", "csv")
    items = df.dropna(subset=["headline", "published_at"]).to_dict("records")
    return ingest(items)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scrape")
    s.add_argument("--loop", type=int, nargs="?", const=NEWS_REFRESH_SEC, default=0,
                   help="repeat every N seconds")
    c = sub.add_parser("import-csv")
    c.add_argument("path")
    args = ap.parse_args()

    db.init_db()
    if args.cmd == "import-csv":
        log.info("inserted %d headlines", import_csv(args.path))
        return
    while True:
        try:
            log.info("inserted %d new headlines", ingest(scrape_headlines()))
        except Exception:
            log.exception("scrape cycle failed")
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
