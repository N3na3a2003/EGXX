from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from config import TIMEFRAMES
from data.cache_manager import FileCache
from data.symbol_mapper import company_name, normalize_symbol, to_yahoo_symbol


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(frozen=True)
class FetchResult:
    symbol: str
    yahoo_symbol: str
    data: pd.DataFrame
    source: str
    warning: str | None = None

    def to_cache_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "yahoo_symbol": self.yahoo_symbol,
            "data": self.data,
            "source": self.source,
            "warning": self.warning,
        }

    @classmethod
    def from_cache_payload(cls, payload: dict[str, object]) -> "FetchResult":
        data = payload.get("data")
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(columns=OHLCV_COLUMNS)
        return cls(
            symbol=str(payload.get("symbol", "")),
            yahoo_symbol=str(payload.get("yahoo_symbol", "")),
            data=data,
            source=str(payload.get("source", "cache")),
            warning=payload.get("warning") if isinstance(payload.get("warning"), str) else None,
        )


class DataFetcher:
    """Best-effort EGX OHLCV fetcher with yfinance primary and free fallbacks."""

    def __init__(self, cache: FileCache | None = None, timeout: int = 12) -> None:
        self.cache = cache or FileCache()
        self.timeout = timeout

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1y", use_cache: bool = True) -> FetchResult:
        bare = normalize_symbol(symbol)
        yahoo_symbol = to_yahoo_symbol(bare)
        if not bare:
            return FetchResult("", "", pd.DataFrame(columns=OHLCV_COLUMNS), "none", "Empty symbol.")
        if timeframe not in TIMEFRAMES:
            timeframe = "1y"

        cache_key = f"{yahoo_symbol}:{timeframe}"
        if use_cache:
            cached = self.cache.get("ohlcv", cache_key)
            if isinstance(cached, dict):
                return FetchResult.from_cache_payload(cached)

        spec = TIMEFRAMES[timeframe]
        warning = None
        source = "yfinance"
        data = self._download_yfinance(yahoo_symbol, spec["period"], spec["interval"])
        if data.empty:
            data = self._download_yahoo_chart(yahoo_symbol, spec["period"], spec["interval"])
            source = "yahoo_chart"
        if data.empty:
            warning = f"No OHLCV data returned for {yahoo_symbol}. The symbol may be missing or temporarily unavailable."
            source = "none"

        result = FetchResult(bare, yahoo_symbol, data, source, warning)
        if use_cache and not data.empty:
            self.cache.set("ohlcv", cache_key, result.to_cache_payload())
        return result

    def fetch_many(self, symbols: Iterable[str], timeframe: str = "1y") -> dict[str, FetchResult]:
        results = {}
        for symbol in symbols:
            result = self.fetch_ohlcv(symbol, timeframe=timeframe)
            results[result.symbol or normalize_symbol(symbol)] = result
        return results

    def latest_price(self, symbol: str) -> tuple[float | None, str | None]:
        result = self.fetch_ohlcv(symbol, timeframe="1d")
        if not result.data.empty:
            price = result.data["Close"].dropna().iloc[-1]
            return float(price), None
        scraped = self._scrape_yahoo_price(to_yahoo_symbol(symbol))
        if scraped is not None:
            return scraped, "Price from Yahoo page snapshot; OHLCV history unavailable."
        return None, result.warning

    def symbol_profile(self, symbol: str) -> dict[str, str]:
        bare = normalize_symbol(symbol)
        return {
            "symbol": bare,
            "yahoo_symbol": to_yahoo_symbol(bare),
            "name": company_name(bare),
        }

    def _download_yfinance(self, yahoo_symbol: str, period: str, interval: str) -> pd.DataFrame:
        try:
            frame = yf.download(
                yahoo_symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return self._clean_ohlcv(frame)

    def _download_yahoo_chart(self, yahoo_symbol: str, period: str, interval: str) -> pd.DataFrame:
        seconds = self._period_to_seconds(period)
        end = int(dt.datetime.now(dt.UTC).timestamp())
        start = end - seconds
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        params = {
            "period1": start,
            "period2": end,
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true",
        }
        try:
            response = requests.get(url, params=params, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            payload = response.json()
            chart = payload.get("chart", {}).get("result", [])
            if not chart:
                return pd.DataFrame(columns=OHLCV_COLUMNS)
            result = chart[0]
            timestamps = result.get("timestamp") or []
            quote = (result.get("indicators", {}).get("quote") or [{}])[0]
            if not timestamps or not quote:
                return pd.DataFrame(columns=OHLCV_COLUMNS)
            frame = pd.DataFrame(
                {
                    "Open": quote.get("open", []),
                    "High": quote.get("high", []),
                    "Low": quote.get("low", []),
                    "Close": quote.get("close", []),
                    "Volume": quote.get("volume", []),
                },
                index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            )
        except Exception:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return self._clean_ohlcv(frame)

    def _scrape_yahoo_price(self, yahoo_symbol: str) -> float | None:
        url = f"https://finance.yahoo.com/quote/{yahoo_symbol}"
        try:
            response = requests.get(url, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            node = soup.select_one('fin-streamer[data-field="regularMarketPrice"]')
            if not node:
                return None
            text = node.get_text(strip=True).replace(",", "")
            return float(text)
        except Exception:
            return None

    @staticmethod
    def _clean_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        available = [column for column in OHLCV_COLUMNS if column in frame.columns]
        if len(available) < 4:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        cleaned = frame.copy()
        if "Volume" not in cleaned.columns:
            cleaned["Volume"] = 0
        cleaned = cleaned[OHLCV_COLUMNS]
        cleaned.index = pd.to_datetime(cleaned.index)
        cleaned = cleaned.sort_index()
        for column in OHLCV_COLUMNS:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"])
        cleaned["Volume"] = cleaned["Volume"].fillna(0)
        return cleaned

    @staticmethod
    def _period_to_seconds(period: str) -> int:
        number = int("".join(ch for ch in period if ch.isdigit()) or "1")
        unit = "".join(ch for ch in period if ch.isalpha()) or "y"
        if unit == "d":
            return number * 24 * 60 * 60
        if unit == "mo":
            return number * 30 * 24 * 60 * 60
        if unit == "y":
            return number * 365 * 24 * 60 * 60
        return 365 * 24 * 60 * 60
