from __future__ import annotations

import pandas as pd
import streamlit as st

from data.data_fetcher import DataFetcher, FetchResult
from data.symbol_mapper import default_symbols, parse_symbol_list
from screener.scanner import MarketScanner


@st.cache_resource
def get_fetcher() -> DataFetcher:
    return DataFetcher()


@st.cache_data(ttl=600, show_spinner=False)
def cached_ohlcv_payload(symbol: str, timeframe: str) -> dict[str, object]:
    return get_fetcher().fetch_ohlcv(symbol, timeframe=timeframe).to_cache_payload()


def cached_ohlcv(symbol: str, timeframe: str) -> FetchResult:
    return FetchResult.from_cache_payload(cached_ohlcv_payload(symbol, timeframe))


@st.cache_data(ttl=600, show_spinner=False)
def cached_scan(symbols: tuple[str, ...], timeframe: str, min_score: float) -> pd.DataFrame:
    scanner = MarketScanner(get_fetcher())
    return scanner.scan(list(symbols), timeframe=timeframe, min_score=min_score)


def symbol_multiline_default() -> str:
    return "\n".join(default_symbols())


def parse_symbols_for_ui(raw: str) -> list[str]:
    symbols = parse_symbol_list(raw)
    return symbols or default_symbols()


def format_egp(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.2f} EGP"


def show_data_warning(message: str | None) -> None:
    if message:
        st.warning(message)
