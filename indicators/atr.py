from __future__ import annotations

import pandas as pd


def calculate_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_atr(frame: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["ATR"] = calculate_atr(enriched, period=period)
    return enriched
