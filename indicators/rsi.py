from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    prices = pd.to_numeric(close, errors="coerce")
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0).clip(0, 100)


def add_rsi(frame: pd.DataFrame, period: int = 14, column_name: str = "RSI") -> pd.DataFrame:
    enriched = frame.copy()
    enriched[column_name] = calculate_rsi(enriched["Close"], period=period)
    return enriched
