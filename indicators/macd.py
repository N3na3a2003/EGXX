from __future__ import annotations

import pandas as pd


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    prices = pd.to_numeric(close, errors="coerce")
    ema_fast = prices.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    histogram = macd - signal
    return pd.DataFrame({"MACD": macd, "MACD_SIGNAL": signal, "MACD_HIST": histogram}, index=close.index)


def add_macd(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    macd = calculate_macd(enriched["Close"])
    return enriched.join(macd)
