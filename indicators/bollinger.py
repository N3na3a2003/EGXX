from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    prices = pd.to_numeric(close, errors="coerce")
    middle = prices.rolling(window=window, min_periods=max(2, window // 2)).mean()
    std = prices.rolling(window=window, min_periods=max(2, window // 2)).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle.replace(0, np.nan)
    return pd.DataFrame(
        {"BB_MIDDLE": middle, "BB_UPPER": upper, "BB_LOWER": lower, "BB_WIDTH": width},
        index=close.index,
    )


def add_bollinger_bands(frame: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    enriched = frame.copy()
    bands = calculate_bollinger_bands(enriched["Close"], window=window, num_std=num_std)
    return enriched.join(bands)
