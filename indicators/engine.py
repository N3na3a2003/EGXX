from __future__ import annotations

import pandas as pd

from indicators.atr import add_atr
from indicators.bollinger import add_bollinger_bands
from indicators.macd import add_macd
from indicators.moving_averages import add_moving_averages, classify_trend
from indicators.rsi import add_rsi
from indicators.support_resistance import add_support_resistance, detect_support_resistance


def enrich_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    enriched = frame.copy()
    enriched = add_rsi(enriched)
    enriched = add_macd(enriched)
    enriched = add_moving_averages(enriched)
    enriched = add_bollinger_bands(enriched)
    enriched = add_atr(enriched)
    enriched = add_support_resistance(enriched)
    enriched["TREND"] = classify_trend(enriched)
    return enriched


def latest_indicator_snapshot(frame: pd.DataFrame) -> dict[str, float | str | None]:
    if frame.empty:
        return {}
    enriched = enrich_indicators(frame)
    latest = enriched.iloc[-1]
    levels = detect_support_resistance(frame)
    trend = classify_trend(enriched)
    fields = ["Close", "Volume", "RSI", "MACD", "MACD_SIGNAL", "MA20", "MA50", "MA100", "MA200", "ATR"]
    snapshot: dict[str, float | str | None] = {"Trend": trend, "Support": levels.support, "Resistance": levels.resistance}
    for field in fields:
        value = latest.get(field)
        snapshot[field] = None if pd.isna(value) else float(value)
    return snapshot
