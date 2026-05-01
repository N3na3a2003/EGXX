from __future__ import annotations

import pandas as pd


def simple_moving_average(close: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(close, errors="coerce").rolling(window=window, min_periods=max(2, window // 2)).mean()


def exponential_moving_average(close: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(close, errors="coerce").ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def add_moving_averages(frame: pd.DataFrame, windows: tuple[int, ...] = (20, 50, 100, 200)) -> pd.DataFrame:
    enriched = frame.copy()
    for window in windows:
        enriched[f"MA{window}"] = simple_moving_average(enriched["Close"], window)
    return enriched


def classify_trend(frame: pd.DataFrame) -> str:
    if frame.empty or "Close" not in frame:
        return "Unknown"
    latest = frame.iloc[-1]
    close = latest.get("Close")
    ma50 = latest.get("MA50")
    ma200 = latest.get("MA200")
    if pd.isna(close) or pd.isna(ma50) or pd.isna(ma200):
        return "Unknown"
    slope_window = frame["Close"].dropna().tail(30)
    slope = 0.0 if len(slope_window) < 2 else (slope_window.iloc[-1] / slope_window.iloc[0] - 1) * 100
    if close > ma50 > ma200 and slope > 2:
        return "Uptrend"
    if close < ma50 < ma200 and slope < -2:
        return "Downtrend"
    return "Sideways"


def moving_average_crosses(frame: pd.DataFrame, fast: str = "MA50", slow: str = "MA200") -> dict[str, bool]:
    if len(frame) < 2 or fast not in frame or slow not in frame:
        return {"bullish_cross": False, "bearish_cross": False}
    recent = frame[[fast, slow]].dropna().tail(2)
    if len(recent) < 2:
        return {"bullish_cross": False, "bearish_cross": False}
    previous, current = recent.iloc[0], recent.iloc[1]
    return {
        "bullish_cross": bool(previous[fast] <= previous[slow] and current[fast] > current[slow]),
        "bearish_cross": bool(previous[fast] >= previous[slow] and current[fast] < current[slow]),
    }
