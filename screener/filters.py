from __future__ import annotations

import pandas as pd

from indicators.support_resistance import detect_support_resistance, is_breakout


def is_oversold(frame: pd.DataFrame, threshold: float = 30.0) -> bool:
    return _latest_value(frame, "RSI") < threshold


def is_overbought(frame: pd.DataFrame, threshold: float = 70.0) -> bool:
    return _latest_value(frame, "RSI") > threshold


def is_strong_trend(frame: pd.DataFrame) -> bool:
    latest = _latest_row(frame)
    if latest is None:
        return False
    close = latest.get("Close")
    ma50 = latest.get("MA50")
    ma200 = latest.get("MA200")
    if pd.isna(close) or pd.isna(ma50) or pd.isna(ma200):
        return False
    return bool(close > ma50 and close > ma200 and ma50 >= ma200 * 0.98)


def has_volume_spike(frame: pd.DataFrame, lookback: int = 20, multiple: float = 1.8) -> bool:
    if frame.empty or "Volume" not in frame or len(frame) < 5:
        return False
    volume = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0)
    latest = float(volume.iloc[-1])
    baseline = float(volume.iloc[-lookback - 1 : -1].mean()) if len(volume) > lookback else float(volume.iloc[:-1].mean())
    return baseline > 0 and latest >= baseline * multiple


def breakout_above_resistance(frame: pd.DataFrame) -> bool:
    levels = detect_support_resistance(frame.iloc[:-1] if len(frame) > 1 else frame)
    return is_breakout(frame, resistance=levels.resistance)


def evaluate_conditions(frame: pd.DataFrame) -> dict[str, bool]:
    return {
        "Oversold": is_oversold(frame),
        "Overbought": is_overbought(frame),
        "Strong trend": is_strong_trend(frame),
        "Breakout": breakout_above_resistance(frame),
        "Volume spike": has_volume_spike(frame),
    }


def _latest_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.iloc[-1]


def _latest_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    value = frame[column].iloc[-1]
    return float(value) if pd.notna(value) else float("nan")
