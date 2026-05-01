from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Levels:
    support: float | None
    resistance: float | None
    support_levels: list[float]
    resistance_levels: list[float]


def detect_support_resistance(
    frame: pd.DataFrame,
    lookback: int = 120,
    pivot_window: int = 3,
    cluster_tolerance_pct: float = 1.5,
) -> Levels:
    """Detect recent support/resistance by clustering local pivots."""
    if frame.empty or len(frame) < pivot_window * 2 + 3:
        return Levels(None, None, [], [])
    recent = frame.tail(lookback).copy()
    highs = pd.to_numeric(recent["High"], errors="coerce")
    lows = pd.to_numeric(recent["Low"], errors="coerce")
    close = float(pd.to_numeric(recent["Close"], errors="coerce").dropna().iloc[-1])

    resistance_pivots: list[float] = []
    support_pivots: list[float] = []
    for idx in range(pivot_window, len(recent) - pivot_window):
        high_slice = highs.iloc[idx - pivot_window : idx + pivot_window + 1]
        low_slice = lows.iloc[idx - pivot_window : idx + pivot_window + 1]
        current_high = highs.iloc[idx]
        current_low = lows.iloc[idx]
        if np.isfinite(current_high) and current_high == high_slice.max():
            resistance_pivots.append(float(current_high))
        if np.isfinite(current_low) and current_low == low_slice.min():
            support_pivots.append(float(current_low))

    supports = _cluster_levels(support_pivots, cluster_tolerance_pct)
    resistances = _cluster_levels(resistance_pivots, cluster_tolerance_pct)
    support = max([level for level in supports if level <= close], default=None)
    resistance = min([level for level in resistances if level >= close], default=None)

    if support is None and len(recent) >= 20:
        support = float(lows.tail(20).min())
    if resistance is None and len(recent) >= 20:
        resistance = float(highs.tail(20).max())

    return Levels(support, resistance, supports, resistances)


def add_support_resistance(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    levels = detect_support_resistance(enriched)
    enriched["SUPPORT"] = levels.support
    enriched["RESISTANCE"] = levels.resistance
    return enriched


def is_breakout(frame: pd.DataFrame, resistance: float | None = None, tolerance_pct: float = 0.5) -> bool:
    if frame.empty or len(frame) < 2:
        return False
    resistance = resistance if resistance is not None else detect_support_resistance(frame.iloc[:-1]).resistance
    if resistance is None or resistance <= 0:
        return False
    previous_close = float(frame["Close"].iloc[-2])
    current_close = float(frame["Close"].iloc[-1])
    trigger = resistance * (1 + tolerance_pct / 100)
    return previous_close <= resistance and current_close > trigger


def is_breakdown(frame: pd.DataFrame, support: float | None = None, tolerance_pct: float = 0.5) -> bool:
    if frame.empty or len(frame) < 2:
        return False
    support = support if support is not None else detect_support_resistance(frame.iloc[:-1]).support
    if support is None or support <= 0:
        return False
    previous_close = float(frame["Close"].iloc[-2])
    current_close = float(frame["Close"].iloc[-1])
    trigger = support * (1 - tolerance_pct / 100)
    return previous_close >= support and current_close < trigger


def _cluster_levels(levels: list[float], tolerance_pct: float) -> list[float]:
    clean = sorted(level for level in levels if np.isfinite(level) and level > 0)
    if not clean:
        return []
    clusters: list[list[float]] = [[clean[0]]]
    for level in clean[1:]:
        anchor = float(np.mean(clusters[-1]))
        if abs(level - anchor) / anchor * 100 <= tolerance_pct:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    return [round(float(np.mean(cluster)), 4) for cluster in clusters if len(cluster) >= 1]
