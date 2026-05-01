from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import DEFAULT_ALERT_CONFIG
from indicators.moving_averages import moving_average_crosses
from indicators.support_resistance import detect_support_resistance, is_breakdown, is_breakout
from screener.filters import has_volume_spike


@dataclass(frozen=True)
class Alert:
    symbol: str
    alert_type: str
    message: str
    severity: str


class AlertEngine:
    def __init__(self, config: dict[str, float] | None = None) -> None:
        self.config = {**DEFAULT_ALERT_CONFIG, **(config or {})}

    def evaluate(self, symbol: str, frame: pd.DataFrame) -> list[Alert]:
        if frame.empty or len(frame) < 2:
            return [Alert(symbol, "Data", "Not enough data to evaluate alerts.", "info")]
        alerts: list[Alert] = []
        latest = frame.iloc[-1]
        previous = frame.iloc[-2]
        close = float(latest["Close"])
        previous_close = float(previous["Close"])
        change_pct = (close / previous_close - 1) * 100 if previous_close else 0.0

        price_threshold = float(self.config["price_change_pct"])
        if abs(change_pct) >= price_threshold:
            direction = "up" if change_pct > 0 else "down"
            alerts.append(
                Alert(symbol, "Price change", f"Price moved {direction} {change_pct:.2f}% from the prior bar.", "warning")
            )

        rsi_low = float(self.config["rsi_low"])
        rsi_high = float(self.config["rsi_high"])
        if "RSI" in frame:
            rsi = float(latest["RSI"])
            prev_rsi = float(previous["RSI"])
            if prev_rsi >= rsi_low > rsi:
                alerts.append(Alert(symbol, "RSI", f"RSI crossed below {rsi_low:.0f}; current RSI is {rsi:.2f}.", "warning"))
            if prev_rsi <= rsi_high < rsi:
                alerts.append(Alert(symbol, "RSI", f"RSI crossed above {rsi_high:.0f}; current RSI is {rsi:.2f}.", "warning"))

        crosses = moving_average_crosses(frame)
        if crosses["bullish_cross"]:
            alerts.append(Alert(symbol, "MA cross", "MA50 crossed above MA200.", "info"))
        if crosses["bearish_cross"]:
            alerts.append(Alert(symbol, "MA cross", "MA50 crossed below MA200.", "warning"))

        levels = detect_support_resistance(frame.iloc[:-1])
        if is_breakout(frame, resistance=levels.resistance):
            alerts.append(Alert(symbol, "Breakout", "Close broke above detected resistance.", "info"))
        if is_breakdown(frame, support=levels.support):
            alerts.append(Alert(symbol, "Breakdown", "Close broke below detected support.", "warning"))

        if has_volume_spike(frame, multiple=float(self.config["volume_spike_multiple"])):
            alerts.append(Alert(symbol, "Volume", "Volume is materially above its recent average.", "info"))

        return alerts


def alerts_to_frame(alerts: list[Alert]) -> pd.DataFrame:
    return pd.DataFrame([alert.__dict__ for alert in alerts])
