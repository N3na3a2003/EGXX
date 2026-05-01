from __future__ import annotations

import pandas as pd

from indicators.moving_averages import classify_trend
from indicators.support_resistance import detect_support_resistance, is_breakout
from screener.filters import evaluate_conditions, has_volume_spike


def score_stock(frame: pd.DataFrame) -> dict[str, float | str | dict[str, bool] | list[str]]:
    if frame.empty or len(frame) < 30:
        return {
            "score": 0.0,
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "risk_score": 0.0,
            "volume_score": 0.0,
            "trend": "Unknown",
            "rating": "Weak",
            "research_stance": "Insufficient data",
            "conditions": {},
            "reasons": [],
            "risks": ["Not enough price history to score the setup."],
        }

    latest = frame.iloc[-1]
    close = _safe_float(latest.get("Close"))
    rsi = _safe_float(latest.get("RSI"), default=50.0)
    ma50 = _safe_float(latest.get("MA50"))
    ma200 = _safe_float(latest.get("MA200"))
    macd = _safe_float(latest.get("MACD"), default=0.0)
    macd_signal = _safe_float(latest.get("MACD_SIGNAL"), default=0.0)
    atr = _safe_float(latest.get("ATR"))
    trend = classify_trend(frame)
    levels = detect_support_resistance(frame)
    conditions = evaluate_conditions(frame)
    breakout = is_breakout(frame, resistance=levels.resistance)
    volume_spike = has_volume_spike(frame)
    reasons: list[str] = []
    risks: list[str] = []

    trend_score = 0.0
    if trend == "Uptrend":
        trend_score += 24
        reasons.append("Uptrend structure.")
    elif trend == "Sideways":
        trend_score += 12
        risks.append("Sideways trend.")
    elif trend == "Downtrend":
        risks.append("Downtrend structure.")
    if ma50 and ma200 and close:
        if close > ma50:
            trend_score += 7
            reasons.append("Price above MA50.")
        else:
            risks.append("Price below MA50.")
        if close > ma200:
            trend_score += 7
            reasons.append("Price above MA200.")
        else:
            risks.append("Price below MA200.")
        if ma50 > ma200:
            trend_score += 4
            reasons.append("MA50 above MA200.")
    trend_score = min(trend_score, 42)

    if 45 <= rsi <= 65:
        momentum_score = 18
        reasons.append(f"RSI balanced at {rsi:.1f}.")
    elif 30 <= rsi < 45 or 65 < rsi <= 70:
        momentum_score = 12
        if rsi < 45:
            reasons.append(f"RSI recovering at {rsi:.1f}.")
        else:
            risks.append(f"RSI elevated at {rsi:.1f}.")
    elif rsi < 30:
        momentum_score = 10
        risks.append(f"RSI oversold at {rsi:.1f}; needs reversal confirmation.")
    else:
        momentum_score = 3
        risks.append(f"RSI overbought at {rsi:.1f}.")
    if macd > macd_signal:
        momentum_score += 8
        reasons.append("MACD positive.")
    else:
        risks.append("MACD not confirmed.")
    if breakout:
        momentum_score += 8
        reasons.append("Breakout above resistance.")
    momentum_score = min(momentum_score, 34)

    risk_score = 8.0
    if levels.support and close:
        distance_to_support = (close / levels.support - 1) * 100
        if 0 <= distance_to_support <= 6:
            risk_score += 10
            reasons.append("Close to support; risk can be defined.")
        elif distance_to_support > 15:
            risk_score -= 4
            risks.append("Price far from support.")
    if levels.resistance and close:
        distance_to_resistance = (levels.resistance / close - 1) * 100
        if 0 <= distance_to_resistance <= 3 and not breakout:
            risk_score += 3
            reasons.append("Near resistance; watch for confirmation.")
        elif distance_to_resistance < 0:
            risk_score += 5
    if atr and close:
        atr_pct = atr / close * 100
        if atr_pct > 6:
            risk_score -= 5
            risks.append(f"High ATR volatility at {atr_pct:.1f}%.")
        elif atr_pct <= 3:
            risk_score += 3
    risk_score = max(0.0, min(16.0, risk_score))

    volume_score = 8.0
    if volume_spike:
        volume_score = 16.0
        reasons.append("Volume spike confirms attention.")
    if "Volume" in frame and len(frame) > 20:
        latest_volume = _safe_float(latest.get("Volume"), default=0.0)
        average_volume = float(pd.to_numeric(frame["Volume"], errors="coerce").tail(20).mean())
        if average_volume > 0 and latest_volume > average_volume:
            volume_score = max(volume_score, 12.0)
            reasons.append("Volume above 20-period average.")
        elif average_volume > 0:
            risks.append("Volume confirmation is limited.")

    score = min(100.0, trend_score + momentum_score + risk_score + volume_score)
    rating = "Strong" if score >= 75 else "Moderate" if score >= 55 else "Weak"
    return {
        "score": round(score, 2),
        "trend_score": round(trend_score, 2),
        "momentum_score": round(momentum_score, 2),
        "risk_score": round(risk_score, 2),
        "volume_score": round(volume_score, 2),
        "trend": trend,
        "rating": rating,
        "research_stance": _stance_from_score(score),
        "conditions": conditions,
        "reasons": _top_items(reasons, 4),
        "risks": _top_items(risks, 4),
    }


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _stance_from_score(score: float) -> str:
    if score >= 80:
        return "Top research candidate"
    if score >= 65:
        return "Good setup, needs plan"
    if score >= 45:
        return "Watchlist / wait"
    return "Weak setup"


def _top_items(items: list[str], limit: int) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) >= limit:
            break
    return result
