from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from indicators.engine import enrich_indicators
from indicators.moving_averages import classify_trend
from indicators.support_resistance import detect_support_resistance, is_breakout
from screener.filters import has_volume_spike


@dataclass(frozen=True)
class TradePlan:
    ticker: str
    entry_reason: str
    entry_price: float
    target_price: float
    stop_loss: float
    side: str = "long"


@dataclass(frozen=True)
class DecisionReview:
    rating: str
    score: float
    risk_reward: float | None
    explanation: list[str]


@dataclass(frozen=True)
class StockDecisionReport:
    ticker: str
    score: float
    rating: str
    research_stance: str
    trend: str
    reasons: list[str]
    risks: list[str]
    next_steps: list[str]
    metrics: dict[str, float | str | None]


class DecisionEngine:
    """Explainable trade-plan reviewer. It does not issue buy/sell instructions."""

    def evaluate_stock(self, ticker: str, frame: pd.DataFrame) -> StockDecisionReport:
        if frame.empty or len(frame) < 50:
            return StockDecisionReport(
                ticker=ticker,
                score=0.0,
                rating="Weak",
                research_stance="Insufficient data",
                trend="Unknown",
                reasons=[],
                risks=["Not enough reliable OHLCV history to evaluate the setup."],
                next_steps=["Try a longer timeframe or verify that the EGX symbol is available from the data source."],
                metrics={},
            )

        enriched = enrich_indicators(frame)
        latest = enriched.iloc[-1]
        close = _safe_float(latest.get("Close"), 0.0)
        rsi = _safe_float(latest.get("RSI"), 50.0)
        macd = _safe_float(latest.get("MACD"), 0.0)
        macd_signal = _safe_float(latest.get("MACD_SIGNAL"), 0.0)
        ma20 = _safe_float(latest.get("MA20"))
        ma50 = _safe_float(latest.get("MA50"))
        ma200 = _safe_float(latest.get("MA200"))
        atr = _safe_float(latest.get("ATR"))
        trend = classify_trend(enriched)
        levels = detect_support_resistance(enriched)
        volume_spike = has_volume_spike(enriched)
        breakout = is_breakout(enriched, resistance=levels.resistance)

        score = 0.0
        reasons: list[str] = []
        risks: list[str] = []
        next_steps: list[str] = []

        if trend == "Uptrend":
            score += 28
            reasons.append("Trend is positive: price is structurally above key moving averages.")
        elif trend == "Sideways":
            score += 14
            risks.append("Trend is sideways, so false breakouts and choppy movement are more likely.")
        elif trend == "Downtrend":
            risks.append("Trend is negative; long setups need stronger confirmation.")
        else:
            risks.append("Trend could not be classified with confidence.")

        if close and ma50 and close > ma50:
            score += 8
            reasons.append("Price is above MA50, showing medium-term strength.")
        elif ma50:
            risks.append("Price is below MA50, so momentum is not yet confirmed.")

        if close and ma200 and close > ma200:
            score += 8
            reasons.append("Price is above MA200, supporting the longer-term structure.")
        elif ma200:
            risks.append("Price is below MA200, which weakens the long-term setup.")

        if ma50 and ma200 and ma50 > ma200:
            score += 6
            reasons.append("MA50 is above MA200, a constructive trend filter.")

        if 40 <= rsi <= 62:
            score += 18
            reasons.append(f"RSI is healthy at {rsi:.1f}; momentum is present without obvious overheating.")
        elif 30 <= rsi < 40:
            score += 12
            reasons.append(f"RSI is recovering from a low zone at {rsi:.1f}.")
        elif 62 < rsi <= 70:
            score += 9
            risks.append(f"RSI is elevated at {rsi:.1f}; waiting for a pullback may improve risk/reward.")
        elif rsi < 30:
            score += 8
            risks.append(f"RSI is oversold at {rsi:.1f}; this is not enough alone without reversal confirmation.")
        else:
            risks.append(f"RSI is overbought at {rsi:.1f}; chase risk is high.")

        if macd > macd_signal:
            score += 10
            reasons.append("MACD is above its signal line, confirming momentum.")
        else:
            risks.append("MACD has not confirmed positive momentum.")

        if breakout:
            score += 10
            reasons.append("Price closed above detected resistance with a breakout condition.")
        elif levels.resistance and close:
            distance_to_resistance = (levels.resistance / close - 1) * 100
            if 0 <= distance_to_resistance <= 3:
                score += 5
                next_steps.append("Watch for a close above resistance with volume confirmation.")
            elif distance_to_resistance < 0:
                score += 6
                reasons.append("Price is trading above the latest detected resistance area.")

        if volume_spike:
            score += 8
            reasons.append("Recent volume is meaningfully above average, which improves confirmation.")
        else:
            risks.append("No strong volume confirmation yet.")

        if levels.support and close:
            distance_to_support = (close / levels.support - 1) * 100
            if 0 <= distance_to_support <= 6:
                score += 8
                reasons.append("Price is still close to support, so risk can be defined more cleanly.")
            elif distance_to_support > 15:
                risks.append("Price is far from detected support, so stop-loss distance may be wide.")

        if atr and close:
            atr_pct = atr / close * 100
            if atr_pct <= 3:
                score += 4
                reasons.append(f"ATR is moderate at {atr_pct:.1f}% of price.")
            elif atr_pct > 6:
                risks.append(f"ATR is high at {atr_pct:.1f}% of price; position sizing should be conservative.")

        if not next_steps:
            if score >= 75:
                next_steps.append("Treat as a high-priority research candidate and define entry, target, and stop before acting.")
            elif score >= 55:
                next_steps.append("Keep on watchlist and wait for cleaner confirmation or better risk/reward.")
            else:
                next_steps.append("Do not force a plan; wait until trend, momentum, or risk location improves.")

        score = round(max(0.0, min(100.0, score)), 2)
        rating = "Strong" if score >= 75 else "Moderate" if score >= 55 else "Weak"
        stance = _stance_from_score(score)
        metrics = {
            "Close": round(close, 4) if close else None,
            "RSI": round(rsi, 2),
            "MA20": _round_or_none(ma20),
            "MA50": _round_or_none(ma50),
            "MA200": _round_or_none(ma200),
            "ATR": _round_or_none(atr),
            "Support": _round_or_none(levels.support),
            "Resistance": _round_or_none(levels.resistance),
            "Volume Spike": "Yes" if volume_spike else "No",
            "Breakout": "Yes" if breakout else "No",
        }
        return StockDecisionReport(
            ticker=ticker,
            score=score,
            rating=rating,
            research_stance=stance,
            trend=trend,
            reasons=_top_items(reasons, 5),
            risks=_top_items(risks, 5),
            next_steps=_top_items(next_steps, 4),
            metrics=metrics,
        )

    def evaluate(self, plan: TradePlan, frame: pd.DataFrame) -> DecisionReview:
        explanation: list[str] = []
        if not plan.entry_reason.strip():
            explanation.append("Entry reason is missing; a disciplined plan needs a clear thesis.")
        if frame.empty:
            return DecisionReview("Weak", 0.0, None, explanation + ["No market data was available for confirmation."])
        if plan.entry_price <= 0 or plan.target_price <= 0 or plan.stop_loss <= 0:
            return DecisionReview("Weak", 0.0, None, explanation + ["Entry, target, and stop loss must be positive numbers."])

        reward = plan.target_price - plan.entry_price
        risk = plan.entry_price - plan.stop_loss
        if plan.side.lower() == "short":
            reward = plan.entry_price - plan.target_price
            risk = plan.stop_loss - plan.entry_price
        risk_reward = reward / risk if risk > 0 else None

        score = 0.0
        if risk_reward is None or risk_reward <= 0:
            explanation.append("Risk/reward is invalid because the stop loss and target do not define positive reward versus risk.")
        elif risk_reward >= 2.0:
            score += 35
            explanation.append(f"Risk/reward is attractive at {risk_reward:.2f}:1.")
        elif risk_reward >= 1.5:
            score += 25
            explanation.append(f"Risk/reward is acceptable at {risk_reward:.2f}:1.")
        elif risk_reward >= 1.0:
            score += 12
            explanation.append(f"Risk/reward is thin at {risk_reward:.2f}:1.")
        else:
            explanation.append(f"Risk/reward is weak at {risk_reward:.2f}:1.")

        enriched = enrich_indicators(frame)
        latest = enriched.iloc[-1]
        trend = classify_trend(enriched)
        rsi = float(latest.get("RSI", 50))
        macd = float(latest.get("MACD", 0)) if pd.notna(latest.get("MACD")) else 0.0
        macd_signal = float(latest.get("MACD_SIGNAL", 0)) if pd.notna(latest.get("MACD_SIGNAL")) else 0.0
        levels = detect_support_resistance(enriched)

        if plan.side.lower() == "long":
            if trend == "Uptrend":
                score += 25
                explanation.append("Trend alignment is positive: price structure is in an uptrend.")
            elif trend == "Sideways":
                score += 12
                explanation.append("Trend is sideways; confirmation matters more than usual.")
            else:
                explanation.append("Trend alignment is weak for a long plan.")
            if 35 <= rsi <= 68:
                score += 15
                explanation.append(f"RSI is balanced at {rsi:.2f}, not obviously stretched.")
            elif rsi < 35:
                score += 10
                explanation.append(f"RSI is low at {rsi:.2f}; this can help mean reversion but needs confirmation.")
            else:
                explanation.append(f"RSI is elevated at {rsi:.2f}; avoid chasing without a strong reason.")
            if macd > macd_signal:
                score += 10
                explanation.append("MACD is above its signal line, supporting momentum.")
            if levels.support and plan.stop_loss < levels.support:
                score += 8
                explanation.append("Stop loss sits below detected support, which gives the plan structural logic.")
            elif levels.support:
                explanation.append("Stop loss is not clearly below detected support.")
        else:
            if trend == "Downtrend":
                score += 25
                explanation.append("Trend alignment is positive for a short plan.")
            else:
                explanation.append("Short plans require extra caution when the broader trend is not down.")
            if rsi > 65:
                score += 10
                explanation.append(f"RSI is elevated at {rsi:.2f}, which may support a pullback thesis.")
            if macd < macd_signal:
                score += 10
                explanation.append("MACD is below its signal line, supporting downside momentum.")

        if len(plan.entry_reason.strip()) >= 20:
            score += 7
            explanation.append("Entry reason is specific enough to review.")

        score = max(0.0, min(100.0, score))
        rating = "Strong" if score >= 75 else "Moderate" if score >= 50 else "Weak"
        explanation.append("This is a plan-quality rating, not an instruction to buy or sell.")
        return DecisionReview(rating, round(score, 2), None if risk_reward is None else round(risk_reward, 2), explanation)


def _stance_from_score(score: float) -> str:
    if score >= 80:
        return "High-priority research candidate"
    if score >= 65:
        return "Constructive setup, needs trade plan"
    if score >= 45:
        return "Watchlist only, wait for confirmation"
    return "Weak setup, avoid forcing a decision"


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


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
