from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from analysis.fundamentals import Fundamentals, FundamentalsFetcher, fundamental_score, valuation_label
from indicators.engine import enrich_indicators
from indicators.moving_averages import classify_trend, moving_average_crosses
from indicators.support_resistance import detect_support_resistance
from ml.predictor import PredictionResult, StockMLPredictor


@dataclass(frozen=True)
class PeerComparison:
    symbol: str
    score: float
    trend: str
    one_month_return: float | None
    volatility: float | None


@dataclass(frozen=True)
class StockAnalysisReport:
    final_score: float
    overall_view: str
    trend: str
    technical_score: float
    momentum_score: float
    fundamental_score: float
    ml_score: float
    ml_result: PredictionResult
    fundamentals: Fundamentals
    valuation: str
    rsi_status: str
    ma_status: str
    volume_trend: str
    one_month_performance: float | None
    three_month_performance: float | None
    liquidity_risk: str
    volatility_risk: str
    market_risk: str
    scenarios: dict[str, float | str]
    entry_zone: str
    strengths: list[str]
    weaknesses: list[str]
    quick_summary: list[str]
    peer_comparison: list[PeerComparison]


class StockAnalysisEngine:
    def __init__(
        self,
        fundamentals_fetcher: FundamentalsFetcher | None = None,
        predictor: StockMLPredictor | None = None,
    ) -> None:
        self.fundamentals_fetcher = fundamentals_fetcher or FundamentalsFetcher()
        self.predictor = predictor or StockMLPredictor()

    def analyze(
        self,
        symbol: str,
        frame: pd.DataFrame,
        peer_frames: dict[str, pd.DataFrame] | None = None,
    ) -> StockAnalysisReport:
        enriched = enrich_indicators(frame)
        fundamentals = self.fundamentals_fetcher.fetch(symbol)
        fund_score, fund_strengths, fund_weaknesses = fundamental_score(fundamentals)
        ml_result = self.predictor.predict_next_period(enriched)
        ml_score = ml_result.probability_up if ml_result.probability_up is not None else 50.0
        technical_score, technical_strengths, technical_weaknesses, ma_status, rsi_status, volume_trend = _technical_score(enriched)
        momentum_score, one_month, three_month, momentum_notes = _momentum_score(enriched)

        final_score = round(
            fund_score * 0.30
            + technical_score * 0.25
            + momentum_score * 0.20
            + ml_score * 0.25,
            2,
        )
        trend = _trend_label(enriched)
        liquidity_risk, volatility_risk, market_risk = _risk_labels(enriched, trend)
        valuation = valuation_label(fundamentals)
        scenarios = _scenarios(enriched, ml_result)
        entry_zone = _entry_zone(enriched)
        peer_comparison = _compare_peers(peer_frames or {})
        strengths = _top_items(fund_strengths + technical_strengths + momentum_notes + _ml_strengths(ml_result), 6)
        weaknesses = _top_items(fund_weaknesses + technical_weaknesses + _ml_weaknesses(ml_result), 6)
        overall_view = _overall_view(final_score)
        quick_summary = _quick_summary(symbol, final_score, trend, ml_result, strengths, weaknesses)

        return StockAnalysisReport(
            final_score=final_score,
            overall_view=overall_view,
            trend=trend,
            technical_score=technical_score,
            momentum_score=momentum_score,
            fundamental_score=fund_score,
            ml_score=round(float(ml_score), 2),
            ml_result=ml_result,
            fundamentals=fundamentals,
            valuation=valuation,
            rsi_status=rsi_status,
            ma_status=ma_status,
            volume_trend=volume_trend,
            one_month_performance=one_month,
            three_month_performance=three_month,
            liquidity_risk=liquidity_risk,
            volatility_risk=volatility_risk,
            market_risk=market_risk,
            scenarios=scenarios,
            entry_zone=entry_zone,
            strengths=strengths,
            weaknesses=weaknesses,
            quick_summary=quick_summary,
            peer_comparison=peer_comparison,
        )


def _technical_score(frame: pd.DataFrame) -> tuple[float, list[str], list[str], str, str, str]:
    if frame.empty:
        return 0.0, [], ["No technical data available."], "Unknown", "Unknown", "Unknown"
    latest = frame.iloc[-1]
    close = _safe_float(latest.get("Close"), 0.0)
    ma50 = _safe_float(latest.get("MA50"))
    ma200 = _safe_float(latest.get("MA200"))
    rsi = _safe_float(latest.get("RSI"), 50.0)
    score = 50.0
    strengths: list[str] = []
    weaknesses: list[str] = []

    if close and ma50 and close > ma50:
        score += 12
        strengths.append("Price is above MA50.")
    elif ma50:
        score -= 10
        weaknesses.append("Price is below MA50.")
    if close and ma200 and close > ma200:
        score += 12
        strengths.append("Price is above MA200.")
    elif ma200:
        score -= 10
        weaknesses.append("Price is below MA200.")
    if ma50 and ma200 and ma50 > ma200:
        score += 10
        ma_status = "Bullish MA structure"
    elif ma50 and ma200 and ma50 < ma200:
        score -= 10
        ma_status = "Bearish MA structure"
    else:
        ma_status = "MA structure unclear"

    crosses = moving_average_crosses(frame)
    if crosses["bullish_cross"]:
        score += 8
        strengths.append("Fresh bullish MA crossover.")
    if crosses["bearish_cross"]:
        score -= 8
        weaknesses.append("Fresh bearish MA crossover.")

    if 40 <= rsi <= 65:
        score += 12
        rsi_status = "Healthy momentum"
    elif rsi < 30:
        score -= 4
        rsi_status = "Oversold, needs confirmation"
    elif rsi > 70:
        score -= 10
        rsi_status = "Overbought"
    else:
        rsi_status = "Neutral"

    volume = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0)
    volume_trend = "Stable"
    if len(volume) >= 20:
        recent = float(volume.tail(5).mean())
        baseline = float(volume.tail(20).mean())
        if baseline > 0 and recent > baseline * 1.25:
            score += 6
            strengths.append("Recent volume is above average.")
            volume_trend = "Rising"
        elif baseline > 0 and recent < baseline * 0.65:
            score -= 5
            weaknesses.append("Recent volume is weak.")
            volume_trend = "Falling"
    return round(max(0.0, min(100.0, score)), 2), strengths, weaknesses, ma_status, rsi_status, volume_trend


def _momentum_score(frame: pd.DataFrame) -> tuple[float, float | None, float | None, list[str]]:
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 22:
        return 50.0, None, None, ["Momentum history is limited."]
    one_month = _period_return(close, 21)
    three_month = _period_return(close, 63)
    score = 50.0
    notes: list[str] = []
    for label, value, weight in [("1M", one_month, 0.55), ("3M", three_month, 0.45)]:
        if value is None:
            continue
        if value > 10:
            score += 22 * weight
            notes.append(f"{label} performance is strong at {value:.1f}%.")
        elif value > 0:
            score += 10 * weight
            notes.append(f"{label} performance is positive at {value:.1f}%.")
        elif value < -10:
            score -= 18 * weight
        else:
            score -= 6 * weight
    return round(max(0.0, min(100.0, score)), 2), one_month, three_month, notes


def _risk_labels(frame: pd.DataFrame, trend: str) -> tuple[str, str, str]:
    volume = pd.to_numeric(frame["Volume"], errors="coerce").dropna()
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    avg_volume = float(volume.tail(30).mean()) if not volume.empty else 0.0
    liquidity = "High" if avg_volume < 50_000 else "Medium" if avg_volume < 250_000 else "Low"
    returns = close.pct_change().dropna()
    vol = float(returns.tail(30).std() * (252 ** 0.5) * 100) if len(returns) >= 10 else 0.0
    volatility = "High" if vol > 45 else "Medium" if vol > 25 else "Low"
    market = "High" if trend == "Bearish" else "Medium" if trend == "Sideways" else "Low"
    return liquidity, volatility, market


def _scenarios(frame: pd.DataFrame, ml_result: PredictionResult) -> dict[str, float | str]:
    close = float(pd.to_numeric(frame["Close"], errors="coerce").dropna().iloc[-1])
    returns = pd.to_numeric(frame["Close"], errors="coerce").pct_change().dropna()
    recent_vol = float(returns.tail(30).std()) if len(returns) >= 10 else 0.03
    if pd.isna(recent_vol) or recent_vol <= 0:
        recent_vol = 0.03
    probability = (ml_result.probability_up or 50.0) / 100
    bias = (probability - 0.5) * 0.08
    swing = max(recent_vol * 8, 0.04)
    return {
        "Best case": round(close * (1 + swing + max(bias, 0)), 4),
        "Most likely": round(close * (1 + bias), 4),
        "Worst case": round(close * (1 - swing + min(bias, 0)), 4),
        "Note": "Scenario levels are risk ranges, not price forecasts.",
    }


def _entry_zone(frame: pd.DataFrame) -> str:
    levels = detect_support_resistance(frame)
    close = float(pd.to_numeric(frame["Close"], errors="coerce").dropna().iloc[-1])
    if levels.support and levels.resistance:
        return f"Support zone {levels.support:.2f}; breakout confirmation above {levels.resistance:.2f}."
    if levels.support:
        return f"Risk-defined support zone near {levels.support:.2f}."
    if levels.resistance:
        return f"Wait for confirmation around resistance near {levels.resistance:.2f}."
    return f"No clean zone detected; current close is {close:.2f}."


def _compare_peers(peer_frames: dict[str, pd.DataFrame]) -> list[PeerComparison]:
    rows: list[PeerComparison] = []
    for symbol, frame in peer_frames.items():
        if frame.empty:
            continue
        enriched = enrich_indicators(frame)
        technical_score, _, _, _, _, _ = _technical_score(enriched)
        momentum_score, one_month, _, _ = _momentum_score(enriched)
        returns = pd.to_numeric(frame["Close"], errors="coerce").pct_change().dropna()
        vol = float(returns.tail(30).std() * (252 ** 0.5) * 100) if len(returns) >= 10 else None
        rows.append(
            PeerComparison(
                symbol=symbol,
                score=round(technical_score * 0.55 + momentum_score * 0.45, 2),
                trend=_trend_label(enriched),
                one_month_return=one_month,
                volatility=None if vol is None else round(vol, 2),
            )
        )
    return sorted(rows, key=lambda item: item.score, reverse=True)[:2]


def _trend_label(frame: pd.DataFrame) -> str:
    trend = classify_trend(frame)
    if trend == "Uptrend":
        return "Bullish"
    if trend == "Downtrend":
        return "Bearish"
    if trend == "Sideways":
        return "Sideways"
    return "Unknown"


def _overall_view(score: float) -> str:
    if score >= 75:
        return "Strong research profile"
    if score >= 60:
        return "Constructive but needs confirmation"
    if score >= 45:
        return "Mixed profile"
    return "Weak profile"


def _quick_summary(
    symbol: str,
    final_score: float,
    trend: str,
    ml_result: PredictionResult,
    strengths: list[str],
    weaknesses: list[str],
) -> list[str]:
    ml_text = f"ML up probability {ml_result.probability_up:.1f}% ({ml_result.confidence})" if ml_result.probability_up is not None else "ML unavailable"
    return [
        f"{symbol} score is {final_score:.0f}/100 with a {trend.lower()} trend profile.",
        f"{ml_text}; use it as one input, not a standalone decision.",
        f"Main strength: {(strengths or ['limited confirmation'])[0]}; main risk: {(weaknesses or ['limited data quality'])[0]}",
    ]


def _ml_strengths(ml_result: PredictionResult) -> list[str]:
    if ml_result.probability_up is not None and ml_result.probability_up >= 60:
        return [f"ML model leans positive at {ml_result.probability_up:.1f}%."]
    return []


def _ml_weaknesses(ml_result: PredictionResult) -> list[str]:
    if not ml_result.available:
        return ml_result.explanation[:1]
    if ml_result.probability_up is not None and ml_result.probability_up < 45:
        return [f"ML model leans cautious at {ml_result.probability_up:.1f}%."]
    if ml_result.confidence == "Low":
        return ["ML confidence is low."]
    return []


def _period_return(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    start = float(close.iloc[-periods - 1])
    end = float(close.iloc[-1])
    if start == 0:
        return None
    return round((end / start - 1) * 100, 2)


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
