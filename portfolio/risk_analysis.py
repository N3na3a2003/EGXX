from __future__ import annotations

import math

import pandas as pd


def allocation_warning(positions: pd.DataFrame, concentration_threshold: float = 35.0) -> list[str]:
    warnings: list[str] = []
    if positions.empty or "Allocation %" not in positions:
        return warnings
    concentrated = positions[positions["Allocation %"] > concentration_threshold]
    for _, row in concentrated.iterrows():
        warnings.append(
            f"{row['Ticker']} is {row['Allocation %']:.1f}% of portfolio value. Consider concentration risk."
        )
    if len(positions) < 4:
        warnings.append("Portfolio has fewer than four holdings. Diversification is limited.")
    losers = positions[positions["P/L %"] < -15]
    for _, row in losers.iterrows():
        warnings.append(f"{row['Ticker']} is down {abs(row['P/L %']):.1f}%. Review thesis and risk controls.")
    return warnings


def health_score_explanation(positions: pd.DataFrame) -> list[str]:
    if positions.empty:
        return ["No holdings entered yet."]
    notes: list[str] = []
    max_allocation = float(positions["Allocation %"].max()) if "Allocation %" in positions else 100.0
    if max_allocation <= 25:
        notes.append("Allocation is reasonably diversified across current holdings.")
    elif max_allocation <= 35:
        notes.append("Largest position is meaningful but still manageable.")
    else:
        notes.append("Largest position is concentrated and dominates portfolio risk.")

    count = len(positions)
    if count >= 5:
        notes.append("Holding count supports basic diversification.")
    elif count >= 3:
        notes.append("Diversification is acceptable but could be broader.")
    else:
        notes.append("Very few holdings means single-stock risk is high.")

    weighted_pl = _weighted_pl(positions)
    if weighted_pl >= 0:
        notes.append("Portfolio P/L is positive or flat on a weighted basis.")
    elif weighted_pl > -10:
        notes.append("Portfolio drawdown is moderate; review weak positions.")
    else:
        notes.append("Portfolio drawdown is material; risk controls should be reviewed.")
    return notes


def portfolio_health_score(positions: pd.DataFrame) -> float:
    if positions.empty:
        return 0.0
    score = 100.0
    max_allocation = float(positions["Allocation %"].max()) if "Allocation %" in positions else 100.0
    if max_allocation > 50:
        score -= 30
    elif max_allocation > 35:
        score -= 18
    elif max_allocation > 25:
        score -= 8

    count = len(positions)
    if count < 3:
        score -= 20
    elif count < 5:
        score -= 8

    weighted_pl = _weighted_pl(positions)
    if weighted_pl < -20:
        score -= 25
    elif weighted_pl < -10:
        score -= 15
    elif weighted_pl < 0:
        score -= 6

    risk_exposure = float(positions["Risk Exposure %"].sum()) if "Risk Exposure %" in positions else 0.0
    if risk_exposure > 40:
        score -= 15
    elif risk_exposure > 25:
        score -= 8
    return round(max(0.0, min(100.0, score)), 2)


def portfolio_volatility(returns: pd.DataFrame, weights: pd.Series) -> float | None:
    if returns.empty or weights.empty:
        return None
    aligned = returns.dropna(how="all").fillna(0)
    weights = weights.reindex(aligned.columns).fillna(0)
    if math.isclose(float(weights.sum()), 0.0):
        return None
    daily_cov = aligned.cov()
    variance = float(weights.T @ daily_cov @ weights)
    if variance < 0:
        return None
    return round((variance ** 0.5) * (252 ** 0.5) * 100, 2)


def _weighted_pl(positions: pd.DataFrame) -> float:
    if positions.empty or "Allocation %" not in positions or "P/L %" not in positions:
        return 0.0
    weights = positions["Allocation %"] / 100
    return float((weights * positions["P/L %"]).sum())
