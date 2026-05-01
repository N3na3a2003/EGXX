from __future__ import annotations

import pandas as pd

from data.data_fetcher import DataFetcher
from data.symbol_mapper import company_name
from indicators.engine import enrich_indicators
from indicators.support_resistance import detect_support_resistance
from screener.scoring import score_stock


class MarketScanner:
    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        self.fetcher = fetcher or DataFetcher()

    def scan(self, symbols: list[str], timeframe: str = "1y", min_score: float = 0.0) -> pd.DataFrame:
        rows = []
        for symbol in symbols:
            result = self.fetcher.fetch_ohlcv(symbol, timeframe=timeframe)
            if result.data.empty:
                rows.append(
                    {
                        "Symbol": symbol,
                        "Name": company_name(symbol),
                        "Score": 0.0,
                        "Rating": "Weak",
                        "Research Stance": "Insufficient data",
                        "Trend": "No data",
                        "Close": None,
                        "RSI": None,
                        "Why": "No data available.",
                        "Risk Notes": result.warning or "No data.",
                        "Conditions": result.warning or "No data",
                        "Source": result.source,
                    }
                )
                continue

            enriched = enrich_indicators(result.data)
            latest = enriched.iloc[-1]
            levels = detect_support_resistance(enriched)
            score = score_stock(enriched)
            conditions = score.get("conditions", {})
            active_conditions = ", ".join([name for name, active in conditions.items() if active]) or "None"
            reasons = "; ".join(score.get("reasons", [])) or "No strong confirmation."
            risks = "; ".join(score.get("risks", [])) or "No major technical risk flagged."
            rows.append(
                {
                    "Symbol": result.symbol,
                    "Name": company_name(result.symbol),
                    "Score": score["score"],
                    "Rating": score["rating"],
                    "Research Stance": score["research_stance"],
                    "Trend": score["trend"],
                    "Close": round(float(latest["Close"]), 4),
                    "RSI": round(float(latest["RSI"]), 2),
                    "MA50": _round_or_none(latest.get("MA50")),
                    "MA200": _round_or_none(latest.get("MA200")),
                    "Support": _round_or_none(levels.support),
                    "Resistance": _round_or_none(levels.resistance),
                    "Why": reasons,
                    "Risk Notes": risks,
                    "Conditions": active_conditions,
                    "Trend Score": score["trend_score"],
                    "Momentum Score": score["momentum_score"],
                    "Risk Score": score["risk_score"],
                    "Volume Score": score["volume_score"],
                    "Source": result.source,
                }
            )

        table = pd.DataFrame(rows)
        if table.empty:
            return table
        table = table[table["Score"] >= min_score]
        return table.sort_values(["Score", "RSI"], ascending=[False, True], na_position="last").reset_index(drop=True)


def _round_or_none(value: object, digits: int = 4) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
