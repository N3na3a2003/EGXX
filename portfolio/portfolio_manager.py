from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.data_fetcher import DataFetcher
from data.symbol_mapper import normalize_symbol
from portfolio.risk_analysis import allocation_warning, health_score_explanation, portfolio_health_score


@dataclass(frozen=True)
class Position:
    ticker: str
    quantity: float
    average_price: float

    @property
    def cost(self) -> float:
        return self.quantity * self.average_price


class PortfolioManager:
    def __init__(self, fetcher: DataFetcher | None = None) -> None:
        self.fetcher = fetcher or DataFetcher()

    def compute(self, positions: list[Position], stop_loss_pct: float = 10.0) -> dict[str, object]:
        rows = []
        for position in positions:
            ticker = normalize_symbol(position.ticker)
            if not ticker or position.quantity <= 0 or position.average_price <= 0:
                continue
            current_price, warning = self.fetcher.latest_price(ticker)
            if current_price is None:
                current_price = position.average_price
            current_value = position.quantity * current_price
            pl_value = current_value - position.cost
            pl_pct = (current_price / position.average_price - 1) * 100
            risk_value = position.cost * (stop_loss_pct / 100)
            rows.append(
                {
                    "Ticker": ticker,
                    "Quantity": position.quantity,
                    "Average Price": position.average_price,
                    "Current Price": round(float(current_price), 4),
                    "Cost": round(position.cost, 2),
                    "Current Value": round(current_value, 2),
                    "P/L": round(pl_value, 2),
                    "P/L %": round(pl_pct, 2),
                    "Risk Value": round(risk_value, 2),
                    "Warning": warning,
                }
            )

        table = pd.DataFrame(rows)
        if table.empty:
            return {
                "positions": table,
                "total_cost": 0.0,
                "total_value": 0.0,
                "total_pl": 0.0,
                "total_pl_pct": 0.0,
                "warnings": [],
                "health_notes": ["No valid positions entered."],
                "health_score": 0.0,
            }

        total_value = float(table["Current Value"].sum())
        total_cost = float(table["Cost"].sum())
        total_pl = total_value - total_cost
        table["Allocation %"] = (table["Current Value"] / total_value * 100).round(2) if total_value else 0.0
        table["Risk Exposure %"] = (table["Risk Value"] / total_value * 100).round(2) if total_value else 0.0
        warnings = allocation_warning(table)
        warnings.extend([warning for warning in table["Warning"].dropna().tolist() if warning])
        return {
            "positions": table.drop(columns=["Warning"]),
            "total_cost": round(total_cost, 2),
            "total_value": round(total_value, 2),
            "total_pl": round(total_pl, 2),
            "total_pl_pct": round((total_pl / total_cost * 100) if total_cost else 0.0, 2),
            "warnings": warnings,
            "health_notes": health_score_explanation(table),
            "health_score": portfolio_health_score(table),
        }


def positions_from_dataframe(frame: pd.DataFrame) -> list[Position]:
    if frame.empty:
        return []
    positions: list[Position] = []
    for _, row in frame.iterrows():
        try:
            positions.append(
                Position(
                    ticker=str(row["Ticker"]),
                    quantity=float(row["Quantity"]),
                    average_price=float(row["Average Price"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return positions
