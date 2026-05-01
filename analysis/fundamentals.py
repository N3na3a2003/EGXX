from __future__ import annotations

from dataclasses import dataclass

import yfinance as yf

from data.cache_manager import FileCache
from data.symbol_mapper import to_yahoo_symbol


@dataclass(frozen=True)
class Fundamentals:
    revenue_growth: float | None
    profit_margin: float | None
    pe_ratio: float | None
    market_cap: float | None
    source: str
    warning: str | None = None

    def to_payload(self) -> dict[str, float | str | None]:
        return {
            "revenue_growth": self.revenue_growth,
            "profit_margin": self.profit_margin,
            "pe_ratio": self.pe_ratio,
            "market_cap": self.market_cap,
            "source": self.source,
            "warning": self.warning,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "Fundamentals":
        return cls(
            revenue_growth=_float_or_none(payload.get("revenue_growth")),
            profit_margin=_float_or_none(payload.get("profit_margin")),
            pe_ratio=_float_or_none(payload.get("pe_ratio")),
            market_cap=_float_or_none(payload.get("market_cap")),
            source=str(payload.get("source", "cache")),
            warning=payload.get("warning") if isinstance(payload.get("warning"), str) else None,
        )


class FundamentalsFetcher:
    def __init__(self, cache: FileCache | None = None) -> None:
        self.cache = cache or FileCache()

    def fetch(self, symbol: str) -> Fundamentals:
        yahoo_symbol = to_yahoo_symbol(symbol)
        cached = self.cache.get("fundamentals", yahoo_symbol)
        if isinstance(cached, dict):
            return Fundamentals.from_payload(cached)
        try:
            info = yf.Ticker(yahoo_symbol).get_info()
        except Exception as exc:
            return Fundamentals(None, None, None, None, "none", f"Fundamentals unavailable: {exc}")
        fundamentals = Fundamentals(
            revenue_growth=_float_or_none(info.get("revenueGrowth")),
            profit_margin=_float_or_none(info.get("profitMargins")),
            pe_ratio=_float_or_none(info.get("trailingPE")),
            market_cap=_float_or_none(info.get("marketCap")),
            source="yfinance_info",
            warning=None,
        )
        self.cache.set("fundamentals", yahoo_symbol, fundamentals.to_payload())
        return fundamentals


def fundamental_score(fundamentals: Fundamentals) -> tuple[float, list[str], list[str]]:
    score = 50.0
    strengths: list[str] = []
    weaknesses: list[str] = []
    if fundamentals.revenue_growth is None and fundamentals.profit_margin is None and fundamentals.pe_ratio is None:
        return 50.0, ["Fundamental data is limited, so the score uses a neutral baseline."], ["Verify fundamentals from company filings."]

    if fundamentals.revenue_growth is not None:
        growth_pct = fundamentals.revenue_growth * 100
        if growth_pct > 15:
            score += 18
            strengths.append(f"Revenue growth is strong at {growth_pct:.1f}%.")
        elif growth_pct > 0:
            score += 8
            strengths.append(f"Revenue growth is positive at {growth_pct:.1f}%.")
        else:
            score -= 12
            weaknesses.append(f"Revenue growth is negative at {growth_pct:.1f}%.")

    if fundamentals.profit_margin is not None:
        margin_pct = fundamentals.profit_margin * 100
        if margin_pct > 15:
            score += 15
            strengths.append(f"Profit margin is healthy at {margin_pct:.1f}%.")
        elif margin_pct > 5:
            score += 7
            strengths.append(f"Profit margin is positive at {margin_pct:.1f}%.")
        else:
            score -= 10
            weaknesses.append(f"Profit margin is thin at {margin_pct:.1f}%.")

    if fundamentals.pe_ratio is not None:
        pe = fundamentals.pe_ratio
        if 4 <= pe <= 14:
            score += 12
            strengths.append(f"P/E is moderate at {pe:.1f}.")
        elif pe > 25:
            score -= 12
            weaknesses.append(f"P/E is elevated at {pe:.1f}.")
        elif pe <= 0:
            score -= 10
            weaknesses.append("P/E is not meaningful or negative.")
    return round(max(0.0, min(100.0, score)), 2), strengths, weaknesses


def valuation_label(fundamentals: Fundamentals) -> str:
    if fundamentals.pe_ratio is None:
        return "Insufficient data"
    pe = fundamentals.pe_ratio
    margin = fundamentals.profit_margin
    growth = fundamentals.revenue_growth
    if pe <= 0:
        return "Not meaningful"
    if pe < 10 and (margin is None or margin > 0.05) and (growth is None or growth >= 0):
        return "Potentially undervalued"
    if pe <= 22:
        return "Fair range"
    return "Potentially overvalued"


def _float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
