from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np

from indicators.moving_averages import simple_moving_average
from indicators.rsi import calculate_rsi


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: dict[str, float]


class Backtester:
    def run_rsi_strategy(
        self,
        frame: pd.DataFrame,
        lower: float = 30,
        upper: float = 70,
        initial_cash: float = 100_000,
    ) -> BacktestResult:
        data = frame.copy()
        if data.empty or len(data) < 30:
            return _empty_result(initial_cash)
        data["RSI"] = calculate_rsi(data["Close"])
        data["signal"] = 0
        data.loc[data["RSI"] < lower, "signal"] = 1
        data.loc[data["RSI"] > upper, "signal"] = 0
        data["position"] = data["signal"].replace(0, np.nan).ffill().fillna(0)
        return self._simulate(data, initial_cash=initial_cash, signal_name="RSI")

    def run_ma_crossover_strategy(
        self,
        frame: pd.DataFrame,
        fast: int = 50,
        slow: int = 200,
        initial_cash: float = 100_000,
    ) -> BacktestResult:
        data = frame.copy()
        if data.empty or len(data) < slow:
            return _empty_result(initial_cash)
        data["fast_ma"] = simple_moving_average(data["Close"], fast)
        data["slow_ma"] = simple_moving_average(data["Close"], slow)
        data["position"] = (data["fast_ma"] > data["slow_ma"]).astype(int)
        return self._simulate(data, initial_cash=initial_cash, signal_name="MA Crossover")

    def _simulate(self, data: pd.DataFrame, initial_cash: float, signal_name: str) -> BacktestResult:
        returns = pd.to_numeric(data["Close"], errors="coerce").pct_change().fillna(0)
        position = data["position"].shift(1).fillna(0)
        strategy_returns = returns * position
        equity = (1 + strategy_returns).cumprod() * initial_cash
        trades = _extract_trades(data, signal_name)
        metrics = _metrics(equity, trades, initial_cash)
        return BacktestResult(trades=trades, equity_curve=equity, metrics=metrics)


def _extract_trades(data: pd.DataFrame, signal_name: str) -> pd.DataFrame:
    changes = data["position"].diff().fillna(data["position"])
    entries = data[changes > 0]
    exits = data[changes < 0]
    rows = []
    exit_iter = iter(exits.iterrows())
    current_exit = next(exit_iter, None)
    for entry_date, entry in entries.iterrows():
        while current_exit is not None and current_exit[0] <= entry_date:
            current_exit = next(exit_iter, None)
        if current_exit is None:
            exit_date = data.index[-1]
            exit_price = float(data["Close"].iloc[-1])
        else:
            exit_date = current_exit[0]
            exit_price = float(current_exit[1]["Close"])
        entry_price = float(entry["Close"])
        rows.append(
            {
                "Strategy": signal_name,
                "Entry Date": entry_date,
                "Entry Price": round(entry_price, 4),
                "Exit Date": exit_date,
                "Exit Price": round(exit_price, 4),
                "Return %": round((exit_price / entry_price - 1) * 100, 2) if entry_price else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _metrics(equity: pd.Series, trades: pd.DataFrame, initial_cash: float) -> dict[str, float]:
    if equity.empty:
        return {"total_return_pct": 0.0, "win_rate_pct": 0.0, "max_drawdown_pct": 0.0, "trades": 0}
    total_return = (float(equity.iloc[-1]) / initial_cash - 1) * 100 if initial_cash else 0.0
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = float((trades["Return %"] > 0).mean() * 100) if not trades.empty else 0.0
    return {
        "total_return_pct": round(total_return, 2),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "trades": int(len(trades)),
    }


def _empty_result(initial_cash: float) -> BacktestResult:
    return BacktestResult(
        trades=pd.DataFrame(columns=["Strategy", "Entry Date", "Entry Price", "Exit Date", "Exit Price", "Return %"]),
        equity_curve=pd.Series([initial_cash], name="Equity"),
        metrics={"total_return_pct": 0.0, "win_rate_pct": 0.0, "max_drawdown_pct": 0.0, "trades": 0},
    )
