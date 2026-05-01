from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_assistant.decision_engine import DecisionEngine, TradePlan
from analysis.stock_analysis import StockAnalysisEngine
from backtesting.backtester import Backtester
from data.csv_loader import load_ohlcv_csv
from data.symbol_mapper import company_name, default_symbols, normalize_symbol
from indicators.engine import enrich_indicators, latest_indicator_snapshot
from indicators.support_resistance import detect_support_resistance
from ui.charts import (
    candlestick_chart,
    equity_curve_chart,
    feature_importance_chart,
    macd_chart,
    rsi_chart,
    scenario_chart,
)
from ui.common import cached_ohlcv, show_data_warning


def render_stock_page() -> None:
    st.title("Stock Analysis")
    top = st.columns([0.35, 0.2, 0.25, 0.2])
    symbol = normalize_symbol(top[0].text_input("Ticker", value="COMI", help="Use EGX ticker with or without .CA"))
    timeframe = top[1].selectbox("Timeframe", ["1d", "1wk", "1mo", "6mo", "1y", "5y"], index=4)
    uploaded_csv = top[2].file_uploader("CSV override", type=["csv"], help="Optional OHLCV CSV with Date, Open, High, Low, Close, Volume")
    show_bollinger = top[3].toggle("Bollinger", value=True)

    source_label = "api"
    warning = None
    if uploaded_csv is not None:
        try:
            base_frame = load_ohlcv_csv(uploaded_csv)
            source_label = "uploaded_csv"
        except Exception as exc:
            st.error(f"CSV could not be loaded: {exc}")
            st.stop()
    else:
        result = cached_ohlcv(symbol, timeframe)
        base_frame = result.data
        source_label = result.source
        warning = result.warning

    show_data_warning(warning)
    if base_frame.empty:
        st.warning("No price data available for this stock.")
        st.stop()

    enriched = enrich_indicators(base_frame)
    snapshot = latest_indicator_snapshot(base_frame)
    levels = detect_support_resistance(enriched)
    latest = enriched.iloc[-1]
    close = float(latest["Close"])
    previous_close = float(enriched["Close"].iloc[-2]) if len(enriched) > 1 else close
    change_pct = (close / previous_close - 1) * 100 if previous_close else 0.0

    peer_frames = _load_peer_frames(symbol)
    analysis_report = StockAnalysisEngine().analyze(symbol, base_frame, peer_frames=peer_frames)
    decision_report = DecisionEngine().evaluate_stock(symbol, enriched)

    st.caption(f"{company_name(symbol)} | Symbol: {symbol}.CA | Source: {source_label} | No direct buy/sell instructions.")
    metrics = st.columns(6)
    metrics[0].metric("Stock Score", f"{analysis_report.final_score:.0f}/100", analysis_report.overall_view)
    metrics[1].metric("ML Up Probability", _format_pct_value(analysis_report.ml_result.probability_up), analysis_report.ml_result.confidence)
    metrics[2].metric("Close", f"{close:,.2f}", f"{change_pct:.2f}%")
    metrics[3].metric("Trend", analysis_report.trend)
    metrics[4].metric("Valuation", analysis_report.valuation)
    metrics[5].metric("Market Risk", analysis_report.market_risk)

    st.subheader("Quick Summary")
    for line in analysis_report.quick_summary:
        st.write(f"- {line}")

    insight_cols = st.columns([0.5, 0.5])
    with insight_cols[0]:
        st.markdown("**Strengths**")
        for item in analysis_report.strengths or ["No strong positive confirmation yet."]:
            st.write(f"- {item}")
    with insight_cols[1]:
        st.markdown("**Weaknesses / Risks**")
        for item in analysis_report.weaknesses or ["No major issue flagged from available data."]:
            st.write(f"- {item}")

    st.plotly_chart(candlestick_chart(enriched, title=f"{symbol} Price, MAs, Volume, and Zones", show_bollinger=show_bollinger), use_container_width=True)

    overview_tab, ml_tab, risk_tab, comparison_tab, assistant_tab, backtest_tab, data_tab = st.tabs(
        ["Overview", "ML Insights", "Risk & Scenarios", "Comparison", "Trade Plan Review", "Backtesting", "Data"]
    )

    with overview_tab:
        score_cols = st.columns(4)
        score_cols[0].metric("Fundamentals", f"{analysis_report.fundamental_score:.0f}/100")
        score_cols[1].metric("Technicals", f"{analysis_report.technical_score:.0f}/100")
        score_cols[2].metric("Momentum", f"{analysis_report.momentum_score:.0f}/100")
        score_cols[3].metric("ML", f"{analysis_report.ml_score:.0f}/100")

        technical_cols = st.columns(3)
        technical_cols[0].metric("MA Status", analysis_report.ma_status)
        technical_cols[1].metric("RSI Status", analysis_report.rsi_status)
        technical_cols[2].metric("Volume Trend", analysis_report.volume_trend)
        st.info(f"Entry zone: {analysis_report.entry_zone}")

        chart_frame = enriched[[column for column in ["Close", "MA50", "MA200"] if column in enriched]].dropna()
        if not chart_frame.empty:
            st.line_chart(chart_frame, use_container_width=True)
        col1, col2 = st.columns(2)
        col1.plotly_chart(rsi_chart(enriched), use_container_width=True)
        col2.plotly_chart(macd_chart(enriched), use_container_width=True)

        fundamentals_table = pd.DataFrame(
            [
                {"Metric": "Revenue growth", "Value": _format_percent_ratio(analysis_report.fundamentals.revenue_growth)},
                {"Metric": "Profit margin", "Value": _format_percent_ratio(analysis_report.fundamentals.profit_margin)},
                {"Metric": "P/E", "Value": _format_number(analysis_report.fundamentals.pe_ratio)},
                {"Metric": "Market cap", "Value": _format_number(analysis_report.fundamentals.market_cap)},
                {"Metric": "Fundamental source", "Value": analysis_report.fundamentals.source},
            ]
        )
        st.dataframe(fundamentals_table, use_container_width=True, hide_index=True)

    with ml_tab:
        ml_cols = st.columns(4)
        ml_cols[0].metric("Probability", _format_pct_value(analysis_report.ml_result.probability_up))
        ml_cols[1].metric("Confidence", analysis_report.ml_result.confidence)
        ml_cols[2].metric("Training rows", analysis_report.ml_result.train_rows)
        ml_cols[3].metric("Validation accuracy", _format_pct_value(analysis_report.ml_result.validation_accuracy))
        for item in analysis_report.ml_result.explanation:
            st.write(f"- {item}")
        st.plotly_chart(feature_importance_chart(analysis_report.ml_result.feature_importance), use_container_width=True)

    with risk_tab:
        risk_cols = st.columns(3)
        risk_cols[0].metric("Liquidity Risk", analysis_report.liquidity_risk)
        risk_cols[1].metric("Volatility Risk", analysis_report.volatility_risk)
        risk_cols[2].metric("Market Risk", analysis_report.market_risk)
        perf_cols = st.columns(2)
        perf_cols[0].metric("1M Performance", _format_pct_value(analysis_report.one_month_performance))
        perf_cols[1].metric("3M Performance", _format_pct_value(analysis_report.three_month_performance))
        st.plotly_chart(scenario_chart(analysis_report.scenarios), use_container_width=True)
        st.dataframe(pd.DataFrame([analysis_report.scenarios]).T.reset_index().rename(columns={"index": "Scenario", 0: "Value"}), use_container_width=True, hide_index=True)

    with comparison_tab:
        if analysis_report.peer_comparison:
            peer_table = pd.DataFrame([peer.__dict__ for peer in analysis_report.peer_comparison])
            st.dataframe(peer_table, use_container_width=True, hide_index=True)
        else:
            st.info("Peer comparison is unavailable because peer data could not be loaded.")
        st.caption("Comparison uses technical and momentum scores only because fundamentals coverage is inconsistent for EGX tickers.")

    with assistant_tab:
        st.caption("This reviews your plan quality. It does not provide a direct buy or sell instruction.")
        st.metric("Research stance", decision_report.research_stance, f"Decision score {decision_report.score:.0f}/100")
        col1, col2 = st.columns(2)
        entry_reason = col1.text_area("Entry reason", placeholder="Example: Pullback to MA50 with volume support and defined stop.", height=130)
        side = col1.selectbox("Plan direction", ["long", "short"])
        entry_price = col2.number_input("Entry price", min_value=0.0, value=float(close), step=0.01)
        target_price = col2.number_input("Target price", min_value=0.0, value=float(close * 1.12), step=0.01)
        stop_loss = col2.number_input("Stop loss", min_value=0.0, value=float(close * 0.94), step=0.01)
        if st.button("Review plan", type="primary"):
            plan = TradePlan(
                ticker=symbol,
                entry_reason=entry_reason,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                side=side,
            )
            review = DecisionEngine().evaluate(plan, enriched)
            st.metric("Plan rating", review.rating, f"Score {review.score:.0f}/100")
            st.metric("Risk/reward", f"{review.risk_reward}:1" if review.risk_reward is not None else "Invalid")
            for item in review.explanation:
                st.write(f"- {item}")

    with backtest_tab:
        strategy = st.radio("Strategy", ["RSI-based", "Moving average crossover"], horizontal=True)
        initial_cash = st.number_input("Initial cash", min_value=1000.0, value=100000.0, step=5000.0)
        tester = Backtester()
        if strategy == "RSI-based":
            lower, upper = st.slider("RSI thresholds", 5, 95, (30, 70), 1)
            backtest = tester.run_rsi_strategy(base_frame, lower=lower, upper=upper, initial_cash=initial_cash)
        else:
            fast = st.number_input("Fast MA", min_value=5, max_value=100, value=50)
            slow = st.number_input("Slow MA", min_value=20, max_value=250, value=200)
            backtest = tester.run_ma_crossover_strategy(base_frame, fast=int(fast), slow=int(slow), initial_cash=initial_cash)
        bcols = st.columns(4)
        bcols[0].metric("Total return", f"{backtest.metrics['total_return_pct']:.2f}%")
        bcols[1].metric("Win rate", f"{backtest.metrics['win_rate_pct']:.2f}%")
        bcols[2].metric("Max drawdown", f"{backtest.metrics['max_drawdown_pct']:.2f}%")
        bcols[3].metric("Trades", backtest.metrics["trades"])
        st.plotly_chart(equity_curve_chart(backtest.equity_curve), use_container_width=True)
        st.dataframe(backtest.trades, use_container_width=True, hide_index=True)

    with data_tab:
        indicator_table = pd.DataFrame([snapshot]).T.reset_index()
        indicator_table.columns = ["Metric", "Value"]
        st.dataframe(indicator_table, use_container_width=True, hide_index=True)
        st.dataframe(enriched.tail(250), use_container_width=True)
        st.download_button(
            "Download enriched data CSV",
            data=enriched.to_csv().encode("utf-8"),
            file_name=f"{symbol}_analysis.csv",
            mime="text/csv",
        )


def _load_peer_frames(symbol: str) -> dict[str, pd.DataFrame]:
    peers = [peer for peer in default_symbols() if peer != symbol][:2]
    frames: dict[str, pd.DataFrame] = {}
    for peer in peers:
        result = cached_ohlcv(peer, "1y")
        if not result.data.empty:
            frames[peer] = result.data
    return frames


def _format_pct_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}%"


def _format_percent_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.1f}%"


def _format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.2f}"
