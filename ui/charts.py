from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators.support_resistance import detect_support_resistance


def candlestick_chart(frame: pd.DataFrame, title: str = "Price", show_bollinger: bool = True) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )
    if frame.empty:
        fig.update_layout(title=title, template="plotly_dark", height=620)
        return fig

    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["Open"],
            high=frame["High"],
            low=frame["Low"],
            close=frame["Close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )
    for column, color in [("MA20", "#38bdf8"), ("MA50", "#f59e0b"), ("MA200", "#e879f9")]:
        if column in frame:
            fig.add_trace(go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column, line=dict(width=1.4, color=color)), row=1, col=1)

    if show_bollinger and {"BB_UPPER", "BB_LOWER"}.issubset(frame.columns):
        fig.add_trace(go.Scatter(x=frame.index, y=frame["BB_UPPER"], mode="lines", name="BB Upper", line=dict(width=1, color="#64748b")), row=1, col=1)
        fig.add_trace(go.Scatter(x=frame.index, y=frame["BB_LOWER"], mode="lines", name="BB Lower", line=dict(width=1, color="#64748b")), row=1, col=1)

    levels = detect_support_resistance(frame)
    if levels.support:
        fig.add_hline(y=levels.support, line_dash="dot", line_color="#22c55e", annotation_text="Support", row=1, col=1)
    if levels.resistance:
        fig.add_hline(y=levels.resistance, line_dash="dot", line_color="#ef4444", annotation_text="Resistance", row=1, col=1)

    volume_color = ["#22c55e" if close >= open_ else "#ef4444" for close, open_ in zip(frame["Close"], frame["Open"])]
    fig.add_trace(go.Bar(x=frame.index, y=frame["Volume"], name="Volume", marker_color=volume_color), row=2, col=1)

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=650,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def rsi_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI" in frame:
        fig.add_trace(go.Scatter(x=frame.index, y=frame["RSI"], mode="lines", name="RSI", line=dict(color="#38bdf8")))
    fig.add_hline(y=70, line_dash="dot", line_color="#ef4444")
    fig.add_hline(y=30, line_dash="dot", line_color="#22c55e")
    fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=25, b=20), yaxis_range=[0, 100])
    return fig


def macd_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if {"MACD", "MACD_SIGNAL", "MACD_HIST"}.issubset(frame.columns):
        fig.add_trace(go.Bar(x=frame.index, y=frame["MACD_HIST"], name="Histogram", marker_color="#64748b"))
        fig.add_trace(go.Scatter(x=frame.index, y=frame["MACD"], mode="lines", name="MACD", line=dict(color="#38bdf8")))
        fig.add_trace(go.Scatter(x=frame.index, y=frame["MACD_SIGNAL"], mode="lines", name="Signal", line=dict(color="#f59e0b")))
    fig.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=25, b=20))
    return fig


def equity_curve_chart(equity: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity", line=dict(color="#22c55e")))
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=25, b=20))
    return fig


def allocation_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty:
        fig.add_trace(go.Pie(labels=frame["Ticker"], values=frame["Current Value"], hole=0.45))
    fig.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=25, b=20), showlegend=True)
    return fig


def feature_importance_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not frame.empty and {"Feature", "Importance"}.issubset(frame.columns):
        ordered = frame.sort_values("Importance", ascending=True).tail(8)
        fig.add_trace(
            go.Bar(
                x=ordered["Importance"],
                y=ordered["Feature"],
                orientation="h",
                marker_color="#38bdf8",
                name="Importance",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=25, b=20),
        xaxis_title="Importance",
        yaxis_title="Feature",
    )
    return fig


def scenario_chart(scenarios: dict[str, float | str]) -> go.Figure:
    labels = [label for label in ["Worst case", "Most likely", "Best case"] if isinstance(scenarios.get(label), (int, float))]
    values = [float(scenarios[label]) for label in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, marker_color=["#ef4444", "#f59e0b", "#22c55e"][: len(labels)]))
    fig.update_layout(template="plotly_dark", height=280, margin=dict(l=20, r=20, t=25, b=20), yaxis_title="Price level")
    return fig
