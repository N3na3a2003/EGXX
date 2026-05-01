from __future__ import annotations

import streamlit as st

from config import DEFAULT_TIMEFRAME
from data.symbol_mapper import default_symbols
from ui.common import cached_scan


def render_dashboard() -> None:
    st.title("EGX Market Dashboard")
    st.caption("Delayed public data via free sources. Use results as research prompts, not execution instructions.")

    symbols = tuple(default_symbols())
    with st.spinner("Scanning default EGX watchlist..."):
        table = cached_scan(symbols, DEFAULT_TIMEFRAME, 0.0)

    if table.empty:
        st.warning("No market data was available. Check internet access or try fewer symbols.")
        return

    valid = table[table["Trend"] != "No data"].copy()
    cols = st.columns(4)
    cols[0].metric("Tracked symbols", len(table))
    cols[1].metric("Data available", len(valid))
    cols[2].metric("Average score", f"{valid['Score'].mean():.1f}" if not valid.empty else "-")
    top_symbol = valid.iloc[0]["Symbol"] if not valid.empty else "-"
    cols[3].metric("Top ranked", top_symbol)

    st.subheader("Best Research Candidates")
    display_columns = [
        "Symbol",
        "Name",
        "Score",
        "Rating",
        "Research Stance",
        "Trend",
        "Close",
        "RSI",
        "Why",
        "Risk Notes",
        "Source",
    ]
    st.dataframe(
        table[[column for column in display_columns if column in table.columns]].head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Full Watchlist")
    full_columns = [
        "Symbol",
        "Score",
        "Rating",
        "Trend",
        "Close",
        "RSI",
        "Support",
        "Resistance",
        "Conditions",
    ]
    st.dataframe(table[[column for column in full_columns if column in table.columns]], use_container_width=True, hide_index=True)

    st.subheader("Trend Mix")
    trend_counts = valid["Trend"].value_counts().reset_index()
    trend_counts.columns = ["Trend", "Count"]
    st.bar_chart(trend_counts, x="Trend", y="Count", use_container_width=True)
