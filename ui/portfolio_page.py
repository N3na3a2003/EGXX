from __future__ import annotations

import pandas as pd
import streamlit as st

from portfolio.portfolio_manager import PortfolioManager, positions_from_dataframe
from ui.charts import allocation_chart
from ui.common import get_fetcher


def render_portfolio_page() -> None:
    st.title("Portfolio")
    st.caption("Enter holdings manually. Values update from the same free, delayed data source used by the analysis pages.")

    if "portfolio_editor" not in st.session_state:
        st.session_state["portfolio_editor"] = pd.DataFrame(
            [
                {"Ticker": "COMI", "Quantity": 100.0, "Average Price": 75.0},
                {"Ticker": "FWRY", "Quantity": 500.0, "Average Price": 7.5},
            ]
        )

    edited = st.data_editor(
        st.session_state["portfolio_editor"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Quantity": st.column_config.NumberColumn(min_value=0.0, step=1.0, required=True),
            "Average Price": st.column_config.NumberColumn(min_value=0.0, step=0.01, required=True),
        },
        hide_index=True,
    )
    st.session_state["portfolio_editor"] = edited
    stop_loss_pct = st.slider("Default risk per holding using stop-loss distance", 1, 40, 10, 1)

    positions = positions_from_dataframe(edited)
    with st.spinner("Valuing portfolio..."):
        result = PortfolioManager(get_fetcher()).compute(positions, stop_loss_pct=float(stop_loss_pct))

    table = result["positions"]
    cols = st.columns(5)
    cols[0].metric("Total cost", f"{result['total_cost']:,.2f}")
    cols[1].metric("Current value", f"{result['total_value']:,.2f}")
    cols[2].metric("P/L", f"{result['total_pl']:,.2f}", f"{result['total_pl_pct']:.2f}%")
    cols[3].metric("Health score", f"{result['health_score']:.0f}/100")
    cols[4].metric("Holdings", len(table))

    if not table.empty:
        left, right = st.columns([0.62, 0.38])
        left.dataframe(table, use_container_width=True, hide_index=True)
        right.plotly_chart(allocation_chart(table), use_container_width=True)
        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button("Download portfolio CSV", data=csv, file_name="egx_portfolio.csv", mime="text/csv")

    warnings = result["warnings"]
    st.subheader("Portfolio Health")
    for note in result["health_notes"]:
        st.write(f"- {note}")

    if warnings:
        st.subheader("Risk Notes")
        for warning in warnings:
            st.warning(warning)
