from __future__ import annotations

import streamlit as st

from config import APP_NAME
from data.cache_manager import FileCache
from ui.alerts_page import render_alerts_page
from ui.dashboard import render_dashboard
from ui.portfolio_page import render_portfolio_page
from ui.screener_page import render_screener_page
from ui.stock_page import render_stock_page


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="EGX",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 0.85rem 1rem;
            border-radius: 8px;
        }
        .small-muted { color: #9ca3af; font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    configure_page()

    st.sidebar.title(APP_NAME)
    st.sidebar.caption("Decision support for Egyptian equities. Data is delayed and best-effort.")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Stock Analysis", "Screener", "Portfolio", "Alerts"],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    st.sidebar.warning(
        "Educational research tool only. No auto-trading, no guaranteed real-time data, "
        "and no direct buy/sell instructions."
    )
    if st.sidebar.button("Clear market data cache", use_container_width=True):
        removed = FileCache().clear_all()
        st.cache_data.clear()
        st.sidebar.success(f"Cache cleared ({removed} files). Refresh or rerun the scan.")

    if page == "Dashboard":
        render_dashboard()
    elif page == "Stock Analysis":
        render_stock_page()
    elif page == "Screener":
        render_screener_page()
    elif page == "Portfolio":
        render_portfolio_page()
    elif page == "Alerts":
        render_alerts_page()


if __name__ == "__main__":
    main()
