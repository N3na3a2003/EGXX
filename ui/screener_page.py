from __future__ import annotations

import streamlit as st

from ui.common import cached_scan, parse_symbols_for_ui, symbol_multiline_default


def render_screener_page() -> None:
    st.title("Smart Screener")
    st.caption("One-click shortlist for EGX research candidates, ranked by trend, momentum, volume, and risk location.")

    left, right = st.columns([0.7, 0.3])
    with left:
        raw_symbols = st.text_area("Symbols", value=symbol_multiline_default(), height=210)
    with right:
        timeframe = st.selectbox("Timeframe", ["6mo", "1y", "5y"], index=1)
        min_score = st.slider("Minimum score", 0, 100, 0, 5)
        top_n = st.slider("Top opportunities", 5, 30, 10, 1)
        st.markdown("<p class='small-muted'>The score explains why a stock is interesting and what can invalidate the setup.</p>", unsafe_allow_html=True)

    symbols = tuple(parse_symbols_for_ui(raw_symbols))
    if st.button("Find best opportunities", type="primary", use_container_width=True):
        st.session_state["last_screener_symbols"] = symbols
        st.session_state["last_screener_timeframe"] = timeframe
        st.session_state["last_screener_min_score"] = float(min_score)
        st.session_state["last_screener_top_n"] = int(top_n)

    symbols_to_scan = st.session_state.get("last_screener_symbols", symbols)
    timeframe_to_scan = st.session_state.get("last_screener_timeframe", timeframe)
    min_score_to_scan = st.session_state.get("last_screener_min_score", float(min_score))
    top_n_to_show = st.session_state.get("last_screener_top_n", int(top_n))

    with st.spinner("Scanning market..."):
        table = cached_scan(symbols_to_scan, timeframe_to_scan, min_score_to_scan)

    if table.empty:
        st.info("No symbols matched the current filters.")
        return

    top_columns = [
        "Symbol",
        "Name",
        "Score",
        "Rating",
        "Research Stance",
        "Trend",
        "Close",
        "RSI",
        "Support",
        "Resistance",
        "Why",
        "Risk Notes",
    ]
    st.subheader(f"Top {min(top_n_to_show, len(table))} Research Candidates")
    st.dataframe(table[[column for column in top_columns if column in table.columns]].head(top_n_to_show), use_container_width=True, hide_index=True)

    with st.expander("Show scoring breakdown"):
        breakdown_columns = [
            "Symbol",
            "Score",
            "Trend Score",
            "Momentum Score",
            "Risk Score",
            "Volume Score",
            "Conditions",
            "Source",
        ]
        st.dataframe(table[[column for column in breakdown_columns if column in table.columns]], use_container_width=True, hide_index=True)

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button("Download screener CSV", data=csv, file_name="egx_screener.csv", mime="text/csv")
