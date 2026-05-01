from __future__ import annotations

import streamlit as st

from alerts.alert_engine import AlertEngine, alerts_to_frame
from indicators.engine import enrich_indicators
from ui.common import cached_ohlcv, parse_symbols_for_ui, symbol_multiline_default


def render_alerts_page() -> None:
    st.title("Alerts")
    st.caption("Configure threshold checks for delayed research alerts. No broker connection or order execution is included.")

    left, right = st.columns([0.65, 0.35])
    with left:
        raw_symbols = st.text_area("Symbols", value=symbol_multiline_default(), height=210)
    with right:
        timeframe = st.selectbox("Alert timeframe", ["6mo", "1y", "5y"], index=1)
        price_change_pct = st.number_input("Price move threshold %", min_value=0.5, max_value=50.0, value=5.0, step=0.5)
        rsi_low = st.number_input("RSI low threshold", min_value=1.0, max_value=49.0, value=30.0, step=1.0)
        rsi_high = st.number_input("RSI high threshold", min_value=51.0, max_value=99.0, value=70.0, step=1.0)
        volume_multiple = st.number_input("Volume spike multiple", min_value=1.1, max_value=10.0, value=1.8, step=0.1)

    config = {
        "price_change_pct": float(price_change_pct),
        "rsi_low": float(rsi_low),
        "rsi_high": float(rsi_high),
        "volume_spike_multiple": float(volume_multiple),
    }
    engine = AlertEngine(config)
    symbols = parse_symbols_for_ui(raw_symbols)

    if st.button("Evaluate alerts", type="primary", use_container_width=True):
        all_alerts = []
        with st.spinner("Evaluating alerts..."):
            for symbol in symbols:
                result = cached_ohlcv(symbol, timeframe)
                if result.data.empty:
                    all_alerts.extend(engine.evaluate(symbol, result.data))
                    continue
                enriched = enrich_indicators(result.data)
                all_alerts.extend(engine.evaluate(symbol, enriched))
        table = alerts_to_frame(all_alerts)
        if table.empty:
            st.success("No alerts triggered for the selected symbols.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download alerts CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="egx_alerts.csv",
                mime="text/csv",
            )
