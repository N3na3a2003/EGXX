# EGX Decision Support Platform

A Streamlit web application for researching Egyptian Stock Exchange equities using free public data sources. It is a decision-support tool only: it does not place trades, provide guaranteed real-time prices, or issue direct buy/sell instructions.

## Features

- EGX ticker handling with Yahoo Finance `.CA` mapping
- Best-effort OHLCV fetching with 10-minute disk and Streamlit cache
- RSI, MACD, MA20/50/100/200, Bollinger Bands, ATR
- Support/resistance detection and trend classification
- Ranked smart screener with trend, momentum, and volume scoring
- Decision score for each stock with plain-English reasons, risks, and disciplined next steps
- RandomForest ML model estimating next-period up probability from RSI, moving averages, volume, and returns
- Feature importance chart explaining which factors mattered most to the model
- Weighted final score: fundamentals 30%, technicals 25%, momentum 20%, ML 25%
- Risk, valuation, scenario analysis, and peer comparison
- Manual portfolio tracking with allocation, P/L, risk exposure, and health score
- Portfolio health explanation covering concentration, diversification, and drawdown risk
- Configurable alerts for price moves, RSI crosses, MA crosses, breakouts, breakdowns, and volume spikes
- Trade-plan review assistant based on risk/reward and indicator alignment
- Backtests for RSI and moving-average crossover strategies
- Interactive Plotly candlestick, volume, indicator, allocation, and equity charts

## Install

```powershell
cd C:\Users\N3na3a\Documents\Codex\2026-04-30\you-are-a-senior-python-engineer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` is not on PATH, install Python 3.11 or newer from python.org, then reopen PowerShell.

## Run

```powershell
streamlit run app.py
```

The app will open at the local URL shown by Streamlit, usually `http://localhost:8501`.

## CSV Dataset Format

You can use APIs or upload a local CSV from the Stock Analysis page. Required columns:

```text
Date,Open,High,Low,Close,Volume
```

Optional columns such as `Ticker`, `RevenueGrowth`, `ProfitMargin`, and `PERatio` may be included for your own records. See:

```text
sample_data/egx_sample_format.csv
```

## ML Notes

The ML layer trains a per-stock `RandomForestClassifier` on historical technical features:

- RSI
- MA50 and MA200 distance
- 1-day, 5-day, and 20-day returns
- 20-period volume ratio
- 20-period volatility

Target is `1` if the next period's close is higher than the current close, else `0`. Output is a probability and confidence label. This is a statistical research signal, not a forecast guarantee or trading instruction.

## Data Notes

The primary source is Yahoo Finance through `yfinance`, using EGX symbols like `COMI.CA`. A requests-based Yahoo chart fallback and a BeautifulSoup quote-page snapshot fallback are included for resilience. Free data can be delayed, missing, or temporarily unavailable, especially for less liquid EGX names.

## Risk Disclaimer

This software is for research and education. It ranks setups and evaluates user-defined plans, but it does not predict prices, guarantee outcomes, or replace licensed financial advice.
