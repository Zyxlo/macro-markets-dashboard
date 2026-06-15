# Macro + Markets Dashboard

This is a Streamlit dashboard I built to track major market assets and get a clearer view of what is happening across markets.

The idea behind the project was not just to show price charts, but to connect different assets together and create a simple macro-style view of the market. The dashboard looks at equities, commodities, crypto, FX, rates, and the dollar, then summarizes whether conditions look more risk-on, risk-off, or mixed.

## What it does

The dashboard tracks:

* S&P 500
* Nasdaq
* Gold
* Crude Oil
* Bitcoin
* EUR/USD
* US Dollar Index
* US 10Y Yield

It shows daily moves, 1-month and 3-month returns, volatility, moving averages, correlations, and relative performance across assets.

I also added a simple macro regime engine that uses rule-based logic to interpret the market environment. For example, weak equities and crypto can point toward a risk-off environment, while stronger equities and weaker defensive pressure can point toward risk-on conditions.

## Main sections

### Overview

The overview gives a quick summary of the market. It shows the best and worst daily performers, market breadth, average volatility, and a simple market regime read.

### Macro Dashboard

This section connects each asset to a macro role. For example, the US Dollar Index is used as a liquidity pressure signal, the US 10Y Yield as a rates pressure signal, Gold as a defensive demand signal, and Oil as an inflation/growth pressure signal.

The macro engine gives a regime score and explains the main pressure in the market.

### Performance

This section compares how different assets performed over the selected time period.

### Correlation

This section shows how the assets are moving relative to each other using a daily return correlation matrix.

### Deep Dive

This section lets you choose one asset and look at its price trend, moving averages, volatility, and a short analyst-style interpretation.

## Tools used

* Python
* Streamlit
* pandas
* NumPy
* yfinance
* Plotly

## How to run it

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Notes

This dashboard is not meant to predict the market or give investment advice. It is a learning project built to practice financial data analysis, dashboard design, and basic macro interpretation.

The macro regime model is intentionally simple and rule-based. The point is to organize market information in a structured way and make cross-asset relationships easier to understand.

## Project status

Version 1 is complete.

The current version includes the market overview, macro dashboard, performance comparison, correlation matrix, asset deep dive, and project notes.
