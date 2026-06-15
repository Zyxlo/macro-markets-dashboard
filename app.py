import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Macro + Markets Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Asset universe
# -----------------------------
assets = {
    "S&P 500": {"ticker": "^GSPC", "type": "Equity Index"},
    "Nasdaq": {"ticker": "^IXIC", "type": "Equity Index"},
    "Gold": {"ticker": "GC=F", "type": "Commodity"},
    "Crude Oil": {"ticker": "CL=F", "type": "Commodity"},
    "Bitcoin": {"ticker": "BTC-USD", "type": "Crypto"},
    "EUR/USD": {"ticker": "EURUSD=X", "type": "FX"},
    "US Dollar Index": {"ticker": "DX-Y.NYB", "type": "Macro"},
    "US 10Y Yield": {"ticker": "^TNX", "type": "Rates"}
}

period_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]


# -----------------------------
# Data functions
# -----------------------------
@st.cache_data(ttl=300)
def get_price_data(ticker_symbol, period="1y", interval="1d"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period, interval=interval)
    return data


def get_close_prices(ticker_symbol, period="1y"):
    data = get_price_data(ticker_symbol, period=period, interval="1d")

    if data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)

    return data["Close"].dropna()


def calculate_asset_snapshot(asset_name, asset_info):
    ticker_symbol = asset_info["ticker"]
    close_prices = get_close_prices(ticker_symbol, period="6mo")

    empty_snapshot = {
        "Asset": asset_name,
        "Type": asset_info["type"],
        "Latest Price": None,
        "Daily Change": None,
        "Daily % Change": None,
        "1M Return": None,
        "3M Return": None,
        "20D Volatility": None,
        "Trend Score": None,
        "Trend Label": "N/A"
    }

    if len(close_prices) < 2:
        return empty_snapshot

    latest_price = float(close_prices.iloc[-1])
    previous_price = float(close_prices.iloc[-2])

    daily_change = latest_price - previous_price
    daily_pct_change = (daily_change / previous_price) * 100

    one_month_return = None
    three_month_return = None

    if len(close_prices) >= 22:
        one_month_return = ((latest_price / float(close_prices.iloc[-22])) - 1) * 100

    if len(close_prices) >= 63:
        three_month_return = ((latest_price / float(close_prices.iloc[-63])) - 1) * 100

    daily_returns = close_prices.pct_change().dropna()
    volatility_20d = None

    if len(daily_returns) >= 20:
        volatility_20d = float(daily_returns.tail(20).std() * np.sqrt(252) * 100)

    ma20 = close_prices.rolling(window=20).mean().iloc[-1] if len(close_prices) >= 20 else np.nan
    ma50 = close_prices.rolling(window=50).mean().iloc[-1] if len(close_prices) >= 50 else np.nan

    trend_score = 0

    if not pd.isna(ma20) and latest_price > ma20:
        trend_score += 1

    if not pd.isna(ma50) and latest_price > ma50:
        trend_score += 1

    if one_month_return is not None and one_month_return > 0:
        trend_score += 1

    if trend_score == 3:
        trend_label = "Strong Uptrend"
    elif trend_score == 2:
        trend_label = "Positive / Mixed"
    elif trend_score == 1:
        trend_label = "Weak / Mixed"
    else:
        trend_label = "Downtrend"

    return {
        "Asset": asset_name,
        "Type": asset_info["type"],
        "Latest Price": latest_price,
        "Daily Change": daily_change,
        "Daily % Change": daily_pct_change,
        "1M Return": one_month_return,
        "3M Return": three_month_return,
        "20D Volatility": volatility_20d,
        "Trend Score": trend_score,
        "Trend Label": trend_label
    }


def build_relative_performance_data(period, selected_assets):
    comparison_data = []

    for asset_name in selected_assets:
        ticker_symbol = assets[asset_name]["ticker"]
        data = get_price_data(ticker_symbol, period=period, interval="1d")

        if data.empty or len(data) < 2:
            continue

        data = data.reset_index()
        date_column = "Datetime" if "Datetime" in data.columns else "Date"

        data = (
            data[[date_column, "Close"]]
            .dropna()
            .rename(columns={date_column: "Date"})
        )

        first_price = float(data["Close"].iloc[0])

        if first_price == 0:
            continue

        data["Performance %"] = ((data["Close"] / first_price) - 1) * 100
        data["Asset"] = asset_name

        comparison_data.append(data[["Date", "Asset", "Performance %"]])

    if not comparison_data:
        return pd.DataFrame()

    return pd.concat(comparison_data, ignore_index=True)


def build_correlation_matrix(period, selected_assets):
    return_series = []

    for asset_name in selected_assets:
        ticker_symbol = assets[asset_name]["ticker"]
        data = get_price_data(ticker_symbol, period=period, interval="1d")

        if data.empty or len(data) < 3:
            continue

        data = data.reset_index()
        date_column = "Datetime" if "Datetime" in data.columns else "Date"

        data["Date"] = pd.to_datetime(data[date_column]).dt.date

        data = (
            data[["Date", "Close"]]
            .dropna()
            .groupby("Date")
            .last()
        )

        returns = data["Close"].pct_change().dropna()
        returns.name = asset_name

        return_series.append(returns)

    if not return_series:
        return pd.DataFrame()

    returns_df = pd.concat(return_series, axis=1, join="outer").sort_index()

    if returns_df.empty:
        return pd.DataFrame()

    return returns_df.corr(min_periods=5)


# -----------------------------
# Interpretation functions
# -----------------------------
def classify_volatility(volatility):
    if volatility is None or pd.isna(volatility):
        return "unknown"

    if volatility < 15:
        return "low"

    if volatility < 30:
        return "moderate"

    return "elevated"


def asset_interpretation(asset_name, snapshot):
    daily = snapshot["Daily % Change"]
    one_month = snapshot["1M Return"]
    three_month = snapshot["3M Return"]
    vol = snapshot["20D Volatility"]
    trend = snapshot["Trend Label"]

    vol_label = classify_volatility(vol)

    points = []

    if daily is not None and not pd.isna(daily):
        if daily > 1:
            points.append(
                f"{asset_name} had a strong positive daily move, suggesting short-term buying pressure."
            )
        elif daily < -1:
            points.append(
                f"{asset_name} had a strong negative daily move, suggesting short-term selling pressure."
            )
        else:
            points.append(
                f"{asset_name} was relatively stable today, so the latest move is not extreme."
            )

    if one_month is not None and not pd.isna(one_month):
        if one_month > 3:
            points.append("One-month momentum is clearly positive.")
        elif one_month < -3:
            points.append("One-month momentum is clearly negative.")
        else:
            points.append("One-month momentum is mostly neutral.")

    if three_month is not None and not pd.isna(three_month):
        if three_month > 5:
            points.append("The three-month trend supports a stronger medium-term move.")
        elif three_month < -5:
            points.append("The three-month trend points to medium-term weakness.")
        else:
            points.append("The three-month trend is not decisive.")

    points.append(f"Trend model: {trend}. Volatility is {vol_label}.")

    if trend in ["Strong Uptrend", "Positive / Mixed"] and vol_label in ["low", "moderate"]:
        decision_read = (
            "Overall read: constructive. Momentum is supportive and risk is not flashing extreme stress."
        )
    elif trend in ["Strong Uptrend", "Positive / Mixed"] and vol_label == "elevated":
        decision_read = (
            "Overall read: positive but unstable. The move is working, but risk is higher."
        )
    elif trend in ["Weak / Mixed", "Downtrend"] and vol_label == "elevated":
        decision_read = (
            "Overall read: cautious. Weak trend plus elevated volatility usually means poor risk/reward."
        )
    else:
        decision_read = "Overall read: mixed. There is no clean signal yet."

    points.append(decision_read)

    return points


def market_regime_read(overview_df):
    if overview_df.empty:
        return "Not enough data to estimate a market regime."

    def get_row(asset):
        rows = overview_df[overview_df["Asset"] == asset]

        if rows.empty:
            return None

        return rows.iloc[0]

    spx = get_row("S&P 500")
    nasdaq = get_row("Nasdaq")
    gold = get_row("Gold")
    dollar = get_row("US Dollar Index")
    ten_y = get_row("US 10Y Yield")
    btc = get_row("Bitcoin")

    risk_score = 0
    defensive_score = 0

    for row in [spx, nasdaq, btc]:
        if row is not None and row["1M Return"] is not None and not pd.isna(row["1M Return"]):
            if row["1M Return"] > 0:
                risk_score += 1
            else:
                defensive_score += 1

    for row in [gold, dollar, ten_y]:
        if row is not None and row["1M Return"] is not None and not pd.isna(row["1M Return"]):
            if row["1M Return"] > 0:
                defensive_score += 1

    if risk_score >= 2 and defensive_score <= 1:
        return "Risk-on regime: equities and/or crypto are leading while defensive pressure looks limited."

    if defensive_score >= 3 and risk_score <= 1:
        return "Risk-off regime: defensive assets or macro stress indicators are leading."

    return "Mixed regime: leadership is split across risk assets and defensive/macro assets."


def macro_regime_engine(overview_df):
    def get_value(asset, column):
        rows = overview_df[overview_df["Asset"] == asset]

        if rows.empty:
            return None

        value = rows.iloc[0][column]

        if value is None or pd.isna(value):
            return None

        return float(value)

    score = 0
    signals = []
    pressures = []
    breakdown = []

    def add_breakdown(signal_name, reading, impact, interpretation):
        breakdown.append({
            "Signal": signal_name,
            "Reading": reading,
            "Score Impact": impact,
            "Interpretation": interpretation
        })

    spx_1m = get_value("S&P 500", "1M Return")
    nasdaq_1m = get_value("Nasdaq", "1M Return")
    btc_1m = get_value("Bitcoin", "1M Return")
    dollar_1m = get_value("US Dollar Index", "1M Return")
    ten_y_1m = get_value("US 10Y Yield", "1M Return")
    gold_1m = get_value("Gold", "1M Return")
    oil_1m = get_value("Crude Oil", "1M Return")

    equity_returns = [
        value for value in [spx_1m, nasdaq_1m]
        if value is not None
    ]

    if equity_returns:
        avg_equity_return = float(np.mean(equity_returns))

        if avg_equity_return > 1:
            score += 2
            signals.append("Equity momentum is positive, which supports risk appetite.")
            add_breakdown(
                "Equities",
                "Positive",
                "+2",
                "S&P 500 and Nasdaq momentum support risk appetite."
            )
        elif avg_equity_return < -1:
            score -= 2
            signals.append("Equity momentum is negative, which weakens risk appetite.")
            pressures.append("weak equities")
            add_breakdown(
                "Equities",
                "Weak",
                "-2",
                "Equity weakness is a major risk-off signal."
            )
        else:
            signals.append("Equity momentum is neutral.")
            add_breakdown(
                "Equities",
                "Neutral",
                "0",
                "Equity momentum is not giving a clear signal."
            )
    else:
        add_breakdown(
            "Equities",
            "No data",
            "0",
            "Not enough equity data to score this signal."
        )

    if btc_1m is not None:
        if btc_1m > 3:
            score += 1
            signals.append("Bitcoin is showing positive momentum, suggesting stronger speculative risk appetite.")
            add_breakdown(
                "Bitcoin",
                "Positive",
                "+1",
                "Crypto strength suggests stronger speculative risk appetite."
            )
        elif btc_1m < -3:
            score -= 1
            signals.append("Bitcoin is weak, suggesting weaker speculative risk appetite.")
            pressures.append("weak crypto")
            add_breakdown(
                "Bitcoin",
                "Weak",
                "-1",
                "Crypto weakness suggests weaker speculative risk appetite."
            )
        else:
            add_breakdown(
                "Bitcoin",
                "Neutral",
                "0",
                "Bitcoin is not giving a strong risk signal."
            )
    else:
        add_breakdown(
            "Bitcoin",
            "No data",
            "0",
            "Not enough Bitcoin data to score this signal."
        )

    if dollar_1m is not None:
        if dollar_1m > 0.5:
            score -= 1
            signals.append("The US Dollar Index is rising, which can pressure global liquidity and risk assets.")
            pressures.append("stronger dollar")
            add_breakdown(
                "US Dollar",
                "Rising",
                "-1",
                "A stronger dollar can tighten global liquidity and pressure risk assets."
            )
        elif dollar_1m < -0.5:
            score += 1
            signals.append("The US Dollar Index is falling, which can support liquidity conditions.")
            add_breakdown(
                "US Dollar",
                "Falling",
                "+1",
                "A weaker dollar can support global liquidity."
            )
        else:
            add_breakdown(
                "US Dollar",
                "Neutral",
                "0",
                "The dollar is not giving a strong liquidity signal."
            )
    else:
        add_breakdown(
            "US Dollar",
            "No data",
            "0",
            "Not enough dollar data to score this signal."
        )

    if ten_y_1m is not None:
        if ten_y_1m > 1:
            score -= 1
            signals.append("The US 10Y Yield is rising, which can pressure equity valuations.")
            pressures.append("rising yields")
            add_breakdown(
                "US 10Y Yield",
                "Rising",
                "-1",
                "Higher yields can pressure equity valuations."
            )
        elif ten_y_1m < -1:
            score += 1
            signals.append("The US 10Y Yield is falling, which can reduce pressure on valuations.")
            add_breakdown(
                "US 10Y Yield",
                "Falling",
                "+1",
                "Lower yields can reduce pressure on valuations."
            )
        else:
            add_breakdown(
                "US 10Y Yield",
                "Neutral",
                "0",
                "Yield movement is not strong enough to change the regime score."
            )
    else:
        add_breakdown(
            "US 10Y Yield",
            "No data",
            "0",
            "Not enough yield data to score this signal."
        )

    if gold_1m is not None:
        if gold_1m > 2:
            score -= 1
            signals.append("Gold is rising strongly, which may show defensive demand or inflation concern.")
            pressures.append("defensive gold bid")
            add_breakdown(
                "Gold",
                "Strong",
                "-1",
                "Strong gold performance may signal defensive demand or inflation concern."
            )
        elif gold_1m < -2:
            score += 1
            signals.append("Gold is weak, suggesting less defensive demand.")
            add_breakdown(
                "Gold",
                "Weak",
                "+1",
                "Weak gold suggests less defensive demand."
            )
        else:
            add_breakdown(
                "Gold",
                "Neutral",
                "0",
                "Gold is not giving a strong defensive signal."
            )
    else:
        add_breakdown(
            "Gold",
            "No data",
            "0",
            "Not enough gold data to score this signal."
        )

    if oil_1m is not None:
        if oil_1m > 5:
            score -= 1
            signals.append("Oil is rising strongly, which may increase inflation pressure.")
            pressures.append("oil / inflation pressure")
            add_breakdown(
                "Oil",
                "Rising strongly",
                "-1",
                "Higher oil can increase inflation pressure."
            )
        elif oil_1m < -5:
            signals.append("Oil is falling strongly, which may reduce inflation pressure but can also suggest weaker demand.")
            add_breakdown(
                "Oil",
                "Falling strongly",
                "0",
                "Lower oil may reduce inflation pressure, but it can also suggest weaker demand."
            )
        else:
            add_breakdown(
                "Oil",
                "Neutral",
                "0",
                "Oil is not giving a strong inflation signal."
            )
    else:
        add_breakdown(
            "Oil",
            "No data",
            "0",
            "Not enough oil data to score this signal."
        )

    if score >= 3:
        label = "Risk-On"
        summary = (
            "The dashboard is showing a risk-on regime. Growth and speculative assets are leading, "
            "while macro pressure looks manageable."
        )
    elif score <= -3:
        label = "Risk-Off"
        summary = (
            "The dashboard is showing a risk-off regime. Defensive or macro pressure signals are stronger "
            "than risk appetite."
        )
    else:
        label = "Mixed"
        summary = (
            "The dashboard is showing a mixed regime. Some risk signals are positive, "
            "but macro pressure is still present."
        )

    main_pressure = ", ".join(pressures[:2]) if pressures else "No major pressure"

    return {
        "label": label,
        "score": score,
        "main_pressure": main_pressure,
        "summary": summary,
        "signals": signals,
        "breakdown": breakdown
    }


def color_change(value):
    if pd.isna(value):
        return ""

    if value > 0:
        return "color: #22c55e; font-weight: 700"

    if value < 0:
        return "color: #ef4444; font-weight: 700"

    return "color: #9ca3af; font-weight: 700"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Dashboard Controls")

selected_asset_types = st.sidebar.multiselect(
    "Asset groups",
    sorted(set(info["type"] for info in assets.values())),
    default=sorted(set(info["type"] for info in assets.values()))
)

filtered_assets = [
    asset_name
    for asset_name, info in assets.items()
    if info["type"] in selected_asset_types
]

if not filtered_assets:
    st.warning("Select at least one asset group from the sidebar.")
    st.stop()

selected_assets = st.sidebar.multiselect(
    "Assets to compare",
    list(assets.keys()),
    default=filtered_assets
)

selected_assets = [
    asset
    for asset in selected_assets
    if asset in filtered_assets
]

if not selected_assets:
    selected_assets = filtered_assets

comparison_period = st.sidebar.selectbox(
    "Performance period",
    period_options,
    index=2
)

correlation_period = st.sidebar.selectbox(
    "Correlation period",
    period_options,
    index=2
)

deep_dive_period = st.sidebar.selectbox(
    "Deep dive chart period",
    period_options,
    index=3
)

selected_asset = st.sidebar.selectbox(
    "Deep dive asset",
    selected_assets
)
st.sidebar.divider()

st.sidebar.caption(
    "Data source: Yahoo Finance via yfinance."
)

st.sidebar.caption(
    "Data refresh: cached for 5 minutes."
)

st.sidebar.caption(
    "Regime model: rule-based, not predictive."
)

# -----------------------------
# Header
# -----------------------------
st.title("Macro + Markets Dashboard")

st.write(
    "Track major assets, cross-asset performance, volatility, trend signals, correlations, "
    "and simple market-regime interpretation."
)

tab_overview, tab_macro, tab_performance, tab_correlation, tab_deep_dive, tab_notes = st.tabs(
    ["Overview", "Macro", "Performance", "Correlation", "Deep Dive", "Project Notes"]
)


# -----------------------------
# Overview tab
# -----------------------------
with tab_overview:
    st.subheader("Market Overview")

    overview_data = [
        calculate_asset_snapshot(asset_name, assets[asset_name])
        for asset_name in selected_assets
    ]

    overview_df = pd.DataFrame(overview_data)
    valid_overview_df = overview_df.dropna(subset=["Daily % Change"])

    if valid_overview_df.empty:
        st.error("No market data found. Try another asset group or refresh later.")
        st.stop()

    best_performer = valid_overview_df.loc[
        valid_overview_df["Daily % Change"].idxmax()
    ]

    worst_performer = valid_overview_df.loc[
        valid_overview_df["Daily % Change"].idxmin()
    ]

    positive_assets = int((valid_overview_df["Daily % Change"] > 0).sum())
    total_assets = len(valid_overview_df)
    avg_volatility = valid_overview_df["20D Volatility"].dropna().mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Performer",
        best_performer["Asset"],
        f"{best_performer['Daily % Change']:.2f}%"
    )

    col2.metric(
        "Worst Performer",
        worst_performer["Asset"],
        f"{worst_performer['Daily % Change']:.2f}%"
    )

    col3.metric(
        "Market Breadth",
        f"{positive_assets}/{total_assets} positive"
    )

    col4.metric(
        "Avg 20D Volatility",
        f"{avg_volatility:.2f}%" if not pd.isna(avg_volatility) else "N/A"
    )

    regime = market_regime_read(overview_df)
    st.info(regime)

    st.subheader("Macro Regime Engine")

    macro_read = macro_regime_engine(overview_df)

    macro_col1, macro_col2, macro_col3 = st.columns(3)

    macro_col1.metric(
        "Macro Regime",
        macro_read["label"]
    )

    macro_col2.metric(
        "Regime Score",
        f"{macro_read['score']:+d}"
    )

    macro_col3.metric(
        "Main Pressure",
        macro_read["main_pressure"]
    )

    st.write(macro_read["summary"])

    

    styled_overview = (
        overview_df.style
        .format({
            "Latest Price": "{:,.2f}",
            "Daily Change": "{:,.2f}",
            "Daily % Change": "{:.2f}%",
            "1M Return": "{:.2f}%",
            "3M Return": "{:.2f}%",
            "20D Volatility": "{:.2f}%",
            "Trend Score": "{:.0f}"
        }, na_rep="N/A")
        .map(
            color_change,
            subset=[
                "Daily Change",
                "Daily % Change",
                "1M Return",
                "3M Return"
            ]
        )
    )

    st.subheader("Asset Snapshot Table")

    st.dataframe(
        styled_overview,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "20D Volatility is annualized from the latest 20 daily returns. "
        "Trend Score ranges from 0 to 3."
    )


# -----------------------------
# Macro tab
# -----------------------------
with tab_macro:
    st.subheader("Macro Dashboard")

    st.write(
        "This section connects market assets to macroeconomic meaning: "
        "rates pressure, dollar liquidity, inflation pressure, defensive demand, "
        "and risk appetite."
    )

    macro_assets = {
        "S&P 500": "Broad equity risk appetite",
        "Nasdaq": "Growth / technology risk appetite",
        "Bitcoin": "Speculative liquidity appetite",
        "US Dollar Index": "Global liquidity pressure",
        "US 10Y Yield": "Interest-rate / valuation pressure",
        "Gold": "Defensive demand / inflation concern",
        "Crude Oil": "Inflation pressure / growth demand"
    }

    macro_rows = []

    for asset_name, role in macro_assets.items():
        snapshot = calculate_asset_snapshot(asset_name, assets[asset_name])

        macro_rows.append({
            "Indicator": asset_name,
            "Macro Role": role,
            "Latest Price": snapshot["Latest Price"],
            "1M Return": snapshot["1M Return"],
            "3M Return": snapshot["3M Return"],
            "20D Volatility": snapshot["20D Volatility"],
            "Trend Label": snapshot["Trend Label"]
        })

    macro_df = pd.DataFrame(macro_rows)

    macro_read = macro_regime_engine(
        pd.DataFrame([
            calculate_asset_snapshot(asset_name, assets[asset_name])
            for asset_name in macro_assets.keys()
        ])
    )

    macro_col1, macro_col2, macro_col3 = st.columns(3)

    macro_col1.metric(
        "Macro Regime",
        macro_read["label"]
    )

    macro_col2.metric(
        "Regime Score",
        f"{macro_read['score']:+d}"
    )

    macro_col3.metric(
        "Main Pressure",
        macro_read["main_pressure"]
    )

    st.info(macro_read["summary"])

    st.subheader("1M Macro Performance")

    macro_chart_df = (
        macro_df
        .dropna(subset=["1M Return"])
        .sort_values("1M Return", ascending=True)
    )

    if macro_chart_df.empty:
        st.warning("Not enough macro data to build the 1M performance chart.")
    else:
        macro_return_fig = px.bar(
            macro_chart_df,
            x="1M Return",
            y="Indicator",
            orientation="h",
            hover_data=["Macro Role", "3M Return", "20D Volatility"],
            title="1-Month Performance by Macro Indicator"
        )

        macro_return_fig.add_vline(
            x=0,
            line_width=1,
            line_dash="dash"
        )

        macro_return_fig.update_layout(
            xaxis_title="1M Return (%)",
            yaxis_title="Macro Indicator",
            height=450
        )

        st.plotly_chart(
            macro_return_fig,
            width="stretch"
        )
        best_macro = macro_chart_df.iloc[-1]
        worst_macro = macro_chart_df.iloc[0]

        col1, col2 = st.columns(2)

        col1.metric(
            "Strongest 1M Macro Indicator",
            best_macro["Indicator"],
            f"{best_macro['1M Return']:.2f}%"
        )

        col2.metric(
            "Weakest 1M Macro Indicator",
            worst_macro["Indicator"],
            f"{worst_macro['1M Return']:.2f}%"
        )

        st.caption(
            "Large negative moves in risk-sensitive assets can pull the regime toward risk-off. "
            "Positive moves in the dollar or yields can also signal tighter macro conditions."
        )

    st.subheader("Macro Indicator Table")

    styled_macro_df = (
        macro_df.style
        .format({
            "Latest Price": "{:,.2f}",
            "1M Return": "{:.2f}%",
            "3M Return": "{:.2f}%",
            "20D Volatility": "{:.2f}%"
        }, na_rep="N/A")
        .map(
            color_change,
            subset=[
                "1M Return",
                "3M Return"
            ]
        )
    )

    st.dataframe(
        styled_macro_df,
        hide_index=True,
        width="stretch"
    )

    st.subheader("Macro Signal Breakdown")

    breakdown_df = pd.DataFrame(macro_read["breakdown"])

    st.dataframe(
        breakdown_df,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "This macro model is rule-based. It does not predict markets. "
        "It summarizes whether current cross-asset behavior looks more risk-on, risk-off, or mixed."
    )


# -----------------------------
# Performance tab
# -----------------------------
with tab_performance:
    st.subheader("Cross-Asset Performance Comparison")

    comparison_df = build_relative_performance_data(
        comparison_period,
        selected_assets
    )

    if comparison_df.empty:
        st.error("No comparison data found. Try another period.")
    else:
        comparison_fig = px.line(
            comparison_df,
            x="Date",
            y="Performance %",
            color="Asset",
            title=f"Relative performance over {comparison_period}"
        )

        comparison_fig.update_layout(
            yaxis_title="Performance since start of period (%)",
            legend_title="Asset",
            height=600
        )

        st.plotly_chart(
            comparison_fig,
            width="stretch"
        )

        final_performance = (
            comparison_df
            .sort_values("Date")
            .groupby("Asset")
            .tail(1)
            .sort_values("Performance %", ascending=False)
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Best Relative Performer",
            final_performance.iloc[0]["Asset"],
            f"{final_performance.iloc[0]['Performance %']:.2f}%"
        )

        col2.metric(
            "Worst Relative Performer",
            final_performance.iloc[-1]["Asset"],
            f"{final_performance.iloc[-1]['Performance %']:.2f}%"
        )


# -----------------------------
# Correlation tab
# -----------------------------
with tab_correlation:
    st.subheader("Correlation Matrix")

    correlation_df = build_correlation_matrix(
        correlation_period,
        selected_assets
    )

    if correlation_df.empty or correlation_df.shape[0] < 2:
        st.error("Not enough assets/data to calculate correlation.")
    else:
        correlation_fig = px.imshow(
            correlation_df,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdYlGn",
            title=f"Daily return correlation over {correlation_period}"
        )

        correlation_fig.update_layout(
            xaxis_title="Asset",
            yaxis_title="Asset",
            height=650
        )

        st.plotly_chart(
            correlation_fig,
            width="stretch"
        )

        corr_pairs = correlation_df.where(
            np.triu(np.ones(correlation_df.shape), k=1).astype(bool)
        ).stack()

        if not corr_pairs.empty:
            strongest_pair = corr_pairs.idxmax()
            weakest_pair = corr_pairs.idxmin()

            col1, col2 = st.columns(2)

            col1.metric(
                "Strongest Relationship",
                f"{strongest_pair[0]} / {strongest_pair[1]}",
                f"{corr_pairs.max():.2f}"
            )

            col2.metric(
                "Weakest Relationship",
                f"{weakest_pair[0]} / {weakest_pair[1]}",
                f"{corr_pairs.min():.2f}"
            )

        st.caption(
            "Correlation near +1 means assets often move together. "
            "Near -1 means they often move opposite. "
            "Near 0 means the relationship is weak."
        )


# -----------------------------
# Deep Dive tab
# -----------------------------
with tab_deep_dive:
    st.subheader("Asset Deep Dive")

    ticker_symbol = assets[selected_asset]["ticker"]

    data = get_price_data(
        ticker_symbol,
        period=deep_dive_period,
        interval="1d"
    )

    if data.empty:
        st.error("No data found. Try another asset or period.")
    else:
        chart_data = data.reset_index()

        date_column = "Datetime" if "Datetime" in chart_data.columns else "Date"

        chart_data["MA20"] = chart_data["Close"].rolling(window=20).mean()
        chart_data["MA50"] = chart_data["Close"].rolling(window=50).mean()

        fig = px.line(
            chart_data,
            x=date_column,
            y=["Close", "MA20", "MA50"],
            title=(
                f"{selected_asset} price with 20D and 50D moving averages "
                f"over {deep_dive_period}"
            )
        )

        fig.update_layout(
            yaxis_title="Price",
            legend_title="Series",
            height=600
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        selected_snapshot = calculate_asset_snapshot(
            selected_asset,
            assets[selected_asset]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Latest Close",
            f"{selected_snapshot['Latest Price']:,.2f}"
            if selected_snapshot["Latest Price"] is not None
            else "N/A"
        )

        col2.metric(
            "Daily % Change",
            f"{selected_snapshot['Daily % Change']:.2f}%"
            if selected_snapshot["Daily % Change"] is not None
            else "N/A"
        )

        col3.metric(
            "1M Return",
            f"{selected_snapshot['1M Return']:.2f}%"
            if selected_snapshot["1M Return"] is not None
            else "N/A"
        )

        col4.metric(
            "20D Volatility",
            f"{selected_snapshot['20D Volatility']:.2f}%"
            if selected_snapshot["20D Volatility"] is not None
            else "N/A"
        )

        st.subheader("Analyst-Style Read")

        interpretation_points = asset_interpretation(
            selected_asset,
            selected_snapshot
        )

        for point in interpretation_points:
            lower_point = point.lower()

            if (
                "cautious" in lower_point
                or "negative" in lower_point
                or "weak" in lower_point
            ):
                st.warning(point)
            elif (
                "constructive" in lower_point
                or "positive" in lower_point
                or "strong" in lower_point
            ):
                st.success(point)
            else:
                st.info(point)


# -----------------------------
# Notes tab
# -----------------------------
with tab_notes:
    st.subheader("Project Notes")

    st.write(
        "This dashboard is a macro and markets analysis tool built with Python and Streamlit. "
        "It collects public market data, calculates return and risk metrics, visualizes cross-asset behavior, "
        "and applies a simple rule-based macro regime model."
    )

    st.subheader("What the dashboard does")

    st.write(
        "- Tracks major market assets including equities, commodities, crypto, FX, rates, and the dollar.\n"
        "- Calculates daily moves, 1-month returns, 3-month returns, volatility, and trend signals.\n"
        "- Compares cross-asset performance over different time periods.\n"
        "- Builds a correlation matrix to show how assets move relative to each other.\n"
        "- Uses a rule-based macro regime engine to classify conditions as Risk-On, Risk-Off, or Mixed.\n"
        "- Provides an asset-level deep dive with moving averages and analyst-style interpretation."
    )

    st.subheader("Tools used")

    st.write(
        "- Python\n"
        "- Streamlit\n"
        "- pandas\n"
        "- NumPy\n"
        "- yfinance\n"
        "- Plotly"
    )

    st.subheader("Important note")

    st.write(
        "This dashboard is not designed to predict markets or give investment advice. "
        "The macro regime model is rule-based and educational. Its purpose is to organize market information "
        "into a structured analytical workflow."
    )

    st.subheader("Skills demonstrated")

    st.write(
        "- Financial data collection\n"
        "- Data cleaning and transformation\n"
        "- Market return and volatility analysis\n"
        "- Cross-asset comparison\n"
        "- Correlation analysis\n"
        "- Dashboard design\n"
        "- Basic macro interpretation logic"
    )