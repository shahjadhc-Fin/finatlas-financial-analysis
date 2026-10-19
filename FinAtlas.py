"""Simple corporate-finance analyzer. Run: streamlit run finance_analyzer_app.py"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import streamlit as st
import yfinance as yf

INDUSTRY_PEER_GROUPS = {
    "Indian IT services — starter basket": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Indian FMCG — starter basket": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "DABUR.NS", "BRITANNIA.NS"],
    "Indian pharma — starter basket": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS"],
    "Indian private banks — starter basket": ["HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "Indian auto makers — starter basket": ["MARUTI.NS", "M&M.NS", "EICHERMOT.NS", "HEROMOTOCO.NS"],
}
NO_BENCHMARK = "None — use only entered tickers"

st.set_page_config(page_title="Company Finance Analyzer", layout="wide")
st.title("FinAtlas - Corporate Finance & Valuation Toolkit")
st.caption("Fetch company data, review financial ratios, compare tickers, and get quick written observations.")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(ticker, period):
    company = yf.Ticker(ticker)
    data = []
    for attribute, fallback in [
        ("history", pd.DataFrame()), ("info", {}),
        ("income_stmt", pd.DataFrame()), ("balance_sheet", pd.DataFrame()),
        ("cashflow", pd.DataFrame()),
    ]:
        try:
            value = company.history(period=period, auto_adjust=True) if attribute == "history" else getattr(company, attribute)
            data.append(value or fallback if isinstance(value, dict) else value)
        except Exception:
            data.append(fallback)
    return tuple(data)


def statement_currency(info):
    """Currency Yahoo reports for financial statements, distinct from quote currency."""
    return (info or {}).get("financialCurrency") or (info or {}).get("currency") or "N/A"


def market_currency(info):
    """Currency used for the listed share price and market capitalization."""
    return (info or {}).get("currency") or "N/A"


def statement_display_scale(currency):
    """Return divisor and display label for absolute financial-statement values."""
    if currency == "INR":
        return 10_000_000, "₹ crore"
    if currency and currency != "N/A":
        return 1_000_000, f"{currency} million"
    return 1, "provider units (currency unknown)"


@st.cache_data(ttl=3600, show_spinner=False)
def get_fx_rate(from_currency, to_currency):
    """Current quote-currency conversion rate: units of target currency per source unit."""
    source = str(from_currency or "N/A").upper()
    target = str(to_currency or "N/A").upper()
    if source == target and source != "N/A":
        return 1.0
    if source == "N/A" or target == "N/A":
        return np.nan

    for pair, invert in [(f"{source}{target}=X", False), (f"{target}{source}=X", True)]:
        try:
            history = yf.Ticker(pair).history(period="5d", auto_adjust=False)
            close = history["Close"].dropna() if not history.empty and "Close" in history else pd.Series(dtype=float)
            if not close.empty and float(close.iloc[-1]) > 0:
                rate = float(close.iloc[-1])
                return 1.0 / rate if invert else rate
        except Exception:
            pass
    return np.nan


def statement_value(statement, labels, year):
    """Return a value only when a source row can be matched with reasonable confidence."""
    if statement is None or statement.empty:
        return np.nan

    year_columns = []
    for column in statement.columns:
        date = pd.to_datetime(column, errors="coerce")
        if not pd.isna(date) and date.year == int(year):
            year_columns.append(column)
    if not year_columns:
        return np.nan
    column = max(year_columns, key=lambda col: pd.to_datetime(col, errors="coerce"))

    clean = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
    row_names = {clean(row): row for row in statement.index}
    chosen = None


    for label in labels:
        if clean(label) in row_names:
            chosen = row_names[clean(label)]
            break


    if chosen is None:
        for label in labels:
            target = clean(label)
            if len(target) < 8:
                continue
            matches = [row for row in statement.index if target in clean(row)]
            if len(matches) == 1:
                chosen = matches[0]
                break

    if chosen is None:
        return np.nan
    value = pd.to_numeric(statement.loc[chosen, column], errors="coerce")
    return float(value) if pd.notna(value) else np.nan


def build_financials(income, balance, cashflow):
    statements = [income, balance, cashflow]
    years = sorted({pd.Timestamp(col).year for frame in statements if frame is not None and not frame.empty
                    for col in frame.columns if not pd.isna(pd.to_datetime(col, errors="coerce"))})
    definitions = {
        "Revenue": (income, ["Total Revenue", "Operating Revenue", "Revenue"]),
        "Gross Profit": (income, ["Gross Profit"]),
        "Net Income": (income, ["Net Income", "Net Income Common Stockholders"]),
        "Operating Income": (income, ["Operating Income", "EBIT"]),
        "Interest Expense": (income, ["Interest Expense Non Operating", "Interest Expense"]),
        "Total Assets": (balance, ["Total Assets"]),
        "Current Assets": (balance, ["Current Assets", "Total Current Assets"]),
        "Current Liabilities": (balance, ["Current Liabilities", "Total Current Liabilities"]),
        "Total Debt": (balance, ["Total Debt"]),
        "Equity": (balance, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Total Stockholder Equity"]),
        "Operating Cash Flow": (cashflow, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
        "Capital Expenditure": (cashflow, ["Capital Expenditure", "Capital Expenditures"]),
    }
    rows = []
    for year in years:
        row = {"Year": year}
        for name, (frame, labels) in definitions.items():
            row[name] = statement_value(frame, labels, year)

        provider_fcf = statement_value(cashflow, ["Free Cash Flow"], year)
        ocf, capex = row["Operating Cash Flow"], row["Capital Expenditure"]
        if pd.notna(ocf) and pd.notna(capex):
            row["Free Cash Flow"] = ocf + capex if capex < 0 else ocf - capex
        else:
            row["Free Cash Flow"] = provider_fcf
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year").dropna(how="all") if rows else pd.DataFrame()


def divide(a, b):
    return a / b if pd.notna(a) and pd.notna(b) and b != 0 else np.nan


def calc_growth(current, previous, consecutive=True):
    return current / previous - 1 if consecutive and pd.notna(current) and pd.notna(previous) and previous > 0 else np.nan


def dcf_enterprise_value(starting_fcf, growth, wacc, terminal_growth, years):
    """Simple unlevered FCF DCF; returns EV and forecast rows, or N/A if assumptions are invalid."""
    if not all(np.isfinite(x) for x in [starting_fcf, growth, wacc, terminal_growth]):
        return np.nan, pd.DataFrame()
    if starting_fcf <= 0 or wacc <= terminal_growth or wacc <= -1 or years < 1:
        return np.nan, pd.DataFrame()

    rows = []
    present_value_forecast = 0.0
    for year in range(1, years + 1):
        forecast_fcf = starting_fcf * (1 + growth) ** year
        present_value = forecast_fcf / (1 + wacc) ** year
        present_value_forecast += present_value
        rows.append({"Forecast year": year, "Forecast free cash flow": forecast_fcf,
                     "Present value": present_value})

    terminal_fcf = rows[-1]["Forecast free cash flow"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    present_value_terminal = terminal_value / (1 + wacc) ** years
    return present_value_forecast + present_value_terminal, pd.DataFrame(rows)


with st.sidebar:
    st.header("Inputs")
    text = st.text_input("Ticker symbols (comma-separated)", "RELIANCE.NS, TCS.NS")
    period = st.selectbox("Share-price history", ["1y", "3y", "5y", "10y", "max"], index=2)
    benchmark_sector = st.selectbox("Optional benchmark basket", [NO_BENCHMARK, *INDUSTRY_PEER_GROUPS.keys()])
    st.caption("Starter baskets add example listed peers automatically. Choose one that matches your company; the app cannot verify business-segment similarity.")
    st.caption("Indian NSE tickers often end in .NS. Check the exact symbol if no data appears.")

entered_tickers = list(dict.fromkeys(t.strip().upper() for t in text.split(",") if t.strip()))[:5]
if not entered_tickers:
    st.info("Enter at least one ticker in the sidebar.")
    st.stop()

benchmark_peer_candidates = INDUSTRY_PEER_GROUPS.get(benchmark_sector, [])
tickers = list(dict.fromkeys(entered_tickers + benchmark_peer_candidates))

prices, infos, annuals, market_stats, ratios_by_ticker, source_statements = {}, {}, {}, {}, {}, {}
with st.spinner("Fetching available data..."):
    for ticker in tickers:
        history, info, income, balance, cashflow = fetch_data(ticker, period)
        prices[ticker] = history.sort_index().dropna(how="all") if not history.empty else pd.DataFrame()
        infos[ticker] = info if isinstance(info, dict) else {}
        source_statements[ticker] = {"income": income, "balance": balance, "cashflow": cashflow}
        annuals[ticker] = build_financials(income, balance, cashflow)

        close = prices[ticker]["Close"].dropna() if "Close" in prices[ticker] else pd.Series(dtype=float)
        if len(close) > 1:
            returns = close.pct_change().dropna()
            market_stats[ticker] = {
                "Latest adjusted close": close.iloc[-1],
                "Adjusted-price return": close.iloc[-1] / close.iloc[0] - 1 if close.iloc[0] != 0 else np.nan,
                "Annualized volatility": returns.std() * np.sqrt(252),
                "Maximum drawdown": (close / close.cummax() - 1).min(),
            }
        else:
            market_stats[ticker] = {}

        ratios = {}
        annual = annuals[ticker]
        if not annual.empty:
            latest_year = int(annual.index[-1])
            has_previous_year = (latest_year - 1) in annual.index
            last = annual.loc[latest_year]
            prev = annual.loc[latest_year - 1] if has_previous_year else pd.Series(dtype=float)
            avg_equity = (last["Equity"] + prev.get("Equity", np.nan)) / 2 if has_previous_year and pd.notna(last["Equity"]) and pd.notna(prev.get("Equity", np.nan)) else np.nan
            avg_assets = (last["Total Assets"] + prev.get("Total Assets", np.nan)) / 2 if has_previous_year and pd.notna(last["Total Assets"]) and pd.notna(prev.get("Total Assets", np.nan)) else np.nan
            ratios = {
                "Revenue growth (YoY)": calc_growth(last["Revenue"], prev.get("Revenue", np.nan), has_previous_year),
                "Net income growth (YoY)": calc_growth(last["Net Income"], prev.get("Net Income", np.nan), has_previous_year),
                "Gross profit margin": divide(last["Gross Profit"], last["Revenue"]),
                "Operating margin": divide(last["Operating Income"], last["Revenue"]),
                "Net profit margin": divide(last["Net Income"], last["Revenue"]),
                "ROE (approx.)": divide(last["Net Income"], avg_equity),
                "ROA (approx.)": divide(last["Net Income"], avg_assets),
                "Current ratio": divide(last["Current Assets"], last["Current Liabilities"]),
                "Debt / equity": divide(last["Total Debt"], last["Equity"]),
                "Interest coverage (approx.)": divide(last["Operating Income"], abs(last["Interest Expense"])) if pd.notna(last["Interest Expense"]) else np.nan,
                "Free cash flow margin": divide(last["Free Cash Flow"], last["Revenue"]),
            }
        ratios_by_ticker[ticker] = ratios

usable = [t for t in tickers if not prices[t].empty or not annuals[t].empty]
entered_usable = [t for t in entered_tickers if t in usable]
for ticker in set(entered_tickers) - set(entered_usable):
    st.warning(f"No data returned for entered ticker {ticker}. Check the ticker or try again later.")
for ticker in set(benchmark_peer_candidates) - set(usable):
    st.caption(f"Benchmark data unavailable for {ticker}; it will be excluded from the basket median.")
if not entered_usable:
    st.error("No usable data was returned for the ticker(s) you entered. The provider may be unavailable or may not cover these tickers right now.")
    st.stop()

main = entered_usable[0]
name = infos[main].get("longName") or infos[main].get("shortName") or main
st.subheader(f"{name} ({main})")
main_statement_currency = statement_currency(infos[main])
main_market_currency = market_currency(infos[main])
st.caption(
    f"Provider financial-statement currency: {main_statement_currency}; listed share-price currency: {main_market_currency}. "
    "Yahoo's standardized statement rows may use a different currency from the listed share. Verify material values against official filings."
)

stats = market_stats[main]
cols = st.columns(4)
metric_items = [
    ("Latest adjusted close", "Latest adjusted close", "{:,.2f}"),
    (f"Adjusted-price return ({period})", "Adjusted-price return", "{:.1%}"),
    (f"Annualized volatility ({period})", "Annualized volatility", "{:.1%}"),
    (f"Maximum drawdown ({period})", "Maximum drawdown", "{:.1%}"),
]
for col, (label, key, fmt) in zip(cols, metric_items):
    value = stats.get(key, np.nan)
    col.metric(label, fmt.format(value) if pd.notna(value) else "N/A")
if infos[main].get("currency"):
    st.caption(f"Market-price values are quoted in {infos[main]['currency']}. Adjusted-price return uses adjusted historical closes; it is backward-looking, not a forecast.")

price_tab, finance_tab, compare_tab, valuation_tab = st.tabs(["Share-price performance", "Financial analysis", "Compare companies", "DCF valuation"])
with price_tab:
    st.write("Adjusted-price lines are indexed to 100 at the start of the selected period, making historical performance easier to compare. Adjusted prices may reflect corporate actions and distributions.")
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = 0
    for ticker in entered_usable:
        close = prices[ticker]["Close"].dropna() if "Close" in prices[ticker] else pd.Series(dtype=float)
        if not close.empty:
            ax.plot(close.index, close / close.iloc[0] * 100, label=ticker)
            plotted += 1
    if plotted:
        ax.axhline(100, linestyle="--", linewidth=0.8)
        ax.set_ylabel("Indexed adjusted close (start = 100)")
        ax.set_xlabel("Date")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    market_table = pd.DataFrame({t: market_stats[t] for t in entered_usable}).T
    if not market_table.empty and len(market_table.columns):
        st.dataframe(market_table.style.format("{:.1%}", subset=["Adjusted-price return", "Annualized volatility", "Maximum drawdown"], na_rep="N/A"), use_container_width=True)
    else:
        st.info("Not enough share-price history was returned to calculate market statistics.")

with finance_tab:
    available = [t for t in entered_usable if not annuals[t].empty]
    if not available:
        st.info("Annual financial statements are not available for these tickers from this provider.")
    else:
        selected = st.selectbox("Company", available)
        annual, ratios = annuals[selected], ratios_by_ticker[selected]
        currency = statement_currency(infos[selected])
        quote_currency = market_currency(infos[selected])
        statement_scale, statement_unit = statement_display_scale(currency)
        latest_year = int(annual.index[-1])
        st.subheader("Annual financial statements (selected rows)")
        display_names = {
            "Revenue": "Provider-reported revenue",
            "Gross Profit": "Provider-reported gross profit",
            "Net Income": "Provider-reported net income",
            "Operating Income": "Provider-reported operating income",
        }
        annual_display = (annual / statement_scale).rename(columns=display_names)
        st.caption(
            f"Values are shown in {statement_unit}. Yahoo financial statements report in {currency}, while the listed share price is quoted in {quote_currency}. "
            "No conversion is applied to the statement table itself. Provider-standardized lines may differ from similarly named items in official filings; verify material values against company reports. The CSV retains raw provider amounts."
        )
        st.dataframe(annual_display.style.format("{:,.2f}", na_rep="N/A"), use_container_width=True)
        st.download_button("Download raw annual data CSV", annual.to_csv().encode("utf-8"), f"{selected}_annual_financials.csv", "text/csv")
        with st.expander("Data coverage and calculation notes"):
            latest_values = annual.loc[latest_year]
            core_fields = ["Revenue", "Net Income", "Operating Income", "Total Assets", "Current Assets",
                           "Current Liabilities", "Total Debt", "Equity", "Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"]
            coverage = pd.DataFrame({
                "Input": core_fields,
                f"FY{latest_year} availability": ["Available" if pd.notna(latest_values.get(field, np.nan)) else "Missing" for field in core_fields]
            })
            st.dataframe(coverage, hide_index=True, use_container_width=True)
            st.caption("Exact provider row names are preferred. Ambiguous partial row matches are left missing rather than guessed. A missing input may make related ratios unavailable.")

        st.subheader("Historical financial trends")
        trend_data = pd.DataFrame(index=annual.index)
        prior_revenue = annual["Revenue"].shift(1)
        prior_income = annual["Net Income"].shift(1)
        consecutive_year = pd.Series(annual.index, index=annual.index).diff().eq(1)
        trend_data["Revenue growth (YoY)"] = (annual["Revenue"] / prior_revenue - 1).where((prior_revenue > 0) & consecutive_year)
        trend_data["Net income growth (YoY)"] = (annual["Net Income"] / prior_income - 1).where((prior_income > 0) & consecutive_year)
        trend_data["Gross profit margin"] = annual["Gross Profit"] / annual["Revenue"].replace(0, np.nan)
        trend_data["Operating margin"] = annual["Operating Income"] / annual["Revenue"].replace(0, np.nan)
        trend_data["Net profit margin"] = annual["Net Income"] / annual["Revenue"].replace(0, np.nan)
        avg_equity_by_year = ((annual["Equity"] + annual["Equity"].shift(1)) / 2).where(consecutive_year)
        avg_assets_by_year = ((annual["Total Assets"] + annual["Total Assets"].shift(1)) / 2).where(consecutive_year)
        trend_data["ROE (approx.)"] = annual["Net Income"] / avg_equity_by_year.replace(0, np.nan)
        trend_data["ROA (approx.)"] = annual["Net Income"] / avg_assets_by_year.replace(0, np.nan)
        trend_data["Current ratio"] = annual["Current Assets"] / annual["Current Liabilities"].replace(0, np.nan)
        trend_data["Debt / equity"] = annual["Total Debt"] / annual["Equity"].replace(0, np.nan)
        trend_data["Free cash flow margin"] = annual["Free Cash Flow"] / annual["Revenue"].replace(0, np.nan)

        trend_scale, trend_unit = statement_display_scale(currency)
        trend_data[f"Provider revenue ({trend_unit})"] = annual["Revenue"] / trend_scale
        trend_data[f"Provider net income ({trend_unit})"] = annual["Net Income"] / trend_scale
        trend_data[f"Operating income ({trend_unit})"] = annual["Operating Income"] / trend_scale
        trend_data[f"Free cash flow ({trend_unit})"] = annual["Free Cash Flow"] / trend_scale
        trend_data = trend_data.replace([np.inf, -np.inf], np.nan)

        trend_metric = st.selectbox("Metric to chart over time", list(trend_data.columns))
        trend_series = trend_data[trend_metric].dropna()
        if len(trend_series) >= 1:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            trend_years = trend_series.index.astype(int)
            ax.plot(trend_years, trend_series.values, marker="o")
            ax.set_title(f"{trend_metric} — {selected}")
            ax.set_xlabel("Fiscal year")
            ax.set_xticks(trend_years)
            if trend_metric in ["Revenue growth (YoY)", "Net income growth (YoY)", "Gross profit margin", "Operating margin", "Net profit margin", "ROE (approx.)", "ROA (approx.)", "Free cash flow margin"]:
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.set_ylabel("Percentage")
            elif trend_metric in ["Current ratio", "Debt / equity"]:
                ax.set_ylabel("Multiple (x)")
            else:
                ax.set_ylabel(trend_unit)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough historical data is available for this metric.")
        st.caption("Trend calculations are historical, not forecasts. Growth is left as N/A when the previous value is zero, negative, or missing.")

        st.subheader(f"Latest available ratios (FY{latest_year})")
        percent_names = [m for m in ratios if any(x in m.lower() for x in ["growth", "margin", "roe", "roa"])]
        ratio_table = pd.DataFrame({"Metric": list(ratios.keys()), "Value": [
            f"{value:.1%}" if name in percent_names and pd.notna(value) else
            (f"{value:.2f}x" if pd.notna(value) else "N/A")
            for name, value in ratios.items()
        ]}).set_index("Metric")
        st.dataframe(ratio_table, use_container_width=True)

        st.subheader("Quick interpretation")
        notes = []
        growth = ratios.get("Revenue growth (YoY)", np.nan)
        if pd.notna(growth):
            notes.append(f"Revenue {'increased' if growth >= 0 else 'decreased'} by {abs(growth):.1%} versus the previous annual period.")
        for label, sentence in [("Operating margin", "Operating margin was"), ("Net profit margin", "Net profit margin was")]:
            value = ratios.get(label, np.nan)
            if pd.notna(value): notes.append(f"{sentence} {value:.1%}.")
        for label, sentence in [("Current ratio", "Current ratio was"), ("Debt / equity", "Debt-to-equity was")]:
            value = ratios.get(label, np.nan)
            if pd.notna(value): notes.append(f"{sentence} {value:.2f}x.")
        cash = annual.iloc[-1].get("Free Cash Flow", np.nan)
        if pd.notna(cash):
            cash_display = cash / statement_scale
            notes.append(f"Latest free cash flow was {cash_display:,.2f} {trend_unit} ({'positive' if cash >= 0 else 'negative'}).")
        ret = market_stats.get(selected, {}).get("Adjusted-price return", np.nan)
        if pd.notna(ret): notes.append(f"Adjusted-price return for the selected chart period was {ret:.1%}; this is historical, not a forecast.")
        st.markdown("\n".join(f"- {note}" for note in notes) if notes else "Not enough data was available to create a summary.")
        st.caption("Ratios are simplified calculations from provider-standardized rows. Missing inputs appear as N/A, not zero; verify line-item definitions against official filings before using the results in research.")

with compare_tab:
    rows = []
    for ticker in entered_usable:
        stats, ratio, info = market_stats[ticker], ratios_by_ticker[ticker], infos[ticker]
        currency = info.get("currency") or "N/A"
        market_cap = info.get("marketCap", np.nan)
        if currency == "INR" and pd.notna(market_cap):
            market_cap = market_cap / 10_000_000
        rows.append({"Ticker": ticker, "Company": info.get("shortName") or info.get("longName") or ticker,
                     "Currency": currency, "Market cap (local units; INR in crore)": market_cap,
                     f"Adjusted-price return ({period})": stats.get("Adjusted-price return", np.nan),
                     "Revenue growth": ratio.get("Revenue growth (YoY)", np.nan),
                     "Operating margin": ratio.get("Operating margin", np.nan),
                     "Net profit margin": ratio.get("Net profit margin", np.nan),
                     "Current ratio": ratio.get("Current ratio", np.nan), "Debt / equity": ratio.get("Debt / equity", np.nan)})
    comparison = pd.DataFrame(rows).set_index("Ticker")
    comparison_currencies = set(comparison["Currency"].dropna())
    if len(comparison_currencies - {"N/A"}) > 1 or "N/A" in comparison_currencies:
        st.warning("These companies report in different or unknown currencies. Market caps and absolute financial values are not FX-adjusted, so do not compare their sizes directly.")
    st.caption("This table shows the tickers you entered. If you selected a starter peer basket, its additional tickers are used in the benchmark table below, not mixed into this table. Financial ratios use each company's latest available annual statement; fiscal year-ends and line-item definitions may differ. Market capitalizations are shown in each provider currency, with INR values converted to crore; no FX conversion is applied.")
    st.dataframe(comparison.style.format({"Market cap (local units; INR in crore)": "{:,.2f}", f"Adjusted-price return ({period})": "{:.1%}", "Revenue growth": "{:.1%}",
                                           "Operating margin": "{:.1%}", "Net profit margin": "{:.1%}",
                                           "Current ratio": "{:.2f}x", "Debt / equity": "{:.2f}x"}, na_rep="N/A"), use_container_width=True)
    st.download_button("Download comparison CSV", comparison.to_csv().encode("utf-8"), "company_comparison.csv", "text/csv")

    st.subheader("Company vs comparison benchmark")
    benchmark_tickers = [t for t in entered_usable if any(pd.notna(v) for v in ratios_by_ticker[t].values())]
    if not benchmark_tickers:
        st.info("No entered company has enough financial data for ratio comparison.")
    else:
        benchmark_company = st.selectbox("Company to benchmark", benchmark_tickers, key="benchmark_company")
        if benchmark_sector == NO_BENCHMARK:
            peer_tickers = [t for t in benchmark_tickers if t != benchmark_company]
            benchmark_label = "Other tickers you entered"
            st.caption(
                "The comparison median uses only the other companies you entered. With one other company, "
                "the median is simply that company's value; with several, it is the middle value. "
                "This is a user-selected comparison set, not an industry benchmark."
            )
        else:
            peer_tickers = [t for t in benchmark_peer_candidates if t in usable and t != benchmark_company and any(pd.notna(v) for v in ratios_by_ticker[t].values())]
            benchmark_label = "Selected basket median"
            if benchmark_company not in benchmark_peer_candidates:
                st.warning("The selected company is not one of the starter-basket tickers. Confirm that its main business is comparable to this basket before interpreting the differences.")
            st.caption(
                f"The {benchmark_sector} median is calculated live from the available tickers in this starter basket: "
                f"{', '.join(benchmark_peer_candidates)}. The selected company is excluded from its own median if it appears in this list. "
                "This is a small selected-company sample, not a whole-industry average. Make sure the basket matches the selected company's main business; this is not automatically verified."
            )

        if not peer_tickers:
            st.warning("No other companies with usable ratios were available for this comparison. Try another ticker or benchmark basket.")
        else:
            st.write("**How to read the table:** Difference = selected-company value minus the comparison median. For growth, margins, ROE and ROA, the difference is in percentage points (pp): 12% versus 10% = +2.0 pp. For ratios such as current ratio and debt/equity, it is shown in x. A positive difference means the number is higher—not automatically better.")
            if benchmark_sector == "Indian private banks — starter basket":
                benchmark_metrics = ["Revenue growth (YoY)", "Net profit margin", "ROE (approx.)", "ROA (approx.)"]
                st.caption("For banks, this comparison limits itself to broad growth and return measures. Bank-specific measures such as net interest margin, non-performing assets and capital adequacy are not reliably mapped in this simple version.")
            else:
                benchmark_metrics = ["Revenue growth (YoY)", "Gross profit margin", "Operating margin", "Net profit margin",
                                     "ROE (approx.)", "ROA (approx.)", "Current ratio", "Debt / equity", "Free cash flow margin"]
            benchmark_rows = []
            for metric_name in benchmark_metrics:
                company_value = ratios_by_ticker[benchmark_company].get(metric_name, np.nan)
                peer_values = pd.Series([ratios_by_ticker[t].get(metric_name, np.nan) for t in peer_tickers], dtype=float).dropna()
                peer_median = peer_values.median() if not peer_values.empty else np.nan
                difference = company_value - peer_median if pd.notna(company_value) and pd.notna(peer_median) else np.nan
                is_percentage = any(word in metric_name.lower() for word in ["growth", "margin", "roe", "roa"])
                if is_percentage:
                    value_fmt = lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
                    diff_fmt = lambda x: f"{x * 100:+.1f} pp" if pd.notna(x) else "N/A"
                    comparison_note = (f"{abs(difference) * 100:.1f} pp {'above' if difference > 0 else 'below'} median" if pd.notna(difference) and difference != 0 else ("At median" if pd.notna(difference) else "N/A"))
                else:
                    value_fmt = lambda x: f"{x:.2f}x" if pd.notna(x) else "N/A"
                    diff_fmt = lambda x: f"{x:+.2f}x" if pd.notna(x) else "N/A"
                    comparison_note = (f"{abs(difference):.2f}x {'above' if difference > 0 else 'below'} median" if pd.notna(difference) and difference != 0 else ("At median" if pd.notna(difference) else "N/A"))
                benchmark_rows.append({"Metric": metric_name, benchmark_company: value_fmt(company_value),
                                       benchmark_label: value_fmt(peer_median), "Difference (selected − median)": diff_fmt(difference),
                                       "Relative position": comparison_note})
            benchmark_table = pd.DataFrame(benchmark_rows).set_index("Metric")
            st.dataframe(benchmark_table, use_container_width=True)
            st.download_button("Download benchmark comparison CSV", benchmark_table.to_csv().encode("utf-8"), f"{benchmark_company}_benchmark_comparison.csv", "text/csv")
            st.caption(f"Median uses up to {len(peer_tickers)} available comparison companies: {', '.join(peer_tickers)}. Ratios come from provider-standardized rows and may use different fiscal periods or definitions. Higher is not inherently better for every metric (for example, leverage).")

    metric = st.selectbox("Annual metric to plot", ["Revenue", "Net Income", "Operating Income", "Free Cash Flow", "Total Assets", "Total Debt", "Equity"])
    plot_tickers = [t for t in entered_usable if metric in annuals[t] and annuals[t][metric].notna().any()]
    plot_currencies = {statement_currency(infos[t]) for t in plot_tickers}
    if len(plot_currencies) > 1 or "N/A" in plot_currencies:
        st.warning("This chart is hidden because the selected companies have different or unknown financial-statement currencies. Compare absolute values only after currency and reporting-basis checks.")
    elif plot_tickers:
        chart_currency = next(iter(plot_currencies))
        scale, chart_unit = statement_display_scale(chart_currency)
        fig, ax = plt.subplots(figsize=(10, 4))
        for ticker in plot_tickers:
            annual = annuals[ticker]
            ax.plot(annual.index, annual[metric] / scale, marker="o", label=ticker)
        ax.set_title(f"Annual {metric} by reported fiscal year")
        ax.set_xlabel("Fiscal year")
        ax.set_ylabel(chart_unit)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No annual values are available for this metric.")

with valuation_tab:
    st.subheader("Simple discounted cash flow (DCF)")
    valuation_tickers = [t for t in entered_usable if not annuals[t].empty]
    if not valuation_tickers:
        st.info("Annual financial statements are required to start a DCF analysis.")
    else:
        dcf_ticker = st.selectbox("Company to value", valuation_tickers, key="dcf_company")
        dcf_annual = annuals[dcf_ticker]
        dcf_currency = statement_currency(infos[dcf_ticker])
        dcf_market_currency = market_currency(infos[dcf_ticker])
        dcf_scale, dcf_unit = statement_display_scale(dcf_currency)
        latest_year = int(dcf_annual.index[-1])
        starting_fcf_raw = dcf_annual.iloc[-1].get("Free Cash Flow", np.nan)
        default_fcf = float(starting_fcf_raw / dcf_scale) if pd.notna(starting_fcf_raw) else 0.0
        st.caption(
            f"Starting annual FCF uses FY{latest_year}. Financial statement values are in Yahoo's reported currency ({dcf_currency}), "
            f"not necessarily the share's listing currency ({dcf_market_currency}). Amounts are displayed in {dcf_unit}. "
            "When both OCF and capex are available, FCF is calculated as OCF minus capex; otherwise the provider FCF row is used. "
            "Review source rows and official filings before relying on the DCF."
        )

        market_cap_quote_currency = infos[dcf_ticker].get("marketCap", np.nan)
        fx_market_to_statement = get_fx_rate(dcf_market_currency, dcf_currency)
        market_equity = (market_cap_quote_currency * fx_market_to_statement
                         if pd.notna(market_cap_quote_currency) and pd.notna(fx_market_to_statement) else np.nan)
        total_debt = dcf_annual.iloc[-1].get("Total Debt", np.nan)
        estimated_wacc_pct = np.nan
        wacc_key = f"dcf_wacc_v2_{dcf_ticker}"

        with st.expander("Check underlying cash-flow source rows"):
            cashflow_source = source_statements.get(dcf_ticker, {}).get("cashflow", pd.DataFrame())
            if cashflow_source is None or cashflow_source.empty:
                st.warning("No raw cash-flow statement was returned by the data provider.")
            else:
                clean_label = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
                patterns = ("freecashflow", "operatingcashflow", "cashflowfromcontinuingoperatingactivities", "capitalexpenditure")
                matching_rows = [row for row in cashflow_source.index if any(pattern in clean_label(row) for pattern in patterns)]
                if matching_rows:
                    raw_cashflow_rows = cashflow_source.loc[matching_rows].copy()
                else:
                    raw_cashflow_rows = cashflow_source.head(25).copy()
                    st.caption("No obvious FCF / OCF / capex row labels matched; showing the first 25 raw rows instead.")
                def date_sort_key(column):
                    date = pd.to_datetime(column, errors="coerce")
                    return date.timestamp() if not pd.isna(date) else float("-inf")
                ordered_cols = sorted(raw_cashflow_rows.columns, key=date_sort_key, reverse=True)
                if ordered_cols:
                    raw_cashflow_rows = raw_cashflow_rows[ordered_cols[:5]]
                st.caption(
                    f"These are raw provider values in Yahoo's financial-statement currency ({dcf_currency}), not {dcf_unit}. "
                    "Inspect the row labels and reporting year; provider-standardized values may differ from official filing definitions."
                )
                st.dataframe(raw_cashflow_rows, use_container_width=True)

                provider_fcf = statement_value(cashflow_source, ["Free Cash Flow"], latest_year)
                ocf = dcf_annual.iloc[-1].get("Operating Cash Flow", np.nan)
                capex = dcf_annual.iloc[-1].get("Capital Expenditure", np.nan)
                calculated_fcf = ocf + capex if pd.notna(ocf) and pd.notna(capex) and capex < 0 else (
                    ocf - capex if pd.notna(ocf) and pd.notna(capex) else np.nan
                )
                if pd.notna(provider_fcf):
                    st.write(f"Provider's direct Free Cash Flow row: **{provider_fcf / dcf_scale:,.2f} {dcf_unit}**")
                if pd.notna(calculated_fcf):
                    st.write(f"Calculated operating cash flow less capex: **{calculated_fcf / dcf_scale:,.2f} {dcf_unit}**")
                if pd.notna(provider_fcf) and pd.notna(calculated_fcf):
                    denominator = max(abs(provider_fcf), abs(calculated_fcf), 1.0)
                    discrepancy = abs(provider_fcf - calculated_fcf) / denominator
                    if discrepancy > 0.20:
                        st.warning(
                            f"The provider's direct FCF row differs from OCF less capex by {discrepancy:.0%}. "
                            "The DCF uses OCF less capex where both inputs are available. Verify units and definitions against the company's filing."
                        )

        if pd.notna(market_equity) and market_equity > 0 and pd.notna(starting_fcf_raw) and starting_fcf_raw > 0:
            fcf_to_market_cap = starting_fcf_raw / market_equity
            if fcf_to_market_cap < 0.005:
                st.warning(
                    f"FCF is only {fcf_to_market_cap:.2%} of market capitalization, unusually low as a scale check. "
                    "Inspect the raw cash-flow rows, statement currency and official filing before relying on this DCF. This threshold is a warning, not a rule."
                )
            elif fcf_to_market_cap > 0.20:
                st.warning(
                    f"FCF is {fcf_to_market_cap:.1%} of market capitalization, unusually high as a scale check. "
                    "Verify the source values, signs and currencies. This threshold is a warning, not a rule."
                )
        elif pd.notna(starting_fcf_raw) and starting_fcf_raw < 0:
            st.warning("Latest FCF is negative. This simple growing-FCF DCF cannot value negative starting cash flow; use a well-supported normalized forecast or a different model.")
        elif dcf_market_currency != dcf_currency:
            st.warning(
                f"Could not obtain a current FX rate from {dcf_market_currency} to {dcf_currency}. "
                "Currency-sensitive checks and the automatic WACC estimate are disabled; enter WACC manually and verify currencies."
            )


        with st.expander("Optional: estimate WACC from capital structure"):
            ke_col, kd_col, tax_col = st.columns(3)
            cost_equity_pct = ke_col.number_input("Cost of equity (%)", min_value=0.1, max_value=50.0, value=12.0, step=0.5, key=f"wacc_cost_equity_{dcf_ticker}")
            cost_debt_pct = kd_col.number_input("Pre-tax cost of debt (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key=f"wacc_cost_debt_{dcf_ticker}")
            tax_pct = tax_col.number_input("Tax rate (%)", min_value=0.0, max_value=60.0, value=25.0, step=1.0, key=f"wacc_tax_rate_{dcf_ticker}")
            estimate_available = pd.notna(market_equity) and market_equity > 0 and pd.notna(total_debt) and total_debt >= 0 and market_equity + total_debt > 0
            if estimate_available:
                equity_weight = market_equity / (market_equity + total_debt)
                debt_weight = total_debt / (market_equity + total_debt)
                estimated_wacc_pct = equity_weight * cost_equity_pct + debt_weight * cost_debt_pct * (1 - tax_pct / 100)
                st.write(f"Estimated WACC: **{estimated_wacc_pct:.2f}%**")
                st.caption(
                    f"Uses market capitalization converted from {dcf_market_currency} to statement currency {dcf_currency} where needed, plus provider-reported debt in {dcf_currency}. "
                    "Review the FX rate and both source values; this is an estimate, not a verified market-data model."
                )
                if st.button("Use estimated WACC", key=f"use_estimated_wacc_{dcf_ticker}"):
                    st.session_state[wacc_key] = float(min(50.0, max(0.1, estimated_wacc_pct)))
                    st.success("Estimated WACC applied to the DCF discount-rate field.")
            else:
                st.warning("Market capitalization or total debt is missing, so an estimated WACC is unavailable. The discount-rate field remains editable.")

        fcf_col, growth_col, wacc_col, terminal_col, years_col = st.columns(5)
        starting_fcf = fcf_col.number_input("Starting annual FCF", value=default_fcf, step=max(1.0, abs(default_fcf) * 0.05), format="%.2f", key=f"dcf_starting_fcf_v3_{dcf_ticker}")
        growth_pct = growth_col.number_input("Annual FCF growth (%)", min_value=-50.0, max_value=100.0, value=5.0, step=1.0, key=f"dcf_growth_{dcf_ticker}")
        initial_wacc = float(min(50.0, max(0.1, estimated_wacc_pct))) if pd.notna(estimated_wacc_pct) else 10.0
        wacc_pct = wacc_col.number_input("WACC / discount rate (%)", min_value=0.1, max_value=50.0, value=initial_wacc, step=0.5, key=wacc_key)
        terminal_pct = terminal_col.number_input("Terminal growth (%)", min_value=-10.0, max_value=20.0, value=3.0, step=0.5, key=f"dcf_terminal_growth_{dcf_ticker}")
        forecast_years = years_col.selectbox("Forecast years", [3, 5, 7, 10], index=1, key=f"dcf_years_{dcf_ticker}")

        growth_rate = growth_pct / 100
        wacc_rate = wacc_pct / 100
        terminal_rate = terminal_pct / 100
        if wacc_rate <= terminal_rate:
            st.error("The discount rate (WACC) must be greater than terminal growth for the DCF formula to work.")
        elif starting_fcf <= 0:
            st.warning("This simple growing-FCF DCF requires a positive starting free cash flow. If current FCF is negative or unreliable, use a justified normalized positive cash-flow estimate or a more suitable forecast model.")
        else:
            enterprise_value, forecast_df = dcf_enterprise_value(starting_fcf, growth_rate, wacc_rate, terminal_rate, forecast_years)
            present_forecast = forecast_df["Present value"].sum()
            last_forecast_fcf = forecast_df.iloc[-1]["Forecast free cash flow"]
            terminal_value = last_forecast_fcf * (1 + terminal_rate) / (wacc_rate - terminal_rate)
            present_terminal = terminal_value / (1 + wacc_rate) ** forecast_years
            forecast_display = forecast_df.round(2)
            dcf_cols = st.columns(3)
            dcf_cols[0].metric("Illustrative enterprise value", f"{enterprise_value:,.2f} {dcf_unit}")
            dcf_cols[1].metric("PV of forecast cash flows", f"{present_forecast:,.2f} {dcf_unit}")
            dcf_cols[2].metric("PV of terminal value", f"{present_terminal:,.2f} {dcf_unit}")
            st.dataframe(forecast_display.set_index("Forecast year"), use_container_width=True)
            st.download_button("Download DCF forecast CSV", forecast_df.to_csv(index=False).encode("utf-8"), f"{dcf_ticker}_dcf_forecast.csv", "text/csv")

            st.subheader("DCF sensitivity analysis")
            st.caption("Enterprise value under different discount-rate and terminal-growth assumptions; all other assumptions remain unchanged.")
            wacc_scenarios = sorted(set([max(0.001, wacc_rate - 0.02), max(0.001, wacc_rate - 0.01), wacc_rate, min(0.50, wacc_rate + 0.01), min(0.50, wacc_rate + 0.02)]))
            terminal_scenarios = sorted(set([terminal_rate - 0.01, terminal_rate, terminal_rate + 0.01]))
            sensitivity = []
            for scenario_wacc in wacc_scenarios:
                result = {"WACC vs terminal growth": f"{scenario_wacc:.1%}"}
                for scenario_terminal in terminal_scenarios:
                    label = f"{scenario_terminal:.1%}"
                    scenario_ev, _ = dcf_enterprise_value(
                        starting_fcf, growth_rate, scenario_wacc, scenario_terminal, forecast_years
                    )
                    result[label] = scenario_ev
                sensitivity.append(result)
            sensitivity_df = pd.DataFrame(sensitivity).set_index("WACC vs terminal growth")
            st.dataframe(sensitivity_df.style.format("{:,.2f}", na_rep="N/A"), use_container_width=True)
            st.download_button("Download DCF sensitivity CSV", sensitivity_df.to_csv().encode("utf-8"), f"{dcf_ticker}_dcf_sensitivity.csv", "text/csv")

        st.caption("This is an assumption-driven illustration, not a price target. The model estimates enterprise value from forecast free cash flow; it does not convert enterprise value into equity value by adding cash and subtracting debt. WACC, growth assumptions, cash-flow quality and provider data require analyst review.")

st.divider()
st.caption("Yahoo Finance data through yfinance is unofficial and may be incomplete. Industry ratios vary; this is a research/education tool, not investment advice.")
