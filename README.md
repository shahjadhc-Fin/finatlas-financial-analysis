# FinAtlas

**Corporate Finance & Equity Research Toolkit**

FinAtlas is a Python dashboard for exploring listed companies, reviewing financial ratios, comparing selected peers, and testing basic discounted cash flow (DCF) assumptions.

## Features

- Historical adjusted-price performance, annual financial statements, and key ratios.
- Multi-year trends in revenue, earnings, margins, returns, debt, and free cash flow.
- Company comparisons and medians for selected peer groups.
- DCF valuation with editable cash-flow assumptions, an optional WACC estimate, and sensitivity analysis.
- CSV downloads for financial data, comparisons, forecasts, and sensitivity tables.

## Run locally (Windows)

Install Python, then open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run finance_analyzer_app.py
```

If PowerShell blocks virtual-environment activation, run this in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The dashboard should open in your browser. Enter ticker symbols separated by commas; examples for NSE-listed companies include `RELIANCE.NS`, `TCS.NS`, and `INFY.NS`.

## Understanding the analysis

When **Entered tickers only** is selected, the app compares the selected company with the other tickers entered. Optional peer groups add a small set of predefined listed companies and calculate a median for each available ratio. These are selected peers, not official industry-wide averages. Comparisons are most useful when companies have similar business models and reporting periods.

A comparison difference is the selected company's figure minus the peer median. Growth, margin, and return differences are measured in percentage points; multiples are shown in `x`. A higher number is not automatically better, especially for leverage ratios.

## Data and valuation limitations

Market and statement data are retrieved through `yfinance`, an unofficial interface to Yahoo Finance. Data may be incomplete, unavailable, delayed, or changed by the provider. The currency used for financial statements may differ from the currency used for the listed share price. Standardized statement rows may not match the line items or definitions in a company's official filings. Verify material values, units, and reporting periods against official reports.

The DCF model estimates enterprise value from assumed future free cash flows. It is not a price target and does not calculate equity value by adjusting for cash, debt, and diluted shares. The optional WACC estimate depends on provider data and editable assumptions. Treat the output as a research aid, not as investment advice or a substitute for analyst review.

## Technology

Python, Streamlit, pandas, NumPy, Matplotlib, and yfinance.
