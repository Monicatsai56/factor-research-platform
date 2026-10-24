## SQL table schema and structure
import pandas as pd
from core.db import engine


def load_universe():
    return pd.read_sql(
        "SELECT ticker, company_name, sector, industry, market "
        "FROM stock_universe;",
        engine,
    )


def load_prices():
    return pd.read_sql(
        'SELECT "Date" AS date, ticker, "Open" AS open, "High" AS high, '
        '"Low" AS low, "Close" AS close, "Volume" AS volume '
        "FROM stock_prices;",
        engine,
        parse_dates=["date"],
    ).sort_values(["ticker", "date"]).reset_index(drop=True)


def load_financials():
    return pd.read_sql(
        "SELECT ticker, fiscal_year, fiscal_quarter, fiscal_period_end, "
        "reported_currency, revenue, cost_of_revenue, gross_profit, "
        "operating_income, depreciation_amortization, net_income, "
        "interest_expense, research_and_development, operating_cash_flow, "
        "capital_expenditures, free_cash_flow, cash_and_equivalents, "
        "total_assets, total_liabilities, total_debt, shares_outstanding, "
        "preferred_equity, ebitda, gross_margin, operating_margin, "
        "net_margin, eps, enterprise_value "
        "FROM stock_fundamentals;",
        engine,
        parse_dates=["fiscal_period_end"],
    ).sort_values(
        ["ticker", "fiscal_year", "fiscal_quarter"]
    ).reset_index(drop=True)


def load_all():
    return load_universe(), load_prices(), load_financials()