#### factor calculation
import pandas as pd
import numpy as np


## Helper for cross-period (history) factors: find same quarter, N years ago
def _same_quarter_n_years_ago(history, current_row, n_years=1):
    target_year = current_row["fiscal_year"] - n_years
    target_q = current_row["fiscal_quarter"]
    match = history[(history["fiscal_year"] == target_year)
                    & (history["fiscal_quarter"] == target_q)]
    if match.empty:
        return None
    return match.iloc[-1]


## Helper for growth factors: YoY growth of one field (same quarter last year).
## NaN if the year-ago quarter is missing, or last year's value <= 0 (growth
## off a non-positive base is meaningless).
def _yoy_growth(history, current_row, field):
    cur = current_row.get(field)
    prev_row = _same_quarter_n_years_ago(history, current_row, n_years=1)
    if prev_row is None:
        return np.nan
    prev = prev_row.get(field)
    if pd.isna(cur) or pd.isna(prev) or prev <= 0:
        return np.nan
    return float(cur) / float(prev) - 1.0


## Helper for price (momentum) factors: close on/before a target date.
## px_hist is this stock's prices up to obs_date (panel feeds it, point-in-time),
## with columns "date" and "close", sorted oldest -> newest.
def _close_on_or_before(px_hist, target_date):
    sub = px_hist[px_hist["date"] <= target_date]
    if sub.empty:
        return None
    val = sub.iloc[-1]["close"]
    return float(val) if pd.notna(val) else None


## Valuation factors  (single-period)
def ev_ebitda(row):
    ev = row.get("enterprise_value")
    ebitda = row.get("ebitda")
    if pd.isna(ev) or pd.isna(ebitda) or ebitda <= 0:
        return np.nan
    return float(ev) / float(ebitda)


## ev / sales. lower = cheaper. revenue is ~always positive, keep negatives if any
def ev_sales(row):
    ev = row.get("enterprise_value")
    revenue = row.get("revenue")
    if pd.isna(ev) or pd.isna(revenue) or revenue <= 0:
        return np.nan
    return float(ev) / float(revenue)


## ev / fcf. NaN when fcf <= 0: a negative denominator flips the ranking
## (the worst cash burners would sort as "cheapest"), which is garbage, not signal.
def ev_fcf(row):
    ev = row.get("enterprise_value")
    fcf = row.get("free_cash_flow")
    if pd.isna(ev) or pd.isna(fcf) or fcf <= 0:
        return np.nan
    return float(ev) / float(fcf)


## earnings yield = net_income / ev. EV ~always positive, so a negative
## numerator (a loss) ranks correctly as a low/negative yield -> keep it.
def earnings_yield(row):
    ni = row.get("net_income")
    ev = row.get("enterprise_value")
    if pd.isna(ni) or pd.isna(ev) or ev <= 0:
        return np.nan
    return float(ni) / float(ev)


## Quality factors  (single-period)
def rnd_intensity(row):
    rnd=row.get("research_and_development")
    revenue=row.get("revenue")
    if pd.isna(rnd) or pd.isna(revenue) or revenue<=0:
        return np.nan
    return float(rnd)/ float(revenue)


## fcf margin = fcf / revenue. negatives kept: a -20% margin IS worse than -5%,
## so the ranking stays meaningful (this is how we see reaction to cash burn).
def fcf_margin(row):
    fcf = row.get("free_cash_flow")
    revenue = row.get("revenue")
    if pd.isna(fcf) or pd.isna(revenue) or revenue <= 0:
        return np.nan
    return float(fcf) / float(revenue)


## gross margin — use the DB's ready-made column
def gross_margin(row):
    gm = row.get("gross_margin")
    return float(gm) if pd.notna(gm) else np.nan


## net margin — use the DB's ready-made column
def net_margin(row):
    nm = row.get("net_margin")
    return float(nm) if pd.notna(nm) else np.nan


## ROA = net_income / total_assets. negatives kept (a loss ranks correctly).
def roa(row):
    ni = row.get("net_income")
    ta = row.get("total_assets")
    if pd.isna(ni) or pd.isna(ta) or ta <= 0:
        return np.nan
    return float(ni) / float(ta)


## ROE = net_income / (total_assets - total_liabilities). NOT subtracting
## preferred_equity (mostly null). NaN if equity <= 0: negative equity makes the
## ratio flip sign and lose meaning.
def roe(row):
    ni = row.get("net_income")
    ta = row.get("total_assets")
    tl = row.get("total_liabilities")
    if pd.isna(ni) or pd.isna(ta) or pd.isna(tl):
        return np.nan
    equity = float(ta) - float(tl)
    if equity <= 0:
        return np.nan
    return float(ni) / equity


## fcf conversion = fcf / net_income. NaN when net_income <= 0: negative
## denominator flips the ranking.
def fcf_conversion(row):
    fcf = row.get("free_cash_flow")
    ni = row.get("net_income")
    if pd.isna(fcf) or pd.isna(ni) or ni <= 0:
        return np.nan
    return float(fcf) / float(ni)


## net debt / ebitda = (total_debt - cash) / ebitda. NaN when ebitda <= 0.
def net_debt_to_ebitda(row):
    td = row.get("total_debt")
    cash = row.get("cash_and_equivalents")
    ebitda = row.get("ebitda")
    if pd.isna(td) or pd.isna(cash) or pd.isna(ebitda) or ebitda <= 0:
        return np.nan
    return (float(td) - float(cash)) / float(ebitda)


## interest coverage = operating_income / interest_expense. NaN when
## interest_expense <= 0 (zero/negative denominator flips the ranking).
def interest_coverage(row):
    oi = row.get("operating_income")
    ie = row.get("interest_expense")
    if pd.isna(oi) or pd.isna(ie) or ie <= 0:
        return np.nan
    return float(oi) / float(ie)


## capex intensity = capex / revenue. capex is stored positive here.
def capex_intensity(row):
    capex = row.get("capital_expenditures")
    revenue = row.get("revenue")
    if pd.isna(capex) or pd.isna(revenue) or revenue <= 0:
        return np.nan
    return float(capex) / float(revenue)


## Growth factors  (cross-period: func(history, current_row))
## history = this stock's PUBLISHED quarters up to & including the current one
## (panel feeds it, point-in-time; no future rows). Compare to same quarter
## last year (fiscal_year/quarter match), so seasonality cancels.
def revenue_growth_yoy(history, current_row):
    return _yoy_growth(history, current_row, "revenue")


def eps_growth_yoy(history, current_row):
    return _yoy_growth(history, current_row, "eps")


def fcf_growth_yoy(history, current_row):
    return _yoy_growth(history, current_row, "free_cash_flow")


def gross_profit_growth_yoy(history, current_row):
    return _yoy_growth(history, current_row, "gross_profit")


## gross margin CHANGE = this quarter's gross_margin minus the same quarter
## last year's. A difference (not a ratio), so negatives are fine. NaN if the
## year-ago quarter is missing.
def gross_margin_change_yoy(history, current_row):
    cur = current_row.get("gross_margin")
    prev_row = _same_quarter_n_years_ago(history, current_row, n_years=1)
    if prev_row is None:
        return np.nan
    prev = prev_row.get("gross_margin")
    if pd.isna(cur) or pd.isna(prev):
        return np.nan
    return float(cur) - float(prev)


## revenue growth ACCELERATION = this quarter's YoY growth minus last quarter's
## YoY growth (is growth speeding up or slowing down?). Needs the year-ago
## quarters for BOTH this quarter and the previous quarter.
def revenue_growth_acceleration(history, current_row):
    cur_g = _yoy_growth(history, current_row, "revenue")
    if pd.isna(cur_g):
        return np.nan
    # previous quarter's row within this point-in-time history
    prevs = history[history[["fiscal_year", "fiscal_quarter"]].apply(tuple, axis=1)
                    < (current_row["fiscal_year"], current_row["fiscal_quarter"])]
    if prevs.empty:
        return np.nan
    prev_q_row = prevs.iloc[-1]
    prev_g = _yoy_growth(history, prev_q_row, "revenue")
    if pd.isna(prev_g):
        return np.nan
    return cur_g - prev_g


## Momentum factors  (price-based: func(px_hist, obs_date))
## px_hist = this stock's prices with date<=obs_date (panel feeds it;
## strictly point-in-time, no future prices). 12-1 momentum = return over
## the 12 months ending ONE month before obs_date (skip the most recent month
## to avoid short-term reversal). NaN if either anchor price is missing.
def momentum_12_1(px_hist, obs_date):
    if px_hist is None or px_hist.empty:
        return np.nan
    end_date = obs_date - pd.DateOffset(months=1)    # skip most recent month
    start_date = obs_date - pd.DateOffset(months=12)
    p_end = _close_on_or_before(px_hist, end_date)
    p_start = _close_on_or_before(px_hist, start_date)
    if p_end is None or p_start is None or p_start <= 0:
        return np.nan
    return p_end / p_start - 1.0



## FACTOR REGISTRY
## "needs": "single"  -> func(row)
## "needs": "history" -> func(history, current_row)   (growth / change factors)
## "needs": "prices"  -> func(px_hist, obs_date)      (momentum / price factors)
FACTOR_REGISTRY = {
    "ev_ebitda": {
        "func": ev_ebitda,
        "needs": "single",
        "industry_neutral": True,
        "description": "EV / EBITDA (lower = cheaper)",
    },
    "ev_sales": {
        "func": ev_sales,
        "needs": "single",
        "industry_neutral": True,
        "description": "EV / sales (lower = cheaper)",
    },
    "ev_fcf": {
        "func": ev_fcf,
        "needs": "single",
        "industry_neutral": True,
        "description": "EV / free cash flow (lower = cheaper; NaN if FCF<=0)",
    },
    "earnings_yield": {
        "func": earnings_yield,
        "needs": "single",
        "industry_neutral": True,
        "description": "net income / EV (higher = cheaper)",
    },
    "rnd_intensity":{
        "func":rnd_intensity,
        "needs": "single",
        "industry_neutral": True,
        "description": "R&D / revenue (higher = more R&D intensity)",
    },
    "fcf_margin": {
        "func": fcf_margin,
        "needs": "single",
        "industry_neutral": True,
        "description": "free cash flow / revenue (negatives kept)",
    },
    "gross_margin": {
        "func": gross_margin,
        "needs": "single",
        "industry_neutral": True,
        "description": "gross margin (DB column)",
    },
    "net_margin": {
        "func": net_margin,
        "needs": "single",
        "industry_neutral": True,
        "description": "net margin (DB column)",
    },
    "roa": {
        "func": roa,
        "needs": "single",
        "industry_neutral": True,
        "description": "net income / total assets",
    },
    "roe": {
        "func": roe,
        "needs": "single",
        "industry_neutral": True,
        "description": "net income / (assets - liabilities)",
    },
    "fcf_conversion": {
        "func": fcf_conversion,
        "needs": "single",
        "industry_neutral": True,
        "description": "free cash flow / net income (NaN if NI<=0)",
    },
    "net_debt_to_ebitda": {
        "func": net_debt_to_ebitda,
        "needs": "single",
        "industry_neutral": True,
        "description": "(total debt - cash) / EBITDA",
    },
    "interest_coverage": {
        "func": interest_coverage,
        "needs": "single",
        "industry_neutral": True,
        "description": "operating income / interest expense",
    },
    "capex_intensity": {
        "func": capex_intensity,
        "needs": "single",
        "industry_neutral": True,
        "description": "capex / revenue",
    },
    "revenue_growth_yoy": {
        "func": revenue_growth_yoy,
        "needs": "history",
        "industry_neutral": True,
        "description": "YoY revenue growth (same quarter vs last year)",
    },
    "eps_growth_yoy": {
        "func": eps_growth_yoy,
        "needs": "history",
        "industry_neutral": True,
        "description": "YoY EPS growth",
    },
    "fcf_growth_yoy": {
        "func": fcf_growth_yoy,
        "needs": "history",
        "industry_neutral": True,
        "description": "YoY free cash flow growth",
    },
    "gross_profit_growth_yoy": {
        "func": gross_profit_growth_yoy,
        "needs": "history",
        "industry_neutral": True,
        "description": "YoY gross profit growth",
    },
    "gross_margin_change_yoy": {
        "func": gross_margin_change_yoy,
        "needs": "history",
        "industry_neutral": True,
        "description": "gross margin change vs same quarter last year",
    },
    "revenue_growth_acceleration": {
        "func": revenue_growth_acceleration,
        "needs": "history",
        "industry_neutral": True,
        "description": "revenue YoY growth this qtr minus last qtr (accel)",
    },
    "momentum_12_1":{
        "func": momentum_12_1,
        "needs": "prices",
        "industry_neutral": False,   # momentum keeps cross-industry signal
        "description": "12-month return, skipping the most recent month",
    },

    }





def get_factor(name):
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"Unknown factor '{name}'. "
                       f"Available: {list(FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[name]


def list_factors():
    return list(FACTOR_REGISTRY)