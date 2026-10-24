"""
Responsibilities of panel.py :
  - define the month-end observation dates (the common grid)
  - for each (observation_date, stock), pick the correct quarter using the
    point-in-time rules (+45 day release lag, 130-day carry-forward limit)
  - apply the factor FORMULA (imported from factors.py), feeding it the shape
    of data it declares in the registry:
        needs = "single"  -> func(row)                 (one quarter)
        needs = "history" -> func(history, current_row)(growth/change factors)
        needs = "prices"  -> func(px_hist, obs_date)   (momentum/price factors)
    history  = that stock's PUBLISHED quarters up to & including the current one
    px_hist  = that stock's prices with date <= obs_date
    (both strictly point-in-time; no future rows.)
  - assemble everything into a long panel: [date, ticker, industry, <factor>]


Point-in-time rules:
  usable    : fiscal_period_end + 45 <= observation_date   (already published)
  not stale : observation_date - release_date <= 130 days
"""

import pandas as pd
import numpy as np
from core.factors import get_factor

RELEASE_LAG_DAYS = 45
CARRY_FORWARD_MAX_DAYS = 130
PERIOD_END_COL = "fiscal_period_end"

## Determine each month-end observing point
def month_end_grid(fins):
    q = fins[PERIOD_END_COL].dropna()
    if q.empty:
        return pd.DatetimeIndex([])
    start = q.min()
    end = q.max() + pd.Timedelta(days=RELEASE_LAG_DAYS + 5)
    return pd.date_range(start=start, end=end, freq="ME")


## That ticker's quarters already published by obs_date (point-in-time history),
## oldest -> newest, with a release_date column.
def published_upto(fins_t, obs_date):
    q = fins_t.copy()
    if q.empty:
        return q
    q["release_date"] = q[PERIOD_END_COL] + pd.Timedelta(days=RELEASE_LAG_DAYS)
    return q[q["release_date"] <= obs_date].sort_values(PERIOD_END_COL)


def latest_published_quarter(fins_t, obs_date):
    published = published_upto(fins_t, obs_date)
    if published.empty:
        return None
    row = published.iloc[-1]   # newest published
    age = (obs_date - row["release_date"]).days
    if age > CARRY_FORWARD_MAX_DAYS:
        return None
    return row


def build_panel(universe, fins, factor_names, prices=None):
    if isinstance(factor_names, str):
        factor_names = [factor_names]

    # look up each factor's metadata (func + needs) once
    metas = {name: get_factor(name) for name in factor_names}

    # only prepare price history if some factor needs it
    needs_prices = any(m.get("needs") == "prices" for m in metas.values())
    if needs_prices:
        if prices is None:
            raise ValueError(
                "A price-based factor needs the `prices` argument, but none "
                "was passed to build_panel()."
            )
        prices_by_tk = {tk: g.sort_values("date").reset_index(drop=True)
                        for tk, g in prices.groupby("ticker")}
    else:
        prices_by_tk = {}

    grid = month_end_grid(fins)
    ind_by_tk = dict(zip(universe["ticker"], universe["industry"]))
    fins_by_tk = {tk: g.sort_values(PERIOD_END_COL)
                  for tk, g in fins.groupby("ticker")}

    rows = []
    for obs_date in grid:
        for tk, fins_t in fins_by_tk.items():
            published = published_upto(fins_t, obs_date)   # point-in-time history

            # current quarter = newest published, if not stale
            if published.empty:
                row = None
            else:
                cand = published.iloc[-1]
                if (obs_date - cand["release_date"]).days > CARRY_FORWARD_MAX_DAYS:
                    row = None
                else:
                    row = cand

            # point-in-time price history (only computed if a price factor exists)
            if needs_prices:
                pt = prices_by_tk.get(tk)
                if pt is None:
                    px_hist = None
                else:
                    px_hist = pt[pt["date"] <= obs_date]
            else:
                px_hist = None

            record = {
                "date": obs_date,
                "ticker": tk,
                "industry": ind_by_tk.get(tk),
            }
            # apply each factor, feeding single / history / prices data per "needs"
            for name, meta in metas.items():
                needs = meta.get("needs")
                if needs == "prices":
                    if px_hist is None or px_hist.empty:
                        record[name] = np.nan
                    else:
                        record[name] = meta["func"](px_hist, obs_date)
                elif needs == "history":
                    record[name] = np.nan if row is None else meta["func"](published, row)
                else:  # "single" (default)
                    record[name] = np.nan if row is None else meta["func"](row)
            rows.append(record)

    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


# Backwards-compatible convenience wrapper for the single ev_ebitda case.
def build_ev_ebitda_panel(universe, fins):
    """Shortcut: build a panel with just the ev_ebitda factor."""
    return build_panel(universe, fins, ["ev_ebitda"])