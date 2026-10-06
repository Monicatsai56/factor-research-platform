## forward return calculation
import pandas as pd
import numpy as np

# Horizons in TRADING days
HORIZONS = {"ret_10d": 10, "ret_30d": 30, "ret_quarter": 63}


def _start_index(dates_sorted, obs_date):
    i = dates_sorted.searchsorted(pd.Timestamp(obs_date), side="left")
    if i >= len(dates_sorted):
        return None
    return i


def add_forward_returns(panel, prices):
    panel = panel.copy()
    for col in HORIZONS:
        panel[col] = np.nan

    # per-ticker sorted price arrays for fast lookups
    px_by_tk = {}
    for tk, g in prices.groupby("ticker"):
        g = g.sort_values("date")
        px_by_tk[tk] = (g["date"].reset_index(drop=True),
                        g["close"].reset_index(drop=True))

    for i, row in panel.iterrows():
        tk = row["ticker"]
        if tk not in px_by_tk:
            continue
        dates_sorted, closes = px_by_tk[tk]

        start_i = _start_index(dates_sorted, row["date"])
        if start_i is None:
            continue
        base_px = closes.iloc[start_i]
        if pd.isna(base_px) or base_px == 0:
            continue

        for col, n in HORIZONS.items():
            end_i = start_i + n
            if end_i >= len(closes):
                continue
            fut_px = closes.iloc[end_i]
            if pd.isna(fut_px):
                continue
            panel.at[i, col] = fut_px / base_px - 1.0

    return panel