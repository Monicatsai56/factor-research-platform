## core/correlation.py
## Used to check factors' correlation before running the regression
import pandas as pd
import numpy as np

from core.standardize import standardize_panel
from core.factors import get_factor

MIN_STOCKS_PER_DATE = 10


def _filter_dates(panel, date_col, start_date, end_date):
    out = panel
    if start_date is not None:
        out = out[out[date_col] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out[date_col] <= pd.Timestamp(end_date)]
    return out


def _standardize_all(panel, factor_cols, winsor_std=3.0):
    out = panel
    z_cols = []
    for f in factor_cols:
        zc = f + "_z"
        neutral = get_factor(f)["industry_neutral"]
        out = standardize_panel(out, f, industry_neutral=neutral,
                                winsor_std=winsor_std, out_col=zc)
        z_cols.append(zc)
    return out, z_cols


def factor_correlation(panel, factor_cols, method="pearson",
                       date_col="date", winsor_std=3.0,
                       start_date=None, end_date=None):

    work, z_cols = _standardize_all(panel, factor_cols, winsor_std)
    sub = _filter_dates(work, date_col, start_date, end_date)

    mats = []
    for d, cs in sub.groupby(date_col):
        block = cs[z_cols].dropna()
        if len(block) < MIN_STOCKS_PER_DATE:
            continue
        if method == "spearman":
            block = block.rank()        # Spearman = Pearson on ranks
        mats.append(block.corr(method="pearson"))

    if not mats:
        return pd.DataFrame(index=factor_cols, columns=factor_cols, dtype=float)
    out = sum(mats) / len(mats)
    rename = dict(zip(z_cols, factor_cols))
    return out.rename(index=rename, columns=rename)


def factor_vif(panel, factor_cols, date_col="date",
               winsor_std=3.0,
               start_date=None, end_date=None):

    work, z_cols = _standardize_all(panel, factor_cols, winsor_std)
    sub = _filter_dates(work, date_col, start_date, end_date)

    if len(z_cols) < 2:
        return pd.DataFrame({"factor": factor_cols, "vif": [np.nan] * len(factor_cols)})

    per_date = {zc: [] for zc in z_cols}
    for d, cs in sub.groupby(date_col):
        block = cs[z_cols].dropna()
        if len(block) < max(MIN_STOCKS_PER_DATE, len(z_cols) + 2):
            continue
        for zc in z_cols:
            y = block[zc].values
            X = block[[c for c in z_cols if c != zc]].values
            Xc = np.column_stack([np.ones(len(X)), X])     # add intercept
            beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
            resid = y - Xc @ beta
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            if ss_tot <= 0:
                continue
            r2 = 1.0 - ss_res / ss_tot
            r2 = min(max(r2, 0.0), 0.999999)               # keep VIF finite
            per_date[zc].append(1.0 / (1.0 - r2))

    rows = [{"factor": f, "vif": (np.mean(per_date[zc]) if per_date[zc] else np.nan)}
            for f, zc in zip(factor_cols, z_cols)]
    return pd.DataFrame(rows)


##customized vif regression
def targeted_vif(panel, dependent, independents, date_col="date",
                 winsor_std=3.0,
                 start_date=None, end_date=None):

    cols = [dependent] + list(independents)
    work = panel
    z_cols = []
    for f in cols:
        zc = f + "_z"
        neutral = get_factor(f)["industry_neutral"]
        work = standardize_panel(work, f, industry_neutral=neutral,
                                 winsor_std=winsor_std, out_col=zc)
        z_cols.append(zc)
    dep_z, indep_z = z_cols[0], z_cols[1:]

    sub = _filter_dates(work, date_col, start_date, end_date)
    r2s = []
    for d, cs in sub.groupby(date_col):
        block = cs[z_cols].dropna()
        if len(block) < max(MIN_STOCKS_PER_DATE, len(z_cols) + 2):
            continue
        y = block[dep_z].values
        X = block[indep_z].values
        Xc = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
        resid = y - Xc @ beta
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            continue
        r2s.append(1.0 - ss_res / ss_tot)

    if not r2s:
        return {"dependent": dependent, "independents": list(independents),
                "r2": np.nan, "vif": np.nan, "n_dates": 0}
    r2 = float(np.mean(r2s))
    r2c = min(max(r2, 0.0), 0.999999)
    return {"dependent": dependent, "independents": list(independents),
            "r2": r2, "vif": 1.0 / (1.0 - r2c), "n_dates": len(r2s)}