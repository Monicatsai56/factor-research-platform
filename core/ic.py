## IC / RankIC / grouped returns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

MIN_STOCKS_PER_DATE = 10   # skip a cross-section with fewer names than this

# Newey-West lag: how many periods the IC autocorrelation extends.
# Month-end grid + quarterly financials -> same report reused ~3 months, so
# adjacent ICs are correlated out to ~2-3 lags. 3 is a safe default; check
# ac_lag1 in the summary to see how strong the overlap actually is.
NW_LAG = 3


def _filter_dates(panel, date_col, start_date, end_date):
    out = panel
    if start_date is not None:
        out = out[out[date_col] >= pd.Timestamp(start_date)]
    if end_date is not None:
        out = out[out[date_col] <= pd.Timestamp(end_date)]
    return out

## Spearman correlation calculation
def rank_ic_series(panel, factor_col, ret_col, date_col="date",
                   start_date=None, end_date=None):
    sub = _filter_dates(panel, date_col, start_date, end_date)
    out = {}
    for d, cs in sub.groupby(date_col):
        pair = cs[[factor_col, ret_col]].dropna()
        if len(pair) < MIN_STOCKS_PER_DATE:
            continue
        ic, _ = spearmanr(pair[factor_col], pair[ret_col])
        if not np.isnan(ic):
            out[d] = ic
    return pd.Series(out).sort_index()


## ---- Newey-West correction for overlapping observations ----
## Overlap (same financials reused across adjacent months) makes adjacent ICs
## correlated, so the plain t-stat (ICIR * sqrt(N)) OVERSTATES significance.
## Newey-West adjusts the STANDARD ERROR for this autocorrelation. It changes
## only the t-stat / p-value (how sure we are) — never ic_mean / ICIR (the
## signal itself). Returns (nw_tstat, nw_pvalue), or (nan, nan) if unavailable.
def nw_tstat(ic_series, lag=NW_LAG):
    s = ic_series.dropna()
    if len(s) < 3:
        return np.nan, np.nan
    try:
        import statsmodels.api as sm
        y = s.values.astype(float)
        X = np.ones(len(y))               # regress IC on a constant = its mean
        model = sm.OLS(y, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": int(lag)})
        return float(model.tvalues[0]), float(model.pvalues[0])
    except Exception:
        return np.nan, np.nan


## lag-1 autocorrelation of the IC series: how similar adjacent-month ICs are.
## ~0 = overlap is a non-issue (plain t-stat trustworthy);
## clearly positive (e.g. 0.3+) = strong overlap, plain t-stat inflated.
def ic_autocorr(ic_series, lag=1):
    s = ic_series.dropna()
    if len(s) <= lag:
        return np.nan
    return float(s.autocorr(lag=lag))


def ic_summary(ic_series, nw_lag=NW_LAG):
    s = ic_series.dropna()
    if s.empty:
        return {"n_dates": 0, "ic_mean": np.nan, "ic_std": np.nan,
                "icir": np.nan, "hit_rate": np.nan,
                "nw_tstat": np.nan, "nw_pvalue": np.nan, "ac_lag1": np.nan}
    mean = s.mean()
    std = s.std()
    t, p = nw_tstat(s, lag=nw_lag)
    return {
        "n_dates": len(s),
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if std and not np.isnan(std) else np.nan,
        "hit_rate": (s > 0).mean(),
        "nw_tstat": t,          # Newey-West t (overlap-corrected significance)
        "nw_pvalue": p,
        "ac_lag1": ic_autocorr(s, lag=1),
    }


def quantile_returns(panel, factor_col, ret_col, n_groups=5, date_col="date",
                     start_date=None, end_date=None, raw_col=None):
    """Mean forward return per quantile, averaged over the date range.
    If raw_col (the un-standardized factor, e.g. 'ev_ebitda') is present, also
    reports each quantile's raw-value range (min/mean/max), averaged per date
    then across dates — so you can see what kind of stock each group holds.
    """
    sub = _filter_dates(panel, date_col, start_date, end_date)
    have_raw = raw_col is not None and raw_col in sub.columns
    cols = [factor_col, ret_col] + ([raw_col] if have_raw else [])
    rows = []
    for d, cs in sub.groupby(date_col):
        pair = cs[cols].dropna(subset=[factor_col, ret_col])
        if len(pair) < n_groups * 2:
            continue
        pair = pair.copy()
        try:
            pair["q"] = pd.qcut(pair[factor_col], n_groups,
                                labels=False, duplicates="drop") + 1
        except Exception:
            continue
        for q, g in pair.groupby("q"):
            rec = {"quantile": q, "ret": g[ret_col].mean()}
            if have_raw:
                rv = g[raw_col].dropna()
                if len(rv):
                    rec["raw_min"] = rv.min()
                    rec["raw_mean"] = rv.mean()
                    rec["raw_max"] = rv.max()
            rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["quantile", "mean_return"])
    df = pd.DataFrame(rows)
    agg = {"ret": "mean"}
    for c in ("raw_min", "raw_mean", "raw_max"):
        if c in df.columns:
            agg[c] = "mean"
    out = df.groupby("quantile").agg(agg).reset_index()
    return out.rename(columns={"ret": "mean_return"})


def evaluate_factor(panel, factor_col, ret_cols=None, n_groups=5,
                    date_col="date", start_date=None, end_date=None,
                    nw_lag=NW_LAG):
    if ret_cols is None:
        ret_cols = ["ret_10d", "ret_30d", "ret_quarter"]

    # raw (un-standardized) factor column, for the per-quantile value range:
    # factor_col is like "ev_ebitda_z"; its raw partner is "ev_ebitda".
    raw_col = factor_col[:-2] if factor_col.endswith("_z") else None

    summaries, ic_series_by_h, quantiles_by_h = [], {}, {}
    for ret_col in ret_cols:
        ics = rank_ic_series(panel, factor_col, ret_col, date_col,
                             start_date, end_date)
        summ = ic_summary(ics, nw_lag=nw_lag)
        summ["horizon"] = ret_col
        summaries.append(summ)
        ic_series_by_h[ret_col] = ics
        quantiles_by_h[ret_col] = quantile_returns(
            panel, factor_col, ret_col, n_groups, date_col,
            start_date, end_date, raw_col=raw_col)

    summary_df = pd.DataFrame(summaries).set_index("horizon")
    return summary_df, ic_series_by_h, quantiles_by_h