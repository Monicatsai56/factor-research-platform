##Standardized
import pandas as pd
import numpy as np

## delete extreme value
def winsorize(series, n_std=3.0):
    if series.dropna().empty:
        return series
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series
    return series.clip(lower=mean - n_std * std, upper=mean + n_std * std)


def industry_neutralize(df, factor_col, industry_col="industry"):
    med = df.groupby(industry_col)[factor_col].transform("median")
    return df[factor_col] - med

## cross-sectional z-score = (x - mean) / std.
def zscore(series):
    s = series.dropna()
    if s.empty:
        return series
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - mean) / std

## combined tool
def standardize_cross_section(df, factor_col, industry_col="industry",
                              industry_neutral=True, winsor_std=3.0):
    if industry_neutral:
        raw = industry_neutralize(df, factor_col, industry_col)
    else:
        raw = df[factor_col].copy()
    raw = winsorize(raw, n_std=winsor_std)
    return zscore(raw)


def standardize_panel(panel, factor_col, date_col="date",
                      industry_col="industry", industry_neutral=True,
                      winsor_std=3.0, out_col=None):
    """Apply standardize_cross_section date-by-date across a whole panel.
    Standardizes WITHIN each date's cross-section — never mixing across dates.
    """
    if out_col is None:
        out_col = factor_col + "_z"
    panel = panel.copy()
    panel[out_col] = np.nan
    for d, idx in panel.groupby(date_col).groups.items():
        cs = panel.loc[idx]
        scores = standardize_cross_section(
            cs, factor_col, industry_col=industry_col,
            industry_neutral=industry_neutral, winsor_std=winsor_std,
        )
        panel.loc[idx, out_col] = scores
    return panel