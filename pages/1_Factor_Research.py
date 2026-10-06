## UI
import streamlit as st
import pandas as pd
import numpy as np

from core.data_loader import load_universe, load_financials, load_prices
from core.factors import list_factors, get_factor
from core.panel import build_panel
from core.standardize import standardize_panel
from core.returns import add_forward_returns, HORIZONS
from core.ic import evaluate_factor

st.title("📊 Factor Research")
st.caption("Measure a single factor's predictive power for next-period returns.")

# friendly labels for the return horizons
H_LABEL = {"ret_10d": "10 days", "ret_30d": "30 days", "ret_quarter": "1 quarter"}
H_KEYS = list(HORIZONS.keys())


@st.cache_data(show_spinner="Building factor panel (runs once)...")
def build_full_panel(factor_name):
    """Load data, build panel, standardize, add forward returns. Cached so it
    only recomputes when the factor changes (not on every date-range tweak)."""
    universe = load_universe()
    fins = load_financials()
    prices = load_prices()

    panel = build_panel(universe, fins, [factor_name], prices=prices)
    meta = get_factor(factor_name)
    panel = standardize_panel(panel, factor_name,
                              industry_neutral=meta["industry_neutral"])
    panel = add_forward_returns(panel, prices)
    return panel


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    factor_name = st.selectbox("Factor", list_factors(),
                               help="More factors get added in core/factors.py")

with col2:
    n_groups = st.selectbox("Quantile groups", [5, 10], index=0,
                            help="Split stocks into this many groups by factor")

# Build (or fetch cached) panel so we know the available date span
panel = build_full_panel(factor_name)
factor_z = factor_name + "_z"

min_date = pd.Timestamp(panel["date"].min()).date()
max_date = pd.Timestamp(panel["date"].max()).date()

with col3:
    st.metric("Month-end observations", panel["date"].nunique())

st.markdown("**Date range** — the period whose IC you want to see")
dcol1, dcol2 = st.columns(2)
with dcol1:
    start_date = st.date_input("From", value=min_date,
                               min_value=min_date, max_value=max_date)
with dcol2:
    end_date = st.date_input("To", value=max_date,
                             min_value=min_date, max_value=max_date)

nw_lag = st.slider("Newey-West lag (overlap correction)", 1, 12, 3,
                   help="Periods of IC autocorrelation to correct for. "
                        "Compare against AC(1) in the results: near 0 means "
                        "overlap is minor; a high AC(1) means it matters.")

meta = get_factor(factor_name)
st.caption(f"**{factor_name}** — {meta['description']}  ·  "
           f"industry-neutralized: {meta['industry_neutral']}")

run = st.button("Run analysis", type="primary")


# ---------------------------------------------------------------------------
# Run + display
# ---------------------------------------------------------------------------
if run:
    summary_df, ic_series_by_h, quantiles_by_h = evaluate_factor(
        panel, factor_z, ret_cols=H_KEYS,
        n_groups=n_groups, start_date=start_date, end_date=end_date,
        nw_lag=nw_lag,
    )
    # stash results so horizon selectors can redraw without re-running
    st.session_state["res"] = (summary_df, ic_series_by_h, quantiles_by_h,
                               factor_name, n_groups)

# use last-run results (survives the per-chart selectbox reruns)
if "res" in st.session_state:
    summary_df, ic_series_by_h, quantiles_by_h, f_name, ng = st.session_state["res"]

    # --- IC summary table ---
    st.subheader("IC summary")
    show = summary_df.copy()
    show.index = [H_LABEL[h] for h in show.index]
    show = show.rename(columns={
        "n_dates": "# obs", "ic_mean": "IC mean", "ic_std": "IC std",
        "icir": "ICIR", "hit_rate": "Hit rate",
        "nw_tstat": "NW t", "ac_lag1": "AC(1)",
    })
    st.dataframe(
        show[["# obs", "IC mean", "IC std", "ICIR", "Hit rate", "NW t", "AC(1)"]]
        .style.format({"IC mean": "{:.4f}", "IC std": "{:.4f}",
                       "ICIR": "{:.3f}", "Hit rate": "{:.1%}",
                       "NW t": "{:.2f}", "AC(1)": "{:.2f}"}),
        use_container_width=True,
    )

    n_obs = int(summary_df["n_dates"].max()) if len(summary_df) else 0
    if n_obs < 12:
        st.warning(
            f"Only {n_obs} month-end observations in this range. "
            "IC statistics from so few points are noisy — treat as indicative, "
            "not conclusive. Widen the date range for a reliable read."
        )
    st.caption(
        "Judge strength by **|IC mean|** (> 0.05 useful) and **|ICIR|** "
        "(> 0.5 good). IC **sign** = direction (positive: high value → high "
        "return; negative: low value → high return). **NW t** = Newey-West "
        "t-stat correcting for overlapping observations (|t| > 2 ≈ significant). "
        "**AC(1)** = lag-1 autocorrelation of the IC series: near 0 = little "
        "overlap; high (0.3+) = overlap inflates the plain ICIR, so trust NW t."
    )

    # --- IC time series (horizon selectable) ---
    h1, h2 = st.columns([3, 1])
    with h1:
        st.subheader("RankIC over time")
    with h2:
        ts_h = st.selectbox("Horizon", H_KEYS, index=1,
                            format_func=lambda k: H_LABEL[k], key="ts_h")
    ics = ic_series_by_h.get(ts_h)
    if ics is not None and not ics.empty:
        st.bar_chart(pd.DataFrame({"RankIC": ics}), height=260, color="#78b2c2")
        st.caption("Each bar = one month-end's RankIC. Consistent sign = stable "
                   "factor; flip-flopping = unreliable.")
    else:
        st.info("Not enough data to plot the IC time series for this range.")

    # --- Quantile returns (horizon selectable) + raw value range ---
    q1, q2 = st.columns([3, 1])
    with q1:
        st.subheader("Quantile returns")
    with q2:
        q_h = st.selectbox("Horizon", H_KEYS, index=2,
                           format_func=lambda k: H_LABEL[k], key="q_h")
    qdf = quantiles_by_h.get(q_h)
    if qdf is not None and not qdf.empty:
        qshow = qdf.set_index("quantile")["mean_return"]
        qshow.index = [f"Q{int(q)}" for q in qshow.index]
        st.bar_chart(qshow, height=260, color="#78b2c2")
        st.caption(f"Q1 = lowest {f_name}, Q{ng} = highest. A monotonic ladder "
                   "= clean factor. Which end outperforms depends on the factor "
                   "(value: Q1; momentum: top group).")

        # raw value range per quantile (what kind of stock each group holds)
        if {"raw_min", "raw_mean", "raw_max"}.issubset(qdf.columns):
            rng = qdf[["quantile", "raw_min", "raw_mean", "raw_max"]].copy()
            rng["quantile"] = [f"Q{int(q)}" for q in rng["quantile"]]
            rng = rng.set_index("quantile").rename(columns={
                "raw_min": "Min", "raw_mean": "Mean", "raw_max": "Max"})
            st.caption(f"Value range of {f_name} per group "
                       "(avg over the selected period):")
            st.dataframe(
                rng.style.format({"Min": "{:.2f}", "Mean": "{:.2f}",
                                  "Max": "{:.2f}"}),
                use_container_width=True,
            )
    else:
        st.info("Not enough data to compute quantile returns for this range.")

else:
    st.info("Set your factor and date range, then click **Run analysis**.")