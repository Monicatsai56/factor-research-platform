## Correlation and Regression UI
import streamlit as st
import pandas as pd
import numpy as np

from core.data_loader import load_universe, load_financials, load_prices
from core.factors import list_factors
from core.panel import build_panel
from core.correlation import factor_correlation, factor_vif, targeted_vif

st.title("🧪 Factor Models")
st.caption("Second filter: of the factors you shortlisted, which are redundant "
           "with each other? Screen collinearity here, then regress on returns.")


@st.cache_data(show_spinner="Building panel for the chosen factors...")
def build_multi_panel(factor_key):
    """factor_key is a sorted tuple of factor names (hashable for the cache)."""
    universe = load_universe()
    fins = load_financials()
    prices = load_prices()
    return build_panel(universe, fins, list(factor_key), prices=prices)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
factors = st.multiselect(
    "Factors to screen", list_factors(),
    help="Pick the handful you shortlisted on the Factor Research page.")

c1, c2 = st.columns(2)
with c1:
    method = st.radio("Correlation type", ["pearson", "spearman"], index=0,
                      help="Pearson = tied to regression collinearity (primary). "
                           "Spearman = outlier-robust rank check.")
with c2:
    winsor_std = st.slider("Winsor (std)", 2.0, 5.0, 3.0, 0.5,
                           help="Clip values beyond this many std from the "
                                "mean before standardizing. 3 is standard.")

panel = None
if len(factors) >= 1:
    panel = build_multi_panel(tuple(sorted(factors)))
    min_d = pd.Timestamp(panel["date"].min()).date()
    max_d = pd.Timestamp(panel["date"].max()).date()
    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input("From", value=min_d,
                                   min_value=min_d, max_value=max_d)
    with d2:
        end_date = st.date_input("To", value=max_d,
                                 min_value=min_d, max_value=max_d)
else:
    start_date = end_date = None

run = st.button("Run collinearity screen", type="primary",
                disabled=len(factors) < 2)
if len(factors) < 2:
    st.info("Pick at least **2** factors to compare.")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if run and panel is not None:
    kw = dict(winsor_std=winsor_std, start_date=start_date, end_date=end_date)

    # --- Correlation matrix (first glance: which pairs look redundant) ---
    st.subheader("Correlation matrix")
    corr = factor_correlation(panel, factors, method=method, **kw)
    st.dataframe(
        corr.style.format("{:.2f}")
            .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
        use_container_width=True,
    )
    st.caption(f"{method.title()} correlation, averaged across dates. "
               "**|r| > 0.7** (deep red/blue) = the pair ranks stocks alike — "
               "candidates for dropping or compositing. The matrix only shows "
               "**pairwise** overlap; use VIF below to catch redundancy spread "
               "across several factors.")

    # --- VIF (verdict: is a factor explained by the others?) ---
    st.subheader("Variance Inflation Factor")
    mode = st.radio("VIF mode", ["All factors", "Targeted (A on B, C…)"],
                    horizontal=True, key="vif_mode",
                    help="All = each factor vs every other. "
                         "Targeted = pick one dependent + specific independents.")

    if mode == "All factors":
        vdf = factor_vif(panel, factors, **kw)

        def _flag(v):
            if pd.isna(v):
                return ""
            return "background-color:#f4cccc" if v > 5 else ""

        st.dataframe(
            vdf.style.format({"vif": "{:.2f}"}).map(_flag, subset=["vif"]),
            use_container_width=True,
        )
        st.caption("Each factor regressed on **all the others you chose**. "
                   "**VIF > 5** (red) = largely explained by the rest → "
                   "double-counting; drop, composite, or orthogonalize before "
                   "the regression. VIF ≈ 1 = independent, carries its own signal.")
    else:
        t1, t2 = st.columns([1, 2])
        with t1:
            dep = st.selectbox("Dependent factor (A)", factors, key="dep")
        with t2:
            indep = st.multiselect(
                "Explained by (B, C, …)",
                [f for f in factors if f != dep], key="indep",
                help="Only these are used — the rest of your pool is ignored.")
        if indep:
            res = targeted_vif(panel, dep, indep, **kw)
            m1, m2, m3 = st.columns(3)
            m1.metric("R²", f"{res['r2']:.3f}" if not pd.isna(res['r2']) else "—")
            m2.metric("VIF", f"{res['vif']:.2f}" if not pd.isna(res['vif']) else "—")
            m3.metric("# dates", res["n_dates"])
            st.caption(
                f"**{dep}** regressed on **{', '.join(indep)}**, per date, "
                "averaged. R² near 1 (high VIF) = those factors reproduce "
                f"`{dep}` → it's redundant given them. Low R² = `{dep}` carries "
                "something they don't → keep it.")
        else:
            st.info("Pick at least one factor on the right to explain the dependent.")

    st.divider()
    st.subheader("Multi-factor regression")
    st.info("Coming next — once the survivors of this screen are chosen, regress "
            "them on forward returns (pick the return horizon as the dependent "
            "variable). Section will be added here.")

elif not run:
    st.caption("Shortlist your factors, set the options, then run the screen.")