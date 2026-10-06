"""
Factor Research Platform — home page / entry point.
Run with:  streamlit run app.py

"""

import streamlit as st
from core.db import test_connection

st.set_page_config(
    page_title="Factor Research Platform",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Factor Research Platform")

st.markdown(
    """
    Welcome. This platform is organized into modules — pick one from the
    sidebar on the left.

    - **📊 Factor Research** — measure a single factor's predictive power
      (RankIC / ICIR / quantile returns) for next-period returns.
    - **🔬 Factor Combination** — correlations and multi-factor blends *(later)*.
    - **📈 Backtest** — build a portfolio from factor scores and test it *(later)*.
    - **🌐 Macro Regime** — how factors behave in different market states *(later)*.
    """
)

st.divider()

# Show database status so you immediately know if the connection works.
st.subheader("System status")
ok, msg = test_connection()
if ok:
    st.success(msg)
else:
    st.error(msg)
    st.info(
        "Check your DB_URL in config.py (or set the FACTOR_DB_URL "
        "environment variable). Make sure PostgreSQL is running."
    )
