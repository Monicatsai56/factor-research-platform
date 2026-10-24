# Factor Research Platform

A point-in-time factor research platform for US technology equities, built to
measure the cross-sectional predictive power of fundamental and price-based
factors on forward returns. The focus is methodological rigor — correct
point-in-time alignment, honest significance testing, and transparent handling
of the pitfalls that make most factor "signals" illusory.

Built with Streamlit (UI) + PostgreSQL (data), ~700 US tech stocks, 2015–2026,
quarterly fundamentals.

---

## What it does

For any factor, the platform computes how well it ranks stocks against their
subsequent returns, across multiple horizons, over a selectable date range:

- **RankIC (Spearman)** per month-end, with ICIR, hit rate, and the IC time series
- **Newey-West–corrected t-statistics** to adjust significance for the
  autocorrelation introduced by overlapping observations
- **Quantile (Q1–Q5) returns** with each group's raw value range, to show the
  *shape* of the factor–return relationship, not just its strength
- A **lag-1 autocorrelation diagnostic** to make the overlap problem visible

Everything is selectable in the UI: factor, horizon (10d / 30d / 1 quarter),
date range, quantile count, and Newey-West lag.

---

## Methodology (the design decisions that matter)

**Point-in-time correctness.** Every factor at a given month-end uses only
financials that were *already published* by that date. A quarter becomes usable
at `fiscal_period_end + 45 days` (an assumed release lag, since no true release
date exists in the data), with a 130-day carry-forward limit beyond which stale
data is dropped. EV is recomputed each quarter rather than taken from a single
stale value, to avoid leaking the future into historical factors.

**Common month-end observation grid.** Because companies report on different
dates, factors are aligned to a shared month-end grid: at each date, every stock
uses its most recent published quarter. This keeps the cross-section clean for
IC, which is inherently a cross-sectional measure.

**Standardization.** Factors are industry-neutralized (subtract the industry
median) then cross-sectionally z-scored across the whole universe — median for
robustness to outliers, whole-universe z-score for sample-size stability.
Momentum is intentionally *not* industry-neutralized, to preserve cross-industry
rotation signal.

**Overlapping observations.** Monthly observation + quarterly financials means
the same report is reused across ~3 month-ends, so adjacent ICs are
autocorrelated and the naive t-stat overstates significance. The platform
reports Newey-West–corrected t-stats and the IC series' lag-1 autocorrelation so
this inflation is explicit, not hidden.

**Forward returns** are measured from the month-end observation date (all stocks
aligned, so return windows don't contaminate the cross-section), as point-to-point
returns over each horizon.

---

## Factor library

~20 factors across four families, each declaring how it's computed so new
factors require no changes elsewhere in the pipeline:

- **Valuation:** EV/EBITDA, EV/Sales, EV/FCF, earnings yield
- **Quality:** FCF margin, gross/net margin, ROA, ROE, FCF conversion,
  net-debt/EBITDA, interest coverage, R&D intensity, capex intensity
- **Growth (cross-period):** YoY revenue / EPS / FCF / gross-profit growth,
  gross-margin change, revenue-growth acceleration
- **Momentum (price-based):** 12-1 month return (skipping the most recent month)

Factors with a negative-possible denominator return NaN when the denominator is
non-positive (a negative denominator flips the ranking into garbage); factors
where negative values are meaningful (margins, growth) keep them.

---

## Architecture

```
app.py                 Streamlit entry point
config.py              DB connection (via env var) + research parameters
core/
  db.py                single shared DB engine
  data_loader.py       the only file that touches SQL
  factors.py           pure factor formulas + registry (add a factor here only)
  panel.py             month-end panel: point-in-time alignment, feeds factors
  standardize.py       industry-neutralize + winsorize + z-score
  returns.py           multi-horizon forward returns
  ic.py                RankIC, ICIR, Newey-West, quantile returns
pages/
  1_Factor_Research.py the research UI
```

The core separation is deliberate: `core/` is pure computation (testable without
the UI), `pages/` only displays. Adding a factor touches only `factors.py`; the
panel, standardization, and IC code never change.


---

## Status & honest limitations

This is a research tool, not a trading system. Single-factor predictive power on
US tech is weak — as it should be in an efficient, heavily-arbitraged market.
The value of the platform is in measuring *what actually predicts* with
discipline, and in being honest about what doesn't.

Known limitations, several of which are active research questions:
- No true earnings release dates (release timing is estimated at +45 days)
- Overlapping observations inflate naive significance (addressed via Newey-West)
- Potential survivorship bias in the current universe construction
- Single-factor only so far; multi-factor regression (handling collinearity and
  out-of-sample validation) is the next stage

---

*Built as a self-directed project to develop quantamental research methodology.*