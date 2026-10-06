### SQL Key
import os

DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_URL = f"postgresql://postgres:{DB_PASSWORD}@localhost:5432/stocks"

# Where precomputed factor panels (parquet files) get stored.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PANEL_PATH = os.path.join(DATA_DIR, "factor_panel.parquet")


# Research parameters (defaults)
REBALANCE_LAG_DAYS = 45      # snapshot date = quarter_end + this many days
FIN_STALE_DAYS     = 100     # financials older than this can't be carried forward
FORWARD_RETURN_DAYS = 63     # ~1 quarter of trading days for "next-period" return

# First snapshot to include
BACKTEST_START = "2015-01-01"
