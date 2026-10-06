### Database Connection
from sqlalchemy import create_engine
from config import DB_URL

# A single shared engine for the whole app.
engine = create_engine(DB_URL, pool_pre_ping=True)


def test_connection():
    """Return (ok: bool, message: str). Used by the UI to show DB status."""
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1;"))
        return True, "Connected to database."
    except Exception as e:
        return False, f"Database connection failed: {e}"
