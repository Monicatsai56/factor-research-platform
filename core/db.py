### Database Connection
from sqlalchemy import create_engine
from config import DB_URL

# A single shared engine
engine = create_engine(DB_URL, pool_pre_ping=True)


def test_connection():
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1;"))
        return True, "Connected to database."
    except Exception as e:
        return False, f"Database connection failed: {e}"
