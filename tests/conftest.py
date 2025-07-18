"""Configuration for pytest tests."""
import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    csv_file = temp_dir / "sample.csv"
    csv_content = """datetime_utc,lat,lon,temp,salinity,rhodamine,ph
1640995200,42.5,-69.5,15.2,35.1,1.2,8.1
1640995201,42.501,-69.501,15.3,35.2,1.3,8.2
1640995202,42.502,-69.502,15.1,35.0,1.1,8.0
"""
    csv_file.write_text(csv_content)
    return csv_file


@pytest.fixture
def sample_sqlite_db(temp_dir):
    """Create a sample SQLite database for testing with real schema."""
    import sqlite3
    db_file = temp_dir / "sample.sqlite"

    from locness_datamanager.setup_db import CREATE_TABLES_TEMPLATE
    conn = sqlite3.connect(db_file)
    conn.executescript(CREATE_TABLES_TEMPLATE)
    conn.commit()
    conn.close()
    return db_file
