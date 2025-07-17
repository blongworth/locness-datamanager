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
    """Create a sample SQLite database for testing."""
    import sqlite3
    db_file = temp_dir / "sample.sqlite"
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute("""
        CREATE TABLE resampled_data (
            datetime_utc INTEGER,
            lat REAL,
            lon REAL,
            rhodamine REAL,
            ph REAL,
            temp REAL,
            salinity REAL,
            ph_ma REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE rhodamine (
            datetime_utc INTEGER,
            latitude REAL,
            longitude REAL,
            gain INTEGER,
            voltage REAL,
            concentration REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE tsg (
            datetime_utc INTEGER,
            scan_no INTEGER,
            cond REAL,
            temp REAL,
            hull_temp REAL,
            time_elapsed REAL,
            nmea_time INTEGER,
            latitude REAL,
            longitude REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE ph (
            datetime_utc TEXT,
            samp_num INTEGER,
            ph_timestamp TEXT,
            v_bat REAL,
            v_bias_pos REAL,
            v_bias_neg REAL,
            t_board REAL,
            h_board REAL,
            vrse REAL,
            vrse_std REAL,
            cevk REAL,
            cevk_std REAL,
            ce_ik REAL,
            i_sub REAL,
            cal_temp REAL,
            cal_sal REAL,
            k0 REAL,
            k2 REAL,
            ph_free REAL,
            ph_total REAL
        )
    """)
    
    conn.commit()
    conn.close()
    
    return db_file
