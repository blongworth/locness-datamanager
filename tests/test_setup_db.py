import os
import sqlite3
from locness_datamanager.setup_db import setup_sqlite_db

def test_setup_sqlite_db_creates_database(tmp_path):
    db_path = tmp_path / "test.db"
    setup_sqlite_db(str(db_path))

    assert os.path.exists(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = {row[0] for row in cursor.fetchall()}

    expected_tables = {'rhodamine', 'ph', 'tsg', 'gps', 'underway_summary'}
    assert expected_tables.issubset(tables)
    conn.close()

def test_setup_sqlite_db_warns_if_tables_exist(tmp_path, capsys):
    db_path = tmp_path / "test.db"

    # First setup to create the tables
    setup_sqlite_db(str(db_path))

    # Second setup to trigger the warning
    setup_sqlite_db(str(db_path))

    captured = capsys.readouterr()
    assert "Warning: The following tables already exist" in captured.out

