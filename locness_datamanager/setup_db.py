#!/usr/bin/env python3
import sys
import sqlite3

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS rhodamine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime_utc INTEGER NOT NULL,
    gain INTEGER, 
    voltage REAL, 
    rho_ppb REAL
);

CREATE TABLE IF NOT EXISTS ph (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime_utc INTEGER NOT NULL,
    samp_num INTEGER,
    ph_timestamp INTEGER, 
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
);

CREATE TABLE IF NOT EXISTS tsg (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime_utc INTEGER NOT NULL,
    scan_no INTEGER,
    cond REAL,
    temp REAL,
    hull_temp REAL,
    time_elapsed REAL,
    nmea_time INTEGER,
    latitude REAL,
    longitude REAL
);

CREATE TABLE IF NOT EXISTS gps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime_utc INTEGER NOT NULL,
    nmea_time_utc INTEGER,
    latitude REAL,
    longitude REAL
);

CREATE TABLE IF NOT EXISTS resampled_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime_utc INTEGER NOT NULL UNIQUE,
    latitude REAL,
    longitude REAL,
    rho_ppb REAL,
    ph REAL,
    temp_c REAL,
    salinity_psu REAL,
    ph_ma REAL
);

CREATE INDEX IF NOT EXISTS idx_rhodamine_datetime_utc ON rhodamine(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_ph_datetime_utc ON ph(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_tsg_datetime_utc ON tsg(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_gps_datetime_utc ON gps(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_resampled_data_datetime_utc ON resampled_data(datetime_utc);
"""
    
def setup_sqlite_db(db_path):
    conn = sqlite3.connect(db_path)
    # Enable WAL mode for concurrency
    conn.executescript(CREATE_TABLES)
    conn.execute('PRAGMA journal_mode=WAL;')
    print(f"SQLite database initialized at {db_path} (WAL mode enabled)")
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: setup_sqlite_db.py <db_path>")
        sys.exit(1)
    db_path = sys.argv[1]
    setup_sqlite_db(db_path)
