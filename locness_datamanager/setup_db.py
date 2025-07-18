#!/usr/bin/env python3
import sys
import sqlite3
import os
from locness_datamanager.config import get_config

CREATE_TABLES_TEMPLATE = """
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
);

CREATE TABLE IF NOT EXISTS tsg (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime_utc INTEGER NOT NULL,
    scan_no INTEGER,
    cond REAL,
    temp REAL,
    salinity REAL,
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

CREATE TABLE IF NOT EXISTS underway_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    datetime_utc INTEGER NOT NULL UNIQUE,
    latitude REAL,
    longitude REAL,
    rho_ppb REAL,
    ph_free REAL,
    temp REAL,
    salinity REAL,
    ph_total_ma REAL
);

CREATE INDEX IF NOT EXISTS idx_rhodamine_datetime_utc ON rhodamine(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_ph_datetime_utc ON ph(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_tsg_datetime_utc ON tsg(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_gps_datetime_utc ON gps(datetime_utc);
CREATE INDEX IF NOT EXISTS idx_underway_summary_datetime_utc ON underway_summary(datetime_utc);
PRAGMA journal_mode=WAL;
"""
    
def setup_sqlite_db(db_path):
    # Create directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    conn = sqlite3.connect(db_path)
    # Enable WAL mode for concurrency
    conn.executescript(CREATE_TABLES_TEMPLATE)
    print(f"SQLite database initialized at {db_path} (WAL mode enabled)")
    conn.close()

def main():
    # Use command line argument if provided, otherwise use config
    if len(sys.argv) >= 2:
        db_path = sys.argv[1]
    else:
        config = get_config()
        db_path = config.get('db_path')
        if not db_path:
            print("Error: db_path not found in config")
            sys.exit(1)
    setup_sqlite_db(db_path)

if __name__ == "__main__":
    main()
