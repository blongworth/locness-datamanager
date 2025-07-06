#!/usr/bin/env python3
import sqlite3
import sys
import os

def setup_sqlite_db(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Enable WAL mode for concurrency
    cur.execute('PRAGMA journal_mode=WAL;')

    # Create fluorometer table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS fluorometer (
          timestamp INTEGER PRIMARY KEY,
          latitude REAL, 
          longitude REAL, 
          gain INTEGER, 
          voltage REAL, 
          concentration REAL)
        );
    ''')

    # Create ph table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS ph (
            pc_timestamp TEXT PRIMARY KEY,
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
    ''')

    # Create tsg table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS tsg (
            timestamp INTEGER PRIMARY KEY,
            scan_no INTEGER,
            cond REAL,
            temp REAL,
            hull_temp REAL,
            time_elapsed REAL,
            nmea_time INTEGER,
            latitude REAL,
            longitude REAL
        );
    ''')

    # Create resampled_data table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS resampled_data (
            timestamp TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            rhodamine REAL,
            ph REAL,
            temp REAL,
            salinity REAL,
            ph_ma REAL
        );
    ''')

    conn.commit()
    print(f"SQLite database initialized at {db_path} (WAL mode enabled)")
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: setup_sqlite_db.py <db_path>")
        sys.exit(1)
    db_path = sys.argv[1]
    setup_sqlite_db(db_path)
