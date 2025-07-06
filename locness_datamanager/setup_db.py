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
            timestamp TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            rhodamine REAL
        );
    ''')

    # Create ph table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS ph (
            timestamp TEXT PRIMARY KEY,
            ph REAL
        );
    ''')

    # Create tsg table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS tsg (
            timestamp TEXT PRIMARY KEY,
            temp REAL,
            salinity REAL
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
