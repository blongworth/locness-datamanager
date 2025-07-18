from locness_datamanager.config import get_config
import sqlite3
import pandas as pd
import time
from typing import Optional
import os
from locness_datamanager import file_writers
import warnings

# TODO: correct field names to match database schema
# TODO: add error handling for database connection and queries
# TODO: use integer timestamps for SQLite compatibility
# TODO: use time bucket resampling in sqlite (test speed)

def read_table(conn, table, columns):
    query = f"SELECT {', '.join(columns)} FROM {table}"
    df = pd.read_sql_query(query, conn)
    # Convert integer timestamps to datetime if needed
    if 'datetime_utc' in df.columns and df['datetime_utc'].dtype in ['int64', 'int32']:
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='s')
    return df

def load_sqlite_tables(sqlite_path):
    """
    Load raw tables from SQLite and return as DataFrames.
    Returns: fluoro, ph, tsg, gps DataFrames
    """
    conn = sqlite3.connect(sqlite_path)
    fluoro = read_table(conn, 'rhodamine', ['datetime_utc','rho_ppb'])
    ph = read_table(conn, 'ph', ['datetime_utc', 'ph_total'])
    tsg = read_table(conn, 'tsg', ['datetime_utc', 'temp', 'salinity'])
    gps = read_table(conn, 'gps', ['datetime_utc', 'latitude', 'longitude'])
    conn.close()
    return fluoro, ph, tsg, gps

def resample_tables(fluoro, ph, tsg, gps, resample_interval=None, config=None):
    """
    Resample each table to the given interval and join into a single DataFrame.
    If resample_interval is None, get it from config['res_int'].
    Handles missing/empty tables by producing a DataFrame with all expected columns, filled with NA where data is missing.
    """
    if config is None:
        config = get_config()
    if resample_interval is None:
        resample_interval = config.get('res_int', '2s')
    # Set datetime_utc as index and ensure datetime
    table_defs = [
        ('rhodamine', fluoro, ['datetime_utc', 'rho_ppb']),
        ('ph', ph, ['datetime_utc', 'ph_total']),
        ('tsg', tsg, ['datetime_utc', 'temp', 'salinity']),
        ('gps', gps, ['datetime_utc', 'latitude', 'longitude'])
    ]
    resampled = {}
    for name, df, cols in table_defs:
        # If df is empty or missing datetime_utc, create empty DataFrame with correct columns
        if df is None or df.empty or 'datetime_utc' not in df.columns:
            df = pd.DataFrame(columns=cols)
        if not df.empty:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            if df['datetime_utc'].duplicated().any():
                warnings.warn(f"Duplicate timestamps found in {name} table. Dropping duplicates.")
            df = df.drop_duplicates(subset='datetime_utc').set_index('datetime_utc')
            # Use mean for resampling instead of nearest
            resampled[name] = df.resample(resample_interval).mean(numeric_only=True)
        else:
            # Create empty DataFrame with datetime_utc index (no rows)
            resampled[name] = pd.DataFrame(columns=cols).set_index('datetime_utc')
    # Find the union of all datetime_utc indices
    all_indices = pd.Index([])
    for df in resampled.values():
        all_indices = all_indices.union(df.index)
    all_indices = all_indices.sort_values()
    # Reindex all DataFrames to the union index
    for name in resampled:
        resampled[name] = resampled[name].reindex(all_indices)
    # Combine all on datetime_utc
    df = pd.concat([
        resampled['rhodamine'],
        resampled['ph'],
        resampled['tsg'],
        resampled['gps']
    ], axis=1)
    df = df.reset_index()
    # Ensure all expected columns exist
    cols = ['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'temp', 'salinity']
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[cols]
    return df

def add_ph_moving_average(df, window_seconds=120, freq_hz=1.0):
    """
    Add a moving average column for pH to the DataFrame.
    window_seconds: window size in seconds
    freq_hz: sampling frequency in Hz
    """
    if 'datetime_utc' in df.columns:
        # Sort by datetime_utc to ensure correct rolling
        df = df.sort_values('datetime_utc')
        df = df.reset_index(drop=True)
    window_size = max(1, int(window_seconds * freq_hz))
    df['ph_total_ma'] = df['ph_total'].rolling(window=window_size, min_periods=1).mean()
    return df

def add_computed_fields(df, config=None):
    """
    Add computed columns (e.g., moving average pH) to the DataFrame.
    """
    if config is None:
        config = get_config()
    window_seconds = config.get('ph_ma_window', 120)
    freq_hz = config.get('ph_freq', 0.5)
    df = add_ph_moving_average(df, window_seconds=window_seconds, freq_hz=freq_hz)
    return df

def load_and_resample_sqlite(sqlite_path, resample_interval=None):
    """
    Load, resample, and add computed fields to sensor data from SQLite.
    Returns a DataFrame with columns: datetime_utc, latitude, longitude, rho_ppb, ph, temp_c, salinity_psu, ph_ma
    """
    config = get_config()
    fluoro, ph, tsg, gps = load_sqlite_tables(sqlite_path)
    df = resample_tables(fluoro, ph, tsg, gps, resample_interval, config)
    df = add_computed_fields(df, config)
    return df

def poll_new_records(
    sqlite_path,
    last_timestamp: Optional[pd.Timestamp] = None,
    poll_interval: float = 2.0,
    resample_interval: str = '2s',
    stop_after: Optional[float] = None
):
    """
    Periodically poll the database for new records after last_timestamp, process, and yield as DataFrame.
    Args:
        sqlite_path: Path to SQLite database
        last_timestamp: Only return records after this timestamp (if None, returns all)
        poll_interval: Seconds between polls
        resample_interval: Resample interval (default '2s')
        stop_after: Stop polling after this many seconds (None = run forever)
    Yields:
        DataFrame of new processed records (may be empty if no new data)
    """
    start_time = time.time()
    last_ts = last_timestamp
    while True:
        df = load_and_resample_sqlite(sqlite_path, resample_interval)
        if last_ts is not None:
            new_df = df[df['datetime_utc'] > last_ts]
        else:
            new_df = df
        if not new_df.empty:
            last_ts = new_df['datetime_utc'].max()
            yield new_df
        if stop_after is not None and (time.time() - start_time) > stop_after:
            break
        time.sleep(poll_interval)

def write_outputs(df, basepath, table_name):
    """Write DataFrame to CSV, Parquet timing each step."""
    import time
    timings = {}
    csv_file = f"{basepath}.csv"
    parquet_file = f"{basepath}.parquet"

    print(f"Writing to {csv_file} (CSV)...")
    t_csv0 = time.perf_counter()
    file_writers.to_csv(df, csv_file, mode='a' if os.path.exists(csv_file) else 'w', header=not os.path.exists(csv_file))
    t_csv1 = time.perf_counter()
    timings['csv'] = t_csv1 - t_csv0

    print(f"Writing to {parquet_file} (Parquet)...")
    t_parquet0 = time.perf_counter()
    file_writers.to_parquet(df, parquet_file, append=True)
    t_parquet1 = time.perf_counter()
    timings['parquet'] = t_parquet1 - t_parquet0

    print("Done.")
    print("Timing summary:")
    for k, v in timings.items():
        print(f"  {k.capitalize()} write: {v:.4f} seconds")

def write_resampled_to_sqlite(df, sqlite_path, output_table):
    """
    Write resampled DataFrame to the underway_summary table in SQLite with integer datetime_utcs.

    Args:
        df: DataFrame with columns: datetime_utc, lat, lon, rhodamine, ph, temp, salinity, ph_ma
        sqlite_path: Path to SQLite database
        output_table: Name of the output table in SQLite
    """
    # Convert datetime timestamps to Unix timestamps (integers)
    df_copy = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_copy['datetime_utc']):
        df_copy['datetime_utc'] = df_copy['datetime_utc'].astype('int64') // 10**9  # Convert to Unix timestamp
    
    conn = sqlite3.connect(sqlite_path)
    try:
        # Insert data into output_table
        df_copy.to_sql(output_table, conn, if_exists='append', index=False)
        print(f"Successfully wrote {len(df_copy)} resampled records to {output_table} table")
    except Exception as e:
        print(f"Error writing to {output_table} table: {e}")
    finally:
        conn.close()

def write_resampled_data_to_sqlite(sqlite_path, resample_interval='2s', output_table='underway_summary'):
    """
    Load, resample data, and write directly to the underway_summary table in SQLite.

    Args:
        sqlite_path: Path to SQLite database
        resample_interval: Resample interval (default '2s')
        output_table: Name of output table (default 'underway_summary')
    """
    # Load and resample data
    df = load_and_resample_sqlite(sqlite_path, resample_interval)

    # Write to the underway_summary table
    write_resampled_to_sqlite(df, sqlite_path, output_table)
    
    return df

def main():
    import argparse
    config = get_config()
    parser = argparse.ArgumentParser(description="Resample and combine SQLite sensor tables, write to CSV, Parquet, and DuckDB.")
    parser.add_argument('--sqlite-path', type=str, default=config.get('db_path'), help='Path to SQLite database (default from config)')
    parser.add_argument('--path', type=str, default=config.get('cloud_path', '.'), help='Directory to write output files (default from config)')
    parser.add_argument('--basename', type=str, default=config.get('basename', 'test_data'), help='Base name for output files (no extension)')
    parser.add_argument('--table', type=str, default=config.get('summary_table', 'underway_summary'), help='SQLite table name (default from config)')
    parser.add_argument('--resample', type=str, default='2s', help='Resample interval (default: 2s)')
    parser.add_argument('--poll', action='store_true', help='Continuously poll for new records')
    parser.add_argument('--interval', type=float, default=2.0, help='Polling interval in seconds (default: 2.0)')
    args = parser.parse_args()

    if not args.sqlite_path:
        print("Error: SQLite database path not specified in config or command line")
        return

    basepath = os.path.join(args.path, args.basename)

    if args.poll:
        print("Polling mode enabled. Press Ctrl+C to stop.")
        last_ts = None
        try:
            for new_df in poll_new_records(args.sqlite_path, last_timestamp=last_ts, poll_interval=args.interval, resample_interval=args.resample, stop_after=args.stop_after):
                if not new_df.empty:
                    print(f"Writing {len(new_df)} new records...")
                    write_outputs(new_df, basepath, args.table)
                    write_resampled_to_sqlite(new_df, args.sqlite_path, output_table=args.table)
                    last_ts = new_df['datetime_utc'].max()
        except KeyboardInterrupt:
            print("\nStopped polling.")
    else:
        df = load_and_resample_sqlite(args.sqlite_path, resample_interval=args.resample)
        write_outputs(df, basepath, args.table)
        write_resampled_to_sqlite(df, args.sqlite_path, output_table=args.table)

if __name__ == "__main__":
    main()

# Example usage:
# df = load_and_resample_sqlite('mydata.sqlite')
# print(df.head())
