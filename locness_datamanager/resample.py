from locness_datamanager.config import get_config
import sqlite3
import pandas as pd
import time
from typing import Optional
import os
from locness_datamanager import file_writers

# TODO: calculate salinity from conductivity and temperature if needed
# TODO: add moving average for pH
# TODO: correct field names to match database schema
# TODO: add error handling for database connection and queries
# TODO: use integer timestamps for SQLite compatibility
# TODO: use time bucket resampling in sqlite (test speed)

def read_table(conn, table, columns):
    query = f"SELECT {', '.join(columns)} FROM {table}"
    df = pd.read_sql_query(query, conn)
    # Convert integer timestamps to datetime if needed
    if 'timestamp' in df.columns and df['timestamp'].dtype in ['int64', 'int32']:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def load_and_resample_sqlite(sqlite_path, resample_interval='2s'):
    """
    Read and resample data from fluorometer, ph, and tsg tables in a SQLite database.
    Returns a DataFrame with columns: timestamp, lat, lon, rhodamine, ph, temp, salinity
    """
    conn = sqlite3.connect(sqlite_path)
    # Read tables
    fluoro = read_table(conn, 'fluorometer', ['timestamp', 'lat', 'lon', 'rhodamine'])
    ph = read_table(conn, 'ph', ['timestamp', 'ph'])
    tsg = read_table(conn, 'tsg', ['timestamp', 'temp', 'salinity'])
    conn.close()

    # Set timestamp as index and ensure datetime
    fluoro['timestamp'] = pd.to_datetime(fluoro['timestamp'])
    ph['timestamp'] = pd.to_datetime(ph['timestamp'])
    tsg['timestamp'] = pd.to_datetime(tsg['timestamp'])
    fluoro = fluoro.set_index('timestamp')
    ph = ph.set_index('timestamp')
    tsg = tsg.set_index('timestamp')

    # Resample to every 2 seconds
    fluoro_res = fluoro.resample(resample_interval).nearest()
    ph_res = ph.resample(resample_interval).nearest()
    tsg_res = tsg.resample(resample_interval).nearest()

    # Combine all on timestamp
    df = fluoro_res.join([ph_res, tsg_res], how='outer')
    df = df.reset_index()
    # Reorder columns
    cols = ['timestamp', 'lat', 'lon', 'rhodamine', 'ph', 'temp', 'salinity']
    df = df[cols]
    # Add moving average of pH
    config = get_config()
    window_seconds = config.get('ph_ma_window', 120)
    freq_hz = config.get('ph_freq', 0.5)
    df = add_ph_moving_average(df, window_seconds=window_seconds, freq_hz=freq_hz)
    return df

def add_ph_moving_average(df, window_seconds=120, freq_hz=1.0):
    """
    Add a moving average column for pH to the DataFrame.
    window_seconds: window size in seconds
    freq_hz: sampling frequency in Hz
    """
    if 'timestamp' in df.columns:
        # Sort by timestamp to ensure correct rolling
        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)
    window_size = max(1, int(window_seconds * freq_hz))
    df['ph_ma'] = df['ph'].rolling(window=window_size, min_periods=1).mean()
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
            new_df = df[df['timestamp'] > last_ts]
        else:
            new_df = df
        if not new_df.empty:
            last_ts = new_df['timestamp'].max()
            yield new_df
        if stop_after is not None and (time.time() - start_time) > stop_after:
            break
        time.sleep(poll_interval)

def write_outputs(df, basepath, table_name):
    """Write DataFrame to CSV, Parquet, and DuckDB, timing each step."""
    import time
    timings = {}
    csv_file = f"{basepath}.csv"
    parquet_file = f"{basepath}.parquet"
    db_file = f"{basepath}.duckdb"

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

    print(f"Writing to {db_file} (DuckDB table: {table_name}) ...")
    t_db0 = time.perf_counter()
    file_writers.to_duckdb(df, db_file, table_name=table_name)
    t_db1 = time.perf_counter()
    timings['duckdb'] = t_db1 - t_db0

    print("Done.")
    print("Timing summary:")
    for k, v in timings.items():
        print(f"  {k.capitalize()} write: {v:.4f} seconds")

def write_resampled_to_sqlite(df, sqlite_path, output_table):
    """
    Write resampled DataFrame to the resampled_data table in SQLite with integer timestamps.
    
    Args:
        df: DataFrame with columns: timestamp, lat, lon, rhodamine, ph, temp, salinity, ph_ma
        sqlite_path: Path to SQLite database
        output_table: Name of the output table in SQLite
    """
    # Convert datetime timestamps to Unix timestamps (integers)
    df_copy = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = df_copy['timestamp'].astype('int64') // 10**9  # Convert to Unix timestamp
    
    conn = sqlite3.connect(sqlite_path)
    try:
        # Insert data into output_table
        df_copy.to_sql(output_table, conn, if_exists='append', index=False)
        print(f"Successfully wrote {len(df_copy)} resampled records to {output_table} table")
    except Exception as e:
        print(f"Error writing to {output_table} table: {e}")
    finally:
        conn.close()

def write_resampled_data_to_sqlite(sqlite_path, resample_interval='2s', output_table='resampled_data'):
    """
    Load, resample data, and write directly to the resampled_data table in SQLite.
    
    Args:
        sqlite_path: Path to SQLite database
        resample_interval: Resample interval (default '2s')
        output_table: Name of output table (default 'resampled_data')
    """
    # Load and resample data
    df = load_and_resample_sqlite(sqlite_path, resample_interval)
    
    # Write to the resampled_data table
    write_resampled_to_sqlite(df, sqlite_path, output_table)
    
    return df

def main():
    import argparse
    config = get_config()
    parser = argparse.ArgumentParser(description="Resample and combine SQLite sensor tables, write to CSV, Parquet, and DuckDB.")
    parser.add_argument('--sqlite-path', type=str, default=config.get('db_path'), help='Path to SQLite database (default from config)')
    parser.add_argument('--path', type=str, default=config.get('cloud_path', '.'), help='Directory to write output files (default from config)')
    parser.add_argument('--basename', type=str, default=config.get('basename', 'resampled_data'), help='Base name for output files (no extension)')
    parser.add_argument('--table', type=str, default=config.get('summary_table', 'sensor_data'), help='DuckDB table name (default from config)')
    parser.add_argument('--resample', type=str, default='2s', help='Resample interval (default: 2s)')
    parser.add_argument('--poll', action='store_true', help='Continuously poll for new records')
    parser.add_argument('--interval', type=float, default=2.0, help='Polling interval in seconds (default: 2.0)')
    parser.add_argument('--stop-after', type=float, default=None, help='Stop polling after this many seconds (default: run forever)')
    parser.add_argument('--write-to-sqlite', action='store_true', help='Write resampled data back to SQLite database')
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
                    if args.write_to_sqlite:
                        write_resampled_to_sqlite(new_df, args.sqlite_path)
                    last_ts = new_df['timestamp'].max()
        except KeyboardInterrupt:
            print("\nStopped polling.")
    else:
        df = load_and_resample_sqlite(args.sqlite_path, resample_interval=args.resample)
        write_outputs(df, basepath, args.table)
        if args.write_to_sqlite:
            write_resampled_to_sqlite(df, args.sqlite_path)

if __name__ == "__main__":
    main()

# Example usage:
# df = load_and_resample_sqlite('mydata.sqlite')
# print(df.head())
