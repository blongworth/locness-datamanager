import sqlite3
import pandas as pd
import time
from typing import Optional
import os
import warnings
from locness_datamanager.config import get_config
from locness_datamanager import file_writers
from isfetphcalc import calc_ph

# TODO: Use mean for resampling?
# TODO: add error handling for database connection and queries
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
    ph = read_table(conn, 'ph', ['datetime_utc', 'vrse', 'ph_total'])
    tsg = read_table(conn, 'tsg', ['datetime_utc', 'temp', 'salinity'])
    gps = read_table(conn, 'gps', ['datetime_utc', 'latitude', 'longitude'])
    conn.close()
    return fluoro, ph, tsg, gps

def resample_raw_data(fluoro, ph, tsg, gps, resample_interval=None):
    """
    Resample each table to the given interval and join into a single DataFrame.
    If resample_interval is None, get it from config['res_int'].
    If all source DataFrames are empty, return an empty DataFrame with expected columns.
    """
    expected_cols = ['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'vrse', 'temp', 'salinity']

    # If all DataFrames are empty, return empty DataFrame with expected columns
    if all(df.empty for df in [fluoro, ph, tsg, gps]):
        return pd.DataFrame(columns=expected_cols)

    # Helper to prepare each DataFrame
    def prep_df(df, name):
        if 'datetime_utc' not in df.columns or df.empty:
            warnings.warn(f"Table '{name}' missing 'datetime_utc' or is empty. Skipping.")
            return pd.DataFrame()
        df = df.copy()
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        df = df.drop_duplicates(subset='datetime_utc').set_index('datetime_utc')
        return df

    fluoro = prep_df(fluoro, 'rhodamine')
    ph = prep_df(ph, 'ph')
    tsg = prep_df(tsg, 'tsg')
    gps = prep_df(gps, 'gps')

    # Resample using nearest if not empty
    def resample_nearest(df):
        return df.resample(resample_interval).nearest() if not df.empty else pd.DataFrame()

    fluoro_res = resample_nearest(fluoro)
    ph_res = resample_nearest(ph)
    tsg_res = resample_nearest(tsg)
    gps_res = resample_nearest(gps)

    # Join all on datetime_utc
    df = fluoro_res.join([ph_res, tsg_res, gps_res], how='outer').reset_index()

    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in df.columns:
            df[col] = pd.NA
            
    # Return only expected columns
    return df[expected_cols]

def add_corrected_ph(df, k0=0.0, k2=0.0):
    """
    Add a corrected pH column to the DataFrame.
    """
    # If temp or salinity are missing, fill ph_corrected with NaN
    if 'temp' not in df.columns or 'salinity' not in df.columns:
        df['ph_corrected'] = pd.NA if hasattr(pd, 'NA') else float('nan')
        return df
    # If any values in temp or salinity are missing, set ph_corrected to NaN for those rows
    mask = df['temp'].isna() | df['salinity'].isna()
    ph_free, ph_total = calc_ph(Vrs=df['vrse'],
                                Press=0,
                                Temp=df['temp'],
                                Salt=df['salinity'],
                                k0=0.0,
                                k2=0.0,
                                Pcoefs=0)
    df['ph_corrected'] = ph_total
    if mask.any():
        df.loc[mask, 'ph_corrected'] = pd.NA if hasattr(pd, 'NA') else float('nan')
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
    # Moving average for ph_corrected
    if 'ph_corrected' in df.columns:
        df['ph_corrected_ma'] = df['ph_corrected'].rolling(window=window_size, min_periods=1).mean()
    # Moving average for ph_total
    if 'ph_total' in df.columns:
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
    ph_k0 = config.get('k0', 0.0)
    ph_k2 = config.get('k2', 0.0)
    # Add corrected pH and moving averages
    df = add_corrected_ph(df, k0=ph_k0, k2=ph_k2)
    df = add_ph_moving_average(df, window_seconds=window_seconds, freq_hz=freq_hz)
    return df

def load_and_resample_sqlite(sqlite_path, resample_interval=None):
    """
    Load, resample, and add computed fields to sensor data from SQLite.
    Returns a DataFrame with columns: datetime_utc, latitude, longitude, rho_ppb, ph, temp_c, salinity_psu, ph_ma
    """
    config = get_config()
    fluoro, ph, tsg, gps = load_sqlite_tables(sqlite_path)
    df = resample_raw_data(fluoro, ph, tsg, gps, resample_interval)
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

def get_last_summary_timestamp(sqlite_path, summary_table='underway_summary'):
    """
    Get the most recent timestamp from the summary table.
    
    Args:
        sqlite_path: Path to SQLite database
        summary_table: Name of the summary table
        
    Returns:
        pandas.Timestamp or None if table is empty
    """
    conn = sqlite3.connect(sqlite_path)
    try:
        query = f"SELECT MAX(datetime_utc) FROM {summary_table}"
        result = pd.read_sql_query(query, conn)
        max_timestamp = result.iloc[0, 0]
        
        if max_timestamp is not None:
            # Convert from Unix timestamp to pandas datetime
            return pd.to_datetime(max_timestamp, unit='s')
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not get last timestamp from {summary_table}: {e}")
        return None
    finally:
        conn.close()

def load_sqlite_tables_after_timestamp(sqlite_path, after_timestamp=None):
    """
    Load raw tables from SQLite, optionally filtering for records after a timestamp.
    
    Args:
        sqlite_path: Path to SQLite database
        after_timestamp: Only return records after this timestamp (if None, returns all)
        
    Returns: 
        fluoro, ph, tsg, gps DataFrames
    """
    conn = sqlite3.connect(sqlite_path)
    
    # Prepare WHERE clause if timestamp filtering is needed
    where_clause = ""
    if after_timestamp is not None:
        # Convert pandas timestamp to Unix timestamp for comparison
        unix_timestamp = int(after_timestamp.timestamp())
        where_clause = f" WHERE datetime_utc > {unix_timestamp}"
    
    fluoro = read_table_with_filter(conn, 'rhodamine', ['datetime_utc','rho_ppb'], where_clause)
    ph = read_table_with_filter(conn, 'ph', ['datetime_utc', 'vrse', 'ph_total'], where_clause)
    tsg = read_table_with_filter(conn, 'tsg', ['datetime_utc', 'temp', 'salinity'], where_clause)
    gps = read_table_with_filter(conn, 'gps', ['datetime_utc', 'latitude', 'longitude'], where_clause)
    
    conn.close()
    return fluoro, ph, tsg, gps

def read_table_with_filter(conn, table, columns, where_clause=""):
    """
    Read a table with optional WHERE filter.
    """
    query = f"SELECT {', '.join(columns)} FROM {table}{where_clause}"
    df = pd.read_sql_query(query, conn)
    # Convert integer timestamps to datetime if needed
    if 'datetime_utc' in df.columns and df['datetime_utc'].dtype in ['int64', 'int32']:
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='s')
    return df

def process_raw_data_incremental(
    sqlite_path, 
    resample_interval='2s',
    summary_table='underway_summary',
    write_csv=False,
    write_parquet=False,
    csv_path=None,
    parquet_path=None,
    replace_all=False
):
    """
    Main function to process raw sensor data and update summary table.
    
    This function:
    1. Checks the timestamp of the most recent entry in the summary table
    2. Extracts more recent records from the raw tables  
    3. Resamples and adds computed fields to new data
    4. Appends new resampled records to the summary table
    5. Optionally writes to CSV and/or Parquet files
    
    Args:
        sqlite_path: Path to SQLite database
        resample_interval: Resampling interval (e.g., '2s')
        summary_table: Name of the summary table (default 'underway_summary')
        write_csv: Whether to append to CSV file
        write_parquet: Whether to append to Parquet file  
        csv_path: Path to CSV file (if None, uses sqlite_path base + '.csv')
        parquet_path: Path to Parquet file (if None, uses sqlite_path base + '.parquet')
        replace_all: If True, reprocess all raw data and replace summary table
        
    Returns:
        DataFrame of processed records (new records only, unless replace_all=True)
    """
    config = get_config()
    
    # Set default file paths if not provided
    if csv_path is None:
        csv_path = sqlite_path.replace('.sqlite', '_resampled.csv').replace('.db', '_resampled.csv')
    if parquet_path is None:
        parquet_path = sqlite_path.replace('.sqlite', '_resampled.parquet').replace('.db', '_resampled.parquet')
    
    if replace_all:
        print("Processing ALL raw data and replacing summary table...")
        # Load all raw data
        fluoro, ph, tsg, gps = load_sqlite_tables(sqlite_path)
        last_timestamp = None
        
        # Clear the summary table
        conn = sqlite3.connect(sqlite_path)
        try:
            conn.execute(f"DELETE FROM {summary_table}")
            conn.commit()
            print(f"Cleared existing data from {summary_table} table")
        except Exception as e:
            print(f"Warning: Could not clear {summary_table} table: {e}")
        finally:
            conn.close()
            
    else:
        print("Processing incremental raw data updates...")
        # Get the most recent timestamp from summary table
        last_timestamp = get_last_summary_timestamp(sqlite_path, summary_table)
        
        if last_timestamp is not None:
            print(f"Last summary timestamp: {last_timestamp}")
        else:
            print("No existing data in summary table, processing all raw data")
            
        # Load raw data after the last timestamp
        fluoro, ph, tsg, gps = load_sqlite_tables_after_timestamp(sqlite_path, last_timestamp)
    
    # Check if there's any new data to process
    total_new_records = sum(len(df) for df in [fluoro, ph, tsg, gps])
    if total_new_records == 0:
        print("No new raw data to process")
        return pd.DataFrame()
    
    print(f"Found {len(fluoro)} new rhodamine, {len(ph)} new pH, {len(tsg)} new TSG, {len(gps)} new GPS records")
    
    # Resample and add computed fields
    print(f"Resampling data with interval: {resample_interval}")
    df = resample_raw_data(fluoro, ph, tsg, gps, resample_interval)
    df = add_computed_fields(df, config)

    # Filter out records with datetime_utc <= last_summary_timestamp (if last_summary_timestamp exists)
    if last_timestamp is not None and not df.empty:
        # Ensure both are comparable (convert to pandas.Timestamp if needed)
        if not pd.api.types.is_datetime64_any_dtype(df['datetime_utc']):
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        df = df[df['datetime_utc'] > last_timestamp]

    if df.empty:
        print("No data after resampling")
        return df

    print(f"Generated {len(df)} resampled records")

    # Write to summary table
    print(f"Writing to {summary_table} table...")
    write_resampled_to_sqlite(df, sqlite_path, summary_table)

    # --- Recalculate and update moving averages for affected rows ---
    # Determine window size and frequency from config
    window_seconds = config.get('ph_ma_window', 120)
    freq_hz = config.get('ph_freq', 0.5)
    window_size = max(1, int(window_seconds * freq_hz))

    # Find the earliest new datetime_utc in this batch
    if not df.empty:
        min_new_dt = df['datetime_utc'].min()
        # Load enough previous rows to cover the moving average window
        with sqlite3.connect(sqlite_path) as conn:
            # Get all rows with datetime_utc >= (min_new_dt - window_seconds)
            min_new_unix = int((min_new_dt - pd.Timedelta(seconds=window_seconds)).timestamp())
            query = f"SELECT * FROM {summary_table} WHERE datetime_utc >= {min_new_unix} ORDER BY datetime_utc"
            df_window = pd.read_sql_query(query, conn)
        # Convert datetime_utc to datetime
        if not pd.api.types.is_datetime64_any_dtype(df_window['datetime_utc']):
            df_window['datetime_utc'] = pd.to_datetime(df_window['datetime_utc'], unit='s')
        # Recalculate moving averages
        df_window = add_ph_moving_average(df_window, window_seconds=window_seconds, freq_hz=freq_hz)
        # Only update rows that are in the new batch (datetime_utc >= min_new_dt)
        update_rows = df_window[df_window['datetime_utc'] >= min_new_dt]
        # Write updated moving averages back to the database
        with sqlite3.connect(sqlite_path) as conn:
            for _, row in update_rows.iterrows():
                dt_unix = int(row['datetime_utc'].timestamp())
                updates = []
                if 'ph_corrected_ma' in row:
                    updates.append(f"ph_corrected_ma = {row['ph_corrected_ma'] if pd.notnull(row['ph_corrected_ma']) else 'NULL'}")
                if 'ph_total_ma' in row:
                    updates.append(f"ph_total_ma = {row['ph_total_ma'] if pd.notnull(row['ph_total_ma']) else 'NULL'}")
                if updates:
                    set_clause = ', '.join(updates)
                    conn.execute(f"UPDATE {summary_table} SET {set_clause} WHERE datetime_utc = {dt_unix}")
            conn.commit()

    # Optionally write to CSV
    if write_csv:
        print(f"Writing to CSV: {csv_path}")
        mode = 'w' if replace_all else 'a'
        header = replace_all or not os.path.exists(csv_path)
        file_writers.to_csv(df, csv_path, mode=mode, header=header)

    # Optionally write to Parquet
    if write_parquet:
        print(f"Writing to Parquet: {parquet_path}")
        append = not replace_all
        partition_hours = config.get('partition_hours', None)
        file_writers.to_parquet(df, parquet_path, append=append, partition_hours=partition_hours)

    print(f"Processing complete. {len(df)} records processed.")
    return df

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

def main():
    import argparse
    config = get_config()
    parser = argparse.ArgumentParser(description="Resample and combine SQLite sensor tables, write to CSV, Parquet, and summary table.")
    parser.add_argument('--sqlite-path', type=str, default=config.get('db_path'), help='Path to SQLite database (default from config)')
    parser.add_argument('--csv-path', type=str, help='Path to CSV output file (default: sqlite_path base + _resampled.csv)')
    parser.add_argument('--parquet-path', type=str, help='Path to Parquet output file (default: sqlite_path base + _resampled.parquet)')
    parser.add_argument('--table', type=str, default='underway_summary', help='Summary table name (default: underway_summary)')
    parser.add_argument('--resample', type=str, default=config.get('res_int', '2s'), help='Resample interval (default from config or 2s)')
    parser.add_argument('--csv', action='store_true', help='Write to CSV file')
    parser.add_argument('--parquet', action='store_true', help='Write to Parquet file')
    parser.add_argument('--replace-all', action='store_true', help='Reprocess all raw data and replace summary table (default: incremental)')
    parser.add_argument('--poll', action='store_true', help='Continuously poll for new records')
    parser.add_argument('--interval', type=float, default=2.0, help='Polling interval in seconds (default: 2.0)')
    parser.add_argument('--stop-after', type=float, help='Stop polling after this many seconds')
    args = parser.parse_args()

    if not args.sqlite_path:
        print("Error: SQLite database path not specified in config or command line")
        return

    if args.poll:
        print("Polling mode enabled. Press Ctrl+C to stop.")
        last_ts = None
        try:
            for new_df in poll_new_records(
                args.sqlite_path, 
                last_timestamp=last_ts, 
                poll_interval=args.interval, 
                resample_interval=args.resample, 
                stop_after=args.stop_after
            ):
                if not new_df.empty:
                    print(f"Writing {len(new_df)} new records...")
                    write_resampled_to_sqlite(new_df, args.sqlite_path, output_table=args.table)
                    
                    # Write to files if requested
                    if args.csv:
                        csv_path = args.csv_path or args.sqlite_path.replace('.sqlite', '_resampled.csv').replace('.db', '_resampled.csv')
                        file_writers.to_csv(new_df, csv_path, mode='a', header=not os.path.exists(csv_path))
                    
                    if args.parquet:
                        parquet_path = args.parquet_path or args.sqlite_path.replace('.sqlite', '_resampled.parquet').replace('.db', '_resampled.parquet')
                        partition_hours = config.get('partition_hours', None)
                        file_writers.to_parquet(new_df, parquet_path, append=True, partition_hours=partition_hours)
                    
                    last_ts = new_df['datetime_utc'].max()
        except KeyboardInterrupt:
            print("\nStopped polling.")
    else:
        # Use the new incremental processing function
        df = process_raw_data_incremental(
            sqlite_path=args.sqlite_path,
            resample_interval=args.resample,
            summary_table=args.table,
            write_csv=args.csv,
            write_parquet=args.parquet,
            csv_path=args.csv_path,
            parquet_path=args.parquet_path,
            replace_all=args.replace_all
        )
        
        if not df.empty:
            print(f"Successfully processed {len(df)} records")
        else:
            print("No records to process")

if __name__ == "__main__":
    main()

# Example usage:
# df = load_and_resample_sqlite('mydata.sqlite')
# print(df.head())
