import os
import sys
import time
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from locness_datamanager.config import get_config
from locness_datamanager import file_writers
from locness_datamanager.resample import add_ph_moving_average

# TODO: add synthetic raw data to ph, tsg, and fluorometer tables

def generate(n_records=1, base_lat=42.5, base_lon=-69.5, start_time=None, frequency_hz=1.0):
    """
    Generate synthetic oceanographic data.
    Args:
        n_records: Number of records to generate
        base_lat: Base latitude
        base_lon: Base longitude
        start_time: Optional datetime to start timestamps
    Returns:
        pandas.DataFrame with synthetic data
    """
    if start_time is None:
        current_time = datetime.now()
    else:
        current_time = start_time
    delta_seconds = 1.0 / frequency_hz if frequency_hz > 0 else 1.0
    data = []
    lat = base_lat
    lon = base_lon
    for i in range(n_records):
        # Move a random number of degrees (small, to simulate realistic movement)
        delta_lat = np.random.uniform(-0.0005, 0.0005)  # ~±50 meters in degrees latitude
        delta_lon = np.random.uniform(-0.0005, 0.0005)  # ~±50 meters in degrees longitude
        if i > 0:
            lat += delta_lat
            lon += delta_lon
        record = {
            "timestamp": current_time + timedelta(seconds=i * delta_seconds),
            "lat": lat,
            "lon": lon,
            "temp": 15 + 5 * np.sin(i * 0.02) + np.random.normal(0, 0.5),
            "salinity": 35 + 2 * np.sin(i * 0.015) + np.random.normal(0, 0.2),
            "rhodamine": min(500, np.random.exponential(scale=1.25)),
            "ph": 8.1 + 0.3 * np.sin(i * 0.025) + np.random.normal(0, 0.05),
        }
        data.append(record)
    return pd.DataFrame(data)

def parse_args():
    """Parse command-line arguments."""
    config = get_config()
    parser = argparse.ArgumentParser(description="Generate synthetic oceanographic data and write to CSV, Parquet, DuckDB, and SQLite.")
    parser.add_argument('--path', type=str, default=config['path'], help='Directory to write output files (default: current directory)')
    parser.add_argument('--basename', type=str, default=config['basename'], help='Base name for output files (no extension)')
    parser.add_argument('--num', type=int, default=config['num'], help='Number of records to generate per batch (default: 1000)')
    parser.add_argument('--freq', type=float, default=config['freq'], help='Sample frequency in Hz (default: 1.0)')
    parser.add_argument('--table', type=str, default=config['table'], help='DuckDB/SQLite table name (default: sensor_data)')
    parser.add_argument('--sqlite', type=str, default=None, help='Path to SQLite output file (default: {basename}.sqlite in output path)')
    parser.add_argument('--continuous', action='store_true', default=config['continuous'], help='Continuously generate and write data every (num * freq) seconds')
    return parser.parse_args()

def write_outputs(df, basepath, table_name, sqlite_file=None):
    """Write DataFrame to CSV, Parquet, DuckDB, and SQLite, timing each step."""
    timings = {}
    csv_file = f"{basepath}.csv"
    parquet_file = f"{basepath}.parquet"
    db_file = f"{basepath}.duckdb"
    sqlite_file = sqlite_file or f"{basepath}.sqlite"

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

    print(f"Writing to {sqlite_file} (SQLite table: {table_name}) ...")
    t_sqlite0 = time.perf_counter()
    file_writers.to_sqlite(df, sqlite_file, table_name=table_name)
    t_sqlite1 = time.perf_counter()
    timings['sqlite'] = t_sqlite1 - t_sqlite0

    print("Done.")
    print("Timing summary:")
    for k, v in timings.items():
        print(f"  {k.capitalize()} write: {v:.4f} seconds")

def generate_batch(num, freq):
    """Generate a batch of synthetic data and return the DataFrame and timing."""
    print(f"Generating {num} samples at {freq} Hz...")
    t0 = time.perf_counter()
    # set start_time so last sample is now
    delta_seconds = 1.0 / freq if freq > 0 else 1.0
    start_time = datetime.now() - timedelta(seconds=(num - 1) * delta_seconds)
    df = generate(n_records=num, frequency_hz=freq, start_time=start_time)
    # Add moving average of pH
    config = get_config()
    window_seconds = config.get('ph_ma_window', 120)
    df = add_ph_moving_average(df, window_seconds=window_seconds, freq_hz=freq)
    t1 = time.perf_counter()
    print(f"  Data generation: {t1-t0:.4f} seconds")
    return df

def main():
    args = parse_args()
    basepath = os.path.join(args.path, args.basename)
    sqlite_file = args.sqlite or f"{basepath}.sqlite"
    if args.continuous:
        print("Continuous mode enabled. Press Ctrl+C to stop.")
        interval = args.num / args.freq if args.freq > 0 else args.num
        try:
            # First batch: generate and write immediately
            print(f"Preparing first batch of {args.num} records...")
            df = generate_batch(args.num, args.freq)
            write_outputs(df, basepath, args.table, sqlite_file)
            next_write_time = time.time() + interval
            while True:
                print(f"Preparing next batch of {args.num} records...")
                df = generate_batch(args.num, args.freq)
                now = time.time()
                sleep_time = next_write_time - now
                if sleep_time > 0:
                    print(f"Sleeping {sleep_time:.2f} seconds before writing batch...")
                    time.sleep(sleep_time)
                print(f"Writing batch at {datetime.now().isoformat(timespec='seconds')}...")
                write_outputs(df, basepath, args.table, sqlite_file)
                next_write_time += interval
        except KeyboardInterrupt:
            print("\nStopped continuous generation.")
            sys.exit(0)
    else:
        df = generate_batch(args.num, args.freq)
        write_outputs(df, basepath, args.table, sqlite_file)


if __name__ == "__main__":
    main()