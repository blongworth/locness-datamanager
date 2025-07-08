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
import sqlite3

# Functions for generating synthetic data for fluorometer, ph, and tsg tables

def generate_fluorometer_data(n_records=1, base_lat=42.5, base_lon=-69.5, start_time=None, frequency_hz=1.0):
    """
    Generate synthetic fluorometer data matching the fluorometer table schema.
    Args:
        n_records: Number of records to generate
        base_lat: Base latitude
        base_lon: Base longitude
        start_time: Optional datetime to start timestamps
        frequency_hz: Sampling frequency in Hz
    Returns:
        pandas.DataFrame with columns: timestamp, latitude, longitude, gain, voltage, concentration
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
        # Small random movement to simulate realistic GPS drift
        if i > 0:
            lat += np.random.uniform(-0.0005, 0.0005)
            lon += np.random.uniform(-0.0005, 0.0005)
        
        # Generate realistic fluorometer readings
        base_concentration = 1.5 + 0.8 * np.sin(i * 0.01) + np.random.normal(0, 0.3)
        concentration = max(0, base_concentration)  # Concentration can't be negative
        
        # Voltage typically correlates with concentration
        voltage = 0.5 + concentration * 0.3 + np.random.normal(0, 0.05)
        voltage = max(0, min(5.0, voltage))  # Voltage between 0-5V
        
        # Gain settings are typically discrete values
        gain = np.random.choice([1, 10, 100])
        
        record = {
            "timestamp": int((current_time + timedelta(seconds=i * delta_seconds)).timestamp()),
            "latitude": lat,
            "longitude": lon,
            "gain": gain,
            "voltage": voltage,
            "concentration": concentration,
        }
        data.append(record)
    
    return pd.DataFrame(data)

def generate_ph_data(n_records=1, start_time=None, frequency_hz=0.1):
    """
    Generate synthetic pH sensor data matching the ph table schema.
    Args:
        n_records: Number of records to generate
        start_time: Optional datetime to start timestamps
        frequency_hz: Sampling frequency in Hz (pH sensors typically sample slowly)
    Returns:
        pandas.DataFrame with pH sensor data columns
    """
    if start_time is None:
        current_time = datetime.now()
    else:
        current_time = start_time
    delta_seconds = 1.0 / frequency_hz if frequency_hz > 0 else 10.0
    
    data = []
    
    for i in range(n_records):
        timestamp = current_time + timedelta(seconds=i * delta_seconds)
        ph_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        pc_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Generate realistic pH sensor readings
        base_ph = 8.1 + 0.2 * np.sin(i * 0.05) + np.random.normal(0, 0.05)
        
        record = {
            "pc_timestamp": pc_timestamp,
            "samp_num": i + 1,
            "ph_timestamp": ph_timestamp,
            "v_bat": 12.0 + np.random.normal(0, 0.5),  # Battery voltage ~12V
            "v_bias_pos": 2.5 + np.random.normal(0, 0.1),  # Positive bias voltage
            "v_bias_neg": -2.5 + np.random.normal(0, 0.1),  # Negative bias voltage
            "t_board": 20.0 + np.random.normal(0, 2.0),  # Board temperature
            "h_board": 50.0 + np.random.normal(0, 10.0),  # Board humidity
            "vrse": 0.5 + np.random.normal(0, 0.05),  # Reference electrode voltage
            "vrse_std": np.random.uniform(0.001, 0.01),  # Standard deviation
            "cevk": 0.8 + np.random.normal(0, 0.1),  # Counter electrode voltage vs K+
            "cevk_std": np.random.uniform(0.001, 0.02),  # Standard deviation
            "ce_ik": np.random.normal(0, 0.01),  # Counter electrode current
            "i_sub": np.random.normal(0, 0.001),  # Substrate current
            "cal_temp": 20.0 + np.random.normal(0, 1.0),  # Calibration temperature
            "cal_sal": 35.0 + np.random.normal(0, 1.0),  # Calibration salinity
            "k0": -0.5 + np.random.normal(0, 0.1),  # Calibration coefficient K0
            "k2": 0.05 + np.random.normal(0, 0.01),  # Calibration coefficient K2
            "ph_free": base_ph + np.random.normal(0, 0.02),  # Free pH scale
            "ph_total": base_ph + 0.1 + np.random.normal(0, 0.02),  # Total pH scale
        }
        data.append(record)
    
    return pd.DataFrame(data)

def generate_tsg_data(n_records=1, base_lat=42.5, base_lon=-69.5, start_time=None, frequency_hz=1.0):
    """
    Generate synthetic TSG (Temperature, Salinity, GPS) data matching the tsg table schema.
    Args:
        n_records: Number of records to generate
        base_lat: Base latitude
        base_lon: Base longitude
        start_time: Optional datetime to start timestamps
        frequency_hz: Sampling frequency in Hz
    Returns:
        pandas.DataFrame with TSG data columns
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
        timestamp_dt = current_time + timedelta(seconds=i * delta_seconds)
        
        # Small random movement to simulate realistic GPS drift
        if i > 0:
            lat += np.random.uniform(-0.0005, 0.0005)
            lon += np.random.uniform(-0.0005, 0.0005)
        
        # Generate realistic oceanographic measurements
        temp = 15.0 + 5.0 * np.sin(i * 0.02) + np.random.normal(0, 0.5)
        hull_temp = temp + np.random.normal(2.0, 0.3)  # Hull temp usually higher
        
        # Conductivity relates to salinity and temperature
        salinity = 35.0 + 2.0 * np.sin(i * 0.015) + np.random.normal(0, 0.2)
        # Rough conversion: conductivity ≈ salinity * 1.7 (simplified)
        cond = salinity * 1.7 + temp * 0.02 + np.random.normal(0, 0.1)
        
        record = {
            "timestamp": int(timestamp_dt.timestamp()),
            "scan_no": i + 1,
            "cond": max(0, cond),  # Conductivity (S/m)
            "temp": temp,  # Temperature (°C)
            "hull_temp": hull_temp,  # Hull temperature (°C)
            "time_elapsed": i * delta_seconds,  # Elapsed time in seconds
            "nmea_time": int(timestamp_dt.timestamp()),  # NMEA timestamp
            "latitude": lat,
            "longitude": lon,
        }
        data.append(record)
    
    return pd.DataFrame(data)

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
    parser.add_argument('--continuous', action='store_true', default=config['continuous'], help='Continuously generate and write data every (num * freq) seconds')
    parser.add_argument('--csv', action='store_true', help='Write CSV output')
    parser.add_argument('--parquet', action='store_true', help='Write Parquet output')
    parser.add_argument('--duckdb', action='store_true', help='Write DuckDB output')
    parser.add_argument('--sqlite', action='store_true', help='Write SQLite output')
    parser.add_argument('--raw-sensors', action='store_true', help='Generate and write raw sensor data to fluorometer, ph, and tsg tables')
    parser.add_argument('--lat', type=float, default=42.5, help='Base latitude for GPS coordinates (default: 42.5)')
    parser.add_argument('--lon', type=float, default=-69.5, help='Base longitude for GPS coordinates (default: -69.5)')
    return parser.parse_args()


def write_outputs(df,
                  basepath,
                  table_name,
                  write_csv=True,
                  write_parquet=True,
                  write_duckdb=True,
                  write_sqlite=True):
    """Write DataFrame to selected outputs, timing each step."""
    timings = {}
    csv_file = f"{basepath}.csv"
    parquet_file = f"{basepath}.parquet"
    db_file = f"{basepath}.duckdb"
    sqlite_file = f"{basepath}.sqlite"

    if write_csv:
        print(f"Writing to {csv_file} (CSV)...")
        t_csv0 = time.perf_counter()
        file_writers.to_csv(df, csv_file, mode='a' if os.path.exists(csv_file) else 'w', header=not os.path.exists(csv_file))
        t_csv1 = time.perf_counter()
        timings['csv'] = t_csv1 - t_csv0

    if write_parquet:
        print(f"Writing to {parquet_file} (Parquet)...")
        t_parquet0 = time.perf_counter()
        # Use default partition_hours if not set
        config = get_config()
        partition_hours = config.get('partition_hours', None)
        file_writers.to_parquet(df, parquet_file, append=True, partition_hours=partition_hours)
        t_parquet1 = time.perf_counter()
        timings['parquet'] = t_parquet1 - t_parquet0

    if write_duckdb:
        print(f"Writing to {db_file} (DuckDB table: {table_name}) ...")
        t_db0 = time.perf_counter()
        file_writers.to_duckdb(df, db_file, table_name=table_name)
        t_db1 = time.perf_counter()
        timings['duckdb'] = t_db1 - t_db0

    if write_sqlite:
        print(f"Writing to {sqlite_file} (SQLite table: {table_name}) ...")
        t_sqlite0 = time.perf_counter()
        # If writing to resampled_data table, use special function with integer timestamps
        if table_name == 'resampled_data':
            write_to_resampled_data_table(df, sqlite_file)
        else:
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

def write_to_resampled_data_table(df, sqlite_path):
    """
    Write DataFrame to the resampled_data table with integer timestamps.
    
    Args:
        df: DataFrame with timestamp column and other sensor data
        sqlite_path: Path to SQLite database
    """
    df_copy = df.copy()
    
    # Convert datetime timestamps to Unix timestamps (integers)
    if pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        df_copy['timestamp'] = df_copy['timestamp'].astype('int64') // 10**9
    
    # Ensure column order matches the resampled_data table
    expected_columns = ['timestamp', 'lat', 'lon', 'rhodamine', 'ph', 'temp', 'salinity', 'ph_ma']
    
    # Only keep columns that exist in the DataFrame and are expected
    available_columns = [col for col in expected_columns if col in df_copy.columns]
    df_copy = df_copy[available_columns]
    
    conn = sqlite3.connect(sqlite_path)
    try:
        df_copy.to_sql('resampled_data', conn, if_exists='append', index=False)
        print(f"Successfully wrote {len(df_copy)} records to resampled_data table")
    except Exception as e:
        print(f"Error writing to resampled_data table: {e}")
    finally:
        conn.close()

def write_to_raw_tables(fluorometer_df=None, ph_df=None, tsg_df=None, sqlite_path="sensors.sqlite"):
    """
    Write synthetic data to the raw fluorometer, ph, and tsg tables in SQLite.
    
    Args:
        fluorometer_df: DataFrame with fluorometer data (optional)
        ph_df: DataFrame with pH data (optional)
        tsg_df: DataFrame with TSG data (optional)
        sqlite_path: Path to SQLite database
    """
    conn = sqlite3.connect(sqlite_path)
    
    try:
        if fluorometer_df is not None and not fluorometer_df.empty:
            fluorometer_df.to_sql('fluorometer', conn, if_exists='append', index=False)
            print(f"Successfully wrote {len(fluorometer_df)} records to fluorometer table")
        
        if ph_df is not None and not ph_df.empty:
            ph_df.to_sql('ph', conn, if_exists='append', index=False)
            print(f"Successfully wrote {len(ph_df)} records to ph table")
        
        if tsg_df is not None and not tsg_df.empty:
            tsg_df.to_sql('tsg', conn, if_exists='append', index=False)
            print(f"Successfully wrote {len(tsg_df)} records to tsg table")
            
    except Exception as e:
        print(f"Error writing to raw tables: {e}")
    finally:
        conn.close()

def generate_raw_sensor_batch(num, freq, start_time=None, base_lat=42.5, base_lon=-69.5):
    """
    Generate a batch of synthetic raw sensor data for all three sensor types.
    
    Args:
        num: Number of records to generate per sensor
        freq: Sampling frequency in Hz
        start_time: Optional start time (defaults to now - duration)
        base_lat: Base latitude
        base_lon: Base longitude
    
    Returns:
        dict: Dictionary containing DataFrames for each sensor type
    """
    print(f"Generating {num} raw sensor samples at {freq} Hz...")
    t0 = time.perf_counter()
    
    # Set start_time so last sample is now
    if start_time is None:
        delta_seconds = 1.0 / freq if freq > 0 else 1.0
        start_time = datetime.now() - timedelta(seconds=(num - 1) * delta_seconds)
    
    # Generate data for each sensor type with appropriate frequencies
    fluorometer_df = generate_fluorometer_data(
        n_records=num, 
        base_lat=base_lat, 
        base_lon=base_lon, 
        start_time=start_time, 
        frequency_hz=freq
    )
    
    # pH sensors typically sample much slower
    ph_freq = min(freq, 0.1)  # Max 0.1 Hz for pH sensor
    ph_records = max(1, int(num * ph_freq / freq))
    ph_df = generate_ph_data(
        n_records=ph_records, 
        start_time=start_time, 
        frequency_hz=ph_freq
    )
    
    tsg_df = generate_tsg_data(
        n_records=num, 
        base_lat=base_lat, 
        base_lon=base_lon, 
        start_time=start_time, 
        frequency_hz=freq
    )
    
    t1 = time.perf_counter()
    print(f"  Raw sensor data generation: {t1-t0:.4f} seconds")
    
    return {
        'fluorometer': fluorometer_df,
        'ph': ph_df,
        'tsg': tsg_df
    }

def main():
    args = parse_args()
    basepath = os.path.join(args.path, args.basename)
    
    # Determine which outputs to write
    any_selected = args.csv or args.parquet or args.duckdb or args.sqlite
    write_csv = args.csv or not any_selected
    write_parquet = args.parquet or not any_selected
    write_duckdb = args.duckdb or not any_selected
    write_sqlite = args.sqlite or not any_selected
    
    # Handle raw sensor data generation
    if args.raw_sensors:
        sqlite_file = f"{basepath}.sqlite"
        print(f"Raw sensor mode: writing to {sqlite_file}")
        
        if args.continuous:
            print("Continuous raw sensor mode enabled. Press Ctrl+C to stop.")
            interval = args.num / args.freq if args.freq > 0 else args.num
            try:
                while True:
                    print(f"Generating raw sensor batch of {args.num} records...")
                    sensor_data = generate_raw_sensor_batch(
                        args.num, args.freq, base_lat=args.lat, base_lon=args.lon
                    )
                    write_to_raw_tables(
                        fluorometer_df=sensor_data['fluorometer'],
                        ph_df=sensor_data['ph'],
                        tsg_df=sensor_data['tsg'],
                        sqlite_path=sqlite_file
                    )
                    print(f"Sleeping {interval:.2f} seconds before next batch...")
                    time.sleep(interval)
            except KeyboardInterrupt:
                print("\nStopped continuous raw sensor generation.")
                sys.exit(0)
        else:
            sensor_data = generate_raw_sensor_batch(
                args.num, args.freq, base_lat=args.lat, base_lon=args.lon
            )
            write_to_raw_tables(
                fluorometer_df=sensor_data['fluorometer'],
                ph_df=sensor_data['ph'],
                tsg_df=sensor_data['tsg'],
                sqlite_path=sqlite_file
            )
        return
    
    # Original resampled data generation
    if args.continuous:
        print("Continuous mode enabled. Press Ctrl+C to stop.")
        interval = args.num / args.freq if args.freq > 0 else args.num
        try:
            # First batch: generate and write immediately
            print(f"Preparing first batch of {args.num} records...")
            df = generate_batch(args.num, args.freq)
            write_outputs(df, basepath, args.table, write_csv, write_parquet, write_duckdb, write_sqlite)
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
                write_outputs(df, basepath, args.table, write_csv, write_parquet, write_duckdb, write_sqlite)
                next_write_time += interval
        except KeyboardInterrupt:
            print("\nStopped continuous generation.")
            sys.exit(0)
    else:
        df = generate_batch(args.num, args.freq)
        write_outputs(df, basepath, args.table, write_csv, write_parquet, write_duckdb, write_sqlite)


if __name__ == "__main__":
    main()

# Example usage of the new sensor-specific functions:
"""
# Generate fluorometer data
fluoro_df = generate_fluorometer_data(n_records=100, frequency_hz=1.0)
print("Fluorometer data shape:", fluoro_df.shape)
print("Fluorometer columns:", list(fluoro_df.columns))

# Generate pH data (slower sampling rate)
ph_df = generate_ph_data(n_records=10, frequency_hz=0.1)
print("pH data shape:", ph_df.shape)
print("pH columns:", list(ph_df.columns))

# Generate TSG data
tsg_df = generate_tsg_data(n_records=100, frequency_hz=1.0)
print("TSG data shape:", tsg_df.shape)
print("TSG columns:", list(tsg_df.columns))

# Generate all sensor types as a batch
sensor_batch = generate_raw_sensor_batch(num=50, freq=1.0)
print("Batch keys:", list(sensor_batch.keys()))

# Write to SQLite database (uncomment to use)
# write_to_raw_tables(
#     fluorometer_df=sensor_batch['fluorometer'],
#     ph_df=sensor_batch['ph'],
#     tsg_df=sensor_batch['tsg'],
#     sqlite_path="test_sensors.sqlite"
# )
"""