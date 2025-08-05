import sys
import time
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from locness_datamanager.config import get_config
from locness_datamanager.resample import process_raw_data_incremental
import sqlite3
from locness_datamanager.setup_db import setup_sqlite_db

# Functions for generating synthetic data for rhodamine, ph, and tsg tables

# TODO: reuse functions from main or resample after generating raw data and writing to SQLite

def generate_rhodamine_data(n_records=1, base_lat=42.5, base_lon=-69.5, start_time=None, frequency_hz=1.0):
    """
    Generate synthetic rhodamine data matching the rhodamine table schema.
    Args:
        n_records: Number of records to generate
        base_lat: Base latitude
        base_lon: Base longitude
        start_time: Optional datetime to start timestamps
        frequency_hz: Sampling frequency in Hz
    Returns:
        pandas.DataFrame with columns: datetime_utc, gain, voltage, rho_ppb
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
        
        # Generate realistic rhodamine readings
        base_concentration = 1.5 + 0.8 * np.sin(i * 0.01) + np.random.normal(0, 0.3)
        concentration = max(0, base_concentration)  # Concentration can't be negative
        
        # Voltage typically correlates with concentration
        voltage = 0.5 + concentration * 0.3 + np.random.normal(0, 0.05)
        voltage = max(0, min(5.0, voltage))  # Voltage between 0-5V
        
        # Gain settings are typically discrete values
        gain = np.random.choice([1, 10, 100])
        
        record = {
            "datetime_utc": int((current_time + timedelta(seconds=i * delta_seconds)).timestamp()),
            "gain": gain,
            "voltage": voltage,
            "rho_ppb": concentration,
        }
        data.append(record)
    
    return pd.DataFrame(data)

def generate_gps_data(n_records=1, base_lat=42.5, base_lon=-69.5, start_time=None, frequency_hz=1.0):
    """
    Generate synthetic GPS data matching the GPS table schema.
    Args:
        n_records: Number of records to generate
        base_lat: Base latitude
        base_lon: Base longitude
        start_time: Optional datetime to start timestamps
        frequency_hz: Sampling frequency in Hz
    Returns:
        pandas.DataFrame with columns: datetime_utc, nmea_time_utc, latitude, longitude
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
        
        record = {
            "datetime_utc": int((current_time + timedelta(seconds=i * delta_seconds)).timestamp()),
            "nmea_time_utc": (current_time + timedelta(seconds=i * delta_seconds)).strftime('%Y-%m-%d %H:%M:%S'),
            "latitude": lat,
            "longitude": lon,
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
        pc_timestamp = int(timestamp.timestamp())
        ph_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate realistic pH sensor readings
        base_ph = 8.1 + 0.2 * np.sin(i * 0.05) + np.random.normal(0, 0.05)
        
        record = {
            "datetime_utc": pc_timestamp,
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
            "cal_temp": 20.0,
            "cal_sal": 35.0,
            "k0": -0.5,
            "k2": 0.05,
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
            "datetime_utc": int(timestamp_dt.timestamp()),
            "scan_no": i + 1,
            "cond": max(0, cond),  # Conductivity (S/m)
            "temp": temp,  # Temperature (°C)
            "salinity": salinity,  # Salinity (PSU)
            "hull_temp": hull_temp,  # Hull temperature (°C)
            "time_elapsed": i * delta_seconds,  # Elapsed time in seconds
            "nmea_time": int(timestamp_dt.timestamp()),  # NMEA timestamp
            "latitude": lat,
            "longitude": lon,
        }
        data.append(record)
    
    return pd.DataFrame(data)

def write_to_raw_tables(rhodamine_df=None, ph_df=None, tsg_df=None, gps_df=None, sqlite_path="sensors.sqlite"):
    """
    Write synthetic data to the raw rhodamine, ph, tsg, and gps tables in SQLite.

    Args:
        rhodamine_df: DataFrame with rhodamine data (optional)
        ph_df: DataFrame with pH data (optional)
        tsg_df: DataFrame with TSG data (optional)
        gps_df: DataFrame with GPS data (optional)
        sqlite_path: Path to SQLite database
    """
    conn = sqlite3.connect(sqlite_path)
    
    try:
        # Create tables using the schema from CREATE_TABLES
        setup_sqlite_db(sqlite_path)
        
        if rhodamine_df is not None and not rhodamine_df.empty:
            rhodamine_df.to_sql('rhodamine', conn, if_exists='append', index=False)
            print(f"Successfully wrote {len(rhodamine_df)} records to rhodamine table")
        
        if ph_df is not None and not ph_df.empty:
            ph_df.to_sql('ph', conn, if_exists='append', index=False)
            print(f"Successfully wrote {len(ph_df)} records to ph table")
        
        if tsg_df is not None and not tsg_df.empty:
            tsg_df.to_sql('tsg', conn, if_exists='append', index=False)
            print(f"Successfully wrote {len(tsg_df)} records to tsg table")

        if gps_df is not None and not gps_df.empty:
            gps_df.to_sql('gps', conn, if_exists='append', index=False)
            print(f"Successfully wrote {len(gps_df)} records to gps table")

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
    rhodamine_df = generate_rhodamine_data(
        n_records=num, 
        base_lat=base_lat, 
        base_lon=base_lon, 
        start_time=start_time, 
        frequency_hz=freq
    )
    
    ph_freq = min(freq, 0.5)  # Max 0.5 Hz for pH sensor
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
    
    # Generate GPS data at the same frequency as TSG
    gps_df = generate_gps_data(
        n_records=num, 
        base_lat=base_lat, 
        base_lon=base_lon, 
        start_time=start_time, 
        frequency_hz=freq
    )
    
    t1 = time.perf_counter()
    print(f"  Raw sensor data generation: {t1-t0:.4f} seconds")
    
    return {
        'rhodamine': rhodamine_df,
        'ph': ph_df,
        'tsg': tsg_df,
        'gps': gps_df
    }

def generate_time_based_sensor_data(duration_seconds, base_lat=42.5, base_lon=-69.5, start_time=None):
    """
    Generate sensor data for a specified duration using default sampling rates for each sensor.
    
    Args:
        duration_seconds: Duration of data to generate in seconds
        base_lat: Base latitude for GPS coordinates
        base_lon: Base longitude for GPS coordinates  
        start_time: Optional start time (defaults to now - duration)
    
    Returns:
        dict: Dictionary containing DataFrames for each sensor type
    """
    print(f"Generating {duration_seconds} seconds of sensor data...")
    t0 = time.perf_counter()
    
    # Set start_time so last sample is now
    if start_time is None:
        start_time = datetime.now() - timedelta(seconds=duration_seconds)
    
    # Default sampling rates for each sensor
    rhodamine_freq = 1.0  # 1 Hz for rhodamine
    ph_freq = 0.5  # 0.5 Hz for pH sensor (moderate sampling)
    tsg_freq = 1.0  # 1 Hz for TSG
    gps_freq = 1.0  # 1 Hz for GPS

    # Calculate number of records for each sensor based on duration and frequency
    rhodamine_records = max(1, int(duration_seconds * rhodamine_freq))
    ph_records = max(1, int(duration_seconds * ph_freq))
    tsg_records = max(1, int(duration_seconds * tsg_freq))
    gps_records = max(1, int(duration_seconds * gps_freq))

    # Generate data for each sensor type
    rhodamine_df = generate_rhodamine_data(
        n_records=rhodamine_records,
        base_lat=base_lat,
        base_lon=base_lon,
        start_time=start_time,
        frequency_hz=rhodamine_freq
    )
    
    ph_df = generate_ph_data(
        n_records=ph_records,
        start_time=start_time,
        frequency_hz=ph_freq
    )
    
    tsg_df = generate_tsg_data(
        n_records=tsg_records,
        base_lat=base_lat,
        base_lon=base_lon,
        start_time=start_time,
        frequency_hz=tsg_freq
    )

    gps_df = generate_gps_data(
        n_records=gps_records,
        base_lat=base_lat,
        base_lon=base_lon,
        start_time=start_time,
        frequency_hz=gps_freq
    )

    t1 = time.perf_counter()
    print(f"  Generated {rhodamine_records} rhodamine, {ph_records} pH, {tsg_records} TSG, {gps_records} GPS records in {t1-t0:.4f} seconds")

    return {
        'rhodamine': rhodamine_df,
        'ph': ph_df,
        'tsg': tsg_df,
        'gps': gps_df
    }

# Argument parsing: load defaults from config.toml, override with CLI
def parse_args():
    config = get_config()

    parser = argparse.ArgumentParser(description="Generate and process synthetic sensor data.")
    parser.add_argument("--db_path", type=str, default=config.get("db_path", "./locness.db"), help="Path to SQLite database")
    parser.add_argument("--parquet_path", type=str, default=config.get("parquet_path", "./locness.parquet"), help="Path to Parquet file")
    parser.add_argument("--time", type=int, default=int(config.get("time", 60)), help="Duration of data to generate in seconds (per batch)")
    parser.add_argument("--resample-interval", type=str, default=config.get("db_res_int", "2s"), help="Resampling interval (e.g., '2s')")
    parser.add_argument("--continuous", action="store_true", default=config.get("continuous", False), help="Run in continuous mode (loop)")
    parser.add_argument("--process", action="store_true", default=False, help="Process/resample data after generation (default: true)")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Only use this module for raw data generation and writing
    def generate_and_write_raw(raw_sqlite_file):
        sensor_data = generate_time_based_sensor_data(
            duration_seconds=args.time,
        )
        print(f"Writing raw sensor data to {raw_sqlite_file}...")
        write_to_raw_tables(
            rhodamine_df=sensor_data['rhodamine'],
            ph_df=sensor_data['ph'],
            tsg_df=sensor_data['tsg'],
            gps_df=sensor_data['gps'],
            sqlite_path=raw_sqlite_file
        )

    # Use resample module's process_raw_data_incremental for resampling and output
    def resample_and_output(raw_sqlite_file, parquet_path, resample_interval):
        process_raw_data_incremental(
            sqlite_path=raw_sqlite_file,
            resample_interval=resample_interval,
            summary_table="underway_summary",
            write_csv=False,
            write_parquet=True,
            parquet_path=parquet_path
        )
        print(f"Resampled data written to underway_summary table in {raw_sqlite_file}")

    if args.continuous:
        print(f"Continuous mode enabled. Generating {args.time} seconds of sensor data every {args.time} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                print(f"Generating {args.time} seconds of sensor data...")
                generate_and_write_raw(raw_sqlite_file=args.db_path)
                if args.process:
                    resample_and_output(raw_sqlite_file=args.db_path, parquet_path=args.parquet_path, resample_interval=args.resample_interval)
                print(f"Sleeping {args.time} seconds before next batch...")
                time.sleep(args.time)
        except KeyboardInterrupt:
            print("\nStopped continuous generation.")
            sys.exit(0)
    else:
        print(f"Generating {args.time} seconds of sensor data...")
        generate_and_write_raw(raw_sqlite_file=args.db_path)
        if args.process:
            resample_and_output(raw_sqlite_file=args.db_path, parquet_path=args.parquet_path, resample_interval=args.resample_interval)
        print("Data generation complete.")


if __name__ == "__main__":
    main()

# Example usage of the new sensor-specific functions:
"""
# Generate rhodamine data
fluoro_df = generate_rhodamine_data(n_records=100, frequency_hz=1.0)
print("rhodamine data shape:", fluoro_df.shape)
print("rhodamine columns:", list(fluoro_df.columns))

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
#     rhodamine_df=sensor_batch['rhodamine'],
#     ph_df=sensor_batch['ph'],
#     tsg_df=sensor_batch['tsg'],
#
#     sqlite_path="test_sensors.sqlite"
# )
"""
