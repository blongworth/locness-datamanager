import os
import tempfile
import shutil
import sqlite3
import pandas as pd
import pytest
from locness_datamanager.synthetic_data import generate_time_based_sensor_data, write_to_raw_tables
from locness_datamanager.resample import process_raw_data_incremental, load_and_resample_sqlite

def create_test_db(tmpdir, duration=120, base_lat=42.5, base_lon=-69.5):
    """Generate raw data and write to a temporary SQLite DB."""
    db_path = os.path.join(tmpdir, "test_incremental.sqlite")
    sensor_data = generate_time_based_sensor_data(duration_seconds=duration, base_lat=base_lat, base_lon=base_lon)
    write_to_raw_tables(
        rhodamine_df=sensor_data['rhodamine'],
        ph_df=sensor_data['ph'],
        tsg_df=sensor_data['tsg'],
        gps_df=sensor_data['gps'],
        sqlite_path=db_path
    )
    return db_path

def test_incremental_vs_batch(tmp_path):
    """
    Test that incrementally resampled data matches batch resampled data.
    """
    # 1. Generate all raw data once
    duration = 120
    base_lat = 42.5
    base_lon = -69.5
    resample_interval = '10s'
    summary_table = 'underway_summary'
    db_path = os.path.join(tmp_path, "test_incremental.sqlite")
    sensor_data = generate_time_based_sensor_data(duration_seconds=duration, base_lat=base_lat, base_lon=base_lon)

    # Split raw data in half for incremental test
    def split_df(df):
        n = len(df)
        half = n // 2
        return df.iloc[:half].copy(), df.iloc[half:].copy()

    rhodamine1, rhodamine2 = split_df(sensor_data['rhodamine'])
    ph1, ph2 = split_df(sensor_data['ph'])
    tsg1, tsg2 = split_df(sensor_data['tsg'])
    gps1, gps2 = split_df(sensor_data['gps'])

    # 2. Batch: process all data at once
    write_to_raw_tables(
        rhodamine_df=sensor_data['rhodamine'],
        ph_df=sensor_data['ph'],
        tsg_df=sensor_data['tsg'],
        gps_df=sensor_data['gps'],
        sqlite_path=db_path
    )
    df_batch = process_raw_data_incremental(
        sqlite_path=db_path,
        resample_interval=resample_interval,
        summary_table=summary_table,
        write_csv=False,
        write_parquet=False,
        replace_all=True
    )
    df_batch_sorted = df_batch.sort_values('datetime_utc').reset_index(drop=True)

    # 3. Clear summary table for incremental test
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DELETE FROM {summary_table}")
        conn.execute(f"DELETE FROM rhodamine")
        conn.execute(f"DELETE FROM ph")
        conn.execute(f"DELETE FROM tsg")
        conn.execute(f"DELETE FROM gps")
        conn.commit()

    # 4. Incremental: process in two halves using the same data
    write_to_raw_tables(
        rhodamine_df=rhodamine1,
        ph_df=ph1,
        tsg_df=tsg1,
        gps_df=gps1,
        sqlite_path=db_path
    )
    df_inc1 = process_raw_data_incremental(
        sqlite_path=db_path,
        resample_interval=resample_interval,
        summary_table=summary_table,
        write_csv=False,
        write_parquet=False,
        replace_all=False
    )
    write_to_raw_tables(
        rhodamine_df=rhodamine2,
        ph_df=ph2,
        tsg_df=tsg2,
        gps_df=gps2,
        sqlite_path=db_path
    )
    df_inc2 = process_raw_data_incremental(
        sqlite_path=db_path,
        resample_interval=resample_interval,
        summary_table=summary_table,
        write_csv=False,
        write_parquet=False,
        replace_all=False
    )
    # 5. Load all resampled data from summary table
    with sqlite3.connect(db_path) as conn:
        df_all = pd.read_sql_query(f"SELECT * FROM {summary_table}", conn)
    df_all_sorted = df_all.sort_values('datetime_utc').reset_index(drop=True)

    # 6. Remove any extra columns (e.g., autoincrement index) from the summary table before comparing
    batch_cols = set(df_batch_sorted.columns)
    df_all_aligned = df_all_sorted[[col for col in df_all_sorted.columns if col in batch_cols]]
    # Also ensure columns are in the same order
    df_all_aligned = df_all_aligned[df_batch_sorted.columns]
    # Convert datetime_utc to datetime64[ns] for comparison
    if pd.api.types.is_integer_dtype(df_all_aligned['datetime_utc']):
        df_all_aligned['datetime_utc'] = pd.to_datetime(df_all_aligned['datetime_utc'], unit='s')
    # Ignore moving average columns for this test
    ignore_cols = [col for col in df_batch_sorted.columns if 'ma' in col]
    compare_cols = [col for col in df_batch_sorted.columns if col not in ignore_cols]
    pd.testing.assert_frame_equal(
        df_batch_sorted[compare_cols].reset_index(drop=True),
        df_all_aligned[compare_cols].reset_index(drop=True),
        check_dtype=False,
        check_like=True
    )


def test_moving_average_consistency(tmp_path):
    """
    Test that the moving average columns are calculated correctly for batch and incremental updates.
    """
    duration = 120
    base_lat = 42.5
    base_lon = -69.5
    resample_interval = '10s'
    summary_table = 'underway_summary'
    db_path = os.path.join(tmp_path, "test_ma.sqlite")
    sensor_data = generate_time_based_sensor_data(duration_seconds=duration, base_lat=base_lat, base_lon=base_lon)

    def split_df(df):
        n = len(df)
        half = n // 2
        return df.iloc[:half].copy(), df.iloc[half:].copy()

    rhodamine1, rhodamine2 = split_df(sensor_data['rhodamine'])
    ph1, ph2 = split_df(sensor_data['ph'])
    tsg1, tsg2 = split_df(sensor_data['tsg'])
    gps1, gps2 = split_df(sensor_data['gps'])

    # Batch
    write_to_raw_tables(
        rhodamine_df=sensor_data['rhodamine'],
        ph_df=sensor_data['ph'],
        tsg_df=sensor_data['tsg'],
        gps_df=sensor_data['gps'],
        sqlite_path=db_path
    )
    df_batch = process_raw_data_incremental(
        sqlite_path=db_path,
        resample_interval=resample_interval,
        summary_table=summary_table,
        write_csv=False,
        write_parquet=False,
        replace_all=True
    )
    df_batch_sorted = df_batch.sort_values('datetime_utc').reset_index(drop=True)

    # Clear for incremental
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DELETE FROM {summary_table}")
        conn.execute(f"DELETE FROM rhodamine")
        conn.execute(f"DELETE FROM ph")
        conn.execute(f"DELETE FROM tsg")
        conn.execute(f"DELETE FROM gps")
        conn.commit()

    # Incremental
    write_to_raw_tables(
        rhodamine_df=rhodamine1,
        ph_df=ph1,
        tsg_df=tsg1,
        gps_df=gps1,
        sqlite_path=db_path
    )
    process_raw_data_incremental(
        sqlite_path=db_path,
        resample_interval=resample_interval,
        summary_table=summary_table,
        write_csv=False,
        write_parquet=False,
        replace_all=False
    )
    write_to_raw_tables(
        rhodamine_df=rhodamine2,
        ph_df=ph2,
        tsg_df=tsg2,
        gps_df=gps2,
        sqlite_path=db_path
    )
    process_raw_data_incremental(
        sqlite_path=db_path,
        resample_interval=resample_interval,
        summary_table=summary_table,
        write_csv=False,
        write_parquet=False,
        replace_all=False
    )
    with sqlite3.connect(db_path) as conn:
        df_all = pd.read_sql_query(f"SELECT * FROM {summary_table}", conn)
    df_all_sorted = df_all.sort_values('datetime_utc').reset_index(drop=True)
    # Align columns and types
    batch_cols = set(df_batch_sorted.columns)
    df_all_aligned = df_all_sorted[[col for col in df_all_sorted.columns if col in batch_cols]]
    df_all_aligned = df_all_aligned[df_batch_sorted.columns]
    if pd.api.types.is_integer_dtype(df_all_aligned['datetime_utc']):
        df_all_aligned['datetime_utc'] = pd.to_datetime(df_all_aligned['datetime_utc'], unit='s')
    # Only compare moving average columns
    ma_cols = [col for col in df_batch_sorted.columns if 'ma' in col]
    pd.testing.assert_frame_equal(
        df_batch_sorted[ma_cols].reset_index(drop=True),
        df_all_aligned[ma_cols].reset_index(drop=True),
        check_dtype=False,
        check_like=True
    )
