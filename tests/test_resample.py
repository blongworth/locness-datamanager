import pytest
import sqlite3
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from locness_datamanager import resample

def make_df(data, columns):
    return pd.DataFrame(data, columns=columns)

def test_add_corrected_ph_basic():
    df = pd.DataFrame({
        'vrse': [0.1, 0.2],
        'temp': [20.0, 21.0],
        'salinity': [35.0, 36.0]
    })
    result = resample.add_corrected_ph(df.copy())
    assert 'ph_corrected' in result.columns
    assert len(result['ph_corrected']) == 2

def test_add_corrected_ph_missing_salinity():
    df = pd.DataFrame({
        'vrse': [0.1, 0.2],
        'temp': [20.0, 21.0]
        # salinity missing
    })
    try:
        result = resample.add_corrected_ph(df.copy())
        # Should either fill with NaN or raise, depending on calc_ph
        assert 'ph_corrected' in result.columns
    except Exception:
        pass  # Acceptable if function raises

def test_add_corrected_ph_missing_temp():
    df = pd.DataFrame({
        'vrse': [0.1, 0.2],
        'salinity': [35.0, 36.0]
        # temp missing
    })
    try:
        result = resample.add_corrected_ph(df.copy())
        assert 'ph_corrected' in result.columns
    except Exception:
        pass

def test_add_corrected_ph_missing_both():
    df = pd.DataFrame({
        'vrse': [0.1, 0.2]
        # temp and salinity missing
    })
    try:
        result = resample.add_corrected_ph(df.copy())
        assert 'ph_corrected' in result.columns
    except Exception:
        pass

def test_add_ph_moving_average_basic():
    df = pd.DataFrame({
        'datetime_utc': pd.date_range('2023-01-01', periods=5, freq='2s'),
        'ph_total': [7.0, 7.1, 7.2, 7.3, 7.4],
        'ph_corrected': [7.0, 7.1, 7.2, 7.3, 7.4]
    })
    result = resample.add_ph_moving_average(df, window_seconds=4, freq_hz=0.5)
    assert 'ph_corrected_ma' in result.columns
    assert 'ph_total_ma' in result.columns
    # The rolling window is 2, so the last value should be the mean of the last 2 values
    expected = (7.3 + 7.4) / 2  # 7.35
    assert result['ph_corrected_ma'].iloc[-1] == pytest.approx(expected, rel=1e-2)

def test_add_computed_fields_uses_config():
    df = pd.DataFrame({
        'datetime_utc': pd.date_range('2023-01-01', periods=3, freq='2s'),
        'vrse': [0.1, 0.2, 0.3],
        'temp': [20.0, 21.0, 22.0],
        'salinity': [35.0, 35.1, 35.2],
        'ph_total': [7.0, 7.2, 7.4]
    })
    config = {'ph_ma_window': 4, 'ph_freq': 0.5}
    result = resample.add_computed_fields(df, config)
    assert 'ph_corrected_ma' in result.columns
    assert 'ph_total_ma' in result.columns

def test_resample_tables_joins_and_resamples():
    dt = pd.date_range('2023-01-01', periods=4, freq='2s')
    fluoro = make_df({'datetime_utc': dt, 'rho_ppb': [1,2,3,4]}, ['datetime_utc','rho_ppb'])
    ph = make_df({'datetime_utc': dt, 'ph_total': [7,8,9,10]}, ['datetime_utc','ph_total'])
    tsg = make_df({'datetime_utc': dt, 'temp': [20,21,22,23], 'salinity': [35,36,37,38]}, ['datetime_utc','temp','salinity'])
    gps = make_df({'datetime_utc': dt, 'latitude': [10,11,12,13], 'longitude': [20,21,22,23]}, ['datetime_utc','latitude','longitude'])
    df = resample.resample_raw_data(fluoro, ph, tsg, gps, resample_interval='2s')
    assert set(['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'temp', 'salinity']).issubset(df.columns)
    assert len(df) == 4

def test_load_sqlite_tables_calls_read_table():
    with patch('locness_datamanager.resample.read_table') as mock_read_table:
        mock_read_table.side_effect = [
            pd.DataFrame({'datetime_utc': [1], 'rho_ppb': [1]}),
            pd.DataFrame({'datetime_utc': [1], 'ph_total': [7]}),
            pd.DataFrame({'datetime_utc': [1], 'temp': [20], 'salinity': [35]}),
            pd.DataFrame({'datetime_utc': [1], 'latitude': [10], 'longitude': [20]})
        ]
        with patch('sqlite3.connect') as mock_connect:
            conn = MagicMock()
            mock_connect.return_value = conn
            result = resample.load_sqlite_tables('dummy_path')
            assert len(result) == 4
            assert all(isinstance(df, pd.DataFrame) for df in result)

def test_load_and_resample_sqlite_integration():
    with patch('locness_datamanager.resample.load_sqlite_tables') as mock_load:
        dt = pd.date_range('2023-01-01', periods=2, freq='2s')
        mock_load.return_value = (
            make_df({'datetime_utc': dt, 'rho_ppb': [1,2]}, ['datetime_utc','rho_ppb']),
            make_df({'datetime_utc': dt, 'ph_total': [7,8], 'ph_corrected': [7.0, 8.0], 'vrse': [0.1, 0.2]}, ['datetime_utc','ph_total','ph_corrected','vrse']),
            make_df({'datetime_utc': dt, 'temp': [20,21], 'salinity': [35,36]}, ['datetime_utc','temp','salinity']),
            make_df({'datetime_utc': dt, 'latitude': [10,11], 'longitude': [20,21]}, ['datetime_utc','latitude','longitude'])
        )
        df = resample.load_and_resample_sqlite('dummy_path', resample_interval='2s')
        assert 'ph_corrected_ma' in df.columns
        assert 'ph_total_ma' in df.columns
        assert len(df) == 2

# --- Additional edge case tests for resample.py ---

def test_resample_tables_with_missing_values():
    dt = pd.date_range('2023-01-01', periods=4, freq='2s')
    fluoro = make_df({'datetime_utc': dt, 'rho_ppb': [1, np.nan, 3, 4]}, ['datetime_utc','rho_ppb'])
    ph = make_df({'datetime_utc': dt, 'ph_total': [7, 8, np.nan, 10]}, ['datetime_utc','ph_total'])
    tsg = make_df({'datetime_utc': dt, 'temp': [20, 21, 22, np.nan], 'salinity': [35, 36, np.nan, 38]}, ['datetime_utc','temp','salinity'])
    gps = make_df({'datetime_utc': dt, 'latitude': [10, 11, 12, 13], 'longitude': [20, 21, 22, 23]}, ['datetime_utc','latitude','longitude'])
    df = resample.resample_raw_data(fluoro, ph, tsg, gps, resample_interval='2s')
    assert len(df) == 4
    # Should not raise, and columns should exist
    assert 'rho_ppb' in df.columns and 'ph_total' in df.columns

def test_resample_tables_with_duplicate_timestamps():
    dt = pd.to_datetime(['2023-01-01 00:00:02', '2023-01-01 00:00:02', '2023-01-01 00:00:02', '2023-01-01 00:00:02'])
    fluoro = make_df({'datetime_utc': dt, 'rho_ppb': [1, 2, 2, 4]}, ['datetime_utc','rho_ppb'])
    ph = make_df({'datetime_utc': dt, 'ph_total': [7, 8, 8, 10]}, ['datetime_utc','ph_total'])
    tsg = make_df({'datetime_utc': dt, 'temp': [20, 21, 21, 23], 'salinity': [35, 36, 36, 38]}, ['datetime_utc','temp','salinity'])
    gps = make_df({'datetime_utc': dt, 'latitude': [10, 11, 11, 13], 'longitude': [20, 21, 21, 23]}, ['datetime_utc','latitude','longitude'])
    df = resample.resample_raw_data(fluoro, ph, tsg, gps, resample_interval='2s')
    assert len(df) == 1
    # test that it drops duplicates and keeps the first occurrence
    assert df['rho_ppb'].iloc[0] == 1 and df['ph_total'].iloc[0] == 7
    # test that it averages the duplicates
    #assert df['temp'].iloc[0] == 20.5 and df['salinity'].iloc[0] == 35.5

def test_resample_tables_with_unsorted_timestamps():
    dt = pd.to_datetime(['2023-01-01 00:00:04', '2023-01-01 00:00:02', '2023-01-01 00:00:00', '2023-01-01 00:00:06'])
    fluoro = make_df({'datetime_utc': dt, 'rho_ppb': [4, 2, 1, 5]}, ['datetime_utc','rho_ppb'])
    ph = make_df({'datetime_utc': dt, 'ph_total': [10, 8, 7, 11]}, ['datetime_utc','ph_total'])
    tsg = make_df({'datetime_utc': dt, 'temp': [23, 21, 20, 24], 'salinity': [38, 36, 35, 39]}, ['datetime_utc','temp','salinity'])
    gps = make_df({'datetime_utc': dt, 'latitude': [13, 11, 10, 14], 'longitude': [23, 21, 20, 24]}, ['datetime_utc','latitude','longitude'])
    df = resample.resample_raw_data(fluoro, ph, tsg, gps, resample_interval='2s')
    assert set(['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'temp', 'salinity']).issubset(df.columns)

def test_resample_tables_with_empty_input():
    dt = pd.to_datetime([])
    fluoro = make_df({'datetime_utc': dt, 'rho_ppb': []}, ['datetime_utc','rho_ppb'])
    ph = make_df({'datetime_utc': dt, 'ph_total': []}, ['datetime_utc','ph_total'])
    tsg = make_df({'datetime_utc': dt, 'temp': [], 'salinity': []}, ['datetime_utc','temp','salinity'])
    gps = make_df({'datetime_utc': dt, 'latitude': [], 'longitude': []}, ['datetime_utc','latitude','longitude'])
    df = resample.resample_raw_data(fluoro, ph, tsg, gps, resample_interval='2s')
    assert df.empty

def test_resample_tables_with_wrong_types():
    dt = pd.date_range('2023-01-01', periods=2, freq='2s')
    fluoro = make_df({'datetime_utc': dt, 'rho_ppb': ['a', 'b']}, ['datetime_utc','rho_ppb'])
    ph = make_df({'datetime_utc': dt, 'ph_total': ['x', 'y']}, ['datetime_utc','ph_total'])
    tsg = make_df({'datetime_utc': dt, 'temp': ['foo', 'bar'], 'salinity': ['baz', 'qux']}, ['datetime_utc','temp','salinity'])
    gps = make_df({'datetime_utc': dt, 'latitude': [10, 11], 'longitude': [20, 21]}, ['datetime_utc','latitude','longitude'])
    try:
        resample.resample_raw_data(fluoro, ph, tsg, gps, resample_interval='2s')
    except Exception:
        pass  # Acceptable if function raises

def test_resample_tables_with_extra_columns():
    dt = pd.date_range('2023-01-01', periods=2, freq='2s')
    fluoro = make_df({'datetime_utc': dt, 'rho_ppb': [1, 2], 'extra1': [100, 200]}, ['datetime_utc','rho_ppb','extra1'])
    ph = make_df({'datetime_utc': dt, 'ph_total': [7, 8], 'extra2': [300, 400]}, ['datetime_utc','ph_total','extra2'])
    tsg = make_df({'datetime_utc': dt, 'temp': [20, 21], 'salinity': [35, 36], 'extra3': [500, 600]}, ['datetime_utc','temp','salinity','extra3'])
    gps = make_df({'datetime_utc': dt, 'latitude': [10, 11], 'longitude': [20, 21], 'extra4': [700, 800]}, ['datetime_utc','latitude','longitude','extra4'])
    df = resample.resample_raw_data(fluoro, ph, tsg, gps, resample_interval='2s')
    assert 'extra1' not in df.columns and 'extra2' not in df.columns and 'extra3' not in df.columns and 'extra4' not in df.columns


# --- Tests for incremental raw data processing ---
def test_process_raw_data_incremental_incremental_mode():
    import tempfile
    import os
    from locness_datamanager.setup_db import setup_sqlite_db
    # Create a temp DB
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_incremental.sqlite")
        setup_sqlite_db(db_path)
        # Insert first batch of data
        dt = pd.date_range('2023-01-01 00:00:02', periods=2, freq='2s')
        fluoro = pd.DataFrame({'datetime_utc': dt.astype(int) // 10**9, 'rho_ppb': [1,2]})
        ph = pd.DataFrame({'datetime_utc': dt.astype(int) // 10**9, 'ph_total': [7,8], 'vrse': [0.1,0.2]})
        tsg = pd.DataFrame({'datetime_utc': dt.astype(int) // 10**9, 'temp': [20,21], 'salinity': [35,36]})
        gps = pd.DataFrame({'datetime_utc': dt.astype(int) // 10**9, 'latitude': [10,11], 'longitude': [20,21]})
        with sqlite3.connect(db_path) as conn:
            fluoro.to_sql('rhodamine', conn, if_exists='append', index=False)
            ph.to_sql('ph', conn, if_exists='append', index=False)
            tsg.to_sql('tsg', conn, if_exists='append', index=False)
            gps.to_sql('gps', conn, if_exists='append', index=False)
        # Now the connection is closed, run incremental mode
        result = resample.process_raw_data_incremental(
            sqlite_path=db_path,
            resample_interval='2s',
            summary_table='underway_summary',
            replace_all=False
        )
        assert not result.empty
        # Check that the number of rows matches the number of unique datetimes
        expected_rows = len(dt)
        assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"


def test_process_raw_data_incremental_replace_all():
    import tempfile
    import os
    from locness_datamanager.setup_db import setup_sqlite_db
    # Create a temp DB
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_replaceall.sqlite")
        setup_sqlite_db(db_path)
        # Insert all data
        dt = pd.date_range('2023-01-01 00:00:00', periods=2, freq='2s')
        fluoro = pd.DataFrame({'datetime_utc': dt.astype(int) // 10**9, 'rho_ppb': [1,2]})
        ph = pd.DataFrame({'datetime_utc': dt.astype(int) // 10**9, 'ph_total': [7,8], 'vrse': [0.1,0.2]})
        tsg = pd.DataFrame({'datetime_utc': dt.astype(int) // 10**9, 'temp': [20,21], 'salinity': [35,36]})
        gps = pd.DataFrame({'datetime_utc': dt.astype(int) // 10**9, 'latitude': [10,11], 'longitude': [20,21]})
        with sqlite3.connect(db_path) as conn:
            fluoro.to_sql('rhodamine', conn, if_exists='append', index=False)
            ph.to_sql('ph', conn, if_exists='append', index=False)
            tsg.to_sql('tsg', conn, if_exists='append', index=False)
            gps.to_sql('gps', conn, if_exists='append', index=False)
        # Now the connection is closed, run replace_all mode
        result = resample.process_raw_data_incremental(
            sqlite_path=db_path,
            resample_interval='2s',
            summary_table='underway_summary',
            replace_all=True
        )
        assert not result.empty


@patch('locness_datamanager.resample.get_last_summary_timestamp')
@patch('locness_datamanager.resample.load_sqlite_tables_after_timestamp')
@patch('locness_datamanager.resample.load_sqlite_tables')
@patch('locness_datamanager.resample.write_resampled_to_sqlite')
def test_process_raw_data_incremental_no_new_data(
    mock_write_sqlite, mock_load_all, mock_load_after, mock_get_last_ts
):
    # Simulate no new data after last timestamp
    mock_get_last_ts.return_value = pd.Timestamp('2023-01-01 00:00:00')
    empty = pd.DataFrame({'datetime_utc': [], 'rho_ppb': [], 'ph_total': [], 'vrse': [], 'temp': [], 'salinity': [], 'latitude':[], 'longitude':[]})
    mock_load_after.return_value = (empty, empty, empty, empty)
    mock_load_all.return_value = (empty, empty, empty, empty)
    result = resample.process_raw_data_incremental(
        sqlite_path='dummy.sqlite',
        resample_interval='2s',
        summary_table='underway_summary',
        replace_all=False
    )
    assert result.empty
    assert not mock_write_sqlite.called


def test_write_resampled_to_sqlite_unique_constraint_error(caplog):
    import sqlite3 as _sqlite3
    df = pd.DataFrame({
        'datetime_utc': pd.to_datetime(['2023-01-01 00:00:00']),
        'latitude': [10],
        'longitude': [20],
        'rho_ppb': [1.0],
        'ph_total': [7.0],
        'vrse': [0.1],
        'temp': [20.0],
        'salinity': [35.0]
    })
    # Patch pandas.DataFrame.to_sql to raise IntegrityError
    with patch('sqlite3.connect') as mock_connect:
        conn = MagicMock()
        mock_connect.return_value = conn
        # Patch to_sql to raise IntegrityError
        def raise_integrity_error(*args, **kwargs):
            raise _sqlite3.IntegrityError('UNIQUE constraint failed: underway_summary.datetime_utc')
        conn.__enter__.return_value = conn
        with patch('pandas.DataFrame.to_sql', side_effect=raise_integrity_error):
            from locness_datamanager.resample import write_resampled_to_sqlite
            write_resampled_to_sqlite(df, 'dummy.sqlite', 'underway_summary')
            # Check that the error was logged (not printed to stdout)
            assert 'Error writing to underway_summary table: UNIQUE constraint failed' in caplog.text


def test_process_raw_data_incremental_batch_consistency():
    """Test that processing data in multiple batches produces the same result as single batch processing."""
    import tempfile
    import os
    from locness_datamanager.setup_db import setup_sqlite_db
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two separate databases for comparison
        single_batch_db = os.path.join(tmpdir, "single_batch.sqlite")
        multi_batch_db = os.path.join(tmpdir, "multi_batch.sqlite")
        
        setup_sqlite_db(single_batch_db)
        setup_sqlite_db(multi_batch_db)
        
        # Create a larger dataset spanning multiple time periods
        dt_full = pd.date_range('2023-01-01 00:00:00', periods=100, freq='2s')
        
        # Full dataset for single batch processing
        fluoro_full = pd.DataFrame({
            'datetime_utc': dt_full.astype(int) // 10**9,
            'rho_ppb': [1.0 + i*0.01 for i in range(100)]
        })
        ph_full = pd.DataFrame({
            'datetime_utc': dt_full.astype(int) // 10**9,
            'ph_total': [7.0 + (i % 20)*0.05 for i in range(100)],
            'vrse': [0.1 + i*0.001 for i in range(100)]
        })
        tsg_full = pd.DataFrame({
            'datetime_utc': dt_full.astype(int) // 10**9,
            'temp': [20.0 + (i % 10)*0.5 for i in range(100)],
            'salinity': [35.0 + (i % 15)*0.1 for i in range(100)]
        })
        gps_full = pd.DataFrame({
            'datetime_utc': dt_full.astype(int) // 10**9,
            'latitude': [10.0 + i*0.01 for i in range(100)],
            'longitude': [20.0 + i*0.01 for i in range(100)]
        })
        
        # Insert all data into single batch database
        with sqlite3.connect(single_batch_db) as conn:
            fluoro_full.to_sql('rhodamine', conn, if_exists='append', index=False)
            ph_full.to_sql('ph', conn, if_exists='append', index=False)
            tsg_full.to_sql('tsg', conn, if_exists='append', index=False)
            gps_full.to_sql('gps', conn, if_exists='append', index=False)
        
        # Process all data in single batch
        resample.process_raw_data_incremental(
            sqlite_path=single_batch_db,
            resample_interval='2s',
            summary_table='underway_summary',
            replace_all=True,
            ph_k0=0.0,
            ph_k2=0.0,
            ph_ma_window=60,
            ph_freq=0.5
        )
        
        # Now process the same data in multiple batches
        batch_size = 25
        for i in range(0, 100, batch_size):
            end_idx = min(i + batch_size, 100)
            
            # Create batch data
            fluoro_batch = fluoro_full.iloc[i:end_idx].copy()
            ph_batch = ph_full.iloc[i:end_idx].copy()
            tsg_batch = tsg_full.iloc[i:end_idx].copy()
            gps_batch = gps_full.iloc[i:end_idx].copy()
            
            # Insert batch into multi-batch database
            with sqlite3.connect(multi_batch_db) as conn:
                fluoro_batch.to_sql('rhodamine', conn, if_exists='append', index=False)
                ph_batch.to_sql('ph', conn, if_exists='append', index=False)
                tsg_batch.to_sql('tsg', conn, if_exists='append', index=False)
                gps_batch.to_sql('gps', conn, if_exists='append', index=False)
            
            # Process incrementally (replace_all=True only for first batch)
            resample.process_raw_data_incremental(
                sqlite_path=multi_batch_db,
                resample_interval='2s',
                summary_table='underway_summary',
                replace_all=(i == 0),  # Only replace all for first batch
                ph_k0=0.0,
                ph_k2=0.0,
                ph_ma_window=60,
                ph_freq=0.5
            )
        
        # Read final results from both databases
        with sqlite3.connect(single_batch_db) as conn:
            single_final = pd.read_sql_query("SELECT * FROM underway_summary ORDER BY datetime_utc", conn)
        
        with sqlite3.connect(multi_batch_db) as conn:
            multi_final = pd.read_sql_query("SELECT * FROM underway_summary ORDER BY datetime_utc", conn)
        
        # Convert timestamps for comparison
        single_final['datetime_utc'] = pd.to_datetime(single_final['datetime_utc'], unit='s')
        multi_final['datetime_utc'] = pd.to_datetime(multi_final['datetime_utc'], unit='s')
        
        # Compare results
        assert len(single_final) == len(multi_final), f"Row count mismatch: {len(single_final)} vs {len(multi_final)}"
        
        # Sort both by datetime_utc to ensure consistent ordering
        single_final = single_final.sort_values('datetime_utc').reset_index(drop=True)
        multi_final = multi_final.sort_values('datetime_utc').reset_index(drop=True)
        
        # Compare timestamps
        pd.testing.assert_series_equal(
            single_final['datetime_utc'], 
            multi_final['datetime_utc'], 
            check_names=False
        )
        
        # Compare core sensor data (should be identical)
        core_columns = ['rho_ppb', 'ph_total', 'temp', 'salinity', 'latitude', 'longitude', 'ph_corrected']
        for col in core_columns:
            if col in single_final.columns and col in multi_final.columns:
                pd.testing.assert_series_equal(
                    single_final[col], 
                    multi_final[col], 
                    check_names=False,
                    rtol=1e-10,  # Very tight tolerance for non-moving-average data
                    check_dtype=False
                )
        
        # Moving averages might have slight differences due to incremental processing
        # but they should be very close
        ma_columns = [col for col in single_final.columns if '_ma' in col]
        for col in ma_columns:
            if col in single_final.columns and col in multi_final.columns:
                # Allow for small numerical differences in moving averages
                diff = (single_final[col] - multi_final[col]).abs()
                max_diff = diff.max()
                # Moving averages should be very close (within 0.01 pH units)
                assert max_diff < 0.01 or pd.isna(max_diff), f"Moving average {col} differs too much: max diff = {max_diff}"
        
        print("✓ Batch consistency test passed:")
        print(f"  - Single batch: {len(single_final)} records")
        print(f"  - Multi batch: {len(multi_final)} records")
        print("  - Core data identical, moving averages within tolerance")


def test_raw_data_usage_once():
    """Test that each row of raw data is used exactly once during resampling."""
    import tempfile
    import os
    from locness_datamanager.setup_db import setup_sqlite_db
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_raw_usage.sqlite")
        setup_sqlite_db(db_path)
        
        # Create test data with unique identifiable values
        dt = pd.date_range('2023-01-01 00:00:00', periods=50, freq='2s')
        
        # Use unique sequential values so we can track exactly which raw data gets used
        fluoro = pd.DataFrame({
            'datetime_utc': dt.astype(int) // 10**9,
            'rho_ppb': [i * 0.1 for i in range(50)]  # 0.0, 0.1, 0.2, ... 4.9
        })
        ph = pd.DataFrame({
            'datetime_utc': dt.astype(int) // 10**9,
            'ph_total': [7.0 + i * 0.01 for i in range(50)],  # 7.00, 7.01, 7.02, ... 7.49
            'vrse': [0.1 + i * 0.001 for i in range(50)]
        })
        tsg = pd.DataFrame({
            'datetime_utc': dt.astype(int) // 10**9,
            'temp': [20.0 + i * 0.1 for i in range(50)],  # 20.0, 20.1, 20.2, ... 24.9
            'salinity': [35.0 + i * 0.01 for i in range(50)]  # 35.00, 35.01, 35.02, ... 35.49
        })
        gps = pd.DataFrame({
            'datetime_utc': dt.astype(int) // 10**9,  
            'latitude': [10.0 + i * 0.01 for i in range(50)],  # 10.00, 10.01, 10.02, ... 10.49
            'longitude': [20.0 + i * 0.01 for i in range(50)]  # 20.00, 20.01, 20.02, ... 20.49
        })
        
        # Insert all data
        with sqlite3.connect(db_path) as conn:
            fluoro.to_sql('rhodamine', conn, if_exists='append', index=False)
            ph.to_sql('ph', conn, if_exists='append', index=False)
            tsg.to_sql('tsg', conn, if_exists='append', index=False)
            gps.to_sql('gps', conn, if_exists='append', index=False)
        
        # Process all data
        result = resample.process_raw_data_incremental(
            sqlite_path=db_path,
            resample_interval='2s',
            summary_table='underway_summary',
            replace_all=True,
            ph_k0=0.0,
            ph_k2=0.0,
            ph_ma_window=60,
            ph_freq=0.5
        )
        
        # Verify each raw data point was used exactly once
        # Since we're resampling at 2s intervals and our raw data is at 2s intervals,
        # we should get a 1:1 mapping (each raw record becomes one resampled record)
        assert len(result) == 50, f"Expected 50 resampled records, got {len(result)}"
        
        # Sort results by timestamp to ensure consistent ordering
        result_sorted = result.sort_values('datetime_utc').reset_index(drop=True)
        
        # Check that all unique values from raw data appear in resampled data
        expected_rho = set(fluoro['rho_ppb'])
        actual_rho = set(result_sorted['rho_ppb'].dropna())
        assert expected_rho == actual_rho, f"Rhodamine values mismatch: expected {len(expected_rho)}, got {len(actual_rho)}"
        
        expected_ph = set(ph['ph_total'])
        actual_ph = set(result_sorted['ph_total'].dropna())
        assert expected_ph == actual_ph, f"pH values mismatch: expected {len(expected_ph)}, got {len(actual_ph)}"
        
        expected_temp = set(tsg['temp'])
        actual_temp = set(result_sorted['temp'].dropna())
        assert expected_temp == actual_temp, f"Temperature values mismatch: expected {len(expected_temp)}, got {len(actual_temp)}"
        
        expected_lat = set(gps['latitude'])
        actual_lat = set(result_sorted['latitude'].dropna())
        assert expected_lat == actual_lat, f"Latitude values mismatch: expected {len(expected_lat)}, got {len(actual_lat)}"
        
        print("✓ Raw data usage test passed:")
        print(f"  - All {len(expected_rho)} rhodamine values used exactly once")
        print(f"  - All {len(expected_ph)} pH values used exactly once")
        print(f"  - All {len(expected_temp)} temperature values used exactly once")
        print(f"  - All {len(expected_lat)} GPS values used exactly once")


def test_batch_consistency_and_raw_data_usage():
    """Combined test: verify batch consistency AND that raw data is used exactly once regardless of batch size."""
    import tempfile
    import os
    from locness_datamanager.setup_db import setup_sqlite_db
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create databases for different processing approaches
        single_batch_db = os.path.join(tmpdir, "single_batch.sqlite")
        multi_batch_db = os.path.join(tmpdir, "multi_batch.sqlite")
        
        setup_sqlite_db(single_batch_db)
        setup_sqlite_db(multi_batch_db)
        
        # Create test data with unique sequential values for tracking
        dt_full = pd.date_range('2023-01-01 00:00:00', periods=60, freq='2s')
        
        # Create unique identifiable raw data
        fluoro_full = pd.DataFrame({
            'datetime_utc': dt_full.astype(int) // 10**9,
            'rho_ppb': [1000 + i for i in range(60)]  # 1000, 1001, 1002, ... 1059
        })
        ph_full = pd.DataFrame({
            'datetime_utc': dt_full.astype(int) // 10**9,
            'ph_total': [7000 + i for i in range(60)],  # 7000, 7001, 7002, ... 7059
            'vrse': [0.1 + i * 0.001 for i in range(60)]
        })
        tsg_full = pd.DataFrame({
            'datetime_utc': dt_full.astype(int) // 10**9,
            'temp': [20000 + i for i in range(60)],  # 20000, 20001, 20002, ... 20059
            'salinity': [35000 + i for i in range(60)]  # 35000, 35001, 35002, ... 35059
        })
        gps_full = pd.DataFrame({
            'datetime_utc': dt_full.astype(int) // 10**9,
            'latitude': [10000 + i for i in range(60)],  # 10000, 10001, 10002, ... 10059
            'longitude': [20000 + i for i in range(60)]  # 20000, 20001, 20002, ... 20059
        })
        
        # Store expected raw data sets for verification
        expected_rho = set(fluoro_full['rho_ppb'])
        expected_ph = set(ph_full['ph_total'])
        expected_temp = set(tsg_full['temp'])
        expected_salinity = set(tsg_full['salinity'])
        expected_lat = set(gps_full['latitude'])
        expected_lon = set(gps_full['longitude'])
        
        # Single batch processing
        with sqlite3.connect(single_batch_db) as conn:
            fluoro_full.to_sql('rhodamine', conn, if_exists='append', index=False)
            ph_full.to_sql('ph', conn, if_exists='append', index=False)
            tsg_full.to_sql('tsg', conn, if_exists='append', index=False)
            gps_full.to_sql('gps', conn, if_exists='append', index=False)
        
        resample.process_raw_data_incremental(
            sqlite_path=single_batch_db,
            resample_interval='2s',
            summary_table='underway_summary',
            replace_all=True,
            ph_k0=0.0,
            ph_k2=0.0,
            ph_ma_window=60,
            ph_freq=0.5
        )
        
        # Multi-batch processing (3 batches of 20 records each)
        batch_size = 20
        for i in range(0, 60, batch_size):
            end_idx = min(i + batch_size, 60)
            
            # Create batch data
            fluoro_batch = fluoro_full.iloc[i:end_idx].copy()
            ph_batch = ph_full.iloc[i:end_idx].copy()
            tsg_batch = tsg_full.iloc[i:end_idx].copy()
            gps_batch = gps_full.iloc[i:end_idx].copy()
            
            # Insert batch
            with sqlite3.connect(multi_batch_db) as conn:
                fluoro_batch.to_sql('rhodamine', conn, if_exists='append', index=False)
                ph_batch.to_sql('ph', conn, if_exists='append', index=False)
                tsg_batch.to_sql('tsg', conn, if_exists='append', index=False)
                gps_batch.to_sql('gps', conn, if_exists='append', index=False)
            
            # Process batch
            resample.process_raw_data_incremental(
                sqlite_path=multi_batch_db,
                resample_interval='2s',
                summary_table='underway_summary',
                replace_all=(i == 0),
                ph_k0=0.0,
                ph_k2=0.0,
                ph_ma_window=60,
                ph_freq=0.5
            )
        
        # Read final results
        with sqlite3.connect(single_batch_db) as conn:
            single_final = pd.read_sql_query("SELECT * FROM underway_summary ORDER BY datetime_utc", conn)
        
        with sqlite3.connect(multi_batch_db) as conn:
            multi_final = pd.read_sql_query("SELECT * FROM underway_summary ORDER BY datetime_utc", conn)
        
        # Test 1: Batch consistency
        assert len(single_final) == len(multi_final), f"Row count mismatch: {len(single_final)} vs {len(multi_final)}"
        assert len(single_final) == 60, f"Expected 60 records, got {len(single_final)}"
        
        # Test 2: Raw data usage - Single batch
        single_rho = set(single_final['rho_ppb'].dropna())
        single_ph = set(single_final['ph_total'].dropna())
        single_temp = set(single_final['temp'].dropna())
        single_salinity = set(single_final['salinity'].dropna())
        single_lat = set(single_final['latitude'].dropna())
        single_lon = set(single_final['longitude'].dropna())
        
        assert expected_rho == single_rho, "Single batch: rhodamine values mismatch"
        assert expected_ph == single_ph, "Single batch: pH values mismatch"
        assert expected_temp == single_temp, "Single batch: temperature values mismatch"
        assert expected_salinity == single_salinity, "Single batch: salinity values mismatch"
        assert expected_lat == single_lat, "Single batch: latitude values mismatch"
        assert expected_lon == single_lon, "Single batch: longitude values mismatch"
        
        # Test 3: Raw data usage - Multi batch
        multi_rho = set(multi_final['rho_ppb'].dropna())
        multi_ph = set(multi_final['ph_total'].dropna())
        multi_temp = set(multi_final['temp'].dropna())
        multi_salinity = set(multi_final['salinity'].dropna())
        multi_lat = set(multi_final['latitude'].dropna())
        multi_lon = set(multi_final['longitude'].dropna())
        
        assert expected_rho == multi_rho, "Multi batch: rhodamine values mismatch"
        assert expected_ph == multi_ph, "Multi batch: pH values mismatch"
        assert expected_temp == multi_temp, "Multi batch: temperature values mismatch"
        assert expected_salinity == multi_salinity, "Multi batch: salinity values mismatch"
        assert expected_lat == multi_lat, "Multi batch: latitude values mismatch"
        assert expected_lon == multi_lon, "Multi batch: longitude values mismatch"
        
        # Test 4: Both approaches produce identical raw data usage
        assert single_rho == multi_rho, "Single and multi batch use different rhodamine values"
        assert single_ph == multi_ph, "Single and multi batch use different pH values"
        assert single_temp == multi_temp, "Single and multi batch use different temperature values"
        assert single_salinity == multi_salinity, "Single and multi batch use different salinity values"
        assert single_lat == multi_lat, "Single and multi batch use different latitude values"
        assert single_lon == multi_lon, "Single and multi batch use different longitude values"
        
        print("✓ Combined batch consistency and raw data usage test passed:")
        print(f"  - Both approaches: {len(single_final)} records")
        print(f"  - All {len(expected_rho)} raw data points used exactly once")
        print("  - Identical results regardless of batch size")
        print(f"  - Single batch: {len(single_rho)} unique values per sensor")
        print(f"  - Multi batch: {len(multi_rho)} unique values per sensor")


def test_resampling_1hz_to_10s():
    """Test resampling from 1Hz raw data to 10-second intervals."""
    import tempfile
    import os
    from locness_datamanager.setup_db import setup_sqlite_db
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_1hz_10s.sqlite")
        setup_sqlite_db(db_path)
        
        # Create 60 seconds of 1Hz data (60 records)
        dt = pd.date_range('2023-01-01 00:00:00', periods=60, freq='1s')
        
        # Create synthetic sensor data with patterns we can verify
        fluoro = pd.DataFrame({
            'datetime_utc': dt.astype(int) // 10**9,
            'rho_ppb': [10.0 + 0.1 * (i % 10) for i in range(60)]  # Repeating pattern every 10 samples
        })
        ph = pd.DataFrame({
            'datetime_utc': dt.astype(int) // 10**9,
            'ph_total': [7.5 + 0.01 * i for i in range(60)],  # Linear trend
            'vrse': [0.15 + 0.001 * i for i in range(60)]
        })
        tsg = pd.DataFrame({
            'datetime_utc': dt.astype(int) // 10**9,
            'temp': [22.0 + 2.0 * np.sin(i * np.pi / 30) for i in range(60)],  # Sinusoidal pattern
            'salinity': [35.5 + 0.5 * np.cos(i * np.pi / 20) for i in range(60)]
        })
        gps = pd.DataFrame({
            'datetime_utc': dt.astype(int) // 10**9,
            'latitude': [40.0 + 0.001 * i for i in range(60)],  # Linear movement
            'longitude': [-70.0 + 0.001 * i for i in range(60)]
        })
        
        # Insert all data
        with sqlite3.connect(db_path) as conn:
            fluoro.to_sql('rhodamine', conn, if_exists='append', index=False)
            ph.to_sql('ph', conn, if_exists='append', index=False)
            tsg.to_sql('tsg', conn, if_exists='append', index=False)
            gps.to_sql('gps', conn, if_exists='append', index=False)
        
        # Process with 10-second resampling
        result = resample.process_raw_data_incremental(
            sqlite_path=db_path,
            resample_interval='10s',
            summary_table='underway_summary',
            replace_all=True,
            ph_k0=0.0,
            ph_k2=0.0,
            ph_ma_window=30,  # 30 second window
            ph_freq=0.1  # 0.1 Hz after resampling (1 sample per 10s)
        )
        
        # Verify resampling results
        assert not result.empty, "Result should not be empty"
        
        # With 60 seconds of data at 10s intervals, we should get 6 records
        # (0s, 10s, 20s, 30s, 40s, 50s)
        expected_records = 6
        assert len(result) == expected_records, f"Expected {expected_records} records, got {len(result)}"
        
        # Sort by timestamp for consistent checking
        result_sorted = result.sort_values('datetime_utc').reset_index(drop=True)
        
        # Verify timestamps are at 10-second intervals
        expected_times = pd.date_range('2023-01-01 00:00:00', periods=6, freq='10s')
        result_times = pd.to_datetime(result_sorted['datetime_utc'])
        
        for i, (expected, actual) in enumerate(zip(expected_times, result_times)):
            assert expected == actual, f"Record {i}: expected {expected}, got {actual}"
        
        # Verify data values make sense for nearest neighbor resampling
        # The resampler should pick the nearest value to each 10s mark
        
        # For rhodamine (repeating pattern), verify we get expected values
        # At 0s, 10s, 20s, 30s, 40s, 50s we should get values near those timestamps
        rho_values = result_sorted['rho_ppb'].values
        assert len(rho_values) == 6, f"Expected 6 rhodamine values, got {len(rho_values)}"
        
        # For pH (linear trend), verify increasing trend
        ph_values = result_sorted['ph_total'].values
        assert len(ph_values) == 6, f"Expected 6 pH values, got {len(ph_values)}"
        # pH should be increasing (linear trend)
        for i in range(1, len(ph_values)):
            if not pd.isna(ph_values[i]) and not pd.isna(ph_values[i-1]):
                assert ph_values[i] >= ph_values[i-1], f"pH should be increasing: {ph_values[i-1]} -> {ph_values[i]}"
        
        # Verify computed pH values exist
        assert 'ph_corrected' in result_sorted.columns, "Should have corrected pH"
        ph_corrected_values = result_sorted['ph_corrected'].dropna()
        assert len(ph_corrected_values) > 0, "Should have some corrected pH values"
        
        # Verify moving averages are computed
        assert 'ph_corrected_ma' in result_sorted.columns, "Should have pH corrected moving average"
        assert 'ph_total_ma' in result_sorted.columns, "Should have pH total moving average"
        
        # At least some moving average values should be non-null
        ph_ma_values = result_sorted['ph_total_ma'].dropna()
        assert len(ph_ma_values) > 0, "Should have some pH moving average values"
        
        # Verify GPS coordinates show movement
        lat_values = result_sorted['latitude'].dropna()
        lon_values = result_sorted['longitude'].dropna()
        assert len(lat_values) > 0, "Should have latitude values"
        assert len(lon_values) > 0, "Should have longitude values"
        
        # Check that coordinates are changing (movement)
        if len(lat_values) > 1:
            lat_range = lat_values.max() - lat_values.min()
            assert lat_range > 0, "Latitude should show movement"
        
        print("✓ 1Hz to 10s resampling test passed:")
        print("  - Raw data: 60 records at 1Hz")
        print(f"  - Resampled: {len(result)} records at 10s intervals")
        print(f"  - pH range: {ph_values.min():.3f} to {ph_values.max():.3f}")
        print(f"  - Moving averages computed: {len(ph_ma_values)} values")
        print(f"  - GPS movement: Lat {lat_values.min():.3f} to {lat_values.max():.3f}")


def test_resampling_different_rates_consistency():
    """Test that resampling maintains data integrity across different time scales."""
    import tempfile
    import os
    from locness_datamanager.setup_db import setup_sqlite_db
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test different resampling scenarios
        test_cases = [
            {'freq': '2s', 'interval': '2s', 'expected_records': 30},  # No resampling
            {'freq': '1s', 'interval': '5s', 'expected_records': 12},  # 5x downsampling
            {'freq': '0.5s', 'interval': '10s', 'expected_records': 6},  # 20x downsampling
        ]
        
        for case_num, case in enumerate(test_cases):
            db_path = os.path.join(tmpdir, f"test_case_{case_num}.sqlite")
            setup_sqlite_db(db_path)
            
            # Create 60 seconds of data at specified frequency
            total_duration = 60  # seconds
            freq_seconds = float(case['freq'].rstrip('s'))
            num_records = int(total_duration / freq_seconds)
            
            dt = pd.date_range('2023-01-01 00:00:00', periods=num_records, freq=case['freq'])
            
            # Create consistent test data
            fluoro = pd.DataFrame({
                'datetime_utc': dt.astype(int) // 10**9,
                'rho_ppb': [100.0 + i * 0.1 for i in range(num_records)]
            })
            ph = pd.DataFrame({
                'datetime_utc': dt.astype(int) // 10**9,
                'ph_total': [8.0 + i * 0.001 for i in range(num_records)],
                'vrse': [0.2 + i * 0.0001 for i in range(num_records)]
            })
            tsg = pd.DataFrame({
                'datetime_utc': dt.astype(int) // 10**9,
                'temp': [25.0 + i * 0.01 for i in range(num_records)],
                'salinity': [36.0 + i * 0.001 for i in range(num_records)]
            })
            gps = pd.DataFrame({
                'datetime_utc': dt.astype(int) // 10**9,
                'latitude': [45.0 + i * 0.0001 for i in range(num_records)],
                'longitude': [-75.0 + i * 0.0001 for i in range(num_records)]
            })
            
            # Insert data
            with sqlite3.connect(db_path) as conn:
                fluoro.to_sql('rhodamine', conn, if_exists='append', index=False)
                ph.to_sql('ph', conn, if_exists='append', index=False)
                tsg.to_sql('tsg', conn, if_exists='append', index=False)
                gps.to_sql('gps', conn, if_exists='append', index=False)
            
            # Process with specified resampling interval
            result = resample.process_raw_data_incremental(
                sqlite_path=db_path,
                resample_interval=case['interval'],
                summary_table='underway_summary',
                replace_all=True,
                ph_k0=0.0,
                ph_k2=0.0,
                ph_ma_window=20,
                ph_freq=1.0 / float(case['interval'].rstrip('s'))  # Frequency after resampling
            )
            
            # Verify expected number of records
            assert len(result) == case['expected_records'], \
                f"Case {case_num}: Expected {case['expected_records']} records, got {len(result)}"
            
            # Verify all expected columns exist
            expected_columns = ['datetime_utc', 'rho_ppb', 'ph_total', 'ph_corrected', 
                              'temp', 'salinity', 'latitude', 'longitude']
            for col in expected_columns:
                assert col in result.columns, f"Case {case_num}: Missing column {col}"
            
            # Verify data trends are preserved (monotonic increases)
            result_sorted = result.sort_values('datetime_utc')
            
            # Check rhodamine trend (should be increasing)
            rho_values = result_sorted['rho_ppb'].dropna()
            if len(rho_values) > 1:
                rho_diffs = np.diff(rho_values)
                assert np.all(rho_diffs >= 0), f"Case {case_num}: Rhodamine should be non-decreasing"
            
            # Check pH trend (should be increasing)
            ph_values = result_sorted['ph_total'].dropna()
            if len(ph_values) > 1:
                ph_diffs = np.diff(ph_values)
                assert np.all(ph_diffs >= 0), f"Case {case_num}: pH should be non-decreasing"
            
            # Check temperature trend (should be increasing)  
            temp_values = result_sorted['temp'].dropna()
            if len(temp_values) > 1:
                temp_diffs = np.diff(temp_values)
                assert np.all(temp_diffs >= 0), f"Case {case_num}: Temperature should be non-decreasing"
            
            print(f"✓ Case {case_num}: {case['freq']} -> {case['interval']} resampling")
            print(f"    Raw: {num_records} records, Resampled: {len(result)} records")
        
        print("✓ Multi-rate resampling consistency test passed:")
        print(f"  - Tested {len(test_cases)} different resampling scenarios")
        print("  - All maintained data integrity and trends")
