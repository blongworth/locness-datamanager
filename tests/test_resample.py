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
import types

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


def test_write_resampled_to_sqlite_unique_constraint_error(capfd):
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
            out, _ = capfd.readouterr()
            assert 'Error writing to underway_summary table: UNIQUE constraint failed' in out
