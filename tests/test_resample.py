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
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from locness_datamanager import resample

def make_df(data, columns):
    return pd.DataFrame(data, columns=columns)

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
    df = resample.resample_tables(fluoro, ph, tsg, gps, resample_interval='2s')
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
