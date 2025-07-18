import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from locness_datamanager import resample

def make_df(data, columns):
    return pd.DataFrame(data, columns=columns)

def test_add_ph_moving_average_basic():
    df = pd.DataFrame({
        'datetime_utc': pd.date_range('2023-01-01', periods=5, freq='2s'),
        'ph_total': [7.0, 7.1, 7.2, 7.3, 7.4]
    })
    result = resample.add_ph_moving_average(df, window_seconds=4, freq_hz=0.5)
    assert 'ph_total_ma' in result.columns
    # The rolling window is 2, so the last value should be the mean of the last 2 values
    expected = (7.3 + 7.4) / 2  # 7.35
    assert result['ph_total_ma'].iloc[-1] == pytest.approx(expected, rel=1e-2)

def test_add_computed_fields_uses_config():
    df = pd.DataFrame({
        'datetime_utc': pd.date_range('2023-01-01', periods=3, freq='2s'),
        'ph_total': [7.0, 7.2, 7.4]
    })
    config = {'ph_ma_window': 4, 'ph_freq': 0.5}
    result = resample.add_computed_fields(df, config)
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
            make_df({'datetime_utc': dt, 'ph_total': [7,8]}, ['datetime_utc','ph_total']),
            make_df({'datetime_utc': dt, 'temp': [20,21], 'salinity': [35,36]}, ['datetime_utc','temp','salinity']),
            make_df({'datetime_utc': dt, 'latitude': [10,11], 'longitude': [20,21]}, ['datetime_utc','latitude','longitude'])
        )
        df = resample.load_and_resample_sqlite('dummy_path', resample_interval='2s')
        assert 'ph_total_ma' in df.columns
        assert len(df) == 2

class TestResampleMissingData:
    @staticmethod
    def make_df(columns, data=None):
        import pandas as pd
        if data is None:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(data, columns=columns)

    def test_resample_all_tables_present(self):
        fluoro = self.make_df(['datetime_utc', 'rho_ppb'], [[1, 10], [2, 20]])
        ph = self.make_df(['datetime_utc', 'ph_total'], [[1, 7.5], [2, 7.6]])
        tsg = self.make_df(['datetime_utc', 'temp', 'salinity'], [[1, 10, 35], [2, 11, 36]])
        gps = self.make_df(['datetime_utc', 'latitude', 'longitude'], [[1, 50, -1], [2, 51, -2]])
        df = resample.resample_tables(fluoro, ph, tsg, gps, resample_interval='1s', config={'res_int': '1s'})
        assert set(df.columns) == {'datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'temp', 'salinity'}
        assert len(df) > 0

    def test_resample_missing_fluoro(self):
        fluoro = self.make_df(['datetime_utc', 'rho_ppb'])
        ph = self.make_df(['datetime_utc', 'ph_total'], [[1, 7.5], [2, 7.6]])
        tsg = self.make_df(['datetime_utc', 'temp', 'salinity'], [[1, 10, 35], [2, 11, 36]])
        gps = self.make_df(['datetime_utc', 'latitude', 'longitude'], [[1, 50, -1], [2, 51, -2]])
        df = resample.resample_tables(fluoro, ph, tsg, gps, resample_interval='1s', config={'res_int': '1s'})
        assert set(df.columns) == {'datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'temp', 'salinity'}
        assert df['rho_ppb'].isna().all()
        assert not df['latitude'].isna().all()

    def test_resample_only_gps(self):
        fluoro = self.make_df(['datetime_utc', 'rho_ppb'])
        ph = self.make_df(['datetime_utc', 'ph_total'])
        tsg = self.make_df(['datetime_utc', 'temp', 'salinity'])
        gps = self.make_df(['datetime_utc', 'latitude', 'longitude'], [[1, 50, -1], [2, 51, -2]])
        df = resample.resample_tables(fluoro, ph, tsg, gps, resample_interval='1s', config={'res_int': '1s'})
        assert set(df.columns) == {'datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'temp', 'salinity'}
        assert df['latitude'].notna().any()
        assert df['rho_ppb'].isna().all()
        assert df['ph_total'].isna().all()
        assert df['temp'].isna().all()
        assert df['salinity'].isna().all()

    def test_resample_all_missing(self):
        fluoro = self.make_df(['datetime_utc', 'rho_ppb'])
        ph = self.make_df(['datetime_utc', 'ph_total'])
        tsg = self.make_df(['datetime_utc', 'temp', 'salinity'])
        gps = self.make_df(['datetime_utc', 'latitude', 'longitude'])
        df = resample.resample_tables(fluoro, ph, tsg, gps, resample_interval='1s', config={'res_int': '1s'})
        assert set(df.columns) == {'datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'temp', 'salinity'}
        assert df.empty or df.isna().all().all()

    def test_resample_missing_gps(self):
        fluoro = self.make_df(['datetime_utc', 'rho_ppb'], [[1, 10], [2, 20]])
        ph = self.make_df(['datetime_utc', 'ph_total'], [[1, 7.5], [2, 7.6]])
        tsg = self.make_df(['datetime_utc', 'temp', 'salinity'], [[1, 10, 35], [2, 11, 36]])
        gps = self.make_df(['datetime_utc', 'latitude', 'longitude'])
        df = resample.resample_tables(fluoro, ph, tsg, gps, resample_interval='1s', config={'res_int': '1s'})
        assert set(df.columns) == {'datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph_total', 'temp', 'salinity'}
        assert df['latitude'].isna().all()
        assert df['rho_ppb'].notna().any()

def test_resample_tables_mean_correctness():
    import pandas as pd
    from locness_datamanager import resample
    # Create data with two points per 2s interval, resample to 4s
    dt = pd.date_range('2023-01-01', periods=4, freq='2s')
    # fluoro: 0,1,2,3 at 0s,2s,4s,6s
    fluoro = pd.DataFrame({'datetime_utc': dt, 'rho_ppb': [0, 1, 2, 3]})
    # ph: 10,20,30,40 at 0s,2s,4s,6s
    ph = pd.DataFrame({'datetime_utc': dt, 'ph_total': [10, 20, 30, 40]})
    # tsg: temp 100,200,300,400; salinity 1,2,3,4
    tsg = pd.DataFrame({'datetime_utc': dt, 'temp': [100, 200, 300, 400], 'salinity': [1, 2, 3, 4]})
    # gps: lat 50,51,52,53; lon -1,-2,-3,-4
    gps = pd.DataFrame({'datetime_utc': dt, 'latitude': [50, 51, 52, 53], 'longitude': [-1, -2, -3, -4]})
    df = resample.resample_tables(fluoro, ph, tsg, gps, resample_interval='4s')
    # There should be two intervals: 0s-4s, 4s-8s
    # For each, mean of two points
    assert len(df) == 2
    # First interval: mean([0,1])=0.5, mean([10,20])=15, mean([100,200])=150, mean([1,2])=1.5, mean([50,51])=50.5, mean([-1,-2])=-1.5
    assert df['rho_ppb'].iloc[0] == pytest.approx(0.5)
    assert df['ph_total'].iloc[0] == pytest.approx(15)
    assert df['temp'].iloc[0] == pytest.approx(150)
    assert df['salinity'].iloc[0] == pytest.approx(1.5)
    assert df['latitude'].iloc[0] == pytest.approx(50.5)
    assert df['longitude'].iloc[0] == pytest.approx(-1.5)
    # Second interval: mean([2,3])=2.5, mean([30,40])=35, mean([300,400])=350, mean([3,4])=3.5, mean([52,53])=52.5, mean([-3,-4])=-3.5
    assert df['rho_ppb'].iloc[1] == pytest.approx(2.5)
    assert df['ph_total'].iloc[1] == pytest.approx(35)
    assert df['temp'].iloc[1] == pytest.approx(350)
    assert df['salinity'].iloc[1] == pytest.approx(3.5)
    assert df['latitude'].iloc[1] == pytest.approx(52.5)
    assert df['longitude'].iloc[1] == pytest.approx(-3.5)
