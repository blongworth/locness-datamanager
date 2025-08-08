"""
Test for the updated resample_and_join method with mean aggregation.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from locness_datamanager.setup_db import setup_sqlite_db
from locness_datamanager.resampler import PersistentResampler


def test_resample_and_join_mean_aggregation():
    """Test that resample_and_join uses mean aggregation and handles NaN correctly."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_mean_agg.sqlite")
        setup_sqlite_db(db_path)
        
        # Create resampler with 2-second interval
        resampler = PersistentResampler(
            sqlite_path=db_path,
            resample_interval='2s',
            ph_ma_window=10,
            ph_freq=1.0
        )
        
        # Create test data with multiple points in each 2-second interval
        # and some NaN values to test handling
        timestamps = pd.to_datetime([
            '2023-01-01 00:00:00',  # Bin 1 start
            '2023-01-01 00:00:01',  # Bin 1
            '2023-01-01 00:00:02',  # Bin 2 start  
            '2023-01-01 00:00:03',  # Bin 2
            '2023-01-01 00:00:04',  # Bin 3 start
            '2023-01-01 00:00:05',  # Bin 3
        ])
        
        # Test data with some NaN values
        raw_data = {
            'rhodamine': pd.DataFrame({
                'datetime_utc': timestamps,
                'rho_ppb': [10.0, 12.0, np.nan, 16.0, 18.0, 20.0]  # NaN in bin 2
            }),
            'ph': pd.DataFrame({
                'datetime_utc': timestamps,
                'ph_total': [7.0, 7.2, 7.4, np.nan, 7.8, 8.0],  # NaN in bin 2
                'vrse': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            }),
            'tsg': pd.DataFrame({
                'datetime_utc': timestamps,
                'temp': [20.0, np.nan, 22.0, 23.0, np.nan, 25.0],  # NaN in bins 1 and 3
                'salinity': [35.0, 35.2, 35.4, 35.6, 35.8, np.nan]  # NaN in bin 3
            }),
            'gps': pd.DataFrame({
                'datetime_utc': timestamps,
                'latitude': [42.0, 42.1, 42.2, 42.3, 42.4, 42.5],
                'longitude': [-71.0, -71.1, -71.2, -71.3, -71.4, -71.5]
            })
        }
        
        # Resample and join
        result = resampler.resample_and_join(raw_data)
        
        # Verify we have the expected number of time bins (3 bins: 0s, 2s, 4s)
        assert len(result) == 3, f"Expected 3 time bins, got {len(result)}"
        
        # Sort result by datetime for consistent testing
        result = result.sort_values('datetime_utc').reset_index(drop=True)
        
        # Test mean aggregation in bin 1 (timestamps 0s, 1s -> should average)
        # rho_ppb: mean of [10.0, 12.0] = 11.0
        assert result['rho_ppb'].iloc[0] == pytest.approx(11.0, abs=1e-10)
        
        # ph_total: mean of [7.0, 7.2] = 7.1  
        assert result['ph_total'].iloc[0] == pytest.approx(7.1, abs=1e-10)
        
        # temp: mean of [20.0, NaN] = 20.0 (NaN dropped)
        assert result['temp'].iloc[0] == pytest.approx(20.0, abs=1e-10)
        
        # Test NaN handling in bin 2 (timestamps 2s, 3s)
        # rho_ppb: mean of [NaN, 16.0] = 16.0 (NaN dropped)
        assert result['rho_ppb'].iloc[1] == pytest.approx(16.0, abs=1e-10)
        
        # ph_total: mean of [7.4, NaN] = 7.4 (NaN dropped)
        assert result['ph_total'].iloc[1] == pytest.approx(7.4, abs=1e-10)
        
        # temp: mean of [22.0, 23.0] = 22.5
        assert result['temp'].iloc[1] == pytest.approx(22.5, abs=1e-10)
        
        # Test bin 3 (timestamps 4s, 5s)
        # rho_ppb: mean of [18.0, 20.0] = 19.0
        assert result['rho_ppb'].iloc[2] == pytest.approx(19.0, abs=1e-10)
        
        # temp: mean of [NaN, 25.0] = 25.0 (NaN dropped)
        assert result['temp'].iloc[2] == pytest.approx(25.0, abs=1e-10)
        
        # salinity: mean of [35.8, NaN] = 35.8 (NaN dropped)
        assert result['salinity'].iloc[2] == pytest.approx(35.8, abs=1e-10)
        
        print("✓ Mean aggregation with NaN handling test passed")


def test_resample_and_join_all_nan_in_bin():
    """Test behavior when all values in a time bin are NaN."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_all_nan.sqlite")
        setup_sqlite_db(db_path)
        
        resampler = PersistentResampler(
            sqlite_path=db_path,
            resample_interval='2s'
        )
        
        timestamps = pd.to_datetime([
            '2023-01-01 00:00:00',
            '2023-01-01 00:00:01',
            '2023-01-01 00:00:02',
            '2023-01-01 00:00:03'
        ])
        
        raw_data = {
            'rhodamine': pd.DataFrame({
                'datetime_utc': timestamps,
                'rho_ppb': [10.0, 12.0, np.nan, np.nan]  # Bin 2 has all NaN
            }),
            'ph': pd.DataFrame({
                'datetime_utc': timestamps,
                'ph_total': [7.0, 7.2, np.nan, np.nan],  # Bin 2 has all NaN
                'vrse': [0.1, 0.2, 0.3, 0.4]
            }),
            'tsg': pd.DataFrame({
                'datetime_utc': timestamps,
                'temp': [20.0, 21.0, 22.0, 23.0],
                'salinity': [35.0, 35.2, 35.4, 35.6]
            }),
            'gps': pd.DataFrame({
                'datetime_utc': timestamps,
                'latitude': [42.0, 42.1, 42.2, 42.3],
                'longitude': [-71.0, -71.1, -71.2, -71.3]
            })
        }
        
        result = resampler.resample_and_join(raw_data)
        result = result.sort_values('datetime_utc').reset_index(drop=True)
        
        # Bin 1: mean of [10.0, 12.0] = 11.0
        assert result['rho_ppb'].iloc[0] == pytest.approx(11.0, abs=1e-10)
        
        # Bin 2: mean of [NaN, NaN] = NaN (all values are NaN)
        assert pd.isna(result['rho_ppb'].iloc[1])
        assert pd.isna(result['ph_total'].iloc[1])
        
        # But temp and salinity should still have values
        assert result['temp'].iloc[1] == pytest.approx(22.5, abs=1e-10)
        
        print("✓ All NaN in bin test passed")


if __name__ == "__main__":
    test_resample_and_join_mean_aggregation()
    test_resample_and_join_all_nan_in_bin()
    print("All resample_and_join tests passed!")
