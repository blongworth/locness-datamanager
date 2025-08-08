"""
Tests for missing data handling in aggregation operations.

This module tests that resampling and moving average calculations
properly handle missing/NaN values by dropping them before computing means.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
import sqlite3
from locness_datamanager import resample, resample_summary
from locness_datamanager.setup_db import setup_sqlite_db
from locness_datamanager.resampler import PersistentResampler


class TestMovingAverageNaNHandling:
    """Test moving average calculations with NaN values."""
    
    def test_add_ph_moving_average_with_nan_values(self):
        """Test that moving averages correctly handle NaN values by dropping them."""
        # Create test data with NaN values
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2023-01-01', periods=6, freq='1s'),
            'ph_total': [7.0, np.nan, 7.2, np.nan, 7.4, 7.5],
            'ph_corrected': [7.1, 7.2, np.nan, 7.4, np.nan, 7.6]
        })
        
        # Apply moving average with window of 3 samples
        result = resample.add_ph_moving_average(df, window_seconds=3, freq_hz=1.0)
        
        # Check that columns were created
        assert 'ph_total_ma' in result.columns
        assert 'ph_corrected_ma' in result.columns
        
        # Test specific calculations
        # At index 2: ph_total window is [7.0, NaN, 7.2] -> should be mean of [7.0, 7.2] = 7.1
        assert result['ph_total_ma'].iloc[2] == pytest.approx(7.1, abs=1e-10)
        
        # At index 4: ph_total window is [7.2, NaN, 7.4] -> should be mean of [7.2, 7.4] = 7.3
        assert result['ph_total_ma'].iloc[4] == pytest.approx(7.3, abs=1e-10)
        
        # At index 2: ph_corrected window is [7.1, 7.2, NaN] -> should be mean of [7.1, 7.2] = 7.15
        assert result['ph_corrected_ma'].iloc[2] == pytest.approx(7.15, abs=1e-10)
        
        # At index 4: ph_corrected window is [NaN, 7.4, NaN] -> should be 7.4
        assert result['ph_corrected_ma'].iloc[4] == pytest.approx(7.4, abs=1e-10)
    
    def test_add_ph_moving_average_all_nan_window(self):
        """Test moving average when entire window contains only NaN values."""
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2023-01-01', periods=5, freq='1s'),
            'ph_total': [np.nan, np.nan, np.nan, np.nan, 7.5],
            'ph_corrected': [np.nan, np.nan, np.nan, np.nan, 7.6]
        })
        
        # Apply moving average with window of 3 samples
        result = resample.add_ph_moving_average(df, window_seconds=3, freq_hz=1.0)
        
        # At index 2: ph_total window is [np.nan, np.nan, np.nan] -> should be NaN
        assert pd.isna(result['ph_total_ma'].iloc[2])
        
        # At index 3: ph_total window is [np.nan, np.nan, np.nan] -> should be NaN  
        assert pd.isna(result['ph_corrected_ma'].iloc[3])
        
        # At index 4: ph_total window contains 7.5, so should not be NaN
        assert not pd.isna(result['ph_total_ma'].iloc[4])
        assert result['ph_total_ma'].iloc[4] == pytest.approx(7.5, abs=1e-10)
    
    def test_add_ph_moving_average_no_nan_values(self):
        """Test moving average works correctly when there are no NaN values."""
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2023-01-01', periods=5, freq='1s'),
            'ph_total': [7.0, 7.1, 7.2, 7.3, 7.4],
            'ph_corrected': [7.1, 7.2, 7.3, 7.4, 7.5]
        })
        
        # Apply moving average with window of 3 samples
        result = resample.add_ph_moving_average(df, window_seconds=3, freq_hz=1.0)
        
        # At index 2: window is [7.0, 7.1, 7.2] -> mean = 7.1
        assert result['ph_total_ma'].iloc[2] == pytest.approx(7.1, abs=1e-10)
        
        # At index 4: window is [7.2, 7.3, 7.4] -> mean = 7.3
        assert result['ph_total_ma'].iloc[4] == pytest.approx(7.3, abs=1e-10)
    
    def test_persistent_resampler_moving_average_nan_handling(self):
        """Test that PersistentResampler correctly handles NaN in moving averages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_nan_ma.sqlite")
            setup_sqlite_db(db_path)
            
            # Create resampler
            resampler = PersistentResampler(
                sqlite_path=db_path,
                resample_interval='1s',
                ph_ma_window=3,  # 3 second window
                ph_freq=1.0,      # 1 Hz
                ph_k0=0.0,
                ph_k2=0.0
            )
            
            # Create test data with NaN values
            dt = pd.date_range('2023-01-01 00:00:00', periods=6, freq='1s')
            dt_unix = dt.astype(int) // 10**9
            
            # Insert pH data with NaN values
            ph_data = pd.DataFrame({
                'datetime_utc': dt_unix,
                'ph_total': [7.0, np.nan, 7.2, np.nan, 7.4, 7.5],
                'vrse': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            })
            
            # Insert some basic data for other sensors
            rhodamine_data = pd.DataFrame({
                'datetime_utc': dt_unix,
                'rho_ppb': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
            })
            
            tsg_data = pd.DataFrame({
                'datetime_utc': dt_unix,
                'temp': [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                'salinity': [35.0, 35.1, 35.2, 35.3, 35.4, 35.5]
            })
            
            gps_data = pd.DataFrame({
                'datetime_utc': dt_unix,
                'latitude': [42.0, 42.1, 42.2, 42.3, 42.4, 42.5],
                'longitude': [-71.0, -71.1, -71.2, -71.3, -71.4, -71.5]
            })
            
            # Insert data into database
            with sqlite3.connect(db_path) as conn:
                rhodamine_data.to_sql('rhodamine', conn, if_exists='append', index=False)
                ph_data.to_sql('ph', conn, if_exists='append', index=False)
                tsg_data.to_sql('tsg', conn, if_exists='append', index=False)
                gps_data.to_sql('gps', conn, if_exists='append', index=False)
            
            # Process data
            result = resampler.process_new_data()
            
            assert not result.empty
            assert 'ph_total_ma' in result.columns
            
            # Verify that moving averages handle NaN correctly
            result_sorted = result.sort_values('datetime_utc').reset_index(drop=True)
            
            # At index 2: ph_total window is [7.0, NaN, 7.2] -> should be mean of [7.0, 7.2] = 7.1
            ma_values = result_sorted['ph_total_ma'].dropna()
            assert len(ma_values) > 0, "Should have some moving average values"
            
            # Find the record at index 2 (third record)
            if len(result_sorted) >= 3:
                # The moving average should handle NaN correctly
                assert not pd.isna(result_sorted['ph_total_ma'].iloc[2]), "Moving average should not be NaN when some values are available"


class TestResampleSummaryNaNHandling:
    """Test resampling operations with NaN values."""
    
    def test_resample_summary_data_with_nan_values(self):
        """Test that resample_summary_data correctly handles NaN values by dropping them."""
        # Create test data with NaN values
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2023-01-01 00:00:00', periods=6, freq='30s'),
            'temp': [20.0, np.nan, 22.0, np.nan, 24.0, 25.0],
            'salinity': [35.0, 35.1, np.nan, 35.3, np.nan, 35.5],
            'ph_total': [7.0, 7.1, 7.2, np.nan, 7.4, 7.5]
        })
        
        # Resample to 1-minute intervals (should group pairs of records)
        result = resample_summary.resample_summary_data(df, resample_interval='1min')
        
        # Should have 3 groups: [0,1], [2,3], [4,5]
        assert len(result) == 3
        
        # Group 1: temp should be mean of [20.0, NaN] = 20.0 (NaN dropped)
        assert result['temp'].iloc[0] == pytest.approx(20.0, abs=1e-10)
        
        # Group 1: salinity should be mean of [35.0, 35.1] = 35.05
        assert result['salinity'].iloc[0] == pytest.approx(35.05, abs=1e-10)
        
        # Group 2: temp should be mean of [22.0, NaN] = 22.0 (NaN dropped)
        assert result['temp'].iloc[1] == pytest.approx(22.0, abs=1e-10)
        
        # Group 2: salinity should be mean of [NaN, 35.3] = 35.3 (NaN dropped)
        assert result['salinity'].iloc[1] == pytest.approx(35.3, abs=1e-10)
        
        # Group 3: temp should be mean of [24.0, 25.0] = 24.5
        assert result['temp'].iloc[2] == pytest.approx(24.5, abs=1e-10)
    
    def test_resample_summary_data_all_nan_group(self):
        """Test resampling when an entire group contains only NaN values."""
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2023-01-01 00:00:00', periods=4, freq='30s'),
            'temp': [20.0, 21.0, np.nan, np.nan],
            'salinity': [35.0, np.nan, np.nan, np.nan]
        })
        
        # Resample to 1-minute intervals
        result = resample_summary.resample_summary_data(df, resample_interval='1min')
        
        # Should have 2 groups: [0,1], [2,3]
        # However, the second group [2,3] has all NaN values, so it gets dropped by dropna(how='all')
        # Therefore we expect only 1 group in the result
        assert len(result) == 1
        
        # Group 1: temp should be mean of [20.0, 21.0] = 20.5
        assert result['temp'].iloc[0] == pytest.approx(20.5, abs=1e-10)
        
        # Group 1: salinity should be mean of [35.0, NaN] = 35.0
        assert result['salinity'].iloc[0] == pytest.approx(35.0, abs=1e-10)
    
    def test_resample_summary_data_no_nan_values(self):
        """Test resampling works correctly when there are no NaN values."""
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2023-01-01 00:00:00', periods=4, freq='30s'),
            'temp': [20.0, 21.0, 22.0, 23.0],
            'salinity': [35.0, 35.1, 35.2, 35.3]
        })
        
        # Resample to 1-minute intervals
        result = resample_summary.resample_summary_data(df, resample_interval='1min')
        
        # Should have 2 groups: [0,1], [2,3]
        assert len(result) == 2
        
        # Group 1: temp should be mean of [20.0, 21.0] = 20.5
        assert result['temp'].iloc[0] == pytest.approx(20.5, abs=1e-10)
        
        # Group 2: temp should be mean of [22.0, 23.0] = 22.5
        assert result['temp'].iloc[1] == pytest.approx(22.5, abs=1e-10)


class TestIntegrationNaNHandling:
    """Integration tests for missing data handling across the system."""
    
    def test_end_to_end_nan_handling(self):
        """Test complete pipeline with NaN values from raw data to final output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_integration_nan.sqlite")
            setup_sqlite_db(db_path)
            
            # Create test data with strategic NaN placement
            dt = pd.date_range('2023-01-01 00:00:00', periods=10, freq='1s')
            dt_unix = dt.astype(int) // 10**9
            
            # Insert data with NaN values in various sensors
            rhodamine_data = pd.DataFrame({
                'datetime_utc': dt_unix,
                'rho_ppb': [10.0, np.nan, 12.0, 13.0, np.nan, 15.0, 16.0, np.nan, 18.0, 19.0]
            })
            
            ph_data = pd.DataFrame({
                'datetime_utc': dt_unix,
                'ph_total': [7.0, 7.1, np.nan, 7.3, 7.4, np.nan, 7.6, 7.7, np.nan, 7.9],
                'vrse': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            })
            
            tsg_data = pd.DataFrame({
                'datetime_utc': dt_unix,
                'temp': [20.0, 21.0, 22.0, np.nan, 24.0, 25.0, np.nan, 27.0, 28.0, 29.0],
                'salinity': [35.0, np.nan, 35.2, 35.3, 35.4, np.nan, 35.6, 35.7, 35.8, np.nan]
            })
            
            gps_data = pd.DataFrame({
                'datetime_utc': dt_unix,
                'latitude': [42.0, 42.1, 42.2, 42.3, 42.4, 42.5, 42.6, 42.7, 42.8, 42.9],
                'longitude': [-71.0, -71.1, -71.2, -71.3, -71.4, -71.5, -71.6, -71.7, -71.8, -71.9]
            })
            
            # Insert data into database
            with sqlite3.connect(db_path) as conn:
                rhodamine_data.to_sql('rhodamine', conn, if_exists='append', index=False)
                ph_data.to_sql('ph', conn, if_exists='append', index=False)
                tsg_data.to_sql('tsg', conn, if_exists='append', index=False)
                gps_data.to_sql('gps', conn, if_exists='append', index=False)
            
            # Step 1: Process raw data with moving averages
            result_raw = resample.process_raw_data_incremental(
                sqlite_path=db_path,
                resample_interval='1s',
                summary_table='underway_summary',
                replace_all=True,
                ph_ma_window=3,
                ph_freq=1.0
            )
            
            assert not result_raw.empty
            assert 'ph_total_ma' in result_raw.columns
            
            # Check that moving averages were computed despite NaN values
            ma_values = result_raw['ph_total_ma'].dropna()
            assert len(ma_values) > 0, "Should have computed some moving averages despite NaN values"
            
            # Step 2: Resample summary data with NaN handling
            result_resampled = resample_summary.resample_summary_data(
                result_raw, 
                resample_interval='2s'
            )
            
            assert not result_resampled.empty
            
            # Verify that resampling handled NaN values correctly
            # All columns should have some non-NaN values
            for col in ['temp', 'salinity', 'ph_total']:
                if col in result_resampled.columns:
                    non_nan_values = result_resampled[col].dropna()
                    assert len(non_nan_values) > 0, f"Column {col} should have some non-NaN values after resampling"
    
    def test_comparison_with_without_nan_handling(self):
        """Compare results with and without proper NaN handling to verify improvement."""
        # Create test data with NaN values
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2023-01-01', periods=6, freq='1s'),
            'ph_total': [7.0, np.nan, 7.2, np.nan, 7.4, 7.5]
        })
        
        # Method 1: Using our improved function (min_periods=1)
        result_improved = resample.add_ph_moving_average(df.copy(), window_seconds=3, freq_hz=1.0)
        
        # Method 2: Using strict rolling (min_periods=window_size)
        df_strict = df.copy()
        window_size = 3
        df_strict['ph_total_ma_strict'] = df_strict['ph_total'].rolling(
            window=window_size  # No min_periods, defaults to window_size
        ).mean()
        
        # Compare results
        # With min_periods=1: should compute moving averages even with some NaN in window
        # With default min_periods=window: needs all values in window to be non-NaN
        
        # Count how many values are non-NaN in each method
        strict_non_nan = df_strict['ph_total_ma_strict'].dropna()
        improved_non_nan = result_improved['ph_total_ma'].dropna()
        
        # The improved method should have more non-NaN values
        assert len(improved_non_nan) > len(strict_non_nan), f"Improved method should preserve more values: {len(improved_non_nan)} vs {len(strict_non_nan)}"
        
        # Strict method should have all NaN due to missing values in windows
        assert len(strict_non_nan) == 0, "Strict method should produce all NaN values due to missing data in windows"
        
        # Improved method should have computed some moving averages
        assert len(improved_non_nan) > 0, "Improved method should compute some moving averages despite missing data"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
