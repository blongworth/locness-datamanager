"""Tests for synthetic_data module."""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import sqlite3


from locness_datamanager.synthetic_data import (
    generate_rhodamine_data,
    generate_ph_data,
    generate_tsg_data,
    generate_raw_sensor_batch,
    write_to_raw_tables,
)
from locness_datamanager.resample import load_and_resample_sqlite, write_resampled_to_sqlite
from locness_datamanager import file_writers

#TODO: fix tests
#TODO: return generated data with pandas timestamp, not int

class TestSyntheticDataGeneration:
    """Test synthetic data generation functions."""

    def test_generate_rhodamine_data_basic(self):
        """Test basic rhodamine data generation."""
        df = generate_rhodamine_data(n_records=10, frequency_hz=1.0)
        
        assert len(df) == 10
        assert list(df.columns) == [
            "datetime_utc", "gain", "voltage", "rho_ppb"
        ]
        
        # Check data types and ranges
        assert df["datetime_utc"].dtype == "int64"
        assert df["gain"].dtype == "int64"
        assert df["voltage"].dtype == "float64"
        assert df["rho_ppb"].dtype == "float64"

        # Check realistic ranges
        assert all(df["gain"].isin([1, 10, 100]))
        assert all(df["voltage"] >= 0)
        assert all(df["voltage"] <= 5.0)
        assert all(df["rho_ppb"] >= 0)

    def test_generate_ph_data_basic(self):
        """Test basic pH data generation."""
        df = generate_ph_data(n_records=5, frequency_hz=0.1)
        
        assert len(df) == 5
        expected_columns = [
            "datetime_utc", "samp_num", "ph_timestamp", "v_bat", "v_bias_pos",
            "v_bias_neg", "t_board", "h_board", "vrse", "vrse_std", "cevk",
            "cevk_std", "ce_ik", "i_sub", "cal_temp", "cal_sal", "k0", "k2",
            "ph_free", "ph_total"
        ]
        assert list(df.columns) == expected_columns
        
        # Check pH ranges
        assert all(df["ph_free"] > 7.0)
        assert all(df["ph_free"] < 9.0)
        assert all(df["ph_total"] > 7.0)
        assert all(df["ph_total"] < 9.0)

    def test_generate_tsg_data_basic(self):
        """Test basic TSG data generation."""
        df = generate_tsg_data(n_records=10, frequency_hz=1.0)
        
        assert len(df) == 10
        expected_columns = [
            "datetime_utc", "scan_no", "cond", "temp", "salinity", "hull_temp",
            "time_elapsed", "nmea_time", "latitude", "longitude"
        ]
        assert list(df.columns) == expected_columns
        
        # Check realistic ranges
        assert all(df["temp"] > 0)  # Temperature in Celsius
        assert all(df["hull_temp"] > df["temp"])  # Hull temp should be higher
        assert all(df["cond"] >= 0)  # Conductivity should be positive
        assert all(df["scan_no"] >= 1)  # Scan numbers start at 1

    def test_generate_with_custom_start_time(self):
        """Test generation with custom start time."""
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        df = generate_rhodamine_data(n_records=3, start_time=start_time, frequency_hz=1.0)
        
        # Check timestamps are sequential
        timestamps = pd.to_datetime(df["datetime_utc"], unit="s")

        # Check that timestamps are spaced approximately 1 second apart
        time_diffs = timestamps.diff().dropna()
        for diff in time_diffs:
            assert abs(diff.total_seconds() - 1.0) < 0.1
            
        # Check that we have 3 records
        assert len(df) == 3

    def test_generate_raw_sensor_batch(self):
        """Test generating a batch of all sensor types."""
        batch = generate_raw_sensor_batch(num=10, freq=1.0)
        
        assert "rhodamine" in batch
        assert "ph" in batch
        assert "tsg" in batch
        
        assert len(batch["rhodamine"]) == 10
        assert len(batch["tsg"]) == 10
        # pH should have fewer records due to lower frequency
        assert len(batch["ph"]) >= 1


class TestDataWriting:
    """Test data writing functions."""

    def test_write_to_raw_tables(self, sample_sqlite_db):
        """Test writing synthetic data to raw tables."""
        # Generate sample data
        rhodamine_df = generate_rhodamine_data(n_records=5)
        ph_df = generate_ph_data(n_records=3)
        tsg_df = generate_tsg_data(n_records=5)
        
        # Write to database
        write_to_raw_tables(
            rhodamine_df=rhodamine_df,
            ph_df=ph_df,
            tsg_df=tsg_df,
            sqlite_path=str(sample_sqlite_db)
        )
        
        # Verify data was written
        import sqlite3
        conn = sqlite3.connect(sample_sqlite_db)
        
        # Check rhodamine table
        cursor = conn.execute("SELECT COUNT(*) FROM rhodamine")
        assert cursor.fetchone()[0] == 5
        
        # Check ph table
        cursor = conn.execute("SELECT COUNT(*) FROM ph")
        assert cursor.fetchone()[0] == 3
        
        # Check tsg table
        cursor = conn.execute("SELECT COUNT(*) FROM tsg")
        assert cursor.fetchone()[0] == 5
        
        conn.close()


class TestFieldMapping:
    """Test that all expected fields are present in raw data, resampled data, and CSV outputs."""

    def test_raw_rhodamine_fields_complete(self):
        """Test that rhodamine raw data has all expected fields."""
        df = generate_rhodamine_data(n_records=5)

        expected_fields = ["datetime_utc", "gain", "voltage", "rho_ppb"]
        assert list(df.columns) == expected_fields
        
        # Verify no null values in critical fields
        assert not df["datetime_utc"].isnull().any()
        assert not df["rho_ppb"].isnull().any()

    def test_raw_ph_fields_complete(self):
        """Test that pH raw data has all expected fields."""
        df = generate_ph_data(n_records=3)
        
        expected_fields = [
            "datetime_utc", "samp_num", "ph_timestamp", "v_bat", "v_bias_pos",
            "v_bias_neg", "t_board", "h_board", "vrse", "vrse_std", "cevk",
            "cevk_std", "ce_ik", "i_sub", "cal_temp", "cal_sal", "k0", "k2",
            "ph_free", "ph_total"
        ]
        assert list(df.columns) == expected_fields
        
        # Verify no null values in critical fields
        assert not df["datetime_utc"].isnull().any()
        assert not df["ph_free"].isnull().any()
        assert not df["ph_total"].isnull().any()

    def test_raw_tsg_fields_complete(self):
        """Test that TSG raw data has all expected fields."""
        df = generate_tsg_data(n_records=5)
        
        expected_fields = [
            "datetime_utc", "scan_no", "cond", "temp", "salinity", "hull_temp",
            "time_elapsed", "nmea_time", "latitude", "longitude"
        ]
        assert list(df.columns) == expected_fields
        
        # Verify no null values in critical fields
        assert not df["datetime_utc"].isnull().any()
        assert not df["temp"].isnull().any()
        assert not df["cond"].isnull().any()
        assert not df["latitude"].isnull().any()
        assert not df["longitude"].isnull().any()


class TestDataValidation:
    """Test data validation and edge cases."""

    def test_zero_frequency_handling(self):
        """Test handling of zero frequency."""
        df = generate_rhodamine_data(n_records=3, frequency_hz=0.0)
        assert len(df) == 3
        
        # Should default to 1 second intervals
        timestamps = pd.to_datetime(df["datetime_utc"], unit="s")
        time_diffs = timestamps.diff().dropna()
        assert all(abs(diff.total_seconds() - 1.0) < 0.1 for diff in time_diffs)

    def test_single_record_generation(self):
        """Test generating a single record."""
        df = generate_tsg_data(n_records=1)
        assert len(df) == 1
        assert df.iloc[0]["scan_no"] == 1
        assert df.iloc[0]["time_elapsed"] == 0

    def test_high_frequency_generation(self):
        """Test high frequency data generation."""
        df = generate_rhodamine_data(n_records=10, frequency_hz=10.0)
        assert len(df) == 10
        
        # Check that timestamps exist and are increasing
        timestamps = pd.to_datetime(df["datetime_utc"], unit="s")
        assert timestamps.is_monotonic_increasing
        
        # Check that we have the correct number of records
        assert len(timestamps) == 10

    @pytest.mark.slow
    def test_large_dataset_generation(self):
        """Test generating larger datasets."""
        df = generate_tsg_data(n_records=1000, frequency_hz=1.0)
        assert len(df) == 1000
        
        # Check that scan numbers are sequential
        assert list(df["scan_no"]) == list(range(1, 1001))
        
        # Check that time_elapsed increases properly
        expected_elapsed = [i * 1.0 for i in range(1000)]
        assert all(abs(actual - expected) < 0.001 
                  for actual, expected in zip(df["time_elapsed"], expected_elapsed))


if __name__ == "__main__":
    pytest.main([__file__])
