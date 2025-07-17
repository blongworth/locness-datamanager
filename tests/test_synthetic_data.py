"""Tests for synthetic_data module."""
import pytest
import pandas as pd
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import sqlite3

from locness_datamanager.synthetic_data import (
    generate_fluorometer_data,
    generate_ph_data,
    generate_tsg_data,
    generate,
    generate_raw_sensor_batch,
    write_to_raw_tables,
    resample_raw_sensor_data,
)
from locness_datamanager.resample import load_and_resample_sqlite, write_resampled_to_sqlite
from locness_datamanager import file_writers


class TestSyntheticDataGeneration:
    """Test synthetic data generation functions."""

    def test_generate_fluorometer_data_basic(self):
        """Test basic fluorometer data generation."""
        df = generate_fluorometer_data(n_records=10, frequency_hz=1.0)
        
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
            "datetime_utc", "scan_no", "cond", "temp", "hull_temp",
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
        df = generate_fluorometer_data(n_records=3, start_time=start_time, frequency_hz=1.0)
        
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
        
        assert "fluorometer" in batch
        assert "ph" in batch
        assert "tsg" in batch
        
        assert len(batch["fluorometer"]) == 10
        assert len(batch["tsg"]) == 10
        # pH should have fewer records due to lower frequency
        assert len(batch["ph"]) >= 1


class TestDataWriting:
    """Test data writing functions."""

    def test_write_to_raw_tables(self, sample_sqlite_db):
        """Test writing synthetic data to raw tables."""
        # Generate sample data
        fluorometer_df = generate_fluorometer_data(n_records=5)
        ph_df = generate_ph_data(n_records=3)
        tsg_df = generate_tsg_data(n_records=5)
        
        # Write to database
        write_to_raw_tables(
            fluorometer_df=fluorometer_df,
            ph_df=ph_df,
            tsg_df=tsg_df,
            sqlite_path=str(sample_sqlite_db)
        )
        
        # Verify data was written
        import sqlite3
        conn = sqlite3.connect(sample_sqlite_db)
        
        # Check fluorometer table
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

    def test_raw_fluorometer_fields_complete(self):
        """Test that fluorometer raw data has all expected fields."""
        df = generate_fluorometer_data(n_records=5)

        expected_fields = ["datetime_utc", "gain", "voltage", "rho_ppb"]
        assert list(df.columns) == expected_fields
        
        # Verify no null values in critical fields
        assert not df["datetime_utc"].isnull().any()
        assert not df["concentration"].isnull().any()

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
            "datetime_utc", "scan_no", "cond", "temp", "hull_temp",
            "time_elapsed", "nmea_time", "latitude", "longitude"
        ]
        assert list(df.columns) == expected_fields
        
        # Verify no null values in critical fields
        assert not df["datetime_utc"].isnull().any()
        assert not df["temp"].isnull().any()
        assert not df["cond"].isnull().any()
        assert not df["latitude"].isnull().any()
        assert not df["longitude"].isnull().any()

    def test_resampled_data_field_mapping(self, sample_sqlite_db):
        """Test that resampled data has all expected fields with correct mapping."""
        # Generate and write raw data
        fluorometer_df = generate_fluorometer_data(n_records=10)
        ph_df = generate_ph_data(n_records=5)
        tsg_df = generate_tsg_data(n_records=10)
        
        write_to_raw_tables(
            fluorometer_df=fluorometer_df,
            ph_df=ph_df,
            tsg_df=tsg_df,
            sqlite_path=str(sample_sqlite_db)
        )
        
        # Test the correct resampling function with field mapping
        df = resample_raw_sensor_data(str(sample_sqlite_db), resample_interval='2s')

        expected_columns = ['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph', 'temp_c', 'salinity_psu', 'ph_ma']
        assert list(df.columns) == expected_columns
        
        # Verify field mappings are correct
        assert not df["latitude"].isnull().all()  # Should have latitude data mapped to latitude
        assert not df["longitude"].isnull().all()  # Should have longitude data mapped to longitude
        assert not df["rho_ppb"].isnull().all()  # Should have concentration mapped to rhodamine
        assert not df["ph"].isnull().all()  # Should have ph_free mapped to ph
        assert not df["temp_c"].isnull().all()  # Should have temp data
        assert not df["salinity_psu"].isnull().all()  # Should have calculated salinity from conductivity
        assert not df["ph_ma"].isnull().all()  # Should have pH moving average

    def test_broken_resampling_function_detection(self, sample_sqlite_db):
        """Test detection of the broken resampling function that expects wrong column names."""
        # Generate and write raw data
        fluorometer_df = generate_fluorometer_data(n_records=5)
        ph_df = generate_ph_data(n_records=3)
        tsg_df = generate_tsg_data(n_records=5)
        
        write_to_raw_tables(
            fluorometer_df=fluorometer_df,
            ph_df=ph_df,
            tsg_df=tsg_df,
            sqlite_path=str(sample_sqlite_db)
        )
        
        # The broken function should fail due to column name mismatches
        with pytest.raises(Exception):  # Should fail due to missing columns
            load_and_resample_sqlite(str(sample_sqlite_db))

    def test_resampled_csv_output_fields(self, sample_sqlite_db, tmp_path):
        """Test that CSV output from resampled data has all expected fields."""
        # Generate and write raw data
        fluorometer_df = generate_fluorometer_data(n_records=10)
        ph_df = generate_ph_data(n_records=5)
        tsg_df = generate_tsg_data(n_records=10)
        
        write_to_raw_tables(
            fluorometer_df=fluorometer_df,
            ph_df=ph_df,
            tsg_df=tsg_df,
            sqlite_path=str(sample_sqlite_db)
        )
        
        # Get resampled data
        df = resample_raw_sensor_data(str(sample_sqlite_db), resample_interval='2s')
        
        # Write to CSV
        csv_path = tmp_path / "test_resampled.csv"
        file_writers.to_csv(df, str(csv_path))
        
        # Read back and verify fields
        df_read = pd.read_csv(csv_path)
        expected_columns = ['timestamp', 'lat', 'lon', 'rhodamine', 'ph', 'temp', 'salinity', 'ph_ma']
        assert list(df_read.columns) == expected_columns
        
        # Verify data integrity
        assert len(df_read) == len(df)
        assert not df_read["lat"].isnull().all()
        assert not df_read["lon"].isnull().all()
        assert not df_read["rhodamine"].isnull().all()

    def test_resampled_sqlite_table_fields(self, sample_sqlite_db):
        """Test that resampled data written back to SQLite has all expected fields."""
        # Generate and write raw data
        fluorometer_df = generate_fluorometer_data(n_records=10)
        ph_df = generate_ph_data(n_records=5)
        tsg_df = generate_tsg_data(n_records=10)
        
        write_to_raw_tables(
            fluorometer_df=fluorometer_df,
            ph_df=ph_df,
            tsg_df=tsg_df,
            sqlite_path=str(sample_sqlite_db)
        )
        
        # Get resampled data and write back to SQLite
        df = resample_raw_sensor_data(str(sample_sqlite_db), resample_interval='2s')
        write_resampled_to_sqlite(df, str(sample_sqlite_db), output_table='resampled_data')
        
        # Read from resampled_data table and verify fields
        conn = sqlite3.connect(sample_sqlite_db)
        df_from_sqlite = pd.read_sql_query("SELECT * FROM resampled_data", conn)
        conn.close()

        expected_columns = ['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph', 'temp_c', 'salinity_psu', 'ph_ma']
        assert list(df_from_sqlite.columns) == expected_columns
        
        # Verify data was written
        assert len(df_from_sqlite) > 0
        assert not df_from_sqlite["latitude"].isnull().all()
        assert not df_from_sqlite["longitude"].isnull().all()
        assert not df_from_sqlite["rho_ppb"].isnull().all()
        assert not df_from_sqlite["ph"].isnull().all()
        assert not df_from_sqlite["temp_c"].isnull().all()
        assert not df_from_sqlite["salinity_psu"].isnull().all()
        assert not df_from_sqlite["ph_ma"].isnull().all()

class TestDataIntegrity:
    """Test data integrity across the pipeline."""

    def test_end_to_end_field_preservation(self, sample_sqlite_db, tmp_path):
        """Test that all fields are preserved through the complete pipeline."""
        # Generate raw data
        fluorometer_df = generate_fluorometer_data(n_records=20, frequency_hz=1.0)
        ph_df = generate_ph_data(n_records=10, frequency_hz=0.5)
        tsg_df = generate_tsg_data(n_records=20, frequency_hz=1.0)
        
        # Write to raw tables
        write_to_raw_tables(
            fluorometer_df=fluorometer_df,
            ph_df=ph_df,
            tsg_df=tsg_df,
            sqlite_path=str(sample_sqlite_db)
        )
        
        # Check raw data in database
        conn = sqlite3.connect(sample_sqlite_db)
        # Verify rhodamine table
        fluoro_from_db = pd.read_sql_query("SELECT * FROM rhodamine", conn)
        assert len(fluoro_from_db) == 20
        assert "rho_ppb" in fluoro_from_db.columns

        # Verify pH table
        ph_from_db = pd.read_sql_query("SELECT * FROM ph", conn)
        assert len(ph_from_db) == 10
        assert "ph_free" in ph_from_db.columns
        assert "ph_total" in ph_from_db.columns
        
        # Verify TSG table
        tsg_from_db = pd.read_sql_query("SELECT * FROM tsg", conn)
        assert len(tsg_from_db) == 20
        assert "cond" in tsg_from_db.columns
        assert "temp" in tsg_from_db.columns
        assert "latitude" in tsg_from_db.columns
        assert "longitude" in tsg_from_db.columns
        
        conn.close()
        
        # Resample data
        resampled_df = resample_raw_sensor_data(str(sample_sqlite_db), resample_interval='2s')
        
        # Verify resampled data has correct mapped fields
        expected_resampled_columns = ['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph', 'temp_c', 'salinity_psu', 'ph_ma']
        assert list(resampled_df.columns) == expected_resampled_columns
        
        # Write to CSV and verify
        csv_path = tmp_path / "end_to_end_test.csv"
        file_writers.to_csv(resampled_df, str(csv_path))
        
        csv_df = pd.read_csv(csv_path)
        assert list(csv_df.columns) == expected_resampled_columns
        assert len(csv_df) > 0
        
        # Verify value ranges are realistic (filter out NaN values)
        valid_lat = csv_df["lat"].dropna()
        valid_lon = csv_df["longitude"].dropna()
        valid_rhodamine = csv_df["rho_ppb"].dropna()
        valid_ph = csv_df["ph"].dropna()
        valid_temp = csv_df["temp_c"].dropna()
        valid_salinity = csv_df["salinity_psu"].dropna()

        assert valid_lat.between(42.4, 42.6).all()  # Around base latitude
        assert valid_lon.between(-69.6, -69.4).all()  # Around base longitude
        assert valid_rhodamine.ge(0).all()  # Non-negative concentrations
        assert valid_ph.between(7.0, 9.0).all()  # Realistic pH range
        assert valid_temp.gt(0).all()  # Positive temperatures
        assert valid_salinity.gt(0).all()  # Positive salinity

    def test_field_consistency_across_formats(self, sample_sqlite_db, tmp_path):
        """Test that field names and values are consistent across different output formats."""
        # Generate and process data
        fluorometer_df = generate_fluorometer_data(n_records=10)
        ph_df = generate_ph_data(n_records=5)
        tsg_df = generate_tsg_data(n_records=10)
        
        write_to_raw_tables(
            fluorometer_df=fluorometer_df,
            ph_df=ph_df,
            tsg_df=tsg_df,
            sqlite_path=str(sample_sqlite_db)
        )
        
        resampled_df = resample_raw_sensor_data(str(sample_sqlite_db), resample_interval='2s')
        
        # Write to different formats
        csv_path = tmp_path / "consistency_test.csv"
        parquet_path = tmp_path / "consistency_test.parquet"
        
        file_writers.to_csv(resampled_df, str(csv_path))
        file_writers.to_parquet(resampled_df, str(parquet_path))
        
        # Write back to SQLite
        write_resampled_to_sqlite(resampled_df, str(sample_sqlite_db), output_table='resampled_data')
        
        # Read from all formats and compare
        csv_df = pd.read_csv(csv_path)
        parquet_df = pd.read_parquet(parquet_path)
        
        conn = sqlite3.connect(sample_sqlite_db)
        sqlite_df = pd.read_sql_query("SELECT * FROM resampled_data", conn)
        conn.close()
        
        # All should have same columns
        expected_columns = ['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 'ph', 'temp_c', 'salinity_psu', 'ph_ma']
        assert list(csv_df.columns) == expected_columns
        assert list(parquet_df.columns) == expected_columns
        assert list(sqlite_df.columns) == expected_columns
        
        # All should have same number of records
        assert len(csv_df) == len(parquet_df) == len(sqlite_df)


class TestDataValidation:
    """Test data validation and edge cases."""

    def test_zero_frequency_handling(self):
        """Test handling of zero frequency."""
        df = generate_fluorometer_data(n_records=3, frequency_hz=0.0)
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
        df = generate_fluorometer_data(n_records=10, frequency_hz=10.0)
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
