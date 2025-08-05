#!/usr/bin/env python3
"""
Test script for the PersistentResampler to verify it produces the correct number of records
without duplicates.
"""

import sqlite3
import pandas as pd
import tempfile
import os
from locness_datamanager.resampler import PersistentResampler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def create_test_database():
    """Create a test database with sample data."""
    # Create temporary database
    fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    conn = sqlite3.connect(db_path)
    
    # Create test data with irregular timestamps that could cause resampling issues
    test_data = []
    base_time = pd.Timestamp('2025-08-05 12:00:00')
    
    # Add some regular data
    for i in range(100):
        timestamp = base_time + pd.Timedelta(seconds=i)
        test_data.append({
            'datetime_utc': int(timestamp.timestamp()),
            'temp': 20.0 + 0.1 * i,
            'salinity': 35.0 + 0.01 * i
        })
    
    # Add some irregular data that might cause resampling duplicates
    irregular_times = [
        base_time + pd.Timedelta(seconds=50.3),
        base_time + pd.Timedelta(seconds=50.7),
        base_time + pd.Timedelta(seconds=51.1),
        base_time + pd.Timedelta(seconds=51.9),
    ]
    
    for i, timestamp in enumerate(irregular_times):
        test_data.append({
            'datetime_utc': int(timestamp.timestamp()),
            'temp': 22.0 + 0.1 * i,
            'salinity': 36.0 + 0.01 * i
        })
    
    # Create tables
    tsg_df = pd.DataFrame(test_data)
    tsg_df.to_sql('tsg', conn, if_exists='replace', index=False)
    
    # Create pH data (every 2 seconds)
    ph_data = []
    for i in range(0, 100, 2):
        timestamp = base_time + pd.Timedelta(seconds=i)
        ph_data.append({
            'datetime_utc': int(timestamp.timestamp()),
            'vrse': 0.4 + 0.001 * i,
            'ph_total': 8.0 + 0.001 * i
        })
    
    ph_df = pd.DataFrame(ph_data)
    ph_df.to_sql('ph', conn, if_exists='replace', index=False)
    
    # Create rhodamine data
    rhodamine_data = []
    for i in range(100):
        timestamp = base_time + pd.Timedelta(seconds=i)
        rhodamine_data.append({
            'datetime_utc': int(timestamp.timestamp()),
            'rho_ppb': 10.0 + 0.1 * i
        })
    
    rhodamine_df = pd.DataFrame(rhodamine_data)
    rhodamine_df.to_sql('rhodamine', conn, if_exists='replace', index=False)
    
    # Create GPS data
    gps_data = []
    for i in range(100):
        timestamp = base_time + pd.Timedelta(seconds=i)
        gps_data.append({
            'datetime_utc': int(timestamp.timestamp()),
            'latitude': 42.0 + 0.001 * i,
            'longitude': -70.0 + 0.001 * i
        })
    
    gps_df = pd.DataFrame(gps_data)
    gps_df.to_sql('gps', conn, if_exists='replace', index=False)
    
    # Create underway_summary table for testing writes
    conn.execute('''
        CREATE TABLE underway_summary (
            datetime_utc INTEGER PRIMARY KEY,
            latitude REAL,
            longitude REAL,
            rho_ppb REAL,
            ph_total REAL,
            vrse REAL,
            temp REAL,
            salinity REAL,
            ph_corrected REAL,
            ph_total_ma REAL,
            ph_corrected_ma REAL
        )
    ''')
    
    conn.close()
    return db_path


def test_resampler_no_duplicates():
    """Test that the resampler doesn't produce duplicate timestamps."""
    print("Testing PersistentResampler for duplicate prevention...")
    
    # Create test database
    db_path = create_test_database()
    
    try:
        # Initialize resampler with 2-second interval (matches config)
        resampler = PersistentResampler(
            sqlite_path=db_path,
            resample_interval='2s',
            ph_ma_window=120,
            ph_freq=0.5,
            ph_k0=-1.39469,
            ph_k2=-0.00107
        )
        
        # Process data multiple times to simulate polling
        all_results = []
        
        for iteration in range(3):
            print(f"\nIteration {iteration + 1}:")
            result = resampler.process_new_data()
            
            if not result.empty:
                print(f"  Generated {len(result)} records")
                print(f"  Time range: {result['datetime_utc'].min()} to {result['datetime_utc'].max()}")
                
                # Check for duplicates within this result
                duplicates = result[result['datetime_utc'].duplicated()]
                if not duplicates.empty:
                    print(f"  ❌ Found {len(duplicates)} duplicate timestamps within result!")
                    print(f"  Duplicate timestamps: {duplicates['datetime_utc'].tolist()}")
                    return False
                else:
                    print("  ✅ No duplicates within result")
                
                all_results.append(result)
            else:
                print("  No new data to process")
        
        # Check for duplicates across all iterations
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            total_duplicates = combined[combined['datetime_utc'].duplicated()]
            
            if not total_duplicates.empty:
                print(f"\n❌ Found {len(total_duplicates)} duplicate timestamps across iterations!")
                print(f"Duplicate timestamps: {total_duplicates['datetime_utc'].tolist()}")
                return False
            else:
                print(f"\n✅ No duplicates across {len(all_results)} iterations")
        
        # Test database writing
        print("\nTesting database writes...")
        from locness_datamanager.resample import write_resampled_to_sqlite
        
        if all_results:
            for i, result in enumerate(all_results):
                if not result.empty:
                    print(f"  Writing iteration {i+1} result ({len(result)} records)...")
                    try:
                        write_resampled_to_sqlite(result, db_path, 'underway_summary')
                        print(f"  ✅ Successfully wrote iteration {i+1}")
                    except Exception as e:
                        print(f"  ❌ Failed to write iteration {i+1}: {e}")
                        return False
        
        # Verify final database state
        conn = sqlite3.connect(db_path)
        final_count = pd.read_sql_query("SELECT COUNT(*) as count FROM underway_summary", conn).iloc[0]['count']
        unique_count = pd.read_sql_query("SELECT COUNT(DISTINCT datetime_utc) as count FROM underway_summary", conn).iloc[0]['count']
        conn.close()
        
        print("\nFinal database state:")
        print(f"  Total records: {final_count}")
        print(f"  Unique timestamps: {unique_count}")
        
        if final_count == unique_count:
            print("  ✅ All records have unique timestamps")
            return True
        else:
            print(f"  ❌ {final_count - unique_count} duplicate timestamps in database!")
            return False
            
    finally:
        # Clean up
        os.unlink(db_path)


def test_resampling_intervals():
    """Test that resampling produces the expected number of records."""
    print("\nTesting resampling interval correctness...")
    
    db_path = create_test_database()
    
    try:
        # Test different intervals
        intervals = ['1s', '2s', '5s', '10s']
        
        for interval in intervals:
            print(f"\nTesting interval: {interval}")
            
            resampler = PersistentResampler(
                sqlite_path=db_path,
                resample_interval=interval,
                ph_ma_window=60,
                ph_freq=0.5
            )
            
            # Reset state for clean test
            resampler.reset_state()
            
            result = resampler.process_new_data()
            
            if not result.empty:
                # Calculate expected number of records
                time_range = result['datetime_utc'].max() - result['datetime_utc'].min()
                interval_seconds = pd.Timedelta(interval).total_seconds()
                expected_records = int(time_range.total_seconds() / interval_seconds) + 1
                
                print(f"  Generated: {len(result)} records")
                print(f"  Expected: ~{expected_records} records")
                print(f"  Time range: {time_range.total_seconds():.1f} seconds")
                print(f"  Interval: {interval_seconds} seconds")
                
                # Check that we're in the right ballpark (within 20% tolerance)
                if abs(len(result) - expected_records) / max(expected_records, 1) < 0.2:
                    print("  ✅ Record count is reasonable")
                else:
                    print("  ⚠️  Record count differs significantly from expected")
            else:
                print("  No data generated")
                
    finally:
        os.unlink(db_path)


def main():
    """Run all tests."""
    print("=" * 60)
    print("PersistentResampler Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test 1: No duplicates
    if not test_resampler_no_duplicates():
        success = False
    
    # Test 2: Correct intervals
    test_resampling_intervals()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All critical tests passed!")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
