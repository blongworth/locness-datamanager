#!/usr/bin/env python3
"""
Test script to validate database integrity and resampling calculations.

This script checks:
1. Raw table completeness (no gaps in datetime_utc)
2. Expected sampling rates for each table
3. Summary table completeness  
4. pH moving average calculations
"""

import sqlite3
import pandas as pd
import sys
from pathlib import Path
from locness_datamanager.config import get_config


def check_table_gaps(conn, table_name, expected_interval_seconds):
    """
    Check for gaps in datetime_utc for a given table.
    
    Args:
        conn: SQLite connection
        table_name: Name of the table to check
        expected_interval_seconds: Expected interval between records in seconds
        
    Returns:
        dict with gap information
    """
    print(f"\nChecking {table_name} table...")
    
    try:
        query = f"SELECT datetime_utc FROM {table_name} ORDER BY datetime_utc"
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return {"table": table_name, "total_records": 0, "gaps": [], "status": "EMPTY"}
            
        # Convert to datetime if stored as unix timestamp
        if df['datetime_utc'].dtype in ['int64', 'int32']:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='s')
        else:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            
        total_records = len(df)
        
        # Calculate time differences
        df['time_diff'] = df['datetime_utc'].diff()
        
        # Find gaps (differences significantly larger than expected)
        tolerance = expected_interval_seconds * 1.5  # 50% tolerance
        gaps = df[df['time_diff'] > pd.Timedelta(seconds=tolerance)]
        
        gap_info = []
        for idx, row in gaps.iterrows():
            gap_info.append({
                'timestamp': row['datetime_utc'],
                'gap_seconds': row['time_diff'].total_seconds(),
                'expected_seconds': expected_interval_seconds
            })
            
        status = "OK" if len(gap_info) == 0 else "GAPS_FOUND"
        
        print(f"  Total records: {total_records}")
        print(f"  Expected interval: {expected_interval_seconds}s")
        print(f"  Gaps found: {len(gap_info)}")
        
        if gap_info:
            print("  Gap details:")
            for gap in gap_info[:5]:  # Show first 5 gaps
                print(f"    At {gap['timestamp']}: {gap['gap_seconds']:.1f}s gap (expected {gap['expected_seconds']}s)")
            if len(gap_info) > 5:
                print(f"    ... and {len(gap_info) - 5} more gaps")
                
        return {
            "table": table_name, 
            "total_records": total_records, 
            "gaps": gap_info, 
            "status": status
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"table": table_name, "total_records": 0, "gaps": [], "status": "ERROR", "error": str(e)}


def check_summary_table(conn, expected_interval_seconds=10):
    """Check the underway_summary table for completeness."""
    return check_table_gaps(conn, 'underway_summary', expected_interval_seconds)


def calculate_ph_moving_average(conn, window_seconds=120, freq_hz=0.5):
    """
    Calculate 2-minute running average on ph_total and compare with ph_total_ma.
    
    Args:
        conn: SQLite connection
        window_seconds: Moving average window in seconds
        freq_hz: Expected frequency in Hz
        
    Returns:
        dict with comparison results
    """
    print("\nChecking pH moving average calculations...")
    
    try:
        # Get pH data from summary table
        query = """
        SELECT datetime_utc, ph_total, ph_total_ma 
        FROM underway_summary 
        WHERE ph_total IS NOT NULL 
        ORDER BY datetime_utc
        """
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return {"status": "NO_DATA", "message": "No pH data found in summary table"}
            
        # Convert datetime if needed
        if df['datetime_utc'].dtype in ['int64', 'int32']:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='s')
        else:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            
        # Calculate our own moving average
        window_size = max(1, int(window_seconds * freq_hz))
        df['ph_total_ma_calculated'] = df['ph_total'].rolling(
            window=window_size, 
            min_periods=1
        ).mean()
        
        # Compare with existing ph_total_ma
        df['ma_diff'] = df['ph_total_ma'] - df['ph_total_ma_calculated']
        
        # Filter out NaN differences
        valid_comparisons = df.dropna(subset=['ma_diff'])
        
        if valid_comparisons.empty:
            return {"status": "NO_COMPARISONS", "message": "No valid comparisons possible"}
            
        max_diff = valid_comparisons['ma_diff'].abs().max()
        mean_diff = valid_comparisons['ma_diff'].abs().mean()
        
        # Check if differences are negligible (within floating point precision)
        tolerance = 1e-10
        matches = (valid_comparisons['ma_diff'].abs() < tolerance).all()
        
        print(f"  Total pH records: {len(df)}")
        print(f"  Valid comparisons: {len(valid_comparisons)}")
        print(f"  Window size: {window_size} samples ({window_seconds}s at {freq_hz}Hz)")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        print(f"  Moving averages match: {'YES' if matches else 'NO'}")
        
        # Show some examples if differences are significant
        if not matches and max_diff > tolerance:
            print("  Examples of differences:")
            large_diffs = valid_comparisons[valid_comparisons['ma_diff'].abs() > tolerance].head(3)
            for idx, row in large_diffs.iterrows():
                print(f"    {row['datetime_utc']}: stored={row['ph_total_ma']:.6f}, calculated={row['ph_total_ma_calculated']:.6f}, diff={row['ma_diff']:.2e}")
        
        return {
            "status": "OK" if matches else "MISMATCH",
            "total_records": len(df),
            "valid_comparisons": len(valid_comparisons),
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "matches": matches
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"status": "ERROR", "error": str(e)}


def main():
    """Main validation function."""
    print("Database Validation Script")
    print("=" * 50)
    
    # Get database path from config
    config = get_config()
    db_path = config.get('db_path')
    
    if not db_path:
        print("ERROR: No database path specified in config")
        sys.exit(1)
        
    if not Path(db_path).exists():
        print(f"ERROR: Database file not found: {db_path}")
        sys.exit(1)
        
    print(f"Checking database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Define expected sampling rates
        table_intervals = {
            'tsg': 1,        # 1 second
            'rhodamine': 1,  # 1 second  
            'gps': 1,        # 1 second
            'ph': 2          # 2 seconds (0.5 Hz)
        }
        
        results = {}
        
        # Check each raw table
        for table, interval in table_intervals.items():
            results[table] = check_table_gaps(conn, table, interval)
            
        # Check summary table (should be every 10 seconds based on config)
        summary_interval = config.get('db_res_int', '10s')
        if summary_interval.endswith('s'):
            summary_seconds = int(summary_interval[:-1])
        else:
            summary_seconds = 10  # default
            
        results['underway_summary'] = check_summary_table(conn, summary_seconds)
        
        # Check pH moving average
        ph_window = config.get('ph_ma_window', 120)
        ph_freq = config.get('ph_freq', 0.5)
        results['ph_moving_average'] = calculate_ph_moving_average(conn, ph_window, ph_freq)
        
        conn.close()
        
        # Summary report
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        all_good = True
        
        for table in table_intervals.keys():
            result = results[table]
            status = result['status']
            if status == 'OK':
                print(f"✓ {table}: OK ({result['total_records']} records)")
            elif status == 'GAPS_FOUND':
                print(f"⚠ {table}: {len(result['gaps'])} gaps found ({result['total_records']} records)")
                all_good = False
            elif status == 'EMPTY':
                print(f"⚠ {table}: EMPTY")
                all_good = False
            else:
                print(f"✗ {table}: ERROR")
                all_good = False
                
        # Summary table
        summary_result = results['underway_summary']
        if summary_result['status'] == 'OK':
            print(f"✓ underway_summary: OK ({summary_result['total_records']} records)")
        else:
            print(f"⚠ underway_summary: {summary_result['status']} ({summary_result['total_records']} records)")
            all_good = False
            
        # pH moving average
        ph_result = results['ph_moving_average']
        if ph_result['status'] == 'OK':
            print(f"✓ pH moving average: OK ({ph_result['valid_comparisons']} comparisons)")
        else:
            print(f"⚠ pH moving average: {ph_result['status']}")
            all_good = False
            
        if all_good:
            print("\n🎉 All validations passed!")
            sys.exit(0)
        else:
            print("\n⚠️  Some issues found. Check details above.")
            sys.exit(1)
            
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
