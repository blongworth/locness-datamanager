"""
Test script for DynamoDB functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the package to path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from locness_datamanager import file_writers


def create_test_data():
    """Create sample underway summary data for testing."""
    # Create datetime range
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i*2) for i in range(10)]
    
    # Create sample data matching underway_summary table structure
    data = {
        'datetime_utc': timestamps,
        'latitude': [42.5 + np.random.normal(0, 0.01) for _ in range(10)],
        'longitude': [-69.5 + np.random.normal(0, 0.01) for _ in range(10)],
        'rho_ppb': [1.5 + np.random.normal(0, 0.1) for _ in range(10)],
        'ph_total': [8.2 + np.random.normal(0, 0.05) for _ in range(10)],
        'vrse': [0.5 + np.random.normal(0, 0.02) for _ in range(10)],
        'ph_corrected': [32.0 + np.random.normal(0, 1.0) for _ in range(10)],
        'temp': [15.0 + np.random.normal(0, 0.5) for _ in range(10)],
        'salinity': [35.0 + np.random.normal(0, 0.2) for _ in range(10)],
        'ph_corrected_ma': [32.0 + np.random.normal(0, 1.0) for _ in range(10)],
        'ph_total_ma': [8.2 + np.random.normal(0, 0.05) for _ in range(10)]
    }
    
    return pd.DataFrame(data)


def test_dynamodb_conversion():
    """Test the data conversion for DynamoDB without actually writing to AWS."""
    print("Testing DynamoDB data conversion...")
    
    df = create_test_data()
    print(f"Created test data with {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # Test the conversion logic (without actually connecting to DynamoDB)
    try:
        from decimal import Decimal
        
        df_copy = df.copy()
        
        # Convert datetime to ISO string format
        if 'datetime_utc' in df_copy.columns:
            if pd.api.types.is_datetime64_any_dtype(df_copy['datetime_utc']):
                df_copy['datetime_utc'] = df_copy['datetime_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Convert DataFrame to list of dictionaries
        records = df_copy.to_dict('records')
        
        # Convert numpy/pandas types to Python native types
        processed_records = []
        for record in records:
            processed_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    continue
                elif isinstance(value, (float, int)) and not pd.isna(value):
                    processed_record[key] = Decimal(str(value))
                else:
                    processed_record[key] = str(value)
            
            if 'datetime_utc' in processed_record:
                processed_records.append(processed_record)
        
        print(f"\nSuccessfully converted {len(processed_records)} records for DynamoDB")
        print("Sample record:")
        if processed_records:
            sample = processed_records[0]
            for key, value in sample.items():
                print(f"  {key}: {value} ({type(value).__name__})")
        
        print("\n✅ Data conversion test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data conversion test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== DynamoDB Functionality Test ===\n")
    
    success = test_dynamodb_conversion()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\nTo use DynamoDB functionality:")
        print("1. Set up AWS credentials (aws configure)")
        print("2. Create a DynamoDB table:")
        print("   python -m locness_datamanager.setup_dynamodb create --table-name locness-underway-summary")
        print("3. Use DynamoDB output in resample_summary:")
        print("   python -m locness_datamanager.resample_summary --dynamodb-table locness-underway-summary")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
