import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from decimal import Decimal
import os
import sys

# Add the locness_datamanager to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from locness_datamanager import file_writers


class TestDynamoDBWriter:
    """Test the DynamoDB writing functionality."""
    
    def create_sample_dataframe(self):
        """Create a sample DataFrame with underway summary data."""
        return pd.DataFrame({
            'datetime_utc': pd.date_range('2025-08-06 12:00:00', periods=3, freq='60s'),
            'latitude': [42.5, 42.51, 42.52],
            'longitude': [-69.5, -69.51, -69.52],
            'rho_ppb': [1.5, 1.6, 1.7],
            'ph_total': [8.1, 8.2, 8.3]
        })
    
    @patch('boto3.resource')
    def test_to_dynamodb_success(self, mock_boto3_resource):
        """Test successful writing to DynamoDB."""
        # Setup mocks
        mock_dynamodb = Mock()
        mock_table = Mock()
        mock_batch_writer = Mock()
        
        mock_boto3_resource.return_value = mock_dynamodb
        mock_dynamodb.Table.return_value = mock_table
        mock_table.batch_writer.return_value = mock_batch_writer
        mock_batch_writer.__enter__ = Mock(return_value=mock_batch_writer)
        mock_batch_writer.__exit__ = Mock(return_value=None)
        
        # Create test data
        df = self.create_sample_dataframe()
        
        # Call the function
        file_writers.to_dynamodb(df, 'test-table', 'us-east-1', batch_size=25)
        
        # Verify boto3 was called correctly
        mock_boto3_resource.assert_called_once_with('dynamodb', region_name='us-east-1')
        mock_dynamodb.Table.assert_called_once_with('test-table')
        
        # Verify batch writer was used
        mock_table.batch_writer.assert_called_once()
        
        # Verify records were written (should be 3 put_item calls)
        assert mock_batch_writer.put_item.call_count == 3
    
    def test_to_dynamodb_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        # Should return early without attempting to connect to DynamoDB
        # This test should pass without mocking since it returns early
        result = file_writers.to_dynamodb(df, 'test-table')
        assert result is None  # Function should return None/early
    
    @patch('boto3.resource')
    def test_to_dynamodb_connection_error(self, mock_boto3_resource):
        """Test handling of DynamoDB connection errors."""
        # Setup mock to raise exception
        mock_boto3_resource.side_effect = Exception("Connection failed")
        
        df = self.create_sample_dataframe()
        
        # Should not raise exception, just log error and return
        result = file_writers.to_dynamodb(df, 'test-table')
        
        # Verify connection was attempted
        mock_boto3_resource.assert_called_once_with('dynamodb', region_name='us-east-1')
        assert result is None  # Should return early on error
    
    @patch('boto3.resource')
    def test_to_dynamodb_data_conversion(self, mock_boto3_resource):
        """Test data type conversion for DynamoDB."""
        # Setup mocks
        mock_dynamodb = Mock()
        mock_table = Mock()
        mock_batch_writer = Mock()
        
        mock_boto3_resource.return_value = mock_dynamodb
        mock_dynamodb.Table.return_value = mock_table
        mock_table.batch_writer.return_value = mock_batch_writer
        mock_batch_writer.__enter__ = Mock(return_value=mock_batch_writer)
        mock_batch_writer.__exit__ = Mock(return_value=None)
        
        # Create test data with various data types and NaN values
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2025-08-06 12:00:00', periods=2, freq='60s'),
            'latitude': [42.5, np.nan],  # Include NaN value
            'longitude': [-69.5, -69.51],
            'rho_ppb': [1.5, 1.6]
        })
        
        # Call the function
        file_writers.to_dynamodb(df, 'test-table')
        
        # Verify records were processed
        assert mock_batch_writer.put_item.call_count == 2
        
        # Get the actual records that were written
        written_records = []
        for call in mock_batch_writer.put_item.call_args_list:
            written_records.append(call[1]['Item'])  # Get the Item argument
        
        # Verify datetime conversion to ISO string format
        assert all('datetime_utc' in record for record in written_records)
        assert all(isinstance(record['datetime_utc'], str) for record in written_records)
        assert '2025-08-06T12:00:00Z' in written_records[0]['datetime_utc']
        
        # Verify numeric values are converted to Decimal
        assert isinstance(written_records[0]['longitude'], Decimal)
        assert isinstance(written_records[0]['rho_ppb'], Decimal)
        
        # Verify NaN values are excluded (second record should not have latitude due to NaN)
        assert 'latitude' not in written_records[1]  # Second record has NaN latitude
        assert 'latitude' in written_records[0]      # First record has valid latitude


def test_dynamodb_data_conversion_standalone():
    """Test the data conversion logic without DynamoDB connection."""
    # Test data conversion manually
    df = pd.DataFrame({
        'datetime_utc': pd.date_range('2025-08-06 12:00:00', periods=2, freq='60s'),
        'latitude': [42.5, np.nan],
        'longitude': [-69.5, -69.51],
        'rho_ppb': [1.5, 1.6]
    })
    
    df_copy = df.copy()
    
    # Convert datetime to ISO string format (mimicking the function logic)
    if 'datetime_utc' in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy['datetime_utc']):
            df_copy['datetime_utc'] = df_copy['datetime_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Convert DataFrame to list of dictionaries
    records = df_copy.to_dict('records')
    
    # Process records (mimicking the function logic)
    processed_records = []
    for record in records:
        processed_record = {}
        for key, value in record.items():
            if pd.isna(value):
                continue  # Skip NaN values
            elif isinstance(value, (float, int)) and not pd.isna(value):
                processed_record[key] = Decimal(str(value))
            else:
                processed_record[key] = str(value)
        
        if 'datetime_utc' in processed_record:
            processed_records.append(processed_record)
    
    # Verify conversion worked
    assert len(processed_records) == 2
    
    # Check first record (has valid latitude)
    assert 'latitude' in processed_records[0]
    assert isinstance(processed_records[0]['latitude'], Decimal)
    assert isinstance(processed_records[0]['datetime_utc'], str)
    assert '2025-08-06T12:00:00Z' == processed_records[0]['datetime_utc']
    
    # Check second record (NaN latitude should be excluded)
    assert 'latitude' not in processed_records[1]
    assert isinstance(processed_records[1]['longitude'], Decimal)
    assert isinstance(processed_records[1]['datetime_utc'], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
