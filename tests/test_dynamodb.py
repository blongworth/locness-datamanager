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
            'datetime_utc': pd.date_range('2025-08-06 12:00:00', periods=5, freq='60s'),
            'latitude': [42.5, 42.51, 42.52, 42.53, 42.54],
            'longitude': [-69.5, -69.51, -69.52, -69.53, -69.54],
            'rho_ppb': [1.5, 1.6, 1.7, 1.8, 1.9],
            'ph_total': [8.1, 8.2, 8.3, 8.4, 8.5],
            'vrse': [0.5, 0.6, 0.7, 0.8, 0.9],
            'ph_corrected': [32.1, 32.2, 32.3, 32.4, 32.5],
            'temp': [15.0, 15.1, 15.2, 15.3, 15.4],
            'salinity': [35.0, 35.1, 35.2, 35.3, 35.4],
            'ph_corrected_ma': [32.1, 32.15, 32.2, 32.25, 32.3],
            'ph_total_ma': [8.1, 8.15, 8.2, 8.25, 8.3]
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
        mock_table.batch_writer.return_value.__enter__.return_value = mock_batch_writer
        mock_table.batch_writer.return_value.__exit__.return_value = None
        
        # Create test data
        df = self.create_sample_dataframe()
        
        # Call the function
        file_writers.to_dynamodb(df, 'test-table', 'us-east-1', batch_size=25)
        
        # Verify boto3 was called correctly
        mock_boto3_resource.assert_called_once_with('dynamodb', region_name='us-east-1')
        mock_dynamodb.Table.assert_called_once_with('test-table')
        
        # Verify batch writer was used
        mock_table.batch_writer.assert_called_once()
        
        # Verify records were written (should be 5 put_item calls)
        assert mock_batch_writer.put_item.call_count == 5
    
    @patch('locness_datamanager.file_writers.boto3')
    @patch('locness_datamanager.file_writers.logging')
    def test_to_dynamodb_empty_dataframe(self, mock_logging, mock_boto3):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        file_writers.to_dynamodb(df, 'test-table')
        
        # Should log info message and return early
        mock_logging.info.assert_called_with("No data to write to DynamoDB")
        # Should not attempt to connect to DynamoDB
        mock_boto3.resource.assert_not_called()
    
    @patch('locness_datamanager.file_writers.boto3')
    @patch('locness_datamanager.file_writers.logging')
    def test_to_dynamodb_connection_error(self, mock_logging, mock_boto3):
        """Test handling of DynamoDB connection errors."""
        # Setup mock to raise exception
        mock_boto3.resource.side_effect = Exception("Connection failed")
        
        df = self.create_sample_dataframe()
        
        file_writers.to_dynamodb(df, 'test-table')
        
        # Should log error and return
        mock_logging.error.assert_called()
        error_call_args = mock_logging.error.call_args[0][0]
        assert "Failed to connect to DynamoDB table test-table" in error_call_args
    
    @patch('locness_datamanager.file_writers.boto3')
    @patch('locness_datamanager.file_writers.logging')
    def test_to_dynamodb_data_conversion(self, mock_logging, mock_boto3):
        """Test data type conversion for DynamoDB."""
        # Setup mocks
        mock_dynamodb = Mock()
        mock_table = Mock()
        mock_batch_writer = Mock()
        
        mock_boto3.resource.return_value = mock_dynamodb
        mock_dynamodb.Table.return_value = mock_table
        mock_table.batch_writer.return_value.__enter__.return_value = mock_batch_writer
        mock_table.batch_writer.return_value.__exit__.return_value = None
        
        # Create test data with various data types and NaN values
        df = pd.DataFrame({
            'datetime_utc': pd.date_range('2025-08-06 12:00:00', periods=2, freq='60s'),
            'latitude': [42.5, np.nan],  # Include NaN value
            'longitude': [-69.5, -69.51],
            'rho_ppb': [1.5, 1.6],
            'integer_col': [10, 20]
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
        
        # Verify NaN values are excluded (first record should not have latitude due to NaN)
        assert 'latitude' not in written_records[1]  # Second record has NaN latitude
        assert 'latitude' in written_records[0]      # First record has valid latitude
    
    @patch('locness_datamanager.file_writers.boto3')
    @patch('locness_datamanager.file_writers.logging')
    def test_to_dynamodb_batch_write_error(self, mock_logging, mock_boto3):
        """Test handling of batch write errors."""
        # Setup mocks
        mock_dynamodb = Mock()
        mock_table = Mock()
        mock_batch_writer = Mock()
        
        mock_boto3.resource.return_value = mock_dynamodb
        mock_dynamodb.Table.return_value = mock_table
        mock_table.batch_writer.return_value.__enter__.return_value = mock_batch_writer
        mock_table.batch_writer.return_value.__exit__.return_value = None
        
        # Make batch writer raise an exception
        mock_batch_writer.put_item.side_effect = Exception("Batch write failed")
        
        df = self.create_sample_dataframe()
        
        file_writers.to_dynamodb(df, 'test-table')
        
        # Should log error for failed batch
        mock_logging.error.assert_called()
        error_calls = [call for call in mock_logging.error.call_args_list 
                      if "Failed to write batch to DynamoDB" in str(call)]
        assert len(error_calls) > 0
        
        # Should also log warning about failed writes
        mock_logging.warning.assert_called()
    
    def test_to_dynamodb_unix_timestamp_conversion(self):
        """Test conversion of Unix timestamps to ISO format."""
        with patch('locness_datamanager.file_writers.boto3') as mock_boto3:
            
            # Setup mocks
            mock_dynamodb = Mock()
            mock_table = Mock()
            mock_batch_writer = Mock()
            
            mock_boto3.resource.return_value = mock_dynamodb
            mock_dynamodb.Table.return_value = mock_table
            mock_table.batch_writer.return_value.__enter__.return_value = mock_batch_writer
            mock_table.batch_writer.return_value.__exit__.return_value = None
            
            # Create test data with Unix timestamps (integers)
            df = pd.DataFrame({
                'datetime_utc': [1722931200, 1722931260],  # Unix timestamps
                'latitude': [42.5, 42.51],
                'longitude': [-69.5, -69.51]
            })
            
            file_writers.to_dynamodb(df, 'test-table')
            
            # Verify records were written
            assert mock_batch_writer.put_item.call_count == 2
            
            # Get the written record
            written_record = mock_batch_writer.put_item.call_args_list[0][1]['Item']
            
            # Verify Unix timestamp was converted to ISO string
            assert isinstance(written_record['datetime_utc'], str)
            assert 'T' in written_record['datetime_utc']  # ISO format indicator
            assert written_record['datetime_utc'].endswith('Z')  # UTC indicator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
