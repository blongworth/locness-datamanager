import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sqlite3

def to_csv(df, filename, mode="w", header=True, index=False):
    """
    Write a DataFrame to a CSV file.
    Args:
        df: pandas.DataFrame
        filename: Output CSV file path
        mode: File mode ('w' for write, 'a' for append)
        header: Write header row (default True)
        index: Write row index (default False)
    """
    df.to_csv(filename, mode=mode, header=header, index=index)

def to_parquet(df, filename, append=False, partition_hours=None):
    """
    Write a DataFrame to a Parquet file, optionally with time-based partitioning.
    Args:
        df: pandas.DataFrame
        filename: Output Parquet file path
        append: If True, append to existing file (default False)
        partition_hours: Hours per partition (default None). If None, no partitioning is done
    """
    df = df.copy()
    table = pa.Table.from_pandas(df)
    
    if partition_hours is not None:
        # Add partition column based on timestamp
        df['partition'] = df['datetime_utc'].dt.floor(f'{partition_hours}h')
        table = pa.Table.from_pandas(df)
        
        if not append or not os.path.exists(filename):
            pq.write_to_dataset(
                table,
                root_path=filename,
                partition_cols=['partition'],
            )
        else:
            # Group data by partition
            for partition_value in df['partition'].unique():
                partition_mask = df['partition'] == partition_value
                partition_table = pa.Table.from_pandas(df[partition_mask])
                partition_path = os.path.join(filename, f"partition={partition_value}")
                
                if os.path.exists(partition_path):
                    # Append to existing partition
                    existing_partition = pq.read_table(partition_path)
                    combined_partition = pa.concat_tables([existing_partition, partition_table])
                    pq.write_table(combined_partition, partition_path)
                else:
                    # Create new partition
                    pq.write_to_dataset(
                        partition_table,
                        root_path=filename,
                        partition_cols=['partition'],
                    )
    else:
        if not append or not os.path.exists(filename):
            pq.write_table(table, filename)
        else:
            existing_table = pq.read_table(filename)
            combined_table = pa.concat_tables([existing_table, table])
            pq.write_table(combined_table, filename)

def to_sqlite(df, db_path, table_name):
    """
    Write a DataFrame to a SQLite3 table.
    Args:
        df: pandas.DataFrame
        db_path: Path to SQLite database file
        table_name: Name of the table to write to
    """
    df_copy = df.copy()
    
    # Convert datetime timestamps to Unix timestamps (integers) for SQLite compatibility
    if 'datetime_utc' in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy['datetime_utc']):
        df_copy['datetime_utc'] = df_copy['datetime_utc'].astype('int64') // 10**9

    conn = sqlite3.connect(db_path)
    df_copy.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()


def to_dynamodb(df, table_name, region_name='us-east-1', batch_size=25):
    """
    Write a DataFrame to a DynamoDB table.
    Args:
        df: pandas.DataFrame with underway summary data
        table_name: Name of the DynamoDB table to write to
        region_name: AWS region for the DynamoDB table
        batch_size: Number of items to write per batch (max 25 for DynamoDB)
    """
    import boto3
    from decimal import Decimal
    import logging
    
    if df.empty:
        logging.info("No data to write to DynamoDB")
        return
    
    # Initialize DynamoDB client
    try:
        dynamodb = boto3.resource('dynamodb', region_name=region_name)
        table = dynamodb.Table(table_name)
    except Exception as e:
        logging.error(f"Failed to connect to DynamoDB table {table_name}: {e}")
        return
    
    df_copy = df.copy()
    
    # Convert datetime to ISO string format for DynamoDB
    if 'datetime_utc' in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy['datetime_utc']):
            df_copy['datetime_utc'] = df_copy['datetime_utc'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        elif df_copy['datetime_utc'].dtype in ['int64', 'int32']:
            # Convert Unix timestamp to ISO string
            df_copy['datetime_utc'] = pd.to_datetime(df_copy['datetime_utc'], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Convert DataFrame to list of dictionaries
    records = df_copy.to_dict('records')
    
    # Convert numpy/pandas types to Python native types and handle NaN values
    processed_records = []
    for record in records:
        processed_record = {}
        for key, value in record.items():
            if pd.isna(value):
                # Skip NaN values (DynamoDB doesn't support null numbers)
                continue
            elif isinstance(value, (float, int)) and not pd.isna(value):
                # Convert to Decimal for DynamoDB numeric types
                processed_record[key] = Decimal(str(value))
            else:
                processed_record[key] = str(value)
        
        # Ensure we have a partition key (datetime_utc is our primary key)
        if 'datetime_utc' in processed_record:
            processed_records.append(processed_record)
    
    if not processed_records:
        logging.warning("No valid records to write to DynamoDB after processing")
        return
    
    # Write records in batches
    total_written = 0
    failed_writes = 0
    
    for i in range(0, len(processed_records), batch_size):
        batch = processed_records[i:i + batch_size]
        
        try:
            # Prepare batch write request
            with table.batch_writer() as batch_writer:
                for record in batch:
                    batch_writer.put_item(Item=record)
            
            total_written += len(batch)
            logging.info(f"Successfully wrote batch of {len(batch)} records to DynamoDB table {table_name}")
            
        except Exception as e:
            failed_writes += len(batch)
            logging.error(f"Failed to write batch to DynamoDB table {table_name}: {e}")
    
    logging.info(f"DynamoDB write complete: {total_written} successful, {failed_writes} failed")
    if failed_writes > 0:
        logging.warning(f"{failed_writes} records failed to write to DynamoDB")
