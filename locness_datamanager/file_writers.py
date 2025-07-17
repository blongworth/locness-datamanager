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
