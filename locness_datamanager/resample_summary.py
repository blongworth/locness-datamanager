"""
resample_summary.py

This module provides functions for resampling the underway_summary table at specified intervals
and writing the results to Parquet and/or CSV files.

The module reads from the underway_summary table, resamples using mean aggregation,
and appends new data to output files at regular intervals.
"""

import sqlite3
import pandas as pd
import time
import logging
import argparse
from typing import Optional
from locness_datamanager.config import get_config
from locness_datamanager import file_writers


def read_summary_table(sqlite_path: str, table_name: str = 'underway_summary', after_timestamp: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Read the underway_summary table from SQLite database.
    
    Args:
        sqlite_path: Path to SQLite database
        table_name: Name of the summary table
        after_timestamp: Only return records after this timestamp (if None, returns all)
        
    Returns:
        DataFrame containing summary data
    """
    try:
        conn = sqlite3.connect(sqlite_path)
    except Exception as e:
        logging.error(f"Error connecting to SQLite database: {e}")
        return pd.DataFrame()
    
    try:
        if after_timestamp is not None:
            unix_timestamp = int(after_timestamp.timestamp())
            query = f"SELECT * FROM {table_name} WHERE datetime_utc > {unix_timestamp} ORDER BY datetime_utc"
        else:
            query = f"SELECT * FROM {table_name} ORDER BY datetime_utc"
            
        df = pd.read_sql_query(query, conn)
        
        # Convert datetime_utc to datetime if it's in unix format
        if 'datetime_utc' in df.columns and df['datetime_utc'].dtype in ['int64', 'int32']:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='s')
            
        return df
        
    except Exception as e:
        logging.error(f"Error reading {table_name} table: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def resample_summary_data(df: pd.DataFrame, resample_interval: str = '60s') -> pd.DataFrame:
    """
    Resample summary data using mean aggregation.
    
    Args:
        df: Input DataFrame with datetime_utc column
        resample_interval: Resampling interval (e.g., '60s', '5min', '1h')
        
    Returns:
        Resampled DataFrame
    """
    if df.empty or 'datetime_utc' not in df.columns:
        return df
    
    # Ensure datetime_utc is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['datetime_utc']):
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    
    # Set datetime_utc as index for resampling
    df_indexed = df.set_index('datetime_utc')
    
    # Resample using mean, forward fill missing values for categorical data
    numeric_columns = df_indexed.select_dtypes(include=['number']).columns
    
    if len(numeric_columns) == 0:
        logging.warning("No numeric columns found for resampling")
        return pd.DataFrame()
    
    # Resample numeric columns with mean
    df_resampled = df_indexed[numeric_columns].resample(resample_interval).mean()
    
    # Reset index to get datetime_utc back as a column
    df_resampled = df_resampled.reset_index()
    
    # Remove rows with all NaN values (can happen if no data in time bin)
    df_resampled = df_resampled.dropna(how='all', subset=numeric_columns)
    
    return df_resampled


def get_last_output_timestamp(file_path: str, file_type: str = 'parquet') -> Optional[pd.Timestamp]:
    """
    Get the most recent timestamp from an output file.
    
    Args:
        file_path: Path to the output file
        file_type: Type of file ('parquet' or 'csv')
        
    Returns:
        Most recent timestamp or None if file doesn't exist or is empty
    """
    try:
        if file_type == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
        else:
            logging.error(f"Unsupported file type: {file_type}")
            return None
            
        if df.empty or 'datetime_utc' not in df.columns:
            return None
            
        # Ensure datetime_utc is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['datetime_utc']):
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            
        return df['datetime_utc'].max()
        
    except FileNotFoundError:
        logging.info(f"Output file {file_path} not found, will create new file")
        return None
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None


def process_summary_incremental(
    sqlite_path: str,
    resample_interval: str = '60s',
    parquet_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    summary_table: str = 'underway_summary',
    partition_hours: Optional[int] = None
) -> pd.DataFrame:
    """
    Process summary data incrementally and write to output files.
    
    Args:
        sqlite_path: Path to SQLite database
        resample_interval: Resampling interval (e.g., '60s', '5min', '1h')
        parquet_path: Path to Parquet output file (None to skip)
        csv_path: Path to CSV output file (None to skip)
        summary_table: Name of the summary table
        
    Returns:
        DataFrame of newly processed records
    """
    config = get_config()
    
    # Use partition_hours from argument if provided, otherwise from config
    if partition_hours is None:
        partition_hours = config.get('partition_hours', None)
    
    # Determine the most recent timestamp from output files
    last_timestamp = None
    
    if parquet_path:
        parquet_ts = get_last_output_timestamp(parquet_path, 'parquet')
        if parquet_ts is not None:
            last_timestamp = parquet_ts
            
    if csv_path:
        csv_ts = get_last_output_timestamp(csv_path, 'csv')
        if csv_ts is not None:
            if last_timestamp is None or csv_ts > last_timestamp:
                last_timestamp = csv_ts
    
    if last_timestamp is not None:
        logging.info(f"Last output timestamp: {last_timestamp}")
    else:
        logging.info("No existing output data found, processing all summary data")
    
    # Read new summary data
    df = read_summary_table(sqlite_path, summary_table, after_timestamp=last_timestamp)
    
    if df.empty:
        logging.info("No new summary data to process")
        return pd.DataFrame()
    
    logging.info(f"Found {len(df)} new summary records")
    
    # Resample the data
    df_resampled = resample_summary_data(df, resample_interval)
    
    if df_resampled.empty:
        logging.info("No data after resampling")
        return df_resampled
    
    logging.info(f"Generated {len(df_resampled)} resampled records")
    
    # Write to Parquet if requested
    if parquet_path:
        logging.info(f"Writing to Parquet: {parquet_path}")
        try:
            file_writers.to_parquet(df_resampled, parquet_path, append=True, partition_hours=partition_hours)
        except Exception as e:
            logging.error(f"Error writing to Parquet: {e}")
    
    # Write to CSV if requested
    if csv_path:
        logging.info(f"Writing to CSV: {csv_path}")
        try:
            import os
            header = not os.path.exists(csv_path)
            file_writers.to_csv(df_resampled, csv_path, mode='a', header=header)
        except Exception as e:
            logging.error(f"Error writing to CSV: {e}")
    
    logging.info(f"Processing complete. {len(df_resampled)} records processed.")
    return df_resampled


def poll_and_resample(
    sqlite_path: str,
    output_poll_interval: int = 60,
    resample_interval: str = '60s',
    parquet_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    summary_table: str = 'underway_summary',
    stop_after: Optional[float] = None
):
    """
    Continuously poll the summary table and resample data at specified intervals.
    
    Args:
        sqlite_path: Path to SQLite database
        output_poll_interval: Seconds between polls
        resample_interval: Resampling interval (e.g., '60s', '5min', '1h') 
        parquet_path: Path to Parquet output file (None to skip)
        csv_path: Path to CSV output file (None to skip)
        summary_table: Name of the summary table
        stop_after: Stop polling after this many seconds (None = run forever)
    """
    start_time = time.time()
    
    logging.info(f"Starting polling mode with {output_poll_interval}s intervals")
    logging.info(f"Resampling interval: {resample_interval}")
    if parquet_path:
        logging.info(f"Parquet output: {parquet_path}")
    if csv_path:
        logging.info(f"CSV output: {csv_path}")
    
    while True:
        try:
            df = process_summary_incremental(
                sqlite_path=sqlite_path,
                resample_interval=resample_interval,
                parquet_path=parquet_path,
                csv_path=csv_path,
                summary_table=summary_table
            )
            
            if not df.empty:
                logging.info(f"Processed {len(df)} new resampled records")
            else:
                logging.debug("No new data to process")
                
        except Exception as e:
            logging.error(f"Error during processing: {e}")
        
        # Check if we should stop
        if stop_after is not None and (time.time() - start_time) > stop_after:
            logging.info("Stop time reached, ending polling")
            break
            
        time.sleep(output_poll_interval)


def main():
    """Command-line entry point for resampling summary data."""
    config = get_config()
    
    # Set up logging
    log_path = config.get('log_path', None)
    log_handlers = [logging.StreamHandler()]
    if log_path:
        try:
            file_handler = logging.FileHandler(log_path)
            log_handlers.append(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging to {log_path}: {e}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=log_handlers
    )
    
    parser = argparse.ArgumentParser(description="Resample underway_summary table and write to output files.")
    parser.add_argument('--db-path', type=str, default=config.get('db_path'), 
                       help='Path to SQLite database (default from config)')
    parser.add_argument('--resample-interval', type=str, default='60s',
                       help='Resampling interval (e.g., 60s, 5min, 1h) (default: 60s)')
    parser.add_argument('--parquet-path', type=str,
                       help='Path to Parquet output file')
    parser.add_argument('--csv-path', type=str,
                       help='Path to CSV output file')
    parser.add_argument('--summary-table', type=str, default='underway_summary',
                       help='Name of the summary table (default: underway_summary)')
    parser.add_argument('--poll', action='store_true',
                       help='Continuously poll for new data')
    parser.add_argument('--output-poll-interval', type=float, default=60.0,
                       help='Polling interval in seconds (default: 60.0)')
    parser.add_argument('--stop-after', type=float,
                       help='Stop polling after this many seconds')
    
    args = parser.parse_args()
    
    if not args.db_path:
        logging.error("Database path not specified in config or command line")
        return
    
    if not args.parquet_path and not args.csv_path:
        logging.error("At least one output path (--parquet-path or --csv-path) must be specified")
        return
    
    if args.poll:
        try:
            poll_and_resample(
                sqlite_path=args.db_path,
                output_poll_interval=args.output_poll_interval,
                resample_interval=args.resample_interval,
                parquet_path=args.parquet_path,
                csv_path=args.csv_path,
                summary_table=args.summary_table,
                stop_after=args.stop_after
            )
        except KeyboardInterrupt:
            logging.info("Stopped polling.")
    else:
        df = process_summary_incremental(
            sqlite_path=args.db_path,
            resample_interval=args.resample_interval,
            parquet_path=args.parquet_path,
            csv_path=args.csv_path,
            summary_table=args.summary_table,
            partition_hours=config.get('partition_hours', None)
        )
        
        if not df.empty:
            logging.info(f"Successfully processed {len(df)} records")
        else:
            logging.info("No records to process")


if __name__ == "__main__":
    main()