import time
import logging
from locness_datamanager.config import get_config
from locness_datamanager.resample import write_resampled_to_sqlite
from locness_datamanager.resample_summary import process_summary_incremental
from locness_datamanager.backup_db import DatabaseBackup
from locness_datamanager.resampler import PersistentResampler
import os

"""
Main module for the LOCNESS Data Manager.

Handles periodic processing, resampling, and backup of underway data using configurable parameters.
Features:
    - Validates database existence.
    - Processes and calibrates new data.
    - Resamples and exports data to Parquet/CSV.
    - Performs scheduled database backups.

Configuration is loaded via `get_config()`. Run as a script to start the data manager loop.
"""

# TODO: Start main from shortcut

def poll_and_process(
    db_path: str = None,
    db_poll_interval: int = 10,
    db_resample_interval: str = '10s',
    parquet_path: str = None,
    parquet_poll_interval: int = 60,
    parquet_resample_interval: str = '60s',
    partition_hours: int = 6,
    backup_interval: float = 12,
    backup_manager: DatabaseBackup = None,
    csv_path: str = None,
    dynamodb_table: str = None,
    dynamodb_region: str = 'us-east-1',
    ph_k0: float = 0.0,
    ph_k2: float = 0.0,
    ph_ma_window: int = 120,
    ph_freq: float = 0.5,
):
    backup_interval_seconds = backup_interval * 3600  # convert hours to seconds

    # Initialize persistent resampler to avoid duplicate data processing
    resampler = PersistentResampler(
        sqlite_path=db_path,
        resample_interval=db_resample_interval,
        ph_ma_window=ph_ma_window,
        ph_freq=ph_freq,
        ph_k0=ph_k0,
        ph_k2=ph_k2
    )

    last_parquet = time.time()
    last_backup = time.time()
    
    while True:
        logging.info("Processing new data...")
        
        # Process new raw data using persistent resampler
        new_df = resampler.process_new_data()
        
        if not new_df.empty:
            # Write new data to summary table
            write_resampled_to_sqlite(new_df, db_path, 'underway_summary')
            logging.info(f"Wrote {len(new_df)} new records to summary table")
        else:
            logging.info("No new data to process")

      # Write new data to DynamoDB if configured
        if not new_df.empty and dynamodb_table:
            logging.info(f"Writing {len(new_df)} new records to DynamoDB table: {dynamodb_table}")
            try:
                from locness_datamanager import file_writers
                file_writers.to_dynamodb(new_df, dynamodb_table, region_name=dynamodb_region)
            except Exception as e:
                logging.error(f"Error writing to DynamoDB: {e}")

        # if time to write parquet
        if time.time() - last_parquet > parquet_poll_interval:
            logging.info("Writing resampled Parquet data...")
            last_parquet = time.time()
            # use main resample_summary interval function
            print("csv_path:", csv_path)
            process_summary_incremental(
                sqlite_path=db_path,
                resample_interval=parquet_resample_interval,
                parquet_path=parquet_path,
                partition_hours=partition_hours,
                csv_path=csv_path,
                # dynamodb writing now handled at full frequency above
                #dynamodb_table=dynamodb_table,
                #dynamodb_region=dynamodb_region
            )

        # if time to backup
        if time.time() - last_backup > backup_interval_seconds:
            logging.info("Backing up database...")
            backup_manager.create_backup()
            last_backup = time.time()

        logging.info(f"Sleeping for {db_poll_interval} seconds...")
        time.sleep(db_poll_interval)

def main():
    config = get_config()
    print("Using configuration:", config)

    # Load configuration parameters
    db_path = config.get('db_path')
    parquet_path = config.get('parquet_path')
    csv_path = config.get('csv_path')
    dynamodb_table = config.get('dynamodb_table')
    dynamodb_region = config.get('dynamodb_region', 'us-east-1')
    log_path = config.get('log_path')
    db_poll_interval = config.get('db_poll_interval')
    db_resample_interval = config.get('db_res_int')
    parquet_poll_interval = config.get('output_poll_interval')
    parquet_resample_interval = config.get('output_res_int')
    partition_hours = config.get('partition_hours')    
    backup_interval = config.get('backup_interval')
    backup_path = config.get('backup_path')
    
    # pH configuration parameters
    ph_k0 = config.get('ph_k0')
    ph_k2 = config.get('ph_k2')
    ph_ma_window = config.get('ph_ma_window')
    ph_freq = config.get('ph_freq')

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
    # Ensure the database exists
    if not os.path.exists(db_path):
        logging.error(f"Database {db_path} does not exist. Please create it first.")
        return
    # test database configuation
    logging.info("Starting data manager...")
    backup_manager = DatabaseBackup(db_path, backup_dir=backup_path)
    while True:
        try:
            poll_and_process(
                db_path=db_path,
                db_poll_interval=db_poll_interval,
                db_resample_interval=db_resample_interval,
                parquet_path=parquet_path,
                parquet_poll_interval=parquet_poll_interval,
                parquet_resample_interval=parquet_resample_interval,
                csv_path=csv_path,
                dynamodb_table=dynamodb_table,
                dynamodb_region=dynamodb_region,
                partition_hours=partition_hours,
                backup_manager=backup_manager,
                backup_interval=backup_interval,
                ph_k0=ph_k0,
                ph_k2=ph_k2,
                ph_ma_window=ph_ma_window,
                ph_freq=ph_freq,
            )
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            time.sleep(db_poll_interval)  # Wait before retrying
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Shutting down gracefully.")
            break

if __name__ == "__main__":
    main()
