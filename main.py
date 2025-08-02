import time
import logging
from locness_datamanager.config import get_config
from locness_datamanager.resample import process_raw_data_incremental
from locness_datamanager.resample_summary import process_summary_incremental
from locness_datamanager.backup_db import DatabaseBackup
import os

''' 
This script should be the main entry point for the data manager.
Use the functions in locness_datamanager to:
1. Check that the database is set up correctly.

2. Process data:
    - Apply calibrations and calculations (e.g., pH, quality flags).

3. Process output:4. Periodically back up the database.
5. Backup/rotate raw csv files.
'''


def poll_and_process(
    db_path: str = None,
    db_poll_interval: int = 10,
    db_resample_interval: str = '10s',
    parquet_path: str = None,
    parquet_poll_interval: int = 60,
    parquet_resample_interval: str = '60s',
    partition_hours: int = 6,
    backup_interval: int = 3600,
    backup_manager: DatabaseBackup = None,
    csv_path: str = None,
    ph_k0: float = 0.0,
    ph_k2: float = 0.0,
    ph_ma_window: int = 120,
    ph_freq: float = 0.5,
):
    last_parquet = time.time()
    last_backup = time.time()
    while True:
        logging.info("Processing new data...")
        # Process raw data with pH configuration parameters
        process_raw_data_incremental(
            sqlite_path=db_path,
            resample_interval=db_resample_interval,
            summary_table='underway_summary',
            replace_all=False,
            ph_k0=ph_k0,
            ph_k2=ph_k2,
            ph_ma_window=ph_ma_window,
            ph_freq=ph_freq,
        )

        # if time to write parquet
        if time.time() - last_parquet > parquet_poll_interval:
            logging.info("Writing resampled Parquet data...")
            last_parquet = time.time()
            # use main resample_summary interval function
            process_summary_incremental(
                sqlite_path=db_path,
                resample_interval=parquet_resample_interval,
                parquet_path=parquet_path,
                partition_hours=partition_hours,
                csv_path=csv_path,
            )

        # if time to backup
        if time.time() - last_backup > backup_interval:
            logging.info("Backing up database...")
            backup_manager.create_backup()
            last_backup = time.time()

        logging.info(f"Sleeping for {db_poll_interval} seconds...")
        time.sleep(db_poll_interval)

def main():
    config = get_config()
    log_path = config.get('log_path', None)
    db_path = config.get('db_path', 'data/locness.db')
    db_poll_interval = config.get('db_poll_interval', 10)
    db_resample_interval = config.get('db_resample_interval', '10s')
    parquet_path = config.get('parquet_path', 'data/locness.parquet')
    parquet_poll_interval = config.get('parquet_poll_interval', 3600)
    parquet_resample_interval = config.get('parquet_resample_interval', '60s')
    partition_hours = config.get('partition_hours', 6)    
    csv_path = config.get('csv_path', 'data/locness.csv')
    backup_path = config.get('backup_path', 'data/backup')
    backup_interval = config.get('backup_interval', 3600)
    
    # pH configuration parameters
    ph_k0 = config.get('ph_k0', 0.0)
    ph_k2 = config.get('ph_k2', 0.0)
    ph_ma_window = config.get('ph_ma_window', 120)
    ph_freq = config.get('ph_freq', 0.5)

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
    db_path = config.get('db_path')
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
