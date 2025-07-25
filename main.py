import time
import logging
from locness_datamanager.config import get_config
from locness_datamanager.resample import resample_data, write_resampled_data_to_sqlite
from locness_datamanager.synthetic_data import add_ph_moving_average
from locness_datamanager.backup_db import DatabaseBackup
import sqlite3
import os

''' 
This script should be the main entry point for the data manager.
Use the functions in locness_datamanager to:
1. Check that the database is set up correctly.
2. Poll for new data at regular intervals.
3. Resample the data.
4. Add calculated filters (e.g., pH moving average).
5. Write the resampled data to a summary table.
6. Write to output file(s) (e.g., parquet)
7. Periodically back up the database.
8. Backup/rotate raw csv files.
'''
# TODO: move these constants to config file
POLL_INTERVAL = 60  # seconds
BACKUP_INTERVAL = 3600  # seconds


def poll_and_process():


    while True:
        logging.info("Polling for new data...")
        # 1. Poll for new data (example: get all new rows since last processed)

        # 2. Resample data
        logging.info("Resampling data...")
        df_resampled = resample_data(rows)

        # 3. Add pH moving average
        logging.info("Adding pH moving average...")
        df_resampled = add_ph_moving_average(df_resampled, window=ph_ma_window, freq=ph_freq)

        # 4. Write to summary table
        logging.info(f"Writing to summary table: {summary_table}")
        write_resampled_data_to_sqlite(db_path, output_table=summary_table)

        # 5. Update file outputs (example: write CSV)
        logging.info("Updating file outputs...")
        df_resampled.to_csv(f"{summary_table}.csv", index=False)

        # 6. Periodically back up the database
        if time.time() - last_backup > BACKUP_INTERVAL:
            logging.info("Backing up database...")
            backup_manager.create_backup()
            last_backup = time.time()

        logging.info(f"Sleeping for {POLL_INTERVAL} seconds...")
        time.sleep(POLL_INTERVAL)

def main():
    config = get_config()
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
    db_path = config.get('db_path')
    # Ensure the database exists
    if not os.path.exists(db_path):
        logging.error(f"Database {db_path} does not exist. Please create it first.")
        return
    # test database configuation
    logging.info("Starting data manager...")
    backup_manager = DatabaseBackup(db_path, backup_dir=backup_dir)
    while True:
        try:
            poll_and_process()
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logging.info("Interrupted by user. Shutting down gracefully.")
            break

if __name__ == "__main__":
    main()
