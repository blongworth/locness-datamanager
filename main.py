import time
from locness_datamanager.config import get_config
from locness_datamanager.resample import resample_data, write_resampled_data_to_sqlite
from locness_datamanager.synthetic_data import add_ph_moving_average
from locness_datamanager.backup_db import DatabaseBackup
import sqlite3
import os

POLL_INTERVAL = 60  # seconds
BACKUP_INTERVAL = 3600  # seconds


def poll_and_process():
    config = get_config()
    db_path = config.get('db_path')
    summary_table = config.get('summary_table', 'underway_summary')
    ph_ma_window = config.get('ph_ma_window', 120)
    ph_freq = config.get('ph_freq', 0.5)
    backup_dir = 'backups'
    last_backup = time.time()

    backup_manager = DatabaseBackup(db_path, backup_dir=backup_dir)

    while True:
        print("Polling for new data...")
        # 1. Poll for new data (example: get all new rows since last processed)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM sensor_data ORDER BY datetime_utc DESC LIMIT 1000")
        rows = cursor.fetchall()
        conn.close()
        if not rows:
            print("No new data found.")
            time.sleep(POLL_INTERVAL)
            continue

        # 2. Resample data
        print("Resampling data...")
        df_resampled = resample_data(rows)

        # 3. Add pH moving average
        print("Adding pH moving average...")
        df_resampled = add_ph_moving_average(df_resampled, window=ph_ma_window, freq=ph_freq)

        # 4. Write to summary table
        print(f"Writing to summary table: {summary_table}")
        write_resampled_data_to_sqlite(db_path, output_table=summary_table)

        # 5. Update file outputs (example: write CSV)
        print("Updating file outputs...")
        df_resampled.to_csv(f"{summary_table}.csv", index=False)

        # 6. Periodically back up the database
        if time.time() - last_backup > BACKUP_INTERVAL:
            print("Backing up database...")
            backup_manager.create_backup()
            last_backup = time.time()

        print(f"Sleeping for {POLL_INTERVAL} seconds...")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    poll_and_process()
