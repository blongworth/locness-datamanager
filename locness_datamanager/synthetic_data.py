import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

class SyntheticDataGenerator:
    @staticmethod
    def to_duckdb(df, db_path, table_name="sensor_data", create_table=True):
        """
        Write a DataFrame to a DuckDB table.
        Args:
            df: pandas.DataFrame
            db_path: Path to DuckDB database file
            table_name: Name of the table to write to
            create_table: If True, create table if it does not exist
        """
        con = duckdb.connect(db_path)
        if create_table:
            con.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TIMESTAMP,
                    lat DOUBLE,
                    lon DOUBLE,
                    temp DOUBLE,
                    salinity DOUBLE,
                    rhodamine DOUBLE,
                    pH DOUBLE
                )
            ''')
        sample_data = [tuple(row) for row in df.itertuples(index=False, name=None)]
        con.executemany(f'''
            INSERT INTO {table_name} (timestamp, lat, lon, temp, salinity, rhodamine, pH)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', sample_data)
        con.close()
    @staticmethod
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

    @staticmethod
    def to_parquet(df, filename, append=False):
        """
        Write a DataFrame to a Parquet file. If append=True, appends to existing file.
        Args:
            df: pandas.DataFrame
            filename: Output Parquet file path
            append: If True, append to existing file (default False)
        """
        table = pa.Table.from_pandas(df)
        if not append or not os.path.exists(filename):
            df.to_parquet(filename, index=False)
        else:
            # Append by reading old, writing both to new temp, then replacing
            tmp_path = filename + ".tmp"
            with pq.ParquetWriter(tmp_path, table.schema) as writer:
                for batch in pq.ParquetFile(filename).iter_batches():
                    writer.write_table(pa.Table.from_batches([batch]))
                writer.write_table(table)
            os.replace(tmp_path, filename)
            
    @staticmethod
    def generate(n_records=1, base_lat=40.7128, base_lon=-74.0060, start_time=None, frequency_hz=1.0):
        """
        Generate synthetic oceanographic data.
        Args:
            n_records: Number of records to generate
            base_lat: Base latitude
            base_lon: Base longitude
            start_time: Optional datetime to start timestamps
        Returns:
            pandas.DataFrame with synthetic data
        """
        if start_time is None:
            current_time = datetime.now()
        else:
            current_time = start_time
        delta_seconds = 1.0 / frequency_hz if frequency_hz > 0 else 1.0
        data = []
        for i in range(n_records):
            record = {
                "timestamp": current_time + timedelta(seconds=i * delta_seconds),
                "lat": base_lat + np.random.normal(0, 0.001),
                "lon": base_lon + np.random.normal(0, 0.001),
                "temp": 15 + 5 * np.sin(i * 0.02) + np.random.normal(0, 0.5),
                "salinity": 35 + 2 * np.sin(i * 0.015) + np.random.normal(0, 0.2),
                "rhodamine": min(500, np.random.exponential(scale=1.25)),
                "ph": 8.1 + 0.3 * np.sin(i * 0.025) + np.random.normal(0, 0.05),
            }
            data.append(record)
        return pd.DataFrame(data)

def create_sample_database(db_path='oceanographic_data.duckdb', sample_frequency_hz=1, num_samples=1000):
    con = duckdb.connect(db_path)
    con.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            timestamp TIMESTAMP,
            lat DOUBLE,
            lon DOUBLE,
            temp DOUBLE,
            salinity DOUBLE,
            rhodamine DOUBLE,
            pH DOUBLE
        )
    ''')
    # Check if table is empty
    count = con.execute('SELECT COUNT(*) FROM sensor_data').fetchone()[0]
    if count == 0:
        delta_seconds = 1.0 / sample_frequency_hz if sample_frequency_hz > 0 else 1.0
        base_time = datetime.now() - timedelta(seconds=delta_seconds * num_samples)
        base_lat, base_lon = 42.3601, -71.0589  # Boston area
        df = SyntheticDataGenerator.generate(num_samples, base_lat=base_lat, base_lon=base_lon, start_time=base_time)
        SyntheticDataGenerator.to_duckdb(df, db_path, table_name="sensor_data", create_table=False)
    con.close()

def add_sample_row(db_path='oceanographic_data.duckdb'):
    con = duckdb.connect(db_path)
    last = con.execute('SELECT * FROM sensor_data ORDER BY id DESC LIMIT 1').fetchone()
    if last:
        _, last_time, last_lat, last_lon, _, _, _, _ = last
        timestamp = datetime.now()
        lat = last_lat + np.random.normal(0, 0.001)
        lon = last_lon + np.random.normal(0, 0.001)
        temp = 15 + 5 * np.sin(timestamp.timestamp() * 0.02) + np.random.normal(0, 0.5)
        salinity = 35 + 2 * np.sin(timestamp.timestamp() * 0.015) + np.random.normal(0, 0.2)
        rhodamine = min(500, np.random.exponential(scale=1.25))
        ph = 8.1 + 0.3 * np.sin(timestamp.timestamp() * 0.025) + np.random.normal(0, 0.05)
        con.execute('''
            INSERT INTO sensor_data (timestamp, lat, lon, temp, salinity, rhodamine, pH)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', [timestamp, lat, lon, temp, salinity, rhodamine, ph])
    con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic oceanographic data and write to CSV, Parquet, and DuckDB.")
    parser.add_argument('--path', type=str, default='.', help='Directory to write output files (default: current directory)')
    parser.add_argument('--basename', type=str, default='synthetic_oceanographic_data', help='Base name for output files (no extension)')
    parser.add_argument('--num', type=int, default=1000, help='Number of records to generate per batch (default: 1000)')
    parser.add_argument('--freq', type=float, default=1.0, help='Sample frequency in Hz (default: 1.0)')
    parser.add_argument('--table', type=str, default='sensor_data', help='DuckDB table name (default: sensor_data)')
    parser.add_argument('--continuous', action='store_true', help='Continuously generate and write data every (num * freq) seconds')
    args = parser.parse_args()

    import time
    import sys

    def run_once():
        print(f"Generating {args.num} samples at {args.freq} Hz...")
        t0 = time.perf_counter()
        df = SyntheticDataGenerator.generate(n_records=args.num, frequency_hz=args.freq)
        t1 = time.perf_counter()
        gen_time = t1 - t0

        basepath = os.path.join(args.path, args.basename)
        csv_file = f"{basepath}.csv"
        parquet_file = f"{basepath}.parquet"
        db_file = f"{basepath}.duckdb"
        table_name = args.table

        print(f"Writing to {csv_file} (CSV)...")
        t_csv0 = time.perf_counter()
        SyntheticDataGenerator.to_csv(df, csv_file, mode='a' if os.path.exists(csv_file) else 'w', header=not os.path.exists(csv_file))
        t_csv1 = time.perf_counter()
        csv_time = t_csv1 - t_csv0

        print(f"Writing to {parquet_file} (Parquet)...")
        t_parquet0 = time.perf_counter()
        SyntheticDataGenerator.to_parquet(df, parquet_file, append=True)
        t_parquet1 = time.perf_counter()
        parquet_time = t_parquet1 - t_parquet0

        print(f"Writing to {db_file} (DuckDB table: {table_name}) ...")
        t_db0 = time.perf_counter()
        SyntheticDataGenerator.to_duckdb(df, db_file, table_name=table_name)
        t_db1 = time.perf_counter()
        db_time = t_db1 - t_db0

        print("Done.")
        print("Timing summary:")
        print(f"  Data generation: {gen_time:.4f} seconds")
        print(f"  CSV write:       {csv_time:.4f} seconds")
        print(f"  Parquet write:   {parquet_time:.4f} seconds")
        print(f"  DuckDB write:    {db_time:.4f} seconds")

    if args.continuous:
        print("Continuous mode enabled. Press Ctrl+C to stop.")
        interval = args.num / args.freq if args.freq > 0 else args.num
        try:
            # First batch: generate and write immediately
            print(f"Preparing first batch of {args.num} records...")
            t0 = time.perf_counter()
            df = SyntheticDataGenerator.generate(n_records=args.num, frequency_hz=args.freq)
            t1 = time.perf_counter()
            gen_time = t1 - t0

            print(f"Writing batch at {datetime.now().isoformat(timespec='seconds')}...")
            basepath = os.path.join(args.path, args.basename)
            csv_file = f"{basepath}.csv"
            parquet_file = f"{basepath}.parquet"
            db_file = f"{basepath}.duckdb"
            table_name = args.table

            t_csv0 = time.perf_counter()
            SyntheticDataGenerator.to_csv(df, csv_file, mode='a' if os.path.exists(csv_file) else 'w', header=not os.path.exists(csv_file))
            t_csv1 = time.perf_counter()
            csv_time = t_csv1 - t_csv0

            t_parquet0 = time.perf_counter()
            SyntheticDataGenerator.to_parquet(df, parquet_file, append=True)
            t_parquet1 = time.perf_counter()
            parquet_time = t_parquet1 - t_parquet0

            t_db0 = time.perf_counter()
            SyntheticDataGenerator.to_duckdb(df, db_file, table_name=table_name)
            t_db1 = time.perf_counter()
            db_time = t_db1 - t_db0

            print("Done.")
            print("Timing summary:")
            print(f"  Data generation: {gen_time:.4f} seconds")
            print(f"  CSV write:       {csv_time:.4f} seconds")
            print(f"  Parquet write:   {parquet_time:.4f} seconds")
            print(f"  DuckDB write:    {db_time:.4f} seconds")

            interval = args.num / args.freq if args.freq > 0 else args.num
            next_write_time = time.time() + interval

            while True:
                # Generate the next batch during the sleep interval
                print(f"Preparing next batch of {args.num} records...")
                t0 = time.perf_counter()
                df = SyntheticDataGenerator.generate(n_records=args.num, frequency_hz=args.freq)
                t1 = time.perf_counter()
                gen_time = t1 - t0

                now = time.time()
                sleep_time = next_write_time - now
                if sleep_time > 0:
                    print(f"Sleeping {sleep_time:.2f} seconds before writing batch...")
                    time.sleep(sleep_time)
                # Write the batch at the correct time
                print(f"Writing batch at {datetime.now().isoformat(timespec='seconds')}...")
                basepath = os.path.join(args.path, args.basename)
                csv_file = f"{basepath}.csv"
                parquet_file = f"{basepath}.parquet"
                db_file = f"{basepath}.duckdb"
                table_name = args.table

                t_csv0 = time.perf_counter()
                SyntheticDataGenerator.to_csv(df, csv_file, mode='a' if os.path.exists(csv_file) else 'w', header=not os.path.exists(csv_file))
                t_csv1 = time.perf_counter()
                csv_time = t_csv1 - t_csv0

                t_parquet0 = time.perf_counter()
                SyntheticDataGenerator.to_parquet(df, parquet_file, append=True)
                t_parquet1 = time.perf_counter()
                parquet_time = t_parquet1 - t_parquet0

                t_db0 = time.perf_counter()
                SyntheticDataGenerator.to_duckdb(df, db_file, table_name=table_name)
                t_db1 = time.perf_counter()
                db_time = t_db1 - t_db0

                print("Done.")
                print("Timing summary:")
                print(f"  Data generation: {gen_time:.4f} seconds")
                print(f"  CSV write:       {csv_time:.4f} seconds")
                print(f"  Parquet write:   {parquet_time:.4f} seconds")
                print(f"  DuckDB write:    {db_time:.4f} seconds")

                next_write_time += interval
        except KeyboardInterrupt:
            print("\nStopped continuous generation.")
            sys.exit(0)
    else:
        run_once()