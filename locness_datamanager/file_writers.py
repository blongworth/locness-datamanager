import os
import duckdb
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
        tmp_path = filename + ".tmp"
        with pq.ParquetWriter(tmp_path, table.schema) as writer:
            for batch in pq.ParquetFile(filename).iter_batches():
                writer.write_table(pa.Table.from_batches([batch]))
            writer.write_table(table)
        os.replace(tmp_path, filename)

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
                ph DOUBLE
            )
        ''')
    sample_data = [tuple(row) for row in df.itertuples(index=False, name=None)]
    con.executemany(f'''
        INSERT INTO {table_name} (timestamp, lat, lon, temp, salinity, rhodamine, ph)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', sample_data)
    con.close()

def to_sqlite(df, db_path, table_name="sensor_data", create_table=True):
    """
    Write a DataFrame to a SQLite3 table.
    Args:
        df: pandas.DataFrame
        db_path: Path to SQLite database file
        table_name: Name of the table to write to
        create_table: If True, create table if it does not exist
    """
    conn = sqlite3.connect(db_path)
    if create_table:
        # Use pandas to_sql with if_exists='append' and let pandas create table if needed
        df.to_sql(table_name, conn, if_exists='append', index=False)
    else:
        # Assume table exists, just append
        df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()
