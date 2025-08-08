import pandas as pd
import sqlite3
from locness_datamanager.config import get_config

# Load the config file
config = get_config()

# Get the parquet path from the config
parquet_path = config.get('parquet_path')

# Read the parquet dataset
df = pd.read_parquet(parquet_path)


def print_datetime_regularity(df, label):
    print(f"\n--- {label} datetime_utc regularity ---")
    if 'datetime_utc' not in df.columns:
        print("datetime_utc column not found!")
        return
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['datetime_utc']):
        # Try integer seconds first, then fallback to generic parse
        if pd.api.types.is_integer_dtype(df['datetime_utc']):
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='s')
        else:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    # Sort by datetime_utc
    df = df.sort_values('datetime_utc')
    # Calculate time differences in seconds
    diffs = df['datetime_utc'].diff().dt.total_seconds().dropna()
    print("Datetime regularity check:")
    print(f"Mean interval: {diffs.mean()} seconds")
    print(f"Median interval: {diffs.median()} seconds")
    print(f"Std deviation: {diffs.std()} seconds")
    print(f"Min interval: {diffs.min()} seconds")
    print(f"Max interval: {diffs.max()} seconds")
    irregular = diffs[(diffs - diffs.mean()).abs() > 2 * diffs.std()]
    total_intervals = len(diffs)
    num_irregular = len(irregular)
    percent_irregular = (num_irregular / total_intervals * 100) if total_intervals > 0 else 0
    if not irregular.empty:
        print(f"Irregular intervals detected: {num_irregular} out of {total_intervals} ({percent_irregular:.2f}%)")
        print(irregular)
    else:
        print("All intervals are regular within 2 standard deviations.")

# Example: print the first few rows
print(df.head())


print_datetime_regularity(df, "Parquet")


# Check regularity of datetime_utc in multiple tables in sqlite
db_path = config.get('db_path')
sqlite_tables = [
    ("underway_summary", "SQLite underway_summary"),
    ("rhodamine", "SQLite rhodamine"),
    ("ph", "SQLite ph"),
    ("tsg", "SQLite tsg"),
    ("gps", "SQLite gps"),
]
with sqlite3.connect(db_path) as conn:
    for table, label in sqlite_tables:
        try:
            df_table = pd.read_sql_query(f"SELECT datetime_utc FROM {table} ORDER BY datetime_utc", conn)
        except Exception as e:
            print(f"Error reading from SQLite table {table}: {e}")
            df_table = pd.DataFrame()
        if df_table.empty or 'datetime_utc' not in df_table.columns:
            print(f"datetime_utc column not found or no data in {table}!")
        else:
            print_datetime_regularity(df_table, label)