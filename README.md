# Locness Data Manager

Locness Data Manager is a python package for managing LOCNESS underway data. Main function is to combine raw data from sql tables, resample to a common frequency, add computed fields, and write to a summary table, AWS DynamoDB, and various file formats (CSV, Parquet, DuckDB). It also includes utilities for generating synthetic oceanographic timeseries data for testing and development.

## Installation

```sh
git clone https://github.com/<username>/locness-datamanager.git
```

Dependency and environment management is easiest handled with uv. 

## Usage

### Setup and startup for underway data management

1. Edit `config.toml` to set database connection parameters, input table names, output file paths, and resampling frequency.
2. Run `uv run setup-db` to create the sqlite database
3. Run `uv run setup-dynamodb create` to create the DynamoDB table (if using)
4. Start the DAQ programs: fluorometer, ph, TSG, GPS.
5. Start the main data manager `uv run datamanager`
6. Start the web dashboard

## Synthetic Data Generation Usage

### CLI

Generate 1000 records at 1 Hz and write to all formats:

```sh
python -m locness_datamanager.synthetic_data --num 1000 --freq 1.0 --basename mydata --path ./output
```

Continuously generate and write a new batch every N seconds:

```sh
python -m locness_datamanager.synthetic_data --num 60 --freq 1 --continuous
```

### As a Library

```python
from locness_datamanager.synthetic_data import SyntheticDataGenerator
df = SyntheticDataGenerator.generate(n_records=100, frequency_hz=1.0)
SyntheticDataGenerator.to_csv(df, 'output.csv')
```

## Installation

Install locally in another project:

```sh
uv pip install -e /path/to/locness-datamanager
```

Or install from GitHub:

```sh
uv pip install 'git+https://github.com/<username>/<repo>.git'
```

## DynamoDB Integration

Store underway summary data to AWS DynamoDB for cloud access and analysis.

### Setup AWS Credentials
Configure AWS credentials using one of these methods:
- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- IAM roles (if running on EC2)

### Create DynamoDB Table
```sh
uv run setup-dynamodb create --table-name locness-underway-summary
```

### Configure Summary Data Output
Add DynamoDB parameters to your `config.toml`:
```toml
dynamodb_table = "locness-underway-summary"
dynamodb_region = "us-east-1"  # optional, defaults to us-east-1
```

Or use CLI flags:
```sh
uv run resample_summary --dynamodb-table locness-underway-summary --poll
```

The main data manager will automatically write summary data to DynamoDB when `dynamodb_table` is configured.

## Requirements

- Python 3.9+
- numpy, pandas, duckdb, pyarrow, boto3 (for DynamoDB)

## License

MIT License