# Locness Data Manager

Locness Data Manager is a Python toolkit for generating, processing, and synchronizing oceanographic and sensor data. It supports robust synthetic data generation, time alignment, resampling, and mirroring between local and remote databases (SQLite, DuckDB, CSV, Parquet).

## Features

- Generate realistic synthetic oceanographic timeseries data
- Output to CSV, Parquet, and DuckDB (append or create)
- Command-line interface (CLI) for batch or continuous data generation
- Modular Python API for integration in other projects
- Time alignment and resampling utilities
- Data mirroring between local and shore (remote) databases

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