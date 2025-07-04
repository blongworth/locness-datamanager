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

## Requirements

- Python 3.9+
- numpy, pandas, duckdb, pyarrow

## License

MIT License