# Testing with pytest and uv

This document describes how to run tests for the locness-datamanager project using pytest and uv.

## Setup

The project is configured with pytest using uv for dependency management. The test configuration is in `pyproject.toml` under the `[tool.pytest.ini_options]` section.

### Development Dependencies

The following test dependencies are included in the `dev` dependency group:

- `pytest>=8.0.0` - Main testing framework
- `pytest-cov>=4.0.0` - Coverage reporting
- `pytest-mock>=3.11.0` - Mocking utilities

### Installing Dependencies

To install the development dependencies:

```bash
uv sync --group dev
```

## Running Tests

### Run All Tests
```bash
uv run pytest
```

### Run Tests with Verbose Output
```bash
uv run pytest -v
```

### Run Specific Test File
```bash
uv run pytest tests/test_config.py
```

### Run Specific Test Class
```bash
uv run pytest tests/test_synthetic_data.py::TestSyntheticDataGeneration
```

### Run Specific Test Method
```bash
uv run pytest tests/test_config.py::TestConfig::test_get_config_returns_dict
```

### Run Tests with Coverage Report
```bash
uv run pytest --cov=locness_datamanager --cov-report=term-missing
```

### Run Tests and Generate HTML Coverage Report
```bash
uv run pytest --cov=locness_datamanager --cov-report=html
```

The HTML coverage report will be generated in the `htmlcov/` directory.

## Test Organization

Tests are organized in the `tests/` directory:

- `tests/test_config.py` - Tests for configuration module
- `tests/test_synthetic_data.py` - Tests for synthetic data generation
- `tests/test_basic.py` - Basic functionality tests
- `tests/conftest.py` - Pytest configuration and fixtures

## Test Markers

The project defines several test markers:

- `@pytest.mark.slow` - For slow-running tests
- `@pytest.mark.integration` - For integration tests  
- `@pytest.mark.unit` - For unit tests

### Running Tests by Marker

```bash
# Run only unit tests
uv run pytest -m unit

# Skip slow tests
uv run pytest -m "not slow"

# Run only integration tests
uv run pytest -m integration
```

## Fixtures

Common test fixtures are defined in `tests/conftest.py`:

- `temp_dir` - Provides a temporary directory for test files
- `sample_csv_file` - Creates a sample CSV file for testing
- `sample_sqlite_db` - Creates a sample SQLite database with test schemas

## Coverage Configuration

Coverage is configured to:

- Include all code in the `locness_datamanager` package
- Exclude test files and cache directories
- Generate both terminal and HTML reports
- Skip common non-testable code patterns (like `if __name__ == "__main__":`)

## Writing Tests

### Test Structure

Follow these conventions:

1. Test files should start with `test_` or end with `_test.py`
2. Test classes should start with `Test`
3. Test methods should start with `test_`

### Example Test

```python
import pytest
from locness_datamanager.config import get_config


class TestConfig:
    """Test configuration functionality."""
    
    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        config = get_config()
        assert isinstance(config, dict)
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """Example of a slow test."""
        # Long-running test code here
        pass
```

### Using Fixtures

```python
def test_with_temp_directory(temp_dir):
    """Test that uses the temp_dir fixture."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.read_text() == "test content"
```

## Continuous Integration

For CI environments, you can run tests with:

```bash
# Exit on first failure, show local variables on failure
uv run pytest -x -l

# Run tests in parallel (if pytest-xdist is installed)
uv run pytest -n auto

# Generate XML report for CI systems
uv run pytest --junitxml=junit.xml
```
