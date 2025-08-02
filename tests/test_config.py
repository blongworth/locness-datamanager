"""Tests for config module."""
import pytest
from locness_datamanager.config import get_config


class TestConfig:
    """Test configuration loading."""

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        config = get_config()
        assert isinstance(config, dict)

    def test_config_has_required_keys(self):
        """Test that config contains expected keys."""
        config = get_config()

        # Check for all expected configuration keys from config.toml
        expected_keys = [
            'db_path',
            'parquet_path',
            'log_path',
            'db_poll_interval',
            'db_res_int',
            'ph_ma_window',
            'ph_freq',
            'ph_k0',
            'ph_k2',
            'output_poll_interval',
            'output_res_int',
            'partition_hours',
            'backup_interval',
            'backup_path',
            'time',
            'num',
            'freq',
            'continuous'
        ]

        for key in expected_keys:
            assert key in config, f"Expected key '{key}' not found in config"

    def test_config_types(self):
        """Test that config values have expected types."""
        config = get_config()
        
        # Test some basic type expectations
        assert isinstance(config.get('num', 0), int)
        assert isinstance(config.get('freq', 0.0), (int, float))
        assert isinstance(config.get('continuous', False), bool)
        assert isinstance(config.get('cloud_path', ''), str)
        assert isinstance(config.get('basename', ''), str)
        assert isinstance(config.get('table', ''), str)
