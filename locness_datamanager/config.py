import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib

CONFIG_TOML = os.path.join(os.path.dirname(__file__), "config.toml")

# Load defaults from config.toml
with open(CONFIG_TOML, "rb") as f:
    _TOML = tomllib.load(f)
DEFAULTS = _TOML.get("defaults", {})

def get_config():
    """
    Load config from config.toml and environment variables.
    Returns a dict of config values.
    """
    config = DEFAULTS.copy()
    config['path'] = os.environ.get('LOCNESS_PATH', config.get('path', '.'))
    config['basename'] = os.environ.get('LOCNESS_BASENAME', config.get('basename', 'synthetic_oceanographic_data'))
    config['num'] = int(os.environ.get('LOCNESS_NUM', config.get('num', 1000)))
    config['freq'] = float(os.environ.get('LOCNESS_FREQ', config.get('freq', 1.0)))
    config['table'] = os.environ.get('LOCNESS_TABLE', config.get('table', 'sensor_data'))
    config['continuous'] = os.environ.get('LOCNESS_CONTINUOUS', str(config.get('continuous', False))).lower() in ('1', 'true', 'yes')
    config['ph_ma_window'] = int(os.environ.get('LOCNESS_PH_MA_WINDOW', config.get('ph_ma_window', 120)))
    config['ph_freq'] = float(os.environ.get('LOCNESS_PH_FREQ', config.get('ph_freq', 0.5)))
    config['partition_hours'] = float(os.environ.get('LOCNESS_PARTITION_HOURS', config.get('partition_hours', 12)))
    return config
