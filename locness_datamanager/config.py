
import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib

CONFIG_TOML = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.toml")

# Load config sections from config.toml
with open(CONFIG_TOML, "rb") as f:
    _TOML = tomllib.load(f)

def get_config():
    """
    Load config from config.toml and environment variables.
    Returns a dict of config values merged from all sections.
    """
    config = {}
    # Merge all sections into one config dict
    for section in _TOML:
        config.update(_TOML[section])

    # Environment variable overrides
    config['cloud_path'] = os.environ.get('LOCNESS_CLOUD_PATH', config.get('cloud_path', '.'))
    config['basename'] = os.environ.get('LOCNESS_BASENAME', config.get('basename', 'synthetic_oceanographic_data'))
    config['db_path'] = os.environ.get('LOCNESS_DB_PATH', config.get('db_path', 'locness.db'))
    config['res_freq'] = os.environ.get('LOCNESS_RES_FREQ', config.get('res_freq', 0.5))
    config['ph_ma_window'] = int(os.environ.get('LOCNESS_PH_MA_WINDOW', config.get('ph_ma_window', 120)))
    config['ph_freq'] = float(os.environ.get('LOCNESS_PH_FREQ', config.get('ph_freq', 0.5)))
    config['partition_hours'] = float(os.environ.get('LOCNESS_PARTITION_HOURS', config.get('partition_hours', 12)))
    config['db_path'] = os.environ.get('LOCNESS_DB_PATH', config.get('db_path', 'locness.db'))
    config['num'] = int(os.environ.get('LOCNESS_NUM', config.get('num', 30)))
    config['freq'] = float(os.environ.get('LOCNESS_FREQ', config.get('freq', 0.5)))
    config['continuous'] = str(os.environ.get('LOCNESS_CONTINUOUS', config.get('continuous', True))).lower() in ('1', 'true', 'yes')
    return config
