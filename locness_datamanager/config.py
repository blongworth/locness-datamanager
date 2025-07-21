
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
    return config
