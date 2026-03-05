# config/__init__.py
"""
Season configuration module.

Each season has its own configuration file with:
- Start/end dates
- Field definitions
- Game times per venue
- Field unavailabilities
- Club days
- No-play preferences

To add a new season:
1. Copy config/season_template.py to config/season_YYYY.py
2. Update all values for the new year
3. The system will automatically detect and load it

Usage:
    from config import load_season_config, load_season_data
    
    config = load_season_config(2025)  # Get raw config dict
    data = load_season_data(2025)      # Get full solver data dict
"""

import importlib


def load_season_config(year: int) -> dict:
    """
    Load configuration dictionary for a specific season year.
    
    Args:
        year: The season year (e.g., 2025, 2026)
        
    Returns:
        SEASON_CONFIG dict from the season file
        
    Raises:
        ValueError: If no configuration file exists for the year
    """
    module_name = f".season_{year}"
    try:
        module = importlib.import_module(module_name, package="config")
        return module.SEASON_CONFIG
    except ModuleNotFoundError:
        raise ValueError(f"No configuration found for season {year}. "
                        f"Create config/season_{year}.py to add support.")


def load_season_data(year: int) -> dict:
    """
    Load complete season data ready for the solver.
    
    This is the main entry point for loading season data. It loads the
    season config and builds all required data structures (teams, grades,
    timeslots, etc.) using build_season_data() from utils.
    
    Args:
        year: The season year (e.g., 2025, 2026)
        
    Returns:
        Complete data dict ready for solver with teams, grades, timeslots, etc.
        
    Raises:
        ValueError: If no configuration file exists for the year
    """
    from utils import build_season_data
    config = load_season_config(year)
    return build_season_data(config)


__all__ = ['load_season_config', 'load_season_data']
