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
"""

def load_season_config(year: int) -> dict:
    """Load configuration for a specific season year."""
    if year == 2025:
        from .season_2025 import SEASON_CONFIG
        return SEASON_CONFIG
    elif year == 2026:
        from .season_2026 import SEASON_CONFIG
        return SEASON_CONFIG
    else:
        raise ValueError(f"No configuration found for season {year}")

__all__ = ['load_season_config']
