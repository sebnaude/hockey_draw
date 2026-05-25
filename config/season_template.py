# config/season_YYYY.py
"""
Season YYYY Configuration Template.

INSTRUCTIONS:
1. Copy this file to config/season_YYYY.py (e.g., season_2027.py)
2. Replace all YYYY with the actual year
3. Update all dates, paths, and settings for your season
4. Ensure team CSV files exist in data/YYYY/teams/
5. The system will automatically detect and load the new season

This file contains all season-specific settings including:
- Season start/end dates
- Playing field definitions  
- Game times by venue/day
- Field unavailabilities (blocked dates)
- Club days (special events)
- No-play preferences (soft constraints)
- PHL-specific configurations
"""

from datetime import datetime, time as tm
from config.defaults import (
    FIELDS, DAY_TIME_MAP, HOME_FIELD_MAP, GRADE_ORDER,
    PERENNIAL_BLOCKED_GAMES,
)

# ============== Playing Fields ==============
# Imported from config/defaults.py (same every year).
# Override here only if venues change for this season.
# FIELDS = [...]

# ============== Game Times by Venue/Day ==============
# Imported from config/defaults.py (same every year).
# Override here only if game times change for this season.
# DAY_TIME_MAP = {...}

# ============== PHL-Specific Game Times ==============
# PHL grade can have different times including Friday nights

PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'Friday': [tm(19, 0)],  # Friday night games
        'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
    },
    'Central Coast Hockey Park': {
        'Friday': [tm(20, 0)],  # 8pm start for away teams
        'Sunday': [tm(15, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    }
}

# ============== Field Unavailabilities ==============
# Blocked weekends/days where NO games can be scheduled (HARD constraints)
# 
# For each venue, specify:
# - weekends: List of Saturdays - the entire Fri-Sun weekend will be blocked
# - whole_days: Specific days that are fully blocked
# - part_days: Specific timeslots that are blocked (datetime with time)

FIELD_UNAVAILABILITIES = {
    'Maitland Park': {
        'weekends': [
            # datetime(YYYY, M, D),  # Description of why blocked
        ],
        'whole_days': [
            # datetime(YYYY, 4, 25),  # ANZAC Day
        ],
        'part_days': [],
    },
    'Newcastle International Hockey Centre': {
        'weekends': [
            # datetime(YYYY, M, D),  # State Championships etc.
        ],
        'whole_days': [
            # datetime(YYYY, 4, 25),  # ANZAC Day
        ],
        'part_days': [
            # datetime(YYYY, 6, 1, 8, 30),  # Specific timeslot blocked
        ],
    },
    'Central Coast Hockey Park': {
        'weekends': [],
        'whole_days': [],
        'part_days': [],
    },
}

# ============== Club Days (Special Events) ==============
# Each club may have a "club day" where all their teams play back-to-back
# at the same venue. Specify the date of each club's event.

CLUB_DAYS = {
    # 'ClubName': datetime(YYYY, M, D),
}

# ============== Blocked Games (Hard No-Play) ==============
# These are HARD constraints - variables are completely removed from the
# game dictionary, making it impossible for the solver to schedule these games.
#
# Use for requests where a team MUST NOT play on a specific date
# (e.g., representative commitments, post-tournament recovery).
#
# Each entry can specify:
# - club: Affects all teams from that club
# - grade/grades: Only affects teams in that grade(s)
# - teams: Specific team names (overrides club resolution)
# - dates: List of date strings ('YYYY-MM-DD') to block
# - reason: Documentation of why this is blocked
#
# See docs/ai/CONFIGURATION_REFERENCE.md for full field reference.

BLOCKED_GAMES = [
    # --- Perennial rules (from config/defaults.py) ---
    # Rounds 1-2 at Broadmeadow only, etc. See docs/PERENNIAL_RULES.md.
    *PERENNIAL_BLOCKED_GAMES,

    # --- Season-specific blocks below ---
    # {
    #     'club': 'ClubName',
    #     'grade': '6th',
    #     'dates': ['YYYY-MM-DD'],
    #     'reason': 'Representative tournament',
    # },
    # {
    #     'club': 'ClubName',
    #     'grades': ['PHL', '2nd'],
    #     'dates': ['YYYY-MM-DD'],
    #     'reason': 'U18 State Championships',
    # },
    # {
    #     'club': 'Gosford',  # Blocks all Gosford teams
    #     'dates': ['YYYY-MM-DD'],
    #     'reason': 'Post-tournament recovery',
    # },
]

# ============== No-Play Preferences (Soft Constraints) ==============
# These are SOFT constraints - the solver will try to avoid scheduling
# these games but may if necessary.
#
# Each entry can specify:
# - club: Affects all teams from that club
# - grade: Only affects teams in that grade  
# - team_name: Specific team (e.g., 'Maitland PHL')
# - dates: List of datetimes to avoid
# - reason: Documentation of why this preference exists

PREFERENCE_NO_PLAY = {
    # 'Unique_Key': {
    #     'club': 'ClubName',
    #     'grade': '6th',  # Optional
    #     'dates': [datetime(YYYY, M, D), datetime(YYYY, M, D)],
    #     'reason': 'NSW Masters Championships',
    # },
}

# ============== Preferred Games (spec-020) ==============
# Soft, weighted FORCED_GAMES analogue (same scope/team/club grammar + optional
# `weight`). Penalty-on-deviation from `count` per `constraint` type. Replaces
# the deleted PHL_PREFERENCES / PreferredDates. Empty = no preferences.
# Example marquee-PHL-date entry:
#   {'grade': 'PHL', 'date': '2027-04-18', 'constraint': 'equal', 'count': 1,
#    'weight': 10000, 'description': 'marquee PHL date'}
# See docs/system/FORCED_GAMES_AS_COUNT_RULES.md.
PREFERRED_GAMES = []

# spec-025: pin pairing to its weekend, free the time. Empty = no pins.
LOCKED_PAIRINGS = []

# ============== Season Configuration ==============
# Main configuration dictionary that pulls everything together

SEASON_CONFIG = {
    # Basic season info
    'year': 9999,  # CHANGE THIS to actual year
    'start_date': datetime(9999, 3, 22),   # First playing day (Sunday)
    'end_date': datetime(9999, 9, 19),     # Last club game before finals
    
    # Schedule parameters
    'max_rounds': 20,  # Maximum number of rounds (4 rounds = 20 matches per team)
    'num_dummy_timeslots': 3,  # For solver flexibility
    
    # Special settings
    'play_anzac_sunday': True,  # Whether to schedule games on ANZAC Sunday
    
    # Data paths (relative to project root)
    'teams_data_path': 'data/9999/teams',  # CHANGE to actual year
    'noplay_data_path': 'data/9999/noplay',
    'field_availability_path': 'data/9999/field_availability',
    
    # Field definitions
    'fields': FIELDS,
    
    # Time configurations
    'day_time_map': DAY_TIME_MAP,
    'phl_game_times': PHL_GAME_TIMES,
    
    # Unavailabilities
    'field_unavailabilities': FIELD_UNAVAILABILITIES,
    
    # Club events
    'club_days': CLUB_DAYS,
    
    # No-play rules
    'blocked_games': BLOCKED_GAMES,
    'preference_no_play': PREFERENCE_NO_PLAY,
    'preferred_games': PREFERRED_GAMES,  # spec-020 soft FORCED analogue
    'locked_pairings': LOCKED_PAIRINGS,  # spec-025 pin pairing to weekend

    # Home field mappings (from config/defaults.py, override if needed)
    'home_field_map': HOME_FIELD_MAP,

    # Grade order (from config/defaults.py, override if needed)
    'grade_order': GRADE_ORDER,

    # Base limits for slack-aware constraints (see CONSTRAINT_DEFAULTS in season_2026.py for docs)
    'constraint_defaults': {
        'spacing_base_slack': 0,
        # spec-033 Unit B: bye-spacing base slack. Default 2 keeps the hard
        # floor 2 rounds below the raw ideal_bye_gap (avoids infeasibility);
        # the soft term pushes byes toward the ideal spread.
        'bye_spacing_base_slack': 2,
        # spec-018: maitland_max_consecutive_home / away_maitland_max_clubs
        # removed — the venue-sequencing rules that read them were deleted.
        # spec-024: max_clubs_per_field removed with MinimiseClubsOnAFieldBroadmeadow.
        'club_game_spread_max_gap': 1,
        # club_game_spread_max_overlap: REMOVED — now dynamic per club: T//2 - 1 where T = team count
        # spec-033 Unit A: club_vs_club_alignment_base_slack removed — alignment is a fixed hard rule with no slack.
    },
}


def get_season_data() -> dict:
    """
    Build complete data dictionary for this season.
    
    This loads teams from CSV files, generates timeslots, and builds
    all data structures needed by the solver.
    
    Returns:
        Complete data dict ready for solver
    """
    from utils import build_season_data
    return build_season_data(SEASON_CONFIG)
