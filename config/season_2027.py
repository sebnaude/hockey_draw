# config/season_2027.py
"""
Season 2027 Configuration.

IMPORTANT: This is a BLANK TEMPLATE. All values must be explicitly set.
DO NOT copy values from previous seasons without verification.

This file contains all season-specific settings for 2027.
"""

from datetime import datetime, time as tm

# ============== Playing Fields ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

FIELDS = [
    # Example format - DELETE and replace with actual 2027 fields:
    # {'location': 'Newcastle International Hockey Centre', 'name': 'SF'},
    # {'location': 'Newcastle International Hockey Centre', 'name': 'EF'},
    # {'location': 'Newcastle International Hockey Centre', 'name': 'WF'},
    # {'location': 'Maitland Park', 'name': 'Maitland Main Field'},
    # {'location': 'Central Coast Hockey Park', 'name': 'Wyong Main Field'},
]

# ============== Game Times by Venue/Day ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

DAY_TIME_MAP = {
    # Example format - DELETE and replace with actual 2027 times:
    # 'Newcastle International Hockey Centre': {
    #     'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
    # },
    # 'Maitland Park': {
    #     'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    # },
    # 'Central Coast Hockey Park': {
    #     'Sunday': [tm(12, 0), tm(13, 30)],
    # }
}

# ============== PHL-Specific Game Times ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

PHL_GAME_TIMES = {
    # Example format - DELETE and replace with actual 2027 PHL times:
    # 'Newcastle International Hockey Centre': {
    #     'Friday': [tm(19, 0)],
    #     'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
    # },
    # 'Central Coast Hockey Park': {
    #     'Friday': [tm(18, 30), tm(20, 0)],
    #     'Sunday': [tm(12, 0), tm(13, 30)]
    # },
    # 'Maitland Park': {
    #     'Sunday': [tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    # }
}

# ============== Field Unavailabilities ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year
# Blocked weekends/days where NO games can be scheduled (HARD constraints)

FIELD_UNAVAILABILITIES = {
    # Example format - DELETE and replace with actual 2027 unavailabilities:
    # 'Maitland Park': {
    #     'weekends': [
    #         # datetime(2027, 4, 3),   # Example: Easter weekend
    #     ],
    #     'whole_days': [
    #         # datetime(2027, 4, 25),  # Example: ANZAC Day
    #     ],
    #     'part_days': [],
    # },
    # 'Newcastle International Hockey Centre': {
    #     'weekends': [],
    #     'whole_days': [],
    #     'part_days': [],
    # },
    # 'Central Coast Hockey Park': {
    #     'weekends': [],
    #     'whole_days': [],
    #     'part_days': [],
    # },
}

# ============== Club Days (Special Events) ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

CLUB_DAYS = {
    # Example format - DELETE and replace with actual 2027 club days:
    # 'Crusaders': datetime(2027, 6, 13),
    # 'Wests': datetime(2027, 7, 11),
}

# ============== No-Play Preferences (Soft Constraints) ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

PREFERENCE_NO_PLAY = {
    # Example format - DELETE and replace with actual 2027 preferences:
    # 'Crusaders_6th_Masters': {
    #     'club': 'Crusaders',
    #     'grade': '6th',
    #     'dates': [datetime(2027, 4, 16), datetime(2027, 4, 17), datetime(2027, 4, 18)],
    #     'reason': 'NSW Masters Men\'s State Championships',
    # },
}

# ============== PHL Preferences ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

PHL_PREFERENCES = {
    'preferred_dates': [],
    'phl_2nd_back_to_back': False,  # Set to True if confirmed for 2027
    'gosford_2nd_grade_bye': False,  # Set to True if confirmed for 2027
}

# ============== Friday Night Configuration ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

FRIDAY_NIGHT_CONFIG = {
    'gosford_friday_count': 0,  # Number of Friday night home games for Gosford
    'friday_clubs': {
        # 'Wests': 0,
        # 'Souths': 0,
        # 'Norths': 0,
        # 'Tigers': 0,
        # 'Maitland': 0,
    },
    'friday_dates': [
        # datetime(2027, 3, 26),  # Example dates
    ],
    'gosford_friday_times': [],  # e.g., [tm(20, 0)]
    'nihc_friday_times': [],     # e.g., [tm(19, 0)]
}

# ============== Special Games ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

SPECIAL_GAMES = {
    # Example format:
    # 'taree_game': {
    #     'teams': ['Tigers', 'Souths'],
    #     'grades': ['PHL', '2nd'],
    #     'month': 5,
    #     'date': None,  # TBC
    # },
}

# ============== Season Configuration ==============
# MUST BE EXPLICITLY SET - Do not inherit from previous year

SEASON_CONFIG = {
    'year': 2027,
    
    # CRITICAL: These dates MUST be set for 2027
    'start_date': None,           # e.g., datetime(2027, 3, 21) - First round Sunday
    'last_round_date': None,      # e.g., datetime(2027, 8, 29) - Last regular round Sunday
    'end_date': None,             # e.g., datetime(2027, 9, 18) - Grand Final date
    
    # CRITICAL: Verify rounds calculation matches available weekends
    'max_rounds': 0,              # Must match: (Sundays between start and last_round) - blocked_weekends
    'num_dummy_timeslots': 3,
    
    # Confirm at AGM
    'play_anzac_sunday': False,   # Set to True if playing ANZAC weekend
    
    # Data paths - update year
    'teams_data_path': 'data/2027/teams',
    'noplay_data_path': 'data/2027/noplay',
    'field_availability_path': 'data/2027/field_availability',
    
    # Field definitions - reference the FIELDS list above
    'fields': FIELDS,
    
    # Time configurations - reference the dicts above
    'day_time_map': DAY_TIME_MAP,
    'phl_game_times': PHL_GAME_TIMES,
    
    # Unavailabilities - reference the dict above
    'field_unavailabilities': FIELD_UNAVAILABILITIES,
    
    # Club events - reference the dict above
    'club_days': CLUB_DAYS,
    
    # Preferences - reference the dicts above
    'preference_no_play': PREFERENCE_NO_PLAY,
    'phl_preferences': PHL_PREFERENCES,
    
    # Friday night settings - reference the dict above
    'friday_night_config': FRIDAY_NIGHT_CONFIG,
    
    # Special games - reference the dict above
    'special_games': SPECIAL_GAMES,
    
    # Home field mappings - MUST BE EXPLICITLY SET
    'home_field_map': {
        # 'Maitland': 'Maitland Park',
        # 'Gosford': 'Central Coast Hockey Park',
        # All others default to Newcastle International Hockey Centre
    },
    
    # Grade order (typically consistent year to year)
    'grade_order': ['PHL', '2nd', '3rd', '4th', '5th', '6th'],
}


def get_season_data() -> dict:
    """
    Build complete data dictionary for the 2027 season.
    
    WARNING: This will fail if SEASON_CONFIG is not properly populated.
    Ensure all required fields are set before running.
    
    Returns:
        Complete data dict ready for solver
    """
    # Validation checks
    if SEASON_CONFIG['start_date'] is None:
        raise ValueError("2027 SEASON_CONFIG['start_date'] is not set!")
    if SEASON_CONFIG['last_round_date'] is None:
        raise ValueError("2027 SEASON_CONFIG['last_round_date'] is not set!")
    if SEASON_CONFIG['end_date'] is None:
        raise ValueError("2027 SEASON_CONFIG['end_date'] is not set!")
    if SEASON_CONFIG['max_rounds'] == 0:
        raise ValueError("2027 SEASON_CONFIG['max_rounds'] is not set!")
    if not FIELDS:
        raise ValueError("2027 FIELDS list is empty!")
    if not DAY_TIME_MAP:
        raise ValueError("2027 DAY_TIME_MAP is empty!")
        
    from utils import build_season_data
    return build_season_data(SEASON_CONFIG)


# ============== CHECKLIST FOR 2027 SEASON SETUP ==============
"""
Before using this config, ensure ALL items are completed:

□ FIELDS - List all playing fields
□ DAY_TIME_MAP - Set game times for each venue/day
□ PHL_GAME_TIMES - Set PHL-specific times
□ FIELD_UNAVAILABILITIES - Block all unavailable weekends/days
□ CLUB_DAYS - Set all club day dates
□ PREFERENCE_NO_PLAY - Add all soft no-play constraints
□ PHL_PREFERENCES - Set PHL configuration
□ FRIDAY_NIGHT_CONFIG - Set Friday night games config
□ SPECIAL_GAMES - Add any special venue games
□ SEASON_CONFIG dates - Set start_date, last_round_date, end_date
□ SEASON_CONFIG max_rounds - Verify calculation matches available weekends
□ home_field_map - Set club home fields

□ Create data/2027/teams/ folder with club CSV files
□ Create data/2027/noplay/ folder (if using Excel files)
□ Create data/2027/field_availability/ folder (if needed)
□ Create reports/2027_club_requests.md to track requests

Run: python run.py preseason --year 2027
This will validate the configuration and show any issues.
"""
