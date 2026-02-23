# config/season_2026.py
"""
Season 2026 Configuration.

This file contains all season-specific settings for 2026.
NOTE: Some values are TBC pending AGM decisions and further information.
"""

from datetime import datetime, time as tm

# ============== Playing Fields ==============
# Same as 2025 unless otherwise specified

FIELDS = [
    {'location': 'Newcastle International Hockey Centre', 'name': 'SF'},  # South Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'EF'},  # East Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'WF'},  # West Field
    {'location': 'Maitland Park', 'name': 'Maitland Main Field'},
    {'location': 'Central Coast Hockey Park', 'name': 'Wyong Main Field'},
]

# ============== Game Times by Venue/Day ==============
# Same as 2025 for Broadmeadow/Maitland, updated for Gosford

DAY_TIME_MAP = {
    'Newcastle International Hockey Centre': {
        'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    },
    'Central Coast Hockey Park': {
        'Sunday': [tm(12, 0), tm(13, 30)],  # Updated: 12pm or 1:30pm per request
    }
}

# ============== PHL-Specific Game Times ==============

PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'Friday': [tm(19, 0)],  # Friday night games continue
        'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
    },
    'Central Coast Hockey Park': {
        'Friday': [tm(18, 30), tm(20, 0)],  # Updated: 6:30pm or 8:00pm per request
        'Sunday': [tm(12, 0), tm(13, 30)]   # Updated: 12pm or 1:30pm
    },
    'Maitland Park': {
        'Sunday': [tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    }
}

# ============== Field Unavailabilities ==============
# TODO: Update with 2026 specific dates

FIELD_UNAVAILABILITIES = {
    'Maitland Park': {
        'weekends': [],  # TBC - add blocked weekends
        'whole_days': [],  # TBC - add ANZAC Day if applicable
        'part_days': [],
    },
    'Newcastle International Hockey Centre': {
        'weekends': [],  # TBC - 2 State Championship weekends (dates needed)
        'whole_days': [],  # TBC
        'part_days': [],
        # Note: NSW Masters Men's training August weekend - date TBC
    },
    'Central Coast Hockey Park': {
        'weekends': [],
        'whole_days': [],
        'part_days': [],
    },
}

# ============== Club Days (Special Events) ==============
# TBC for 2026

CLUB_DAYS = {
    # 'Crusaders': datetime(2026, X, X),
    # 'Wests': datetime(2026, X, X),
    # etc.
}

# ============== No-Play Preferences (Soft Constraints) ==============

PREFERENCE_NO_PLAY = {
    # TBC for 2026
}

# ============== PHL Preferences ==============

PHL_PREFERENCES = {
    'preferred_dates': [],
    
    # PHL/2nd back-to-back confirmed for 2026
    'phl_2nd_back_to_back': True,
    
    # Teams playing Gosford have 2nd grade bye (confirmed)
    'gosford_2nd_grade_bye': True,
}

# ============== Friday Night Configuration ==============
# Per request from Central Coast Hockey Association

FRIDAY_NIGHT_CONFIG = {
    # Gosford requested 8 home Friday night matches
    'gosford_friday_count': 8,
    
    # Clubs confirmed for Friday nights
    'friday_clubs': {
        'Wests': 2,       # 2 matches
        'Souths': 2,      # 2 matches
        'Norths': 1,      # 1 match
        'Tigers': 2,      # 2 matches
        'Maitland': 0,    # TBC
    },
    
    # Gosford Friday times
    'gosford_friday_times': [tm(18, 30), tm(20, 0)],  # 6:30pm or 8:00pm
}

# ============== Special Games ==============

SPECIAL_GAMES = {
    # Tigers vs Souths at Taree (PHL & 2nd Grade) - May, date TBC
    'taree_game': {
        'teams': ['Tigers', 'Souths'],
        'grades': ['PHL', '2nd'],
        'month': 5,  # May
        'date': None,  # TBC
    },
}

# ============== Season Configuration ==============

SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),   # Sunday 22nd March
    'end_date': datetime(2026, 9, 19),     # Saturday 19th September (Grand Final)
    
    # TBC: 3 rounds (15 matches) or 4 rounds (20 matches) - pending AGM
    'max_rounds': 20,  # Default to 4 rounds, change to 15 if 3 rounds approved
    'num_dummy_timeslots': 3,
    
    # Confirmed: Playing ANZAC weekend Sunday
    'play_anzac_sunday': True,
    
    # Data paths
    'teams_data_path': 'data/2026/teams',
    'noplay_data_path': 'data/2026/noplay',
    'field_availability_path': 'data/2026/field_availability',
    
    # Field definitions
    'fields': FIELDS,
    
    # Time configurations
    'day_time_map': DAY_TIME_MAP,
    'phl_game_times': PHL_GAME_TIMES,
    
    # Unavailabilities
    'field_unavailabilities': FIELD_UNAVAILABILITIES,
    
    # Club events
    'club_days': CLUB_DAYS,
    
    # Preferences
    'preference_no_play': PREFERENCE_NO_PLAY,
    'phl_preferences': PHL_PREFERENCES,
    
    # Friday night settings
    'friday_night_config': FRIDAY_NIGHT_CONFIG,
    
    # Special games
    'special_games': SPECIAL_GAMES,
    
    # Home field mappings
    'home_field_map': {
        'Maitland': 'Maitland Park',
        'Gosford': 'Central Coast Hockey Park',
        # All others default to Newcastle International Hockey Centre
    },
    
    # Grade order (for adjacency constraints)
    'grade_order': ['PHL', '2nd', '3rd', '4th', '5th', '6th'],
}


# ============== STILL NEEDED FOR 2026 ==============
"""
BLOCKERS (cannot generate draw without):
1. Team CSV files for 2026 (data/2026/teams/*.csv)
2. State Championship weekend dates (2 weekends blocked)
3. AGM decision: 3 rounds (15 matches) or 4 rounds (20 matches)
4. Maitland Friday night confirmation

CAN ADD LATER:
5. NSW Masters Men's training weekend (August)
6. Tigers/Souths Taree game date (May)
7. Specific Friday night dates for Gosford
8. Club day dates
"""
