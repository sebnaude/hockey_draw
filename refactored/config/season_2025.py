# config/season_2025.py
"""
Season 2025 Configuration.

This file contains all season-specific settings for 2025.
"""

from datetime import datetime, time as tm

# ============== Playing Fields ==============

FIELDS = [
    {'location': 'Newcastle International Hockey Centre', 'name': 'SF'},  # South Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'EF'},  # East Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'WF'},  # West Field
    {'location': 'Maitland Park', 'name': 'Maitland Main Field'},
    {'location': 'Central Coast Hockey Park', 'name': 'Wyong Main Field'},
]

# ============== Game Times by Venue/Day ==============

DAY_TIME_MAP = {
    'Newcastle International Hockey Centre': {
        'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    }
}

# ============== PHL-Specific Game Times ==============

PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'Friday': [tm(19, 0)], 
        'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
    },
    'Central Coast Hockey Park': {
        'Friday': [tm(20, 0)], 
        'Sunday': [tm(15, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    }
}

# ============== Field Unavailabilities ==============

FIELD_UNAVAILABILITIES = {
    'Maitland Park': {
        'weekends': [
            datetime(2025, 4, 19), datetime(2025, 4, 12), datetime(2025, 5, 10),
            datetime(2025, 5, 24), datetime(2025, 6, 28), datetime(2025, 5, 3), datetime(2025, 6, 7)
        ],
        'whole_days': [datetime(2025, 4, 25)],  # ANZAC Day
        'part_days': [],
    },
    'Newcastle International Hockey Centre': {
        'weekends': [datetime(2025, 4, 19), datetime(2025, 5, 3), datetime(2025, 6, 7)],
        'whole_days': [datetime(2025, 4, 25), datetime(2025, 5, 31)],
        'part_days': [
            datetime(2025, 6, 1, 8, 30), 
            datetime(2025, 6, 1, 10, 0), 
            datetime(2025, 6, 1, 11, 30)
        ],
    },
    'Central Coast Hockey Park': {
        'weekends': [datetime(2025, 4, 19), datetime(2025, 4, 5), datetime(2025, 5, 3), datetime(2025, 6, 7)],
        'whole_days': [datetime(2025, 4, 25)],
        'part_days': [],
    },
}

# ============== Club Days (Special Events) ==============

CLUB_DAYS = {
    'Crusaders': datetime(2025, 6, 22),
    'Wests': datetime(2025, 7, 13),
    'University': datetime(2025, 7, 27),
    'Tigers': datetime(2025, 7, 6),
    'Port Stephens': datetime(2025, 7, 20)
}

# ============== No-Play Preferences (Soft Constraints) ==============

PREFERENCE_NO_PLAY = {
    'Maitland': [
        {'date': '2025-07-20', 'field_location': 'Newcastle International Hockey Centre'},
        {'date': '2025-08-24', 'field_location': 'Newcastle International Hockey Centre'}
    ],
    'Norths': [
        {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '11:30'},
        {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '13:00'},
        {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '14:30'},
        {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '16:00'}
    ]
}

# ============== PHL Preferences ==============

PHL_PREFERENCES = {
    'preferred_dates': []  # Can add specific date preferences here
}

# ============== Season Configuration ==============

SEASON_CONFIG = {
    'year': 2025,
    'start_date': datetime(2025, 3, 21),
    'end_date': datetime(2025, 9, 2),
    'max_rounds': 21,
    'num_dummy_timeslots': 3,
    
    # Data paths
    'teams_data_path': 'data/2025/teams',
    'noplay_data_path': 'data/2025/noplay',
    'field_availability_path': 'data/2025/field_availability',
    
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
    
    # Home field mappings
    'home_field_map': {
        'Maitland': 'Maitland Park',
        'Gosford': 'Central Coast Hockey Park',
        # All others default to Newcastle International Hockey Centre
    },
    
    # Grade order (for adjacency constraints)
    'grade_order': ['PHL', '2nd', '3rd', '4th', '5th', '6th'],
}
