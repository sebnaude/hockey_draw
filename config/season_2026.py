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
        'Friday': [tm(19, 0)],  # Friday night games at NIHC - 7pm
        'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]  # Standard PHL times
    },
    'Central Coast Hockey Park': {
        'Friday': [tm(18, 30), tm(20, 0)],  # Gosford: 6:30pm or 8:00pm (8pm confirmed at AGM)
        'Sunday': [tm(12, 0), tm(13, 30)]   # Gosford: 12pm or 1:30pm ONLY
    },
    'Maitland Park': {
        'Sunday': [tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    }
}

# ============== PHL State Championship Weekend Slots ==============
# SC weekends are blocked for regular games, but PHL can use "back end" (Sunday afternoon)
# These dates override the blocked weekends for PHL grade ONLY
# 3 slots per SC weekend: 2 on East Field, 1 on West Field (can adjust times later)

PHL_SC_WEEKEND_SLOTS = {
    # May 17, 2026 (Sunday) - NSW Masters State Championships weekend
    datetime(2026, 5, 17): {
        'EF': [tm(14, 30), tm(16, 0)],   # East Field - 2 PHL games (2:30pm, 4pm)
        'WF': [tm(14, 30)],              # West Field - 1 PHL game (2:30pm)
    },
    # June 21, 2026 (Sunday) - Girls U16's State Championships weekend  
    datetime(2026, 6, 21): {
        'EF': [tm(14, 30), tm(16, 0)],   # East Field - 2 PHL games (2:30pm, 4pm)
        'WF': [tm(14, 30)],              # West Field - 1 PHL game (2:30pm)
    },
}

# ============== Field Unavailabilities ==============
# Blocked weekends/days where NO games can be scheduled (HARD constraints)
# Email states: "20 playing weekends" from Mar 22 to Aug 30
# 24 total Sundays - 4 blocked = 20 available

FIELD_UNAVAILABILITIES = {
    'Maitland Park': {
        'weekends': [
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            datetime(2026, 5, 16),  # May 15-17 (Masters SC Newcastle)
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
            datetime(2026, 6, 20),  # Jun 19-21 (U16 Girls SC Newcastle)
        ],
        'whole_days': [datetime(2026, 4, 25)],  # ANZAC Day (Saturday)
        'part_days': [],
    },
    'Newcastle International Hockey Centre': {
        'weekends': [
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            datetime(2026, 5, 16),  # May 15-17 (Masters SC Newcastle)
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
            datetime(2026, 6, 20),  # Jun 19-21 (U16 Girls SC Newcastle)
        ],
        'whole_days': [datetime(2026, 4, 25)],  # ANZAC Day (Saturday)
        'part_days': [],
    },
    'Central Coast Hockey Park': {
        'weekends': [
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            datetime(2026, 5, 16),  # May 15-17 (Masters SC Newcastle)
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
            datetime(2026, 6, 20),  # Jun 19-21 (U16 Girls SC Newcastle)
        ],
        'whole_days': [datetime(2026, 4, 25)],  # ANZAC Day (Saturday)
        'part_days': [],
    },
}

# ============== Club Days (Special Events) ==============

CLUB_DAYS = {
    'Crusaders': datetime(2026, 6, 14),  # All 4 teams back-to-back on same field
    # 'Souths_Norths_Derby': TBD,  # Red & Blue Derby Day - August, date TBC
}

# ============== No-Play Preferences (Soft Constraints) ==============

PREFERENCE_NO_PLAY = {
    # Crusaders 6th Grade - Masters State Championships
    'Crusaders_6th_Masters_Moorebank': {
        'club': 'Crusaders',
        'grade': '6th',
        'dates': [datetime(2026, 4, 17), datetime(2026, 4, 18), datetime(2026, 4, 19)],
        'reason': 'NSW Masters Men\'s at Moorebank',
    },
    'Crusaders_6th_Masters_Tamworth': {
        'club': 'Crusaders',
        'grade': '6th',
        'dates': [datetime(2026, 6, 26), datetime(2026, 6, 27), datetime(2026, 6, 28)],
        'reason': 'NSW Masters Men\'s at Tamworth',
    },
    # Souths PHL/2nd - U18's State Championships
    'Souths_U18_SC': {
        'club': 'Souths',
        'grades': ['PHL', '2nd'],
        'dates': [datetime(2026, 5, 24)],  # Sunday, includes Friday night before
        'reason': 'U18\'s State Championships',
    },
    # Gosford - Weekend after Men's SC
    'Gosford_Post_SC': {
        'club': 'Gosford',
        'dates': [datetime(2026, 6, 21)],  # Weekend after June 14 Men's SC
        'reason': 'Recovery weekend after Men\'s State Championships',
    },
}

# ============== PHL Preferences ==============

PHL_PREFERENCES = {
    'preferred_dates': [],
    
    # PHL/2nd back-to-back confirmed for 2026
    'phl_2nd_back_to_back': True,
    
    # Teams playing Gosford have 2nd grade bye (confirmed)
    'gosford_2nd_grade_bye': True,
}

# ============== PHL SCHEDULE SUMMARY ==============
# 
# FRIDAY NIGHTS AT GOSFORD (Central Coast):
#   - Confirmed dates: Mar 27, Apr 17, Apr 24, May 29, Jun 12
#   - Times: 8:00pm (confirmed at AGM)
#   - Clubs agreed to play: Wests x2, Souths x2, Norths x1, Tigers x2, Maitland x1
#   - Special: June 12 = Norths 80th Anniversary
#
# FRIDAY NIGHTS AT NIHC (Newcastle):
#   - Time: 7:00pm
#   - Aligned with Junior Boys program
#   - Dates TBC
#
# SUNDAY AT GOSFORD:
#   - Times: 12:00pm or 1:30pm ONLY
#   - Used when Friday night not available
#
# SUNDAY AT NIHC/MAITLAND:
#   - Standard PHL times apply
#   - PHL & 2nd grade run back-to-back
#
# STATE CHAMPIONSHIP WEEKENDS (PHL can play at back end):
#   - May 15-17: Masters SC (Newcastle) - can schedule PHL Sunday afternoon
#   - Jun 19-21: U16 Girls SC (Newcastle) - can schedule PHL Sunday afternoon
#   Note: Currently blocked entirely - need to add part_days if using back end
#

# ============== Friday Night Configuration ==============
# Per request from Central Coast Hockey Association (confirmed at AGM)

FRIDAY_NIGHT_CONFIG = {
    # Gosford requested 8 home Friday night matches - CONFIRMED at AGM
    'gosford_friday_count': 8,
    
    # Clubs confirmed for Friday nights at 8pm start (total = 8)
    'friday_clubs': {
        'Wests': 2,       # 2 matches - CONFIRMED
        'Souths': 2,      # 2 matches - CONFIRMED
        'Norths': 1,      # 1 match - June 12 (80th anniversary)
        'Tigers': 2,      # 2 matches - CONFIRMED
        'Maitland': 1,    # 1 match - CONFIRMED (happy to play both vs Gosford Friday)
    },
    
    # Confirmed Friday night dates at Gosford
    'friday_dates': [
        datetime(2026, 3, 27),   # March 27
        datetime(2026, 4, 17),   # April 17
        datetime(2026, 4, 24),   # April 24
        datetime(2026, 5, 29),   # May 29
        datetime(2026, 6, 12),   # June 12 - Norths 80th Anniversary
        # Need 3 more dates for 8 total
    ],
    
    # Gosford Friday times - 8pm confirmed for away teams
    'gosford_friday_times': [tm(20, 0)],  # 8:00pm confirmed at AGM
    
    # NIHC Friday night time
    'nihc_friday_times': [tm(19, 0)],  # 7:00pm
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
    'last_round_date': datetime(2026, 8, 30),  # Sunday 30th August (last regular round)
    'end_date': datetime(2026, 9, 19),     # Saturday 19th September (Grand Final)
    
    # CONFIRMED at AGM: 4 rounds (20 matches) - NOT reducing to 3 rounds
    # 24 Sundays from Mar 22 to Aug 30, minus 4 blocked weekends = 20 playing weekends
    'max_rounds': 20,  # 4 rounds confirmed
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
    'phl_sc_weekend_slots': PHL_SC_WEEKEND_SLOTS,  # SC weekend PHL-only slots
    
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


def get_season_data() -> dict:
    """
    Build complete data dictionary for the 2026 season.
    
    This loads teams from CSV files, generates timeslots, and builds
    all data structures needed by the solver.
    
    Returns:
        Complete data dict ready for solver
    """
    from utils import build_season_data
    return build_season_data(SEASON_CONFIG)


# ============== STILL NEEDED FOR 2026 ==============
"""
COMPLETED:
✅ Team CSV files updated (except Norths - awaiting nomination)
✅ AGM decision: 4 rounds (20 matches) confirmed
✅ Maitland Friday night: 1 match confirmed
✅ Friday night dates locked in
✅ Crusaders club day: June 14

BLOCKERS:
1. Norths team nominations - MISSING

STATE CHAMPIONSHIPS (PHL can play at end):
- May 15-17: NSW Masters (Newcastle)
- June 19-21: Girls U16's (Newcastle)

CAN ADD LATER:
1. NSW Masters Men's training weekend (August) - date TBC
2. Tigers/Souths Taree game exact date (May)
3. Red & Blue Derby date (August) - Souths/Norths to confirm
4. August catch-up weekend for wet weather deferrals
"""
