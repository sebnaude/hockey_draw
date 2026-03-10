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

# ============== PHL Game Variable Generation Dictionary ==============
# THIS DICT CONTROLS WHICH GAME VARIABLES ARE CREATED FOR PHL
# Only timeslots matching (venue, field, day, time) tuples here will have PHL vars generated.
# This dramatically reduces solver variables - PHL can ONLY play at these specific slots.
#
# Key rules:
# - PHL cannot play on South Field (SF) at NIHC - only EF and WF
# - Gosford: 1 slot per week max (away venue) - Friday OR Sunday, not both
# - PHL times are restricted to specific windows (not early morning/late evening)
#
# Structure: { venue: { field: { day: [times] } } }

PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {  # East Field only (no SF for PHL)
            'Friday': [tm(19, 0)],  # 7pm Friday night
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        'WF': {  # West Field only (no SF for PHL)
            'Friday': [tm(19, 0)],  # 7pm Friday night  
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        # NOTE: SF (South Field) deliberately excluded - PHL cannot play there
    },
    'Central Coast Hockey Park': {
        'Wyong Main Field': {
            # Gosford: 1 game per week max (away venue)
            # Friday 8pm (confirmed at AGM) OR Sunday 12pm/1:30pm
            'Friday': [tm(20, 0)],  # 8pm only (AGM confirmed)
            'Sunday': [tm(12, 0), tm(13, 30)]  # 12pm or 1:30pm ONLY
        },
    },
    'Maitland Park': {
        'Maitland Main Field': {
            'Sunday': [tm(12, 0), tm(13, 0), tm(15, 0), tm(16, 30)]
        },
    },
}

# ============== 2nd Grade Game Variable Generation Dictionary ==============
# THIS DICT CONTROLS WHICH GAME VARIABLES ARE CREATED FOR 2ND GRADE
# Only venues listed here will have 2nd grade variables created.
#
# Key rules:
# - Cannot play on South Field (SF) at NIHC - only EF and WF listed
# - Gosford not listed (PHL-only venue)
# - Times: PHL slots PLUS one slot before/after (where available in DAY_TIME_MAP)
#
# IMPORTANT: Cannot create NEW timeslots - only existing DAY_TIME_MAP slots.
#
# Structure: { venue: { field: { day: [times] } } }

SECOND_GRADE_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {  # East Field (no SF)
            # PHL times + 10:00 (before) + 17:30 (after)
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
        'WF': {  # West Field (no SF)
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
    },
    # Gosford not listed - PHL-only venue
    'Maitland Park': {
        'Maitland Main Field': {
            # PHL: 12:00, 13:00, 15:00, 16:30
            # +10:30 (before 12:00), no slot after 16:30 exists in DAY_TIME_MAP
            'Sunday': [tm(10, 30), tm(12, 0), tm(13, 0), tm(15, 0), tm(16, 30)]
        },
    },
}

# ============== Field Unavailabilities ==============
# Blocked weekends/days where NO games can be scheduled (HARD constraints)
# Email states: "20 playing weekends" from Mar 22 to Aug 30
# 24 total Sundays - 4 blocked = 20 available (BUT PHL has 22 - see below)
#
# IMPORTANT: State Championship weekends (May 15-17, Jun 19-21) are special:
# - Friday night at Gosford is available for PHL (rescues the weekend)
# - Sunday afternoon (12pm+) at NIHC is available for PHL ONLY
# - All other grades are fully blocked for these weekends
# - This gives PHL 22 available weekends (20 regular + 2 rescued)

FIELD_UNAVAILABILITIES = {
    'Maitland Park': {
        'weekends': [
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            datetime(2026, 4, 11),  # Apr 10-12 (U14 State Champs at Maitland)
            datetime(2026, 5, 16),  # May 15-17 (Masters SC Newcastle) - fully blocked
            datetime(2026, 5, 23),  # May 22-24 (U18 State Champs at Maitland)
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
            datetime(2026, 6, 20),  # Jun 19-21 (U16 Girls SC Newcastle) - fully blocked
            datetime(2026, 7, 18),  # Jul 17-19 (U16 State Champs at Maitland)
        ],
        'whole_days': [
            datetime(2026, 4, 25),  # ANZAC Day (Saturday)
            datetime(2026, 4, 19),  # Steamfest Maitland (Sunday)
            datetime(2026, 6, 7),   # Maitland Running Festival (Sunday)
        ],
        'part_days': [],
    },
    'Newcastle International Hockey Centre': {
        'weekends': [
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            # NOTE: State championship weekends NOT listed here - handled via part_days
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
        ],
        'whole_days': [
            datetime(2026, 4, 25),  # ANZAC Day (Saturday)
            # State championship Fridays - daytime blocked but evening available for PHL
            # (5/15 and 6/19 evenings can host PHL games - handled via part_days)
            datetime(2026, 5, 16),  # Masters SC Saturday
            datetime(2026, 6, 20),  # U16 Girls SC Saturday
        ],
        'part_days': [
            # State Championship Fridays - block daytime only (allow 7pm PHL game)
            datetime(2026, 5, 15, 8, 30),   # Masters SC Friday morning
            datetime(2026, 5, 15, 10, 0),
            datetime(2026, 5, 15, 11, 30),
            datetime(2026, 5, 15, 13, 0),
            datetime(2026, 5, 15, 14, 30),
            datetime(2026, 5, 15, 16, 0),
            datetime(2026, 5, 15, 17, 30),  # Block up to 5:30pm, allow 7pm
            datetime(2026, 6, 19, 8, 30),   # U16 Girls SC Friday morning
            datetime(2026, 6, 19, 10, 0),
            datetime(2026, 6, 19, 11, 30),
            datetime(2026, 6, 19, 13, 0),
            datetime(2026, 6, 19, 14, 30),
            datetime(2026, 6, 19, 16, 0),
            datetime(2026, 6, 19, 17, 30),  # Block up to 5:30pm, allow 7pm
            # State Championship Sundays - block MORNING slots only (allow afternoon for PHL)
            # May 17 (Masters SC Sunday morning)
            datetime(2026, 5, 17, 8, 30),
            datetime(2026, 5, 17, 10, 0),
            datetime(2026, 5, 17, 11, 30),
            # Jun 21 (U16 Girls SC Sunday morning)
            datetime(2026, 6, 21, 8, 30),
            datetime(2026, 6, 21, 10, 0),
            datetime(2026, 6, 21, 11, 30),
        ],
    },
    'Central Coast Hockey Park': {
        'weekends': [
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            # NOTE: State championship weekends - Friday is available for PHL
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
        ],
        'whole_days': [
            datetime(2026, 4, 25),  # ANZAC Day (Saturday)
            # State championship Saturdays and Sundays (but NOT Fridays - PHL plays there)
            datetime(2026, 5, 16),  # Masters SC Saturday
            datetime(2026, 5, 17),  # Masters SC Sunday
            datetime(2026, 6, 20),  # U16 Girls SC Saturday
            datetime(2026, 6, 21),  # U16 Girls SC Sunday
        ],
        'part_days': [],
    },
}

# ============== PHL-Only Afternoon Dates ==============
# Dates where Sunday afternoon (12pm+) is available for PHL ONLY at NIHC
# Other grades are blocked from these timeslots via generate_X() filtering
# These are the "rescued" state championship weekends
PHL_ONLY_AFTERNOON_DATES = [
    datetime(2026, 5, 17).date(),  # Masters SC Sunday - PHL can play PM at NIHC
    datetime(2026, 6, 21).date(),  # U16 Girls SC Sunday - PHL can play PM at NIHC
]

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
# Note: PHL_PREFERENCES only supports 'preferred_dates' key for the constraint system.
# Other settings are documented in comments below.
#
# Additional PHL Rules (enforced via constraints, not this dict):
#   - PHL/2nd back-to-back: CONFIRMED for 2026
#   - Teams playing Gosford have 2nd grade bye: CONFIRMED

PHL_PREFERENCES = {
    'preferred_dates': [],  # Add specific date preferences here if needed
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
    
    # NIHC (Broadmeadow) Friday night games - SPECIFIC matchups only
    # Only these exact matchups are allowed on Friday nights at NIHC
    # Date format: 'YYYY-MM-DD' -> list of allowed club pairs (alphabetical order)
    # For dates with 'Norths', only matchups INCLUDING Norths are allowed
    'nihc_friday_games': {
        '2026-05-08': [('Maitland', 'Souths')],       # Souths vs Maitland
        '2026-06-19': [('Tigers', 'Wests')],          # Tigers vs Wests
        '2026-07-24': 'norths_only',                   # Norths vs TBC (any opponent)
    },
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

# ============== Maximum Available Weekends Per Grade ==============
# These are the MAXIMUM AVAILABLE WEEKENDS a grade can play (hard ceiling).
# This is NOT the same as actual rounds played - that's calculated from team count.
#
# ⚠️ CRITICAL: Friday nights are NOT additional weekends!
# Friday games at Gosford are PART OF that weekend (team plays Friday OR Sunday, not both).
# PHL gets extra weekends because Friday at Gosford can "rescue" weekends that are 
# otherwise blocked for Sunday play (e.g., State Championships on Saturday/Sunday).
# 
# Calculation for 2026:
# - 24 total Sundays (Mar 22 - Aug 30)
# - 4 blocked weekends (Easter, 2x State Championships, 1x blocked)
# - = 20 available Sundays
# - PHL: 2 additional weekends rescued via Friday at Gosford = 22 total

MAX_WEEKENDS_PER_GRADE = {
    'PHL': 22,   # 20 Sundays + 2 rescued via Friday (State Champ weekends)
    '2nd': 20,   # 20 Sundays only
    '3rd': 20,   # 20 Sundays only
    '4th': 20,   # 20 Sundays only
    '5th': 20,   # 20 Sundays only
    '6th': 20,   # 20 Sundays only
}

# ============== Fill All Weekends (Formula Selection) ==============
# Determines how actual played rounds are calculated from max weekends.
#
# False = Formula 1 (Strict Equal Matchups):
#   - Every matchup occurs exactly the same number of times
#   - May result in "no-play" weekends where teams have byes
#   - Formula: floor(weekends/(T-1)) × (T-1) games per team
#
# True = Formula 2 (Fill All Weekends):
#   - Play every available weekend (no byes)
#   - Matchups slightly uneven: each pair meets base or base+1 times
#   - Solver distributes the +1 matchups optimally
#
# Example with 18 weekends, 6 teams:
#   - 6 teams (even): can play all 18 weekends
#   - Formula: g0 = floor(2 * 18 * 3 / 6) = 18 games per team

# ============== Grade Rounds Override ==============
# Set EXACT number of rounds for specific grades.
# This OVERRIDES both the formula calculation AND max_weekends.
# Use when a grade needs a specific number of rounds (e.g., per AGM decision).
#
# 2nd Grade 2026: 4 teams → formula gives 20 games (each opponent 6-7×)
# ACTUAL: TBC - update when confirmed

GRADE_ROUNDS_OVERRIDE = {
    # '2nd': 18,  # Example: force exactly 18 rounds (uncomment when confirmed)
}

# ============== Season Configuration ==============

SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),   # Sunday 22nd March
    'last_round_date': datetime(2026, 8, 30),  # Sunday 30th August (last regular round)
    'end_date': datetime(2026, 9, 19),     # Saturday 19th September (Grand Final)
    
    # Default max rounds (used as fallback if grade not in MAX_WEEKENDS_PER_GRADE)
    # This is the default MAXIMUM weekends any grade can play
    'max_rounds': 20,
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
    'phl_game_times': PHL_GAME_TIMES,  # Controls PHL variable generation (venue/field/day/time)
    'second_grade_times': SECOND_GRADE_TIMES,  # Controls 2nd grade variable generation
    
    # Grade-specific round configuration
    'max_weekends_per_grade': MAX_WEEKENDS_PER_GRADE,  # Max available weekends per grade
    'grade_rounds_override': GRADE_ROUNDS_OVERRIDE,     # Exact round counts (overrides formula)
    
    # Unavailabilities
    'field_unavailabilities': FIELD_UNAVAILABILITIES,
    
    # PHL-only afternoon dates (state championship weekends)
    # On these dates, Sunday afternoon (12pm+) at NIHC is PHL-only
    'phl_only_afternoon_dates': PHL_ONLY_AFTERNOON_DATES,
    
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
