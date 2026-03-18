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
            # Friday excluded - only EF gets Friday night at Broadmeadow
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
            # Jun 19-21 (U16 Girls SC) - NOT in weekends so Friday 7pm PHL game allowed
        ],
        'whole_days': [
            datetime(2026, 4, 25),  # ANZAC Day (Saturday)
            datetime(2026, 6, 20),  # U16 Girls SC Saturday
            datetime(2026, 6, 21),  # U16 Girls SC Sunday - fully blocked
        ],
        'part_days': [
            # U16 Girls SC Friday Jun 19 - block daytime, allow 7pm PHL
            datetime(2026, 6, 19, 8, 30),
            datetime(2026, 6, 19, 10, 0),
            datetime(2026, 6, 19, 11, 30),
            datetime(2026, 6, 19, 13, 0),
            datetime(2026, 6, 19, 14, 30),
            datetime(2026, 6, 19, 16, 0),
            datetime(2026, 6, 19, 17, 30),  # Block up to 5:30pm, allow 7pm
        ],
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
    'University': datetime(2026, 7, 26),  # University of Newcastle Hockey Club day
    # 'Souths_Norths_Derby': TBD,  # Red & Blue Derby Day - August, date TBC
}

# ============== No-Play Preferences (Soft Constraints) ==============

PREFERENCE_NO_PLAY = {
    # Crusaders 6th Grade - Masters State Championships (Moorebank)
    'Crusaders_6th_Masters_Moorebank': {
        'club': 'Crusaders',
        'grade': '6th',
        'dates': [datetime(2026, 4, 17), datetime(2026, 4, 18), datetime(2026, 4, 19)],
        'reason': 'NSW Masters Men\'s at Moorebank',
    },
}

# ============== Blocked Games (Hard No-Play Variable Removal) ==============
# Sister mechanism to FORCED_GAMES. Same config format but opposite logic:
# FORCED_GAMES: variables matching scope but NOT matching teams → eliminated
# BLOCKED_GAMES: variables matching scope AND matching teams → eliminated
#
# Use this for no-play requests where games must NOT exist (player unavailability,
# state championships, recovery weekends, etc.)
#
# Supported fields: same as FORCED_GAMES — club, teams, grade, grades, date, day, etc.

BLOCKED_GAMES = [
    # Crusaders 6th Grade - NSW Masters at Tamworth (Jun 26-28)
    # All Crusaders 6th affected — key players away at state championship
    {
        'club': 'Crusaders',
        'grade': '6th',
        'date': '2026-06-28',  # Sunday Jun 28 is the playing date
        'description': 'Crusaders 6th - NSW Masters at Tamworth',
        'reason': 'NSW Masters Men\'s at Tamworth',
    },
    # Souths PHL & 2nd Grade - U18 State Championships (May 24)
    # Both Souths PHL and Souths 2nd affected
    {
        'club': 'Souths',
        'grades': ['PHL', '2nd'],
        'date': '2026-05-24',
        'description': 'Souths PHL/2nd - U18 State Championships',
        'reason': 'U18\'s State Championships',
    },
    # Gosford - Recovery weekend after Men's SC (Jun 21)
    # All Gosford teams affected (only PHL exists for Gosford)
    {
        'club': 'Gosford',
        'date': '2026-06-21',
        'description': 'Gosford - Recovery after Men\'s State Championships',
        'reason': 'Recovery weekend after Men\'s State Championships',
    },
]

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
}

# ============== Forced Games (Partial Key Matching) ==============
# Each entry is a partial key specification for decision variables.
# Variables matching the SCOPE fields (date, day, field_location, etc.) but NOT matching
# the GAME fields (teams, grade) are eliminated from the model.
#
# Supported fields (matching the 11-tuple key):
#   Game fields:  teams (list of 1-2 club names, auto-sorted to team1/team2)
#   Scope fields: grade, day, day_slot, time, week, date, round_no, field_name, field_location
#
# If 'teams' has 2 names: both team1 AND team2 must match (order doesn't matter)
# If 'teams' has 1 name:  either team1 OR team2 must match (any opponent)
# Any field can be a single value or a list (any-of match)
#
# Multiple entries with the same scope are OR'd: a variable survives if it matches ANY entry.
#
# Example: Force Norths vs Souths PHL on a specific Sunday at NIHC (any time, any field)
#   {'teams': ['Norths', 'Souths'], 'grade': 'PHL', 'date': '2026-05-10',
#    'field_location': 'Newcastle International Hockey Centre'}

FORCED_GAMES = [
    # NIHC Friday May 8: Souths vs Maitland PHL only
    {
        'teams': ['Maitland', 'Souths'],
        'grade': 'PHL',
        'date': '2026-05-08',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Souths vs Maitland',
    },
    # NIHC Friday Jun 19: Tigers vs Wests PHL
    {
        'teams': ['Tigers', 'Wests'],
        'grade': 'PHL',
        'date': '2026-06-19',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Tigers vs Wests (State Champ weekend)',
    },
    # NIHC Friday Jul 24: Norths vs TBC PHL (any opponent)
    {
        'teams': ['Norths'],
        'grade': 'PHL',
        'date': '2026-07-24',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Norths home (opponent TBC)',
    },
    # Blue v Red Derby - Norths vs Souths, Sunday May 10
    {
        'teams': ['Norths', 'Souths'],
        'grade': 'PHL',
        'date': '2026-05-10',
        'day': 'Sunday',
        'description': 'Blue v Red Derby - PHL',
    },
    {
        'teams': ['Norths', 'Souths'],
        'grade': '2nd',
        'date': '2026-05-10',
        'day': 'Sunday',
        'description': 'Blue v Red Derby - 2nd Grade',
    },
    {
        'teams': ['Norths', 'Souths'],
        'grade': '3rd',
        'date': '2026-05-10',
        'day': 'Sunday',
        'description': 'Blue v Red Derby - 3rd Grade',
    },
    {
        'teams': ['Norths', 'Souths'],
        'grade': '4th',
        'date': '2026-05-10',
        'day': 'Sunday',
        'description': 'Blue v Red Derby - 4th Grade',
    },
]

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
# These are the AVAILABLE weekends for each grade to play (not actual rounds played).
# PHL has extra weekends due to Friday nights at Gosford (8) and NIHC.
# Actual rounds played is calculated from team count, capped by this number.

MAX_WEEKENDS_PER_GRADE = {
    'PHL': 22,   # 20 Sundays + Friday nights (Gosford 8 + NIHC)
    '2nd': 20,   # 20 Sundays only
    '3rd': 20,   # 20 Sundays only
    '4th': 20,   # 20 Sundays only
    '5th': 20,   # 20 Sundays only
    '6th': 20,   # 20 Sundays only
}

# ============== Grade Rounds Override ==============
# Set EXACT number of rounds for specific grades.
# This overrides the calculated value from max_games_per_grade().
# Use when a grade needs a specific number of rounds (e.g., per AGM decision).
#
# 2nd Grade 2026: 4 teams → play each opponent multiple times
# With 4 teams and 20 weekends, formula would give ~20 games.
# ACTUAL: Set to X rounds per AGM decision (TBC - update when confirmed)

GRADE_ROUNDS_OVERRIDE = {
    # '2nd': 18,  # Example: if 2nd grade plays exactly 18 rounds (uncomment when confirmed)
    # Note: Uncomment and set value once AGM confirms 2nd grade round count
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
    
    # Club events
    'club_days': CLUB_DAYS,
    
    # Preferences
    'preference_no_play': PREFERENCE_NO_PLAY,
    'phl_preferences': PHL_PREFERENCES,
    
    # Friday night settings
    'friday_night_config': FRIDAY_NIGHT_CONFIG,
    
    # Special games
    'special_games': SPECIAL_GAMES,
    
    # Forced games (partial key variable elimination)
    'forced_games': FORCED_GAMES,
    
    # Blocked games (no-play variable elimination)
    'blocked_games': BLOCKED_GAMES,
    
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
