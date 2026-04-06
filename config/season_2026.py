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
            'Friday': [tm(19, 0)],  # 7pm - Gosford vs Maitland only (other clubs blocked)
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
            datetime(2026, 3, 28),  # Mar 28-29 (Round 2 - blocked at Maitland)
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            # May 15-17 (Masters SC) - moved to whole_days, Sunday open for PHL
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
            # Jun 19-21 (U16 Girls SC) - moved to whole_days, Sunday open for PHL
        ],
        'whole_days': [
            datetime(2026, 4, 25),  # ANZAC Day (Saturday)
            datetime(2026, 5, 15),  # Masters SC Friday - blocked at Maitland
            datetime(2026, 5, 16),  # Masters SC Saturday - blocked at Maitland
            # May 17 (Sunday) OPEN for PHL at Maitland
            datetime(2026, 6, 19),  # U16 Girls SC Friday - blocked at Maitland
            datetime(2026, 6, 20),  # U16 Girls SC Saturday - blocked at Maitland
            # Jun 21 (Sunday) OPEN for PHL at Maitland
        ],
        'part_days': [],
    },
    'Newcastle International Hockey Centre': {
        'weekends': [
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            # May 15-17 (Masters SC) - moved to selective blocking, Sunday open for PHL
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
            # Jun 19-21 (U16 Girls SC) - NOT in weekends so Friday 7pm PHL game allowed
        ],
        'whole_days': [
            datetime(2026, 4, 25),  # ANZAC Day (Saturday)
            datetime(2026, 5, 15),  # Masters SC Friday - blocked at NIHC
            datetime(2026, 5, 16),  # Masters SC Saturday - blocked at NIHC
            # May 17 (Sunday) OPEN for PHL at NIHC (non-PHL blocked via BLOCKED_GAMES)
            datetime(2026, 6, 20),  # U16 Girls SC Saturday
            # Jun 21 (Sunday) OPEN for PHL at NIHC (non-PHL blocked via BLOCKED_GAMES)
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
            # May 15-17 (Masters SC) - moved to whole_days, Friday open for PHL
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
            # Jun 19-21 (U16 Girls SC) - moved to whole_days, Friday open for PHL
        ],
        'whole_days': [
            datetime(2026, 4, 25),  # ANZAC Day (Saturday)
            datetime(2026, 5, 16),  # Masters SC Saturday - blocked at Gosford
            datetime(2026, 5, 17),  # Masters SC Sunday - blocked at Gosford
            datetime(2026, 6, 20),  # U16 Girls SC Saturday - blocked at Gosford
            datetime(2026, 6, 21),  # U16 Girls SC Sunday - blocked at Gosford
            # Jun 19 (Friday) OPEN for forced PHL 8pm game
            # May 15 (Friday) OPEN for forced PHL 8pm game
        ],
        'part_days': [],
    },
}

# ============== Club Days (Special Events) ==============

CLUB_DAYS = {
    'Crusaders': datetime(2026, 6, 14),  # All 4 teams back-to-back on same field
    'University': datetime(2026, 7, 26),  # University of Newcastle Hockey Club day
    # 'Souths_Norths_Derby': datetime(2026, 5, 10),  # Red & Blue Derby Day - May 10 confirmed
}

# ============== No-Play Preferences (Soft Constraints) ==============

PREFERENCE_NO_PLAY = {
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
    # === NSW Masters at Moorebank/Illawarra (Apr 17-19) ===
    # Crusaders 6th blocked
    {
        'club': 'Crusaders',
        'grade': '6th',
        'date': '2026-04-19',  # Sunday Apr 19 is the playing date
        'description': 'Crusaders 6th - NSW Masters at Moorebank',
        'reason': 'NSW Masters Men\'s at Moorebank',
    },
    # Colts: 4th, 5th Gold, 6th blocked
    {
        'teams': ['Colts 4th'],
        'grade': '4th',
        'date': '2026-04-19',
        'description': 'Colts 4th - NSW Masters at Moorebank',
        'reason': 'NSW Masters Men\'s at Moorebank',
    },
    {
        'teams': ['Colts Gold 5th'],
        'grade': '5th',
        'date': '2026-04-19',
        'description': 'Colts Gold 5th - NSW Masters at Moorebank',
        'reason': 'NSW Masters Men\'s at Moorebank',
    },
    {
        'teams': ['Colts 6th'],
        'grade': '6th',
        'date': '2026-04-19',
        'description': 'Colts 6th - NSW Masters at Moorebank',
        'reason': 'NSW Masters Men\'s at Moorebank',
    },
    # === NSW Masters at Tamworth (Jun 26-28) ===
    # Crusaders 6th blocked
    {
        'club': 'Crusaders',
        'grade': '6th',
        'date': '2026-06-28',  # Sunday Jun 28 is the playing date
        'description': 'Crusaders 6th - NSW Masters at Tamworth',
        'reason': 'NSW Masters Men\'s at Tamworth',
    },
    # Colts: 4th, 5th Green, 6th blocked
    {
        'teams': ['Colts 4th'],
        'grade': '4th',
        'date': '2026-06-28',
        'description': 'Colts 4th - NSW Masters at Tamworth',
        'reason': 'NSW Masters Men\'s at Tamworth',
    },
    {
        'teams': ['Colts Green 5th'],
        'grade': '5th',
        'date': '2026-06-28',
        'description': 'Colts Green 5th - NSW Masters at Tamworth',
        'reason': 'NSW Masters Men\'s at Tamworth',
    },
    {
        'teams': ['Colts 6th'],
        'grade': '6th',
        'date': '2026-06-28',
        'description': 'Colts 6th - NSW Masters at Tamworth',
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
    # === Gosford Friday nights - block next two non-confirmed Fridays only ===
    # Confirmed dates (in FORCED_GAMES): Mar 27, Apr 17, Apr 24, May 15, May 29.
    # Only blocking the immediate upcoming non-confirmed Fridays.
    {
        'club': 'Gosford',
        'date': '2026-04-03',
        'day': 'Friday',
        'description': 'Gosford Friday blocked - not a confirmed Friday night date',
        'reason': 'Only confirmed dates allowed at Gosford',
    },
    {
        'club': 'Gosford',
        'date': '2026-04-10',
        'day': 'Friday',
        'description': 'Gosford Friday blocked - not a confirmed Friday night date',
        'reason': 'Only confirmed dates allowed at Gosford',
    },
    # === Maitland Friday nights - only Gosford vs Maitland allowed ===
    # PHL_GAME_TIMES adds Friday 7pm at Maitland Park. home_field_map ensures only
    # Maitland-involved games exist there. These blocks remove all non-Gosford opponents.
    {'club': 'Norths', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'description': 'Maitland Friday - only Gosford allowed', 'reason': 'Gosford vs Maitland only on Friday nights'},
    {'club': 'Souths', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'description': 'Maitland Friday - only Gosford allowed', 'reason': 'Gosford vs Maitland only on Friday nights'},
    {'club': 'Wests', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'description': 'Maitland Friday - only Gosford allowed', 'reason': 'Gosford vs Maitland only on Friday nights'},
    {'club': 'Tigers', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'description': 'Maitland Friday - only Gosford allowed', 'reason': 'Gosford vs Maitland only on Friday nights'},
    {'club': 'Crusaders', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'description': 'Maitland Friday - only Gosford allowed', 'reason': 'Gosford vs Maitland only on Friday nights'},
    {'club': 'Colts', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'description': 'Maitland Friday - only Gosford allowed', 'reason': 'Gosford vs Maitland only on Friday nights'},
    {'club': 'University', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'description': 'Maitland Friday - only Gosford allowed', 'reason': 'Gosford vs Maitland only on Friday nights'},
    {'club': 'Port Stephens', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'description': 'Maitland Friday - only Gosford allowed', 'reason': 'Gosford vs Maitland only on Friday nights'},
    # === State Championship Sundays at Maitland - PHL only ===
    # No teams/club = blocks ALL variables matching the scope (grade + date + location).
    # May 17 (Masters SC) and Jun 21 (U16 Girls SC) - block non-PHL grades at Maitland.
    {'grade': '2nd', 'date': '2026-05-17', 'field_location': 'Maitland Park',
     'description': 'Masters SC weekend - Maitland PHL only'},
    {'grade': '3rd', 'date': '2026-05-17', 'field_location': 'Maitland Park',
     'description': 'Masters SC weekend - Maitland PHL only'},
    {'grade': '4th', 'date': '2026-05-17', 'field_location': 'Maitland Park',
     'description': 'Masters SC weekend - Maitland PHL only'},
    {'grade': '5th', 'date': '2026-05-17', 'field_location': 'Maitland Park',
     'description': 'Masters SC weekend - Maitland PHL only'},
    {'grade': '6th', 'date': '2026-05-17', 'field_location': 'Maitland Park',
     'description': 'Masters SC weekend - Maitland PHL only'},
    {'grade': '2nd', 'date': '2026-06-21', 'field_location': 'Maitland Park',
     'description': 'U16 Girls SC weekend - Maitland PHL only'},
    {'grade': '3rd', 'date': '2026-06-21', 'field_location': 'Maitland Park',
     'description': 'U16 Girls SC weekend - Maitland PHL only'},
    {'grade': '4th', 'date': '2026-06-21', 'field_location': 'Maitland Park',
     'description': 'U16 Girls SC weekend - Maitland PHL only'},
    {'grade': '5th', 'date': '2026-06-21', 'field_location': 'Maitland Park',
     'description': 'U16 Girls SC weekend - Maitland PHL only'},
    {'grade': '6th', 'date': '2026-06-21', 'field_location': 'Maitland Park',
     'description': 'U16 Girls SC weekend - Maitland PHL only'},
    # === State Championship Sundays at NIHC - PHL only ===
    # May 17 (Masters SC) and Jun 21 (U16 Girls SC) - block non-PHL grades at NIHC.
    {'grade': '2nd', 'date': '2026-05-17', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'Masters SC weekend - NIHC PHL only'},
    {'grade': '3rd', 'date': '2026-05-17', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'Masters SC weekend - NIHC PHL only'},
    {'grade': '4th', 'date': '2026-05-17', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'Masters SC weekend - NIHC PHL only'},
    {'grade': '5th', 'date': '2026-05-17', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'Masters SC weekend - NIHC PHL only'},
    {'grade': '6th', 'date': '2026-05-17', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'Masters SC weekend - NIHC PHL only'},
    {'grade': '2nd', 'date': '2026-06-21', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'U16 Girls SC weekend - NIHC PHL only'},
    {'grade': '3rd', 'date': '2026-06-21', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'U16 Girls SC weekend - NIHC PHL only'},
    {'grade': '4th', 'date': '2026-06-21', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'U16 Girls SC weekend - NIHC PHL only'},
    {'grade': '5th', 'date': '2026-06-21', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'U16 Girls SC weekend - NIHC PHL only'},
    {'grade': '6th', 'date': '2026-06-21', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'U16 Girls SC weekend - NIHC PHL only'},
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
#   - Controlled by: PHL_GAME_TIMES (timeslots), BLOCKED_GAMES (date filtering),
#     FORCED_GAMES (specific matchups), CONSTRAINT_DEFAULTS['gosford_friday_games'] (count)
#   - Times: 8:00pm (confirmed at AGM, set in PHL_GAME_TIMES)
#   - Confirmed dates: Mar 27, Apr 17, Apr 24, May 29, Jun 12
#     (non-confirmed Fridays blocked via BLOCKED_GAMES)
#
# FRIDAY NIGHTS AT NIHC (Newcastle):
#   - Time: 7:00pm (set in PHL_GAME_TIMES)
#   - Max games: CONSTRAINT_DEFAULTS['max_friday_broadmeadow']
#   - Specific matchups forced via FORCED_GAMES
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
    # === NIHC Friday Nights (3 games at Broadmeadow) ===
    # May 8: Changed to Norths vs Maitland (was Souths vs Maitland, conflicted with derby)
    {
        'teams': ['Norths', 'Maitland'],
        'grade': 'PHL',
        'date': '2026-05-08',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Norths vs Maitland',
    },
    # Jun 19: Tigers vs Wests PHL (State Champ weekend - 7pm allowed)
    {
        'teams': ['Tigers', 'Wests'],
        'grade': 'PHL',
        'date': '2026-06-19',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Tigers vs Wests (State Champ weekend)',
    },
    # Jul 24: Norths vs TBC PHL (any opponent)
    {
        'teams': ['Norths'],
        'grade': 'PHL',
        'date': '2026-07-24',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Norths home (opponent TBC)',
    },
    # === Blue v Red Derby - Sunday May 10 (PHL removed: Norths plays Friday May 8) ===
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
    # === Gosford Friday Nights (5 forced dates) ===
    # PHL_GAME_TIMES already restricts Gosford variables to Gosford-involved games,
    # so no team specification needed — just date + venue forces a game there.
    {
        'grade': 'PHL',
        'date': '2026-03-27',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Mar 27',
    },
    {
        'grade': 'PHL',
        'date': '2026-04-17',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Apr 17',
    },
    {
        'grade': 'PHL',
        'date': '2026-04-24',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Apr 24 (ANZAC)',
    },
    {
        'grade': 'PHL',
        'date': '2026-05-15',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - May 15 (Masters SC weekend)',
    },
    {
        'grade': 'PHL',
        'date': '2026-05-29',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - May 29',
    },
    # Jun 19 at Gosford is OPEN (not forced) — unblocked via FIELD_UNAVAILABILITIES
    # === State Championship Sundays at NIHC - max 1 PHL game ===
    # These Sundays are open for PHL only (non-PHL blocked via BLOCKED_GAMES).
    # Use 'lesse' constraint: sum <= 1 (at most 1 game, solver may choose 0).
    {
        'grade': 'PHL',
        'date': '2026-05-17',
        'day': 'Sunday',
        'field_location': 'Newcastle International Hockey Centre',
        'constraint': 'lesse',
        'description': 'Masters SC weekend - max 1 PHL game at NIHC',
    },
    {
        'grade': 'PHL',
        'date': '2026-06-21',
        'day': 'Sunday',
        'field_location': 'Newcastle International Hockey Centre',
        'constraint': 'lesse',
        'description': 'U16 Girls SC weekend - max 1 PHL game at NIHC',
    },
    # === Norths v Wests Weekend - June 14 (week 13) ===
    # From Norths request: all grades Norths v Wests play that weekend.
    # Pairing forced, timeslot/field left open for solver.
    # No Norths 6th grade team exists, so 6th excluded.
    {
        'teams': ['Norths', 'Wests'],
        'grade': 'PHL',
        'date': '2026-06-14',
        'day': 'Sunday',
        'description': 'Norths v Wests Weekend - PHL',
    },
    {
        'teams': ['Norths', 'Wests'],
        'grade': '2nd',
        'date': '2026-06-14',
        'day': 'Sunday',
        'description': 'Norths v Wests Weekend - 2nd Grade',
    },
    {
        'teams': ['Norths', 'Wests'],
        'grade': '3rd',
        'date': '2026-06-14',
        'day': 'Sunday',
        'description': 'Norths v Wests Weekend - 3rd Grade',
    },
    {
        'teams': ['Norths', 'Wests'],
        'grade': '4th',
        'date': '2026-06-14',
        'day': 'Sunday',
        'description': 'Norths v Wests Weekend - 4th Grade',
    },
    {
        'teams': ['Norths', 'Wests'],
        'grade': '5th',
        'date': '2026-06-14',
        'day': 'Sunday',
        'description': 'Norths v Wests Weekend - 5th Grade',
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
    'PHL': 21,   # 20 Sundays + 1 Friday-only week (season ends Aug 30)
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
    '3rd': 18,  # Override: 8 teams with 20 weeks leaves 0 byes, too tight with blocked games
}

# ============== Grade Scheduling Method ==============
# Controls how games-per-team is calculated for each grade.
#
# Method 1 (default): Balanced round-robin. Games = largest multiple of (T-1)
#   that fits in available weekends, so each team plays every opponent the
#   SAME number of times. Best for small grades where equal matchups matter.
#   E.g. PHL (6 teams, 22 weekends) → 20 games (4× each opponent)
#
# Method 2: Maximize games. Fits as many games as possible into available
#   weekends, allowing base/base+1 matchup frequency. Better for larger
#   grades where forcing equal matchups would waste too many weekends.
#   E.g. 3rd (8 teams, 20 weekends) → 20 games (some opponents 2×, some 3×)
#
# Grades not listed here use the default (method 1).

GRADE_SCHEDULING_METHOD = {
    'PHL': 1,
    '2nd': 1,
    '3rd': 2,
    '4th': 2,
    '5th': 2,
    '6th': 1,
}

# ============== Constraint Base Limits ==============
# These are the base (default) hard limits for each slack-aware constraint.
# The --slack N flag loosens from these values. All are configurable per season.
#
# Formula per constraint:
#   EqualMatchUpSpacing:  min_gap = max(T//2+1, T-2 - spacing_base_slack - slack)
#   MaitlandHomeGrouping: max_consecutive = maitland_max_consecutive_home + slack
#   AwayAtMaitlandGrouping: max_away_clubs = away_maitland_max_clubs + slack
#   ClubVsClubAlignment:  (no base limit config — slack reduces required coincidences)
#   MaximiseClubsPerTimeslotBroadmeadow: min_clubs = floor(games/2) - slack
#   MinimiseClubsOnAFieldBroadmeadow: max_clubs = max_clubs_per_field + slack
#   ClubGameSpread: upper = club_game_spread_max_gap + slack, lower = -(club_game_spread_max_overlap + slack)

CONSTRAINT_DEFAULTS = {
    'spacing_base_slack': 2,               # EqualMatchUpSpacing: additional base slack (0 = start at ideal)
    'maitland_max_consecutive_home': 3,    # MaitlandHomeGrouping: max consecutive home weeks (1 = no back-to-back)
    'away_maitland_max_clubs': 2,          # AwayAtMaitlandGrouping: max away clubs at Maitland per week
    'max_clubs_per_field': 5,              # MinimiseClubsOnAFieldBroadmeadow: max clubs sharing a field per day
    'club_game_spread_max_gap': 2,         # ClubGameSpread: max allowed gap (spread) per club per day
    'club_game_spread_max_overlap': 1,     # ClubGameSpread: max allowed double-ups (0 = no two games at same slot)
    'gosford_friday_games': 8,             # PHLAndSecondGradeTimes: exact number of Friday PHL games at Gosford (AGM decision)
    'maitland_friday_games': 2,            # PHLAndSecondGradeTimes: exact number of Friday PHL games at Maitland (Gosford vs Maitland only)
    'max_friday_broadmeadow': 3,           # PHLAndSecondGradeTimes: max Friday PHL games at NIHC (Broadmeadow)
}

# ============== Soft Constraint Penalty Weights ==============
# These weights control relative priority between soft constraints.
#
# At objective-building time, each weight is NORMALIZED by dividing by the
# number of penalty variables that constraint created. This means:
#   - A weight of 100,000 always contributes ~100,000 to the objective at
#     full violation, regardless of whether it has 24 vars or 4,692 vars.
#   - You can reason about these as "how bad is a full violation of this
#     constraint compared to others" without worrying about var counts.
#
# Higher weight = solver tries harder to satisfy this constraint.
# The solver maximizes: scheduled_games - sum(normalized_penalties).

PENALTY_WEIGHTS = {
    'MaitlandHomeGrouping':             1_000_000,
    'AwayAtMaitlandGrouping':             100_000,
    'ClubVsClubAlignment':                 50_000,
    'EqualMatchUpSpacing':                100_000,
    'ClubGameSpread':                     100_000,
    'ClubFieldConcentration':              80_000,
    'PreferredTimesConstraint':           200_000,
    'ClubVsClubAlignmentField':                 0,  # Superseded by ClubFieldConcentration
    'ClubGradeAdjacencyConstraint':        50_000,
    'phl_preferences':                     10_000,
    'MaximiseClubsPerTimeslotBroadmeadow':  5_000,
    'MinimiseClubsOnAFieldBroadmeadow':     5_000,
}

# ============== Season Configuration ==============

SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),   # Sunday 22nd March (first playing day)
    'end_date': datetime(2026, 8, 30),     # Sunday 30th August (last club game before finals)
    
    # Default max rounds (used as fallback if grade not in MAX_WEEKENDS_PER_GRADE)
    # This is the default MAXIMUM weekends any grade can play
    'max_rounds': 20,
    
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
    'grade_scheduling_method': GRADE_SCHEDULING_METHOD,  # Method 1 (balanced) or 2 (maximize games) per grade
    
    # Unavailabilities
    'field_unavailabilities': FIELD_UNAVAILABILITIES,
    
    # Club events
    'club_days': CLUB_DAYS,
    
    # Preferences
    'preference_no_play': PREFERENCE_NO_PLAY,
    'phl_preferences': PHL_PREFERENCES,
    
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

    # Base limits for slack-aware constraints
    'constraint_defaults': CONSTRAINT_DEFAULTS,

    # Penalty weights for soft constraints
    'penalty_weights': PENALTY_WEIGHTS,

    # Solver timing
    'max_time_per_stage': 172800,  # 2 days per stage (seconds)
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
3. Red & Blue Derby date: May 10 (CONFIRMED - in FORCED_GAMES)
4. August catch-up weekend for wet weather deferrals
"""
