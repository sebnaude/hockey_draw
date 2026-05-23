# config/season_2026.py
"""
Season 2026 Configuration.

This file contains all season-specific settings for 2026.
NOTE: Some values are TBC pending AGM decisions and further information.
"""

from datetime import datetime, time as tm
from config.defaults import PERENNIAL_BLOCKED_GAMES

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
            'Sunday': [tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
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
            # PHL: 12:00, 13:30, 15:00, 16:30
            # +10:30 (before 12:00), no slot after 16:30 exists in DAY_TIME_MAP
            'Sunday': [tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
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
            # Jun 19-21 (U16 Girls SC) - no matches at Newcastle on Friday or Saturday
        ],
        'whole_days': [
            datetime(2026, 4, 25),  # ANZAC Day (Saturday)
            datetime(2026, 5, 15),  # Masters SC Friday - blocked at NIHC
            datetime(2026, 5, 16),  # Masters SC Saturday - blocked at NIHC
            # May 17 (Sunday) OPEN for PHL at NIHC (non-PHL blocked via BLOCKED_GAMES)
            datetime(2026, 6, 19),  # U16 Girls SC Friday - fully blocked at NIHC
            datetime(2026, 6, 20),  # U16 Girls SC Saturday - blocked at NIHC
            # Jun 21 (Sunday) OPEN for PHL at NIHC (non-PHL blocked via BLOCKED_GAMES)
        ],
        'part_days': [],
    },
    'Central Coast Hockey Park': {
        'weekends': [
            datetime(2026, 4, 4),   # Apr 3-5 (Easter weekend)
            # May 15-17 (Masters SC) - moved to whole_days, Friday open for PHL
            datetime(2026, 6, 6),   # Jun 5-7 (confirmed blocked)
            datetime(2026, 6, 13),  # Jun 12-14 - Gosford unavailable entire weekend
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
# spec-012: time-only preferences. Maitland and Port Stephens both ask not to
# play at 08:30. Entries without a 'date' apply across every playable week —
# any X-var for the named club at the named time accrues a PreferredTimes
# penalty in the soft stage.
#
# Entry shape (2026 structured format, see utils.normalize_preference_no_play):
#   'key': {
#       'club': 'ClubName',                # required (or 'dates' format)
#       'time': 'HH:MM',                   # optional, time-only filter
#       'date'/'dates': ...,               # optional, date filter
#       'grade'/'grades': ...,             # optional, grade filter
#       'description': '...',              # ignored by solver
#   }

PREFERENCE_NO_PLAY = {
    'maitland_no_8_30am': {
        'club': 'Maitland',
        'time': '08:30',
        'description': 'Maitland teams prefer not to play at 08:30 (spec-012)',
    },
    'port_stephens_no_8_30am': {
        'club': 'Port Stephens',
        'time': '08:30',
        'description': 'Port Stephens teams prefer not to play at 08:30 (spec-012)',
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
    # --- Perennial rules (from config/defaults.py) ---
    # Rounds 1-2 at Broadmeadow only, etc. See docs/PERENNIAL_RULES.md.
    *PERENNIAL_BLOCKED_GAMES,

    # === NSW Masters at Moorebank/Illawarra (Apr 17-19) ===
    # Crusaders 6th blocked
    {
        'club': 'Crusaders',
        'grade': '6th',
        'date': '2026-04-19',  # Sunday Apr 19 is the playing date
        'description': 'Crusaders 6th - NSW Masters at Moorebank',
        'reason': 'NSW Masters Men\'s at Moorebank',
        'note': True,  # spec-028: surface in published draw notes column
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
    # === Gosford unavailable weekend Jun 12-14 ===
    {'club': 'Gosford', 'date': '2026-06-12', 'day': 'Friday',
     'description': 'Gosford unavailable - Jun 12 Friday', 'reason': 'Gosford unavailable entire weekend Jun 12-14'},
    {'club': 'Gosford', 'date': '2026-06-14', 'day': 'Sunday',
     'description': 'Gosford unavailable - Jun 14 Sunday', 'reason': 'Gosford unavailable entire weekend Jun 12-14'},

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
    # Souths PHL & 2nd Grade - U18 State Championships (May 22-24 weekend)
    # Both Souths PHL and Souths 2nd affected. Block full weekend (Fri + Sun).
    {
        'club': 'Souths',
        'grade': 'PHL',
        'date': '2026-05-22',
        'day': 'Friday',
        'description': 'Souths PHL - U18 State Championships weekend (Friday)',
        'reason': 'U18 State Championships',
    },
    {
        'club': 'Souths',
        'grades': ['PHL', '2nd'],
        'date': '2026-05-24',
        'description': 'Souths PHL/2nd - U18 State Championships (Sunday)',
        'reason': 'U18 State Championships',
    },
    # Gosford - Recovery weekend after Men's SC (Jun 21)
    # All Gosford teams affected (only PHL exists for Gosford)
    {
        'club': 'Gosford',
        'date': '2026-06-21',
        'description': 'Gosford - Recovery after Men\'s State Championships',
        'reason': 'Recovery weekend after Men\'s State Championships',
    },
    # === NIHC Friday nights — block all except allowed dates ===
    # Allowed NIHC Friday dates: May 8, Jun 5, Jun 12, Jun 26, Jul 24, Jul 31
    # Apr 3 already blocked via Easter FIELD_UNAVAILABILITIES.
    # Before May
    {'grade': 'PHL', 'date': '2026-03-27', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-04-10', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-04-17', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-04-24', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    # May (only May 8 allowed)
    {'grade': 'PHL', 'date': '2026-05-01', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-05-15', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-05-22', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-05-29', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    # June (only Jun 5, 12, 26 allowed)
    {'grade': 'PHL', 'date': '2026-06-19', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    # July (only Jul 24, 31 allowed)
    {'grade': 'PHL', 'date': '2026-07-03', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-07-10', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-07-17', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    # August (none allowed)
    {'grade': 'PHL', 'date': '2026-08-07', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-08-14', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-08-21', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    {'grade': 'PHL', 'date': '2026-08-28', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
     'description': 'NIHC Friday blocked', 'reason': 'Not an allowed NIHC Friday date'},
    # === Gosford Friday nights - block non-game Fridays ===
    # All 8 Gosford Friday dates are now forced in FORCED_GAMES.
    # Block remaining Fridays where Gosford should NOT play.
    # Apr 3 = Easter (already blocked via FIELD_UNAVAILABILITIES).
    {'club': 'Gosford', 'date': '2026-04-10', 'day': 'Friday',
     'description': 'Gosford Friday blocked', 'reason': 'Not a Gosford Friday night date'},
    {'club': 'Gosford', 'date': '2026-04-24', 'day': 'Friday',
     'description': 'Gosford Friday blocked - ANZAC', 'reason': 'PHL bye weekend (published in revo)'},
    {'club': 'Gosford', 'date': '2026-05-01', 'day': 'Friday',
     'description': 'Gosford Friday blocked', 'reason': 'Not a Gosford Friday night date'},
    {'club': 'Gosford', 'date': '2026-05-08', 'day': 'Friday',
     'description': 'Gosford Friday blocked', 'reason': 'Not a Gosford Friday night date'},
    {'club': 'Gosford', 'date': '2026-05-22', 'day': 'Friday',
     'description': 'Gosford Friday blocked', 'reason': 'Not a Gosford Friday night date'},
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
    # === Aug 16 (week 22, State Masters) — no morning games at Broadmeadow ===
    # Games only from 1pm onwards. Block 8:30, 10:00, 11:30.
    {'date': '2026-08-16', 'time': ['08:30', '10:00', '11:30'],
     'field_location': 'Newcastle International Hockey Centre',
     'description': 'Week 22 State Masters - no games before 1pm at NIHC',
     'reason': 'State Masters weekend - fields unavailable until ~12:30'},
    # === May 24 — no late games at Broadmeadow ===
    # Block 4pm, 5:30pm, 7pm. Allow up to 2:30pm.
    {'date': '2026-05-24', 'time': ['16:00', '17:30', '19:00'],
     'field_location': 'Newcastle International Hockey Centre',
     'description': 'May 24 - no late games at NIHC',
     'reason': 'Restricted day - no games past 2:30pm slot'},
    # === Gosford no late games at Broadmeadow ===
    # Gosford teams must not play at NIHC past the 2:30pm slot (no 4pm, 5:30pm, 7pm).
    {'club': 'Gosford', 'time': ['16:00', '17:30', '19:00'],
     'field_location': 'Newcastle International Hockey Centre',
     'description': 'Gosford - no late Broadmeadow games',
     'reason': 'Gosford teams cannot play at NIHC past 2:30pm slot'},
]

# ============== PHL Preferences ==============
# spec-020: PHL_PREFERENCES (and its narrow `preferred_dates` soft constraint,
# the deleted PreferredDates atom) removed. To softly prefer "exactly one PHL
# game on date X", add a PREFERRED_GAMES entry below:
#   {'grade': 'PHL', 'date': '2026-04-19', 'constraint': 'equal', 'count': 1,
#    'weight': 10000, 'description': 'marquee PHL date'}
#
# Additional PHL Rules (enforced via constraints, not this dict):
#   - PHL/2nd back-to-back: CONFIRMED for 2026
#   - Teams playing Gosford have 2nd grade bye: CONFIRMED

# ============== Preferred Games (spec-020) ==============
# Soft, weighted analogue of FORCED_GAMES (same scope/team/club grammar +
# optional `weight`). Penalty-on-deviation from `count` per `constraint` type
# (equal/lesse/less/greater/greatere) instead of a hard rule. Single shared
# bucket weighted by PENALTY_WEIGHTS['preferred_games']; per-entry `weight` is a
# multiplier. Empty list = no preferences (no penalty). See
# docs/system/FORCED_GAMES_AS_COUNT_RULES.md.
PREFERRED_GAMES = []

# ============== PHL SCHEDULE SUMMARY ==============
#
# FRIDAY NIGHTS AT GOSFORD (Central Coast):
#   - Controlled by: PHL_GAME_TIMES (timeslots), BLOCKED_GAMES (date filtering),
#     FORCED_GAMES (specific matchups), CONSTRAINT_DEFAULTS['gosford_friday_games'] (count)
#   - Times: 8:00pm (confirmed at AGM, set in PHL_GAME_TIMES)
#   - Confirmed dates: Mar 27, Apr 17, Apr 24, May 29, Jun 12
#     (non-confirmed Fridays blocked via BLOCKED_GAMES)
#   - Norths plays Gosford on exactly 1 Friday (via FORCED_GAMES)
#
# FRIDAY NIGHTS AT NIHC (Newcastle):
#   - Time: 7:00pm (set in PHL_GAME_TIMES)
#   - Max games: CONSTRAINT_DEFAULTS['max_friday_broadmeadow'] (3)
#   - Allowed dates: May 8, Jun 5, Jun 12, Jun 26, Jul 24, Jul 31 (all others blocked)
#   - Jun 12: Norths vs Wests (80th Anniversary, forced date via FORCED_GAMES)
#   - Maitland vs Souths: exactly 1 NIHC Friday (solver picks date)
#   - Wests vs Tigers: exactly 1 NIHC Friday (solver picks date)
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
    # === Per-venue PHL Friday count caps ===
    # These three entries express the per-season Friday-night totals at each
    # venue as FORCED_GAMES count rules — see docs/FORCED_GAMES_AS_COUNT_RULES.md.
    # Per-venue counts are budgets, not structural constraints; they belong
    # here in season config, not in hardcoded constraint atoms. Per-pair Friday
    # entries below (e.g. "Maitland vs Souths" / "Norths vs Gosford") combine
    # with these caps via multi-scope variable registration (commit cd8a338),
    # so each Friday game counts toward every matching scope simultaneously.
    {
        'grade': 'PHL', 'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'count': 3, 'constraint': 'lesse',
        'description': 'Max 3 PHL Friday games at Broadmeadow per season',
    },
    {
        'grade': 'PHL', 'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'count': 8, 'constraint': 'equal',
        'description': 'Exactly 8 PHL Friday games at Gosford per season (AGM 2026)',
    },
    {
        'grade': 'PHL', 'day': 'Friday',
        'field_location': 'Maitland Park',
        'count': 2, 'constraint': 'equal',
        'description': 'Exactly 2 PHL Friday games at Maitland Park per season',
    },
    # === NIHC Friday Nights ===
    # Jun 12: Norths vs Wests PHL (Norths 80th Anniversary Friday) — fixed date
    {
        'teams': ['Norths', 'Wests'],
        'grade': 'PHL',
        'date': '2026-06-12',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Norths vs Wests (80th Anniversary)',
        'note': True,  # spec-028: surface in published draw notes column
    },
    # Maitland vs Souths PHL — exactly 1 NIHC Friday night (solver picks date)
    {
        'teams': ['Maitland', 'Souths'],
        'grade': 'PHL',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Maitland vs Souths (exactly 1)',
    },
    # Wests vs Tigers PHL — exactly 1 NIHC Friday night (solver picks date)
    {
        'teams': ['Wests', 'Tigers'],
        'grade': 'PHL',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Wests vs Tigers (exactly 1)',
    },
    # === Norths vs Gosford - exactly 1 Friday PHL game at Gosford ===
    {
        'teams': ['Norths', 'Gosford'],
        'grade': 'PHL',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Norths vs Gosford - exactly 1 Friday night at Gosford',
    },
    # === Blue v Red Derby - Sunday May 10 (all grades including PHL) ===
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
    # 3rd/4th Derby already covered by locked wk8 entries below
    # === Gosford Friday Nights (7 forced dates; 8th chosen by solver) ===
    # PHL_GAME_TIMES already restricts Gosford variables to Gosford-involved games,
    # so no team specification needed — just date + venue forces a game there.
    # Apr 24 (ANZAC) removed — all PHL have bye that weekend (published in revo).
    # NOTE: Mar 27 removed — falls in round 1 which is blocked at Gosford
    # by PERENNIAL_BLOCKED_GAMES (rounds 1-2 at Broadmeadow only).
    {
        'grade': 'PHL',
        'date': '2026-04-17',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Apr 17',
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
    {
        'grade': 'PHL',
        'date': '2026-06-19',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Jun 19',
    },
    {
        'grade': 'PHL',
        'date': '2026-07-10',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Jul 10',
    },
    {
        'grade': 'PHL',
        'date': '2026-08-14',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Aug 14',
    },
    {
        'grade': 'PHL',
        'date': '2026-08-28',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Aug 28',
    },
    # === State Championship Sundays at NIHC - max 2 PHL games ===
    # These Sundays are open for PHL only (non-PHL blocked via BLOCKED_GAMES).
    # Use 'lesse' constraint: sum <= 2 (at most 2 games, solver may choose fewer).
    # HACK: Changed from 1 to 2 to give PHL teams enough weeks to reach 20 games
    # when combined with locked weeks. With sum<=1, teams without a Friday game
    # on these weekends lose the week entirely, dropping below 20 capacity.
    # TODO: Move this limit to CONSTRAINT_DEFAULTS (e.g. 'max_phl_sc_weekend_games')
    # instead of hardcoding in FORCED_GAMES. That way it's configurable per season.
    {
        'grade': 'PHL',
        'date': '2026-05-17',
        'day': 'Sunday',
        'field_location': 'Newcastle International Hockey Centre',
        'constraint': 'lesse',
        'count': 2,
        'description': 'Masters SC weekend - max 2 PHL games at NIHC',
    },
    {
        'grade': 'PHL',
        'date': '2026-06-21',
        'day': 'Sunday',
        'field_location': 'Newcastle International Hockey Centre',
        'constraint': 'lesse',
        'count': 2,
        'description': 'U16 Girls SC weekend - max 2 PHL games at NIHC',
    },
    # Norths v Wests Weekend Jun 14: 2nd-5th already covered by locked wk13 entries below
    # PHL is on Friday Jun 12 at NIHC (above)
    # === Locked draw pairings (3rd-6th grade, week 7+) ===
    # Lock team pairings + dates from current draw v10.6. Solver picks field/time.
    # 246 games locked.
    {'teams': ['Crusaders 3rd', 'Souths 3rd'], 'grade': '3rd', 'date': '2026-05-03', 'description': 'Locked wk7: Crusaders 3rd vs Souths 3rd'},
    {'teams': ['Maitland 3rd', 'Port Stephens 3rd'], 'grade': '3rd', 'date': '2026-05-03', 'description': 'Locked wk7: Maitland 3rd vs Port Stephens 3rd'},
    {'teams': ['Norths 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-05-03', 'description': 'Locked wk7: Norths 3rd vs University 3rd'},
    {'teams': ['Tigers 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-05-03', 'description': 'Locked wk7: Tigers 3rd vs Wests 3rd'},
    {'teams': ['Colts 4th', 'Souths 4th'], 'grade': '4th', 'date': '2026-05-03', 'description': 'Locked wk7: Colts 4th vs Souths 4th'},
    {'teams': ['Maitland 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-05-03', 'description': 'Locked wk7: Maitland 4th vs Tigers 4th'},
    {'teams': ['Norths 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-05-03', 'description': 'Locked wk7: Norths 4th vs University Seapigs 4th'},
    {'teams': ['Port Stephens 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-05-03', 'description': 'Locked wk7: Port Stephens 4th vs University Redhogs 4th'},
    {'teams': ['Wests Green 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-05-03', 'description': 'Locked wk7: Wests Green 4th vs Wests Red 4th'},
    {'teams': ['Colts Gold 5th', 'Tigers 5th'], 'grade': '5th', 'date': '2026-05-03', 'description': 'Locked wk7: Colts Gold 5th vs Tigers 5th'},
    {'teams': ['Colts Green 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-05-03', 'description': 'Locked wk7: Colts Green 5th vs Wests Red 5th'},
    {'teams': ['Crusaders 5th', 'Norths 5th'], 'grade': '5th', 'date': '2026-05-03', 'description': 'Locked wk7: Crusaders 5th vs Norths 5th'},
    {'teams': ['Maitland 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-05-03', 'description': 'Locked wk7: Maitland 5th vs Wests Green 5th'},
    {'teams': ['Colts 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-05-03', 'description': 'Locked wk7: Colts 6th vs University Seapigs 6th'},
    {'teams': ['Crusaders 6th', 'Souths 6th'], 'grade': '6th', 'date': '2026-05-03', 'description': 'Locked wk7: Crusaders 6th vs Souths 6th'},
    {'teams': ['Maitland 6th', 'Port Stephens 6th'], 'grade': '6th', 'date': '2026-05-03', 'description': 'Locked wk7: Maitland 6th vs Port Stephens 6th'},
    {'teams': ['Tigers Black 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-05-03', 'description': 'Locked wk7: Tigers Black 6th vs Wests 6th'},
    {'teams': ['Tigers Yellow 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-05-03', 'description': 'Locked wk7: Tigers Yellow 6th vs University Gentlemen 6th'},
    {'teams': ['Maitland 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-05-10', 'description': 'Locked wk8: Maitland 3rd vs Tigers 3rd'},
    {'teams': ['Norths 3rd', 'Souths 3rd'], 'grade': '3rd', 'date': '2026-05-10', 'description': 'Locked wk8: Norths 3rd vs Souths 3rd'},
    {'teams': ['Port Stephens 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-05-10', 'description': 'Locked wk8: Port Stephens 3rd vs University 3rd'},
    {'teams': ['Colts 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-05-10', 'description': 'Locked wk8: Colts 4th vs University Seapigs 4th'},
    {'teams': ['Crusaders 4th', 'Maitland 4th'], 'grade': '4th', 'date': '2026-05-10', 'description': 'Locked wk8: Crusaders 4th vs Maitland 4th'},
    {'teams': ['Norths 4th', 'Souths 4th'], 'grade': '4th', 'date': '2026-05-10', 'description': 'Locked wk8: Norths 4th vs Souths 4th'},
    {'teams': ['Tigers 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-05-10', 'description': 'Locked wk8: Tigers 4th vs Wests Green 4th'},
    {'teams': ['University Redhogs 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-05-10', 'description': 'Locked wk8: University Redhogs 4th vs Wests Red 4th'},
    {'teams': ['Colts Gold 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-05-10', 'description': 'Locked wk8: Colts Gold 5th vs Wests Green 5th'},
    {'teams': ['Colts Green 5th', 'Crusaders 5th'], 'grade': '5th', 'date': '2026-05-10', 'description': 'Locked wk8: Colts Green 5th vs Crusaders 5th'},
    {'teams': ['Norths 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-05-10', 'description': 'Locked wk8: Norths 5th vs Wests Red 5th'},
    {'teams': ['Tigers 5th', 'University 5th'], 'grade': '5th', 'date': '2026-05-10', 'description': 'Locked wk8: Tigers 5th vs University 5th'},
    {'teams': ['Colts 6th', 'Port Stephens 6th'], 'grade': '6th', 'date': '2026-05-10', 'description': 'Locked wk8: Colts 6th vs Port Stephens 6th'},
    {'teams': ['Crusaders 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-05-10', 'description': 'Locked wk8: Crusaders 6th vs University Gentlemen 6th'},
    {'teams': ['Maitland 6th', 'Souths 6th'], 'grade': '6th', 'date': '2026-05-10', 'description': 'Locked wk8: Maitland 6th vs Souths 6th'},
    {'teams': ['Tigers Yellow 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-05-10', 'description': 'Locked wk8: Tigers Yellow 6th vs Wests 6th'},
    {'teams': ['Crusaders 3rd', 'Port Stephens 3rd'], 'grade': '3rd', 'date': '2026-05-24', 'description': 'Locked wk10: Crusaders 3rd vs Port Stephens 3rd'},
    {'teams': ['Maitland 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-05-24', 'description': 'Locked wk10: Maitland 3rd vs Wests 3rd'},
    {'teams': ['Norths 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-05-24', 'description': 'Locked wk10: Norths 3rd vs Tigers 3rd'},
    {'teams': ['Souths 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-05-24', 'description': 'Locked wk10: Souths 3rd vs University 3rd'},
    {'teams': ['Colts 4th', 'Port Stephens 4th'], 'grade': '4th', 'date': '2026-05-24', 'description': 'Locked wk10: Colts 4th vs Port Stephens 4th'},
    {'teams': ['Crusaders 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-05-24', 'description': 'Locked wk10: Crusaders 4th vs Wests Red 4th'},
    {'teams': ['Maitland 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-05-24', 'description': 'Locked wk10: Maitland 4th vs Wests Green 4th'},
    {'teams': ['Norths 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-05-24', 'description': 'Locked wk10: Norths 4th vs Tigers 4th'},
    {'teams': ['Souths 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-05-24', 'description': 'Locked wk10: Souths 4th vs University Seapigs 4th'},
    {'teams': ['Colts Gold 5th', 'Maitland 5th'], 'grade': '5th', 'date': '2026-05-24', 'description': 'Locked wk10: Colts Gold 5th vs Maitland 5th'},
    {'teams': ['Colts Green 5th', 'University 5th'], 'grade': '5th', 'date': '2026-05-24', 'description': 'Locked wk10: Colts Green 5th vs University 5th'},
    {'teams': ['Wests Green 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-05-24', 'description': 'Locked wk10: Wests Green 5th vs Wests Red 5th'},
    {'teams': ['Colts 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-05-24', 'description': 'Locked wk10: Colts 6th vs University Gentlemen 6th'},
    {'teams': ['Crusaders 6th', 'Tigers Black 6th'], 'grade': '6th', 'date': '2026-05-24', 'description': 'Locked wk10: Crusaders 6th vs Tigers Black 6th'},
    {'teams': ['Maitland 6th', 'Tigers Yellow 6th'], 'grade': '6th', 'date': '2026-05-24', 'description': 'Locked wk10: Maitland 6th vs Tigers Yellow 6th'},
    {'teams': ['Port Stephens 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-05-24', 'description': 'Locked wk10: Port Stephens 6th vs University Seapigs 6th'},
    {'teams': ['Souths 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-05-24', 'description': 'Locked wk10: Souths 6th vs Wests 6th'},
    {'teams': ['Maitland 3rd', 'Souths 3rd'], 'grade': '3rd', 'date': '2026-05-31', 'description': 'Locked wk11: Maitland 3rd vs Souths 3rd'},
    {'teams': ['Port Stephens 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-05-31', 'description': 'Locked wk11: Port Stephens 3rd vs Tigers 3rd'},
    {'teams': ['University 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-05-31', 'description': 'Locked wk11: University 3rd vs Wests 3rd'},
    {'teams': ['Colts 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-05-31', 'description': 'Locked wk11: Colts 4th vs Tigers 4th'},
    {'teams': ['Crusaders 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-05-31', 'description': 'Locked wk11: Crusaders 4th vs Wests Green 4th'},
    {'teams': ['Maitland 4th', 'Souths 4th'], 'grade': '4th', 'date': '2026-05-31', 'description': 'Locked wk11: Maitland 4th vs Souths 4th'},
    {'teams': ['Norths 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-05-31', 'description': 'Locked wk11: Norths 4th vs University Redhogs 4th'},
    {'teams': ['University Seapigs 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-05-31', 'description': 'Locked wk11: University Seapigs 4th vs Wests Red 4th'},
    {'teams': ['Colts Gold 5th', 'Colts Green 5th'], 'grade': '5th', 'date': '2026-05-31', 'description': 'Locked wk11: Colts Gold 5th vs Colts Green 5th'},
    {'teams': ['Maitland 5th', 'Norths 5th'], 'grade': '5th', 'date': '2026-05-31', 'description': 'Locked wk11: Maitland 5th vs Norths 5th'},
    {'teams': ['Colts 6th', 'Crusaders 6th'], 'grade': '6th', 'date': '2026-05-31', 'description': 'Locked wk11: Colts 6th vs Crusaders 6th'},
    {'teams': ['Maitland 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-05-31', 'description': 'Locked wk11: Maitland 6th vs University Seapigs 6th'},
    {'teams': ['Port Stephens 6th', 'Tigers Black 6th'], 'grade': '6th', 'date': '2026-05-31', 'description': 'Locked wk11: Port Stephens 6th vs Tigers Black 6th'},
    {'teams': ['Tigers Yellow 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-05-31', 'description': 'Locked wk11: Tigers Yellow 6th vs University Gentlemen 6th'},
    {'teams': ['Crusaders 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-06-14', 'description': 'Locked wk13: Crusaders 3rd vs Tigers 3rd'},
    {'teams': ['Norths 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-06-14', 'description': 'Locked wk13: Norths 3rd vs Wests 3rd'},
    {'teams': ['Port Stephens 3rd', 'Souths 3rd'], 'grade': '3rd', 'date': '2026-06-14', 'description': 'Locked wk13: Port Stephens 3rd vs Souths 3rd'},
    {'teams': ['Colts 4th', 'Crusaders 4th'], 'grade': '4th', 'date': '2026-06-14', 'description': 'Locked wk13: Colts 4th vs Crusaders 4th'},
    {'teams': ['Maitland 4th', 'Port Stephens 4th'], 'grade': '4th', 'date': '2026-06-14', 'description': 'Locked wk13: Maitland 4th vs Port Stephens 4th'},
    {'teams': ['Norths 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-06-14', 'description': 'Locked wk13: Norths 4th vs Wests Green 4th'},
    {'teams': ['Souths 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-06-14', 'description': 'Locked wk13: Souths 4th vs University Redhogs 4th'},
    {'teams': ['University Seapigs 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-06-14', 'description': 'Locked wk13: University Seapigs 4th vs Wests Red 4th'},
    {'teams': ['Colts Gold 5th', 'Crusaders 5th'], 'grade': '5th', 'date': '2026-06-14', 'description': 'Locked wk13: Colts Gold 5th vs Crusaders 5th'},
    {'teams': ['Maitland 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-06-14', 'description': 'Locked wk13: Maitland 5th vs Wests Red 5th'},
    {'teams': ['Norths 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-06-14', 'description': 'Locked wk13: Norths 5th vs Wests Green 5th'},
    {'teams': ['Tigers 5th', 'University 5th'], 'grade': '5th', 'date': '2026-06-14', 'description': 'Locked wk13: Tigers 5th vs University 5th'},
    {'teams': ['Colts 6th', 'Tigers Black 6th'], 'grade': '6th', 'date': '2026-06-14', 'description': 'Locked wk13: Colts 6th vs Tigers Black 6th'},
    {'teams': ['Crusaders 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-06-14', 'description': 'Locked wk13: Crusaders 6th vs University Seapigs 6th'},
    {'teams': ['Maitland 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-06-14', 'description': 'Locked wk13: Maitland 6th vs Wests 6th'},
    {'teams': ['Port Stephens 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-06-14', 'description': 'Locked wk13: Port Stephens 6th vs University Gentlemen 6th'},
    {'teams': ['Souths 6th', 'Tigers Yellow 6th'], 'grade': '6th', 'date': '2026-06-14', 'description': 'Locked wk13: Souths 6th vs Tigers Yellow 6th'},
    {'teams': ['Crusaders 3rd', 'Norths 3rd'], 'grade': '3rd', 'date': '2026-06-28', 'description': 'Locked wk15: Crusaders 3rd vs Norths 3rd'},
    {'teams': ['Maitland 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-06-28', 'description': 'Locked wk15: Maitland 3rd vs Tigers 3rd'},
    {'teams': ['Port Stephens 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-06-28', 'description': 'Locked wk15: Port Stephens 3rd vs University 3rd'},
    {'teams': ['Souths 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-06-28', 'description': 'Locked wk15: Souths 3rd vs Wests 3rd'},
    {'teams': ['Crusaders 4th', 'Souths 4th'], 'grade': '4th', 'date': '2026-06-28', 'description': 'Locked wk15: Crusaders 4th vs Souths 4th'},
    {'teams': ['Maitland 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-06-28', 'description': 'Locked wk15: Maitland 4th vs University Seapigs 4th'},
    {'teams': ['Norths 4th', 'Port Stephens 4th'], 'grade': '4th', 'date': '2026-06-28', 'description': 'Locked wk15: Norths 4th vs Port Stephens 4th'},
    {'teams': ['Tigers 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-06-28', 'description': 'Locked wk15: Tigers 4th vs University Redhogs 4th'},
    {'teams': ['Wests Green 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-06-28', 'description': 'Locked wk15: Wests Green 4th vs Wests Red 4th'},
    {'teams': ['Colts Gold 5th', 'Norths 5th'], 'grade': '5th', 'date': '2026-06-28', 'description': 'Locked wk15: Colts Gold 5th vs Norths 5th'},
    {'teams': ['Crusaders 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-06-28', 'description': 'Locked wk15: Crusaders 5th vs Wests Green 5th'},
    {'teams': ['Tigers 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-06-28', 'description': 'Locked wk15: Tigers 5th vs Wests Red 5th'},
    {'teams': ['Maitland 6th', 'Tigers Black 6th'], 'grade': '6th', 'date': '2026-06-28', 'description': 'Locked wk15: Maitland 6th vs Tigers Black 6th'},
    {'teams': ['Port Stephens 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-06-28', 'description': 'Locked wk15: Port Stephens 6th vs Wests 6th'},
    {'teams': ['Souths 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-06-28', 'description': 'Locked wk15: Souths 6th vs University Gentlemen 6th'},
    {'teams': ['Tigers Yellow 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-06-28', 'description': 'Locked wk15: Tigers Yellow 6th vs University Seapigs 6th'},
    {'teams': ['Crusaders 3rd', 'Maitland 3rd'], 'grade': '3rd', 'date': '2026-07-05', 'description': 'Locked wk16: Crusaders 3rd vs Maitland 3rd'},
    {'teams': ['Norths 3rd', 'Port Stephens 3rd'], 'grade': '3rd', 'date': '2026-07-05', 'description': 'Locked wk16: Norths 3rd vs Port Stephens 3rd'},
    {'teams': ['Souths 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-07-05', 'description': 'Locked wk16: Souths 3rd vs University 3rd'},
    {'teams': ['Tigers 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-07-05', 'description': 'Locked wk16: Tigers 3rd vs Wests 3rd'},
    {'teams': ['Colts 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-07-05', 'description': 'Locked wk16: Colts 4th vs University Redhogs 4th'},
    {'teams': ['Crusaders 4th', 'Maitland 4th'], 'grade': '4th', 'date': '2026-07-05', 'description': 'Locked wk16: Crusaders 4th vs Maitland 4th'},
    {'teams': ['Norths 4th', 'Souths 4th'], 'grade': '4th', 'date': '2026-07-05', 'description': 'Locked wk16: Norths 4th vs Souths 4th'},
    {'teams': ['Port Stephens 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-07-05', 'description': 'Locked wk16: Port Stephens 4th vs Wests Green 4th'},
    {'teams': ['Tigers 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-07-05', 'description': 'Locked wk16: Tigers 4th vs Wests Red 4th'},
    {'teams': ['Colts Gold 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-07-05', 'description': 'Locked wk16: Colts Gold 5th vs Wests Red 5th'},
    {'teams': ['Colts Green 5th', 'Maitland 5th'], 'grade': '5th', 'date': '2026-07-05', 'description': 'Locked wk16: Colts Green 5th vs Maitland 5th'},
    {'teams': ['Crusaders 5th', 'University 5th'], 'grade': '5th', 'date': '2026-07-05', 'description': 'Locked wk16: Crusaders 5th vs University 5th'},
    {'teams': ['Norths 5th', 'Tigers 5th'], 'grade': '5th', 'date': '2026-07-05', 'description': 'Locked wk16: Norths 5th vs Tigers 5th'},
    {'teams': ['Colts 6th', 'Maitland 6th'], 'grade': '6th', 'date': '2026-07-05', 'description': 'Locked wk16: Colts 6th vs Maitland 6th'},
    {'teams': ['Crusaders 6th', 'Souths 6th'], 'grade': '6th', 'date': '2026-07-05', 'description': 'Locked wk16: Crusaders 6th vs Souths 6th'},
    {'teams': ['Port Stephens 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-07-05', 'description': 'Locked wk16: Port Stephens 6th vs University Seapigs 6th'},
    {'teams': ['Tigers Black 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-07-05', 'description': 'Locked wk16: Tigers Black 6th vs University Gentlemen 6th'},
    {'teams': ['Tigers Yellow 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-07-05', 'description': 'Locked wk16: Tigers Yellow 6th vs Wests 6th'},
    {'teams': ['Crusaders 3rd', 'Souths 3rd'], 'grade': '3rd', 'date': '2026-07-12', 'description': 'Locked wk17: Crusaders 3rd vs Souths 3rd'},
    {'teams': ['Maitland 3rd', 'Norths 3rd'], 'grade': '3rd', 'date': '2026-07-12', 'description': 'Locked wk17: Maitland 3rd vs Norths 3rd'},
    {'teams': ['Port Stephens 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-07-12', 'description': 'Locked wk17: Port Stephens 3rd vs Wests 3rd'},
    {'teams': ['Tigers 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-07-12', 'description': 'Locked wk17: Tigers 3rd vs University 3rd'},
    {'teams': ['Colts 4th', 'Norths 4th'], 'grade': '4th', 'date': '2026-07-12', 'description': 'Locked wk17: Colts 4th vs Norths 4th'},
    {'teams': ['Crusaders 4th', 'Port Stephens 4th'], 'grade': '4th', 'date': '2026-07-12', 'description': 'Locked wk17: Crusaders 4th vs Port Stephens 4th'},
    {'teams': ['Maitland 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-07-12', 'description': 'Locked wk17: Maitland 4th vs Wests Red 4th'},
    {'teams': ['Tigers 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-07-12', 'description': 'Locked wk17: Tigers 4th vs University Seapigs 4th'},
    {'teams': ['University Redhogs 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-07-12', 'description': 'Locked wk17: University Redhogs 4th vs Wests Green 4th'},
    {'teams': ['Colts Gold 5th', 'Tigers 5th'], 'grade': '5th', 'date': '2026-07-12', 'description': 'Locked wk17: Colts Gold 5th vs Tigers 5th'},
    {'teams': ['Colts Green 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-07-12', 'description': 'Locked wk17: Colts Green 5th vs Wests Red 5th'},
    {'teams': ['Maitland 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-07-12', 'description': 'Locked wk17: Maitland 5th vs Wests Green 5th'},
    {'teams': ['Norths 5th', 'University 5th'], 'grade': '5th', 'date': '2026-07-12', 'description': 'Locked wk17: Norths 5th vs University 5th'},
    {'teams': ['Colts 6th', 'Souths 6th'], 'grade': '6th', 'date': '2026-07-12', 'description': 'Locked wk17: Colts 6th vs Souths 6th'},
    {'teams': ['Crusaders 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-07-12', 'description': 'Locked wk17: Crusaders 6th vs Wests 6th'},
    {'teams': ['Maitland 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-07-12', 'description': 'Locked wk17: Maitland 6th vs University Gentlemen 6th'},
    {'teams': ['Port Stephens 6th', 'Tigers Yellow 6th'], 'grade': '6th', 'date': '2026-07-12', 'description': 'Locked wk17: Port Stephens 6th vs Tigers Yellow 6th'},
    {'teams': ['Tigers Black 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-07-12', 'description': 'Locked wk17: Tigers Black 6th vs University Seapigs 6th'},
    {'teams': ['Crusaders 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-07-19', 'description': 'Locked wk18: Crusaders 3rd vs Wests 3rd'},
    {'teams': ['Maitland 3rd', 'Souths 3rd'], 'grade': '3rd', 'date': '2026-07-19', 'description': 'Locked wk18: Maitland 3rd vs Souths 3rd'},
    {'teams': ['Norths 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-07-19', 'description': 'Locked wk18: Norths 3rd vs University 3rd'},
    {'teams': ['Port Stephens 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-07-19', 'description': 'Locked wk18: Port Stephens 3rd vs Tigers 3rd'},
    {'teams': ['Colts 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-07-19', 'description': 'Locked wk18: Colts 4th vs Wests Green 4th'},
    {'teams': ['Crusaders 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-07-19', 'description': 'Locked wk18: Crusaders 4th vs University Seapigs 4th'},
    {'teams': ['Maitland 4th', 'Norths 4th'], 'grade': '4th', 'date': '2026-07-19', 'description': 'Locked wk18: Maitland 4th vs Norths 4th'},
    {'teams': ['Port Stephens 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-07-19', 'description': 'Locked wk18: Port Stephens 4th vs University Redhogs 4th'},
    {'teams': ['Souths 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-07-19', 'description': 'Locked wk18: Souths 4th vs Tigers 4th'},
    {'teams': ['Colts Green 5th', 'University 5th'], 'grade': '5th', 'date': '2026-07-19', 'description': 'Locked wk18: Colts Green 5th vs University 5th'},
    {'teams': ['Crusaders 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-07-19', 'description': 'Locked wk18: Crusaders 5th vs Wests Red 5th'},
    {'teams': ['Maitland 5th', 'Norths 5th'], 'grade': '5th', 'date': '2026-07-19', 'description': 'Locked wk18: Maitland 5th vs Norths 5th'},
    {'teams': ['Tigers 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-07-19', 'description': 'Locked wk18: Tigers 5th vs Wests Green 5th'},
    {'teams': ['Colts 6th', 'Port Stephens 6th'], 'grade': '6th', 'date': '2026-07-19', 'description': 'Locked wk18: Colts 6th vs Port Stephens 6th'},
    {'teams': ['Crusaders 6th', 'Tigers Yellow 6th'], 'grade': '6th', 'date': '2026-07-19', 'description': 'Locked wk18: Crusaders 6th vs Tigers Yellow 6th'},
    {'teams': ['Maitland 6th', 'Souths 6th'], 'grade': '6th', 'date': '2026-07-19', 'description': 'Locked wk18: Maitland 6th vs Souths 6th'},
    {'teams': ['University Seapigs 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-07-19', 'description': 'Locked wk18: University Seapigs 6th vs Wests 6th'},
    {'teams': ['Crusaders 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-07-26', 'description': 'Locked wk19: Crusaders 3rd vs University 3rd'},
    {'teams': ['Maitland 3rd', 'Port Stephens 3rd'], 'grade': '3rd', 'date': '2026-07-26', 'description': 'Locked wk19: Maitland 3rd vs Port Stephens 3rd'},
    {'teams': ['Norths 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-07-26', 'description': 'Locked wk19: Norths 3rd vs Wests 3rd'},
    {'teams': ['Souths 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-07-26', 'description': 'Locked wk19: Souths 3rd vs Tigers 3rd'},
    {'teams': ['Colts 4th', 'Port Stephens 4th'], 'grade': '4th', 'date': '2026-07-26', 'description': 'Locked wk19: Colts 4th vs Port Stephens 4th'},
    {'teams': ['Crusaders 4th', 'Norths 4th'], 'grade': '4th', 'date': '2026-07-26', 'description': 'Locked wk19: Crusaders 4th vs Norths 4th'},
    {'teams': ['Maitland 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-07-26', 'description': 'Locked wk19: Maitland 4th vs Tigers 4th'},
    {'teams': ['Souths 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-07-26', 'description': 'Locked wk19: Souths 4th vs Wests Red 4th'},
    {'teams': ['University Redhogs 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-07-26', 'description': 'Locked wk19: University Redhogs 4th vs University Seapigs 4th'},
    {'teams': ['Colts Gold 5th', 'Maitland 5th'], 'grade': '5th', 'date': '2026-07-26', 'description': 'Locked wk19: Colts Gold 5th vs Maitland 5th'},
    {'teams': ['Crusaders 5th', 'Tigers 5th'], 'grade': '5th', 'date': '2026-07-26', 'description': 'Locked wk19: Crusaders 5th vs Tigers 5th'},
    {'teams': ['University 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-07-26', 'description': 'Locked wk19: University 5th vs Wests Green 5th'},
    {'teams': ['Colts 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-07-26', 'description': 'Locked wk19: Colts 6th vs Wests 6th'},
    {'teams': ['Crusaders 6th', 'Tigers Black 6th'], 'grade': '6th', 'date': '2026-07-26', 'description': 'Locked wk19: Crusaders 6th vs Tigers Black 6th'},
    {'teams': ['Souths 6th', 'Tigers Yellow 6th'], 'grade': '6th', 'date': '2026-07-26', 'description': 'Locked wk19: Souths 6th vs Tigers Yellow 6th'},
    {'teams': ['University Gentlemen 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-07-26', 'description': 'Locked wk19: University Gentlemen 6th vs University Seapigs 6th'},
    {'teams': ['Crusaders 3rd', 'Maitland 3rd'], 'grade': '3rd', 'date': '2026-08-02', 'description': 'Locked wk20: Crusaders 3rd vs Maitland 3rd'},
    {'teams': ['Norths 3rd', 'Souths 3rd'], 'grade': '3rd', 'date': '2026-08-02', 'description': 'Locked wk20: Norths 3rd vs Souths 3rd'},
    {'teams': ['University 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-08-02', 'description': 'Locked wk20: University 3rd vs Wests 3rd'},
    {'teams': ['Colts 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-08-02', 'description': 'Locked wk20: Colts 4th vs University Seapigs 4th'},
    {'teams': ['Crusaders 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-08-02', 'description': 'Locked wk20: Crusaders 4th vs Tigers 4th'},
    {'teams': ['Maitland 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-08-02', 'description': 'Locked wk20: Maitland 4th vs Wests Green 4th'},
    {'teams': ['Port Stephens 4th', 'Souths 4th'], 'grade': '4th', 'date': '2026-08-02', 'description': 'Locked wk20: Port Stephens 4th vs Souths 4th'},
    {'teams': ['University Redhogs 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-08-02', 'description': 'Locked wk20: University Redhogs 4th vs Wests Red 4th'},
    {'teams': ['Colts Gold 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-08-02', 'description': 'Locked wk20: Colts Gold 5th vs Wests Green 5th'},
    {'teams': ['Colts Green 5th', 'Tigers 5th'], 'grade': '5th', 'date': '2026-08-02', 'description': 'Locked wk20: Colts Green 5th vs Tigers 5th'},
    {'teams': ['Crusaders 5th', 'Maitland 5th'], 'grade': '5th', 'date': '2026-08-02', 'description': 'Locked wk20: Crusaders 5th vs Maitland 5th'},
    {'teams': ['University 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-08-02', 'description': 'Locked wk20: University 5th vs Wests Red 5th'},
    {'teams': ['Colts 6th', 'Crusaders 6th'], 'grade': '6th', 'date': '2026-08-02', 'description': 'Locked wk20: Colts 6th vs Crusaders 6th'},
    {'teams': ['Maitland 6th', 'Tigers Black 6th'], 'grade': '6th', 'date': '2026-08-02', 'description': 'Locked wk20: Maitland 6th vs Tigers Black 6th'},
    {'teams': ['Port Stephens 6th', 'Souths 6th'], 'grade': '6th', 'date': '2026-08-02', 'description': 'Locked wk20: Port Stephens 6th vs Souths 6th'},
    {'teams': ['Tigers Yellow 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-08-02', 'description': 'Locked wk20: Tigers Yellow 6th vs University Seapigs 6th'},
    {'teams': ['University Gentlemen 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-08-02', 'description': 'Locked wk20: University Gentlemen 6th vs Wests 6th'},
    {'teams': ['Crusaders 3rd', 'Norths 3rd'], 'grade': '3rd', 'date': '2026-08-09', 'description': 'Locked wk21: Crusaders 3rd vs Norths 3rd'},
    {'teams': ['Maitland 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-08-09', 'description': 'Locked wk21: Maitland 3rd vs Tigers 3rd'},
    {'teams': ['Port Stephens 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-08-09', 'description': 'Locked wk21: Port Stephens 3rd vs University 3rd'},
    {'teams': ['Souths 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-08-09', 'description': 'Locked wk21: Souths 3rd vs Wests 3rd'},
    {'teams': ['Colts 4th', 'Maitland 4th'], 'grade': '4th', 'date': '2026-08-09', 'description': 'Locked wk21: Colts 4th vs Maitland 4th'},
    {'teams': ['Crusaders 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-08-09', 'description': 'Locked wk21: Crusaders 4th vs Wests Red 4th'},
    {'teams': ['Norths 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-08-09', 'description': 'Locked wk21: Norths 4th vs Tigers 4th'},
    {'teams': ['Port Stephens 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-08-09', 'description': 'Locked wk21: Port Stephens 4th vs University Seapigs 4th'},
    {'teams': ['Souths 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-08-09', 'description': 'Locked wk21: Souths 4th vs Wests Green 4th'},
    {'teams': ['Colts Gold 5th', 'University 5th'], 'grade': '5th', 'date': '2026-08-09', 'description': 'Locked wk21: Colts Gold 5th vs University 5th'},
    {'teams': ['Crusaders 5th', 'Norths 5th'], 'grade': '5th', 'date': '2026-08-09', 'description': 'Locked wk21: Crusaders 5th vs Norths 5th'},
    {'teams': ['Maitland 5th', 'Tigers 5th'], 'grade': '5th', 'date': '2026-08-09', 'description': 'Locked wk21: Maitland 5th vs Tigers 5th'},
    {'teams': ['Colts 6th', 'Tigers Yellow 6th'], 'grade': '6th', 'date': '2026-08-09', 'description': 'Locked wk21: Colts 6th vs Tigers Yellow 6th'},
    {'teams': ['Crusaders 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-08-09', 'description': 'Locked wk21: Crusaders 6th vs University Gentlemen 6th'},
    {'teams': ['Maitland 6th', 'Port Stephens 6th'], 'grade': '6th', 'date': '2026-08-09', 'description': 'Locked wk21: Maitland 6th vs Port Stephens 6th'},
    {'teams': ['Souths 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-08-09', 'description': 'Locked wk21: Souths 6th vs University Seapigs 6th'},
    {'teams': ['Tigers Black 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-08-09', 'description': 'Locked wk21: Tigers Black 6th vs Wests 6th'},
    {'teams': ['Crusaders 3rd', 'Port Stephens 3rd'], 'grade': '3rd', 'date': '2026-08-16', 'description': 'Locked wk22: Crusaders 3rd vs Port Stephens 3rd'},
    {'teams': ['Norths 3rd', 'Tigers 3rd'], 'grade': '3rd', 'date': '2026-08-16', 'description': 'Locked wk22: Norths 3rd vs Tigers 3rd'},
    {'teams': ['Colts 4th', 'Souths 4th'], 'grade': '4th', 'date': '2026-08-16', 'description': 'Locked wk22: Colts 4th vs Souths 4th'},
    {'teams': ['Maitland 4th', 'Port Stephens 4th'], 'grade': '4th', 'date': '2026-08-16', 'description': 'Locked wk22: Maitland 4th vs Port Stephens 4th'},
    {'teams': ['Norths 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-08-16', 'description': 'Locked wk22: Norths 4th vs University Redhogs 4th'},
    {'teams': ['Tigers 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-08-16', 'description': 'Locked wk22: Tigers 4th vs Wests Green 4th'},
    {'teams': ['Colts Gold 5th', 'Colts Green 5th'], 'grade': '5th', 'date': '2026-08-16', 'description': 'Locked wk22: Colts Gold 5th vs Colts Green 5th'},
    {'teams': ['Crusaders 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-08-16', 'description': 'Locked wk22: Crusaders 5th vs Wests Green 5th'},
    {'teams': ['Norths 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-08-16', 'description': 'Locked wk22: Norths 5th vs Wests Red 5th'},
    {'teams': ['Crusaders 6th', 'Maitland 6th'], 'grade': '6th', 'date': '2026-08-16', 'description': 'Locked wk22: Crusaders 6th vs Maitland 6th'},
    {'teams': ['Souths 6th', 'Tigers Black 6th'], 'grade': '6th', 'date': '2026-08-16', 'description': 'Locked wk22: Souths 6th vs Tigers Black 6th'},
    {'teams': ['Crusaders 3rd', 'Souths 3rd'], 'grade': '3rd', 'date': '2026-08-23', 'description': 'Locked wk23: Crusaders 3rd vs Souths 3rd'},
    {'teams': ['Maitland 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-08-23', 'description': 'Locked wk23: Maitland 3rd vs University 3rd'},
    {'teams': ['Tigers 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-08-23', 'description': 'Locked wk23: Tigers 3rd vs Wests 3rd'},
    {'teams': ['Colts 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-08-23', 'description': 'Locked wk23: Colts 4th vs Tigers 4th'},
    {'teams': ['Crusaders 4th', 'Souths 4th'], 'grade': '4th', 'date': '2026-08-23', 'description': 'Locked wk23: Crusaders 4th vs Souths 4th'},
    {'teams': ['Maitland 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-08-23', 'description': 'Locked wk23: Maitland 4th vs University Redhogs 4th'},
    {'teams': ['Port Stephens 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-08-23', 'description': 'Locked wk23: Port Stephens 4th vs Wests Red 4th'},
    {'teams': ['University Seapigs 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-08-23', 'description': 'Locked wk23: University Seapigs 4th vs Wests Green 4th'},
    {'teams': ['Colts Gold 5th', 'Norths 5th'], 'grade': '5th', 'date': '2026-08-23', 'description': 'Locked wk23: Colts Gold 5th vs Norths 5th'},
    {'teams': ['Colts Green 5th', 'Wests Green 5th'], 'grade': '5th', 'date': '2026-08-23', 'description': 'Locked wk23: Colts Green 5th vs Wests Green 5th'},
    {'teams': ['Crusaders 5th', 'University 5th'], 'grade': '5th', 'date': '2026-08-23', 'description': 'Locked wk23: Crusaders 5th vs University 5th'},
    {'teams': ['Tigers 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-08-23', 'description': 'Locked wk23: Tigers 5th vs Wests Red 5th'},
    {'teams': ['Colts 6th', 'Souths 6th'], 'grade': '6th', 'date': '2026-08-23', 'description': 'Locked wk23: Colts 6th vs Souths 6th'},
    {'teams': ['Crusaders 6th', 'Port Stephens 6th'], 'grade': '6th', 'date': '2026-08-23', 'description': 'Locked wk23: Crusaders 6th vs Port Stephens 6th'},
    {'teams': ['Maitland 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-08-23', 'description': 'Locked wk23: Maitland 6th vs University Gentlemen 6th'},
    {'teams': ['Tigers Black 6th', 'Tigers Yellow 6th'], 'grade': '6th', 'date': '2026-08-23', 'description': 'Locked wk23: Tigers Black 6th vs Tigers Yellow 6th'},
    {'teams': ['University Seapigs 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-08-23', 'description': 'Locked wk23: University Seapigs 6th vs Wests 6th'},
    {'teams': ['Crusaders 3rd', 'University 3rd'], 'grade': '3rd', 'date': '2026-08-30', 'description': 'Locked wk24: Crusaders 3rd vs University 3rd'},
    {'teams': ['Maitland 3rd', 'Norths 3rd'], 'grade': '3rd', 'date': '2026-08-30', 'description': 'Locked wk24: Maitland 3rd vs Norths 3rd'},
    {'teams': ['Port Stephens 3rd', 'Wests 3rd'], 'grade': '3rd', 'date': '2026-08-30', 'description': 'Locked wk24: Port Stephens 3rd vs Wests 3rd'},
    {'teams': ['Colts 4th', 'Wests Red 4th'], 'grade': '4th', 'date': '2026-08-30', 'description': 'Locked wk24: Colts 4th vs Wests Red 4th'},
    {'teams': ['Crusaders 4th', 'Wests Green 4th'], 'grade': '4th', 'date': '2026-08-30', 'description': 'Locked wk24: Crusaders 4th vs Wests Green 4th'},
    {'teams': ['Norths 4th', 'University Seapigs 4th'], 'grade': '4th', 'date': '2026-08-30', 'description': 'Locked wk24: Norths 4th vs University Seapigs 4th'},
    {'teams': ['Port Stephens 4th', 'Tigers 4th'], 'grade': '4th', 'date': '2026-08-30', 'description': 'Locked wk24: Port Stephens 4th vs Tigers 4th'},
    {'teams': ['Souths 4th', 'University Redhogs 4th'], 'grade': '4th', 'date': '2026-08-30', 'description': 'Locked wk24: Souths 4th vs University Redhogs 4th'},
    {'teams': ['Colts Green 5th', 'Norths 5th'], 'grade': '5th', 'date': '2026-08-30', 'description': 'Locked wk24: Colts Green 5th vs Norths 5th'},
    {'teams': ['Crusaders 5th', 'Tigers 5th'], 'grade': '5th', 'date': '2026-08-30', 'description': 'Locked wk24: Crusaders 5th vs Tigers 5th'},
    {'teams': ['Maitland 5th', 'University 5th'], 'grade': '5th', 'date': '2026-08-30', 'description': 'Locked wk24: Maitland 5th vs University 5th'},
    {'teams': ['Wests Green 5th', 'Wests Red 5th'], 'grade': '5th', 'date': '2026-08-30', 'description': 'Locked wk24: Wests Green 5th vs Wests Red 5th'},
    {'teams': ['Colts 6th', 'Tigers Black 6th'], 'grade': '6th', 'date': '2026-08-30', 'description': 'Locked wk24: Colts 6th vs Tigers Black 6th'},
    {'teams': ['Crusaders 6th', 'Tigers Yellow 6th'], 'grade': '6th', 'date': '2026-08-30', 'description': 'Locked wk24: Crusaders 6th vs Tigers Yellow 6th'},
    {'teams': ['Maitland 6th', 'University Seapigs 6th'], 'grade': '6th', 'date': '2026-08-30', 'description': 'Locked wk24: Maitland 6th vs University Seapigs 6th'},
    {'teams': ['Port Stephens 6th', 'University Gentlemen 6th'], 'grade': '6th', 'date': '2026-08-30', 'description': 'Locked wk24: Port Stephens 6th vs University Gentlemen 6th'},
    {'teams': ['Souths 6th', 'Wests 6th'], 'grade': '6th', 'date': '2026-08-30', 'description': 'Locked wk24: Souths 6th vs Wests 6th'},
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
    '2nd': 20,   # 20 Sundays only (22 calendar Sundays minus 2 PHL-only weekends)
    '3rd': 20,   # 20 Sundays only
    '4th': 20,   # 20 Sundays only
    '5th': 20,   # 20 Sundays only
    '6th': 20,   # 20 Sundays only
}
# PHL-only Sundays (non-PHL blocked via BLOCKED_GAMES):
#   - May 17 (week 9)  — Masters State Championship
#   - Jun 21 (week 14) — U16 Girls State Championship

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
#   ClubVsClubAlignment:  (no base limit config — slack reduces required coincidences)
#   ClubGameSpread (spec-024): per-field contiguity, gap_cap = max(0, min(1, n-3)) + slack;
#                              plus an off-primary-field soft penalty per (club, week, day)

CONSTRAINT_DEFAULTS = {
    'spacing_base_slack': 2,               # EqualMatchUpSpacing: additional base slack (0 = start at ideal)
    # spec-018: maitland_max_consecutive_home / away_maitland_max_clubs removed
    # (venue-sequencing rules deleted).
    # spec-024: max_clubs_per_field removed with MinimiseClubsOnAFieldBroadmeadow.
    'club_game_spread_max_gap': 1,         # ClubGameSpread: max allowed gap (spread) per club per day (+ slack at runtime)
    'club_game_spread_max_overlap': 0,     # ClubGameSpread: max allowed double-ups (+ slack at runtime; 0 = no overlap)
    'club_vs_club_alignment_base_slack': 1, # ClubVsClubAlignment: with --slack 1, effective slack = 2, min_required = num_games - 2
    'gosford_friday_games': 8,             # PHLAndSecondGradeTimes: exact number of Friday PHL games at Gosford (AGM decision)
    'maitland_friday_games': 2,            # PHLAndSecondGradeTimes: exact number of Friday PHL games at Maitland (Gosford vs Maitland only)
    'max_friday_broadmeadow': 3,           # PHLAndSecondGradeTimes: max Friday PHL games at NIHC (Broadmeadow)
}

# ============== Preferred / Avoided Weekends at Away Grounds (spec-006) ==============
# Soft preferences for scheduling (or not scheduling) games at a specific
# away venue on specific dates.  Read by the `PreferredWeekendsAwayGround`
# atom in the `soft_optimisation` stage.  Never blocks feasibility.
#
# Entry format:
#   'date'           — single date string 'YYYY-MM-DD'
#   'dates'          — OR a list of date strings (mutually exclusive with 'date')
#   'field_location' — venue string matching field_location in decision variable keys
#   'field_name'     — optional; venue-level if omitted (any field at the location)
#   'mode'           — 'avoid': penalty per game scheduled there on that date
#                      'prefer': penalty for each game MISSING (target_count default = 1)
#   'weight'         — optional per-entry override; default from PENALTY_WEIGHTS
#   'description'    — human-readable label (ignored by solver)
#
# 2026 NRL Knights home games at Maitland Park
# Maitland HC does not want to play on these dates due to NRL home matches.
# Source: docs/seasonal/2026/operational_TODO.md [DONE: implemented in spec-006]

PREFERRED_WEEKENDS = [
    {
        'date': '2026-04-05',
        'field_location': 'Maitland Park',
        'mode': 'avoid',
        'description': 'NRL Knights vs Raiders at Maitland (Maitland HC prefers not to play)',
        'note': True,  # spec-028: surface in published draw notes column
    },
    {
        'date': '2026-04-26',
        'field_location': 'Maitland Park',
        'mode': 'avoid',
        'description': 'NRL Knights vs Panthers at Maitland (Maitland HC prefers not to play)',
    },
    {
        'date': '2026-05-03',
        'field_location': 'Maitland Park',
        'mode': 'avoid',
        'description': 'NRL Knights vs Rabbitohs at Maitland (Maitland HC prefers not to play)',
    },
    {
        'date': '2026-06-28',
        'field_location': 'Maitland Park',
        'mode': 'avoid',
        'description': 'NRL Knights vs Wests Tigers at Maitland (Maitland HC prefers not to play)',
    },
    {
        'date': '2026-07-05',
        'field_location': 'Maitland Park',
        'mode': 'avoid',
        'description': 'NRL Knights vs Dolphins at Maitland (Maitland HC prefers not to play)',
    },
    {
        'date': '2026-08-16',
        'field_location': 'Maitland Park',
        'mode': 'avoid',
        'description': 'NRL Knights vs Titans at Maitland (Maitland HC prefers not to play)',
    },
]

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
    # spec-018: MaitlandHomeGrouping / AwayAtMaitlandGrouping penalty weights
    # removed — venue-sequencing soft penalties deleted.
    'ClubVsClubAlignment':                 50_000,
    'EqualMatchUpSpacing':                100_000,
    'ClubGameSpread':                     100_000,
    'ClubFieldConcentration':              80_000,
    'PreferredTimesConstraint':           200_000,
    'ClubVsClubAlignmentField':                 0,  # Superseded by ClubFieldConcentration
    'ClubGradeAdjacencyConstraint':        50_000,
    # spec-020: 'phl_preferences' weight removed — PreferredDates deleted; the
    # marquee-PHL-date behaviour is now a PREFERRED_GAMES entry using the
    # 'preferred_games' weight below.
    # spec-024: MaximiseClubsPerTimeslotBroadmeadow / MinimiseClubsOnAFieldBroadmeadow
    # weights removed (constraints deleted). Club-spread pressure is now the
    # ClubGameSpread weight (per-field holes + off-primary-field games).
    # Penalty per dummy slot used. Higher = solver avoids dummy slots more strongly.
    # Set to 0 to allow free use of dummy slots (no penalty).
    'dummy_slots':                      1_000_000,
    # Soft lex matchup ordering: tiny tie-break, never overrides real constraints.
    'soft_lex_ordering':                        1,
    # spec-016: NIHC field-fill order (WF→EF→SF) soft symmetry-breaker.
    # Penalty per out-of-order fill (EF without WF, or SF without EF). Small —
    # just above soft_lex_ordering so field order wins over the pure
    # alphabetical tie-break, but never dominates real soft constraints.
    'nihc_fill_order':                          5,
    # spec-006: preferred / avoided away-ground weekends (e.g. NRL clash dates).
    # Each 'avoid' entry incurs this penalty per game scheduled at the venue on that date.
    # Each 'prefer' entry incurs this penalty per game MISSING from the venue on that date.
    'preferred_weekends_away_ground':           1_000,
    # spec-020: soft, weighted FORCED_GAMES analogue. Single shared bucket for
    # all PREFERRED_GAMES entries; per-entry `weight` acts as a multiplier on
    # top of this default. Matches the old `phl_preferences` weight (the
    # marquee-PHL-date behaviour PreferredDates used to provide).
    'preferred_games':                         10_000,
    # spec-018: 'maitland_alternate_home_away' (spec-012) removed — the
    # alternation soft penalty was deleted.
}

# ============== Season Configuration ==============

SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),   # Sunday 22nd March (first playing day)
    'end_date': datetime(2026, 8, 30),     # Sunday 30th August (last club game before finals)
    
    # Default max rounds (used as fallback if grade not in MAX_WEEKENDS_PER_GRADE)
    # This is the default MAXIMUM weekends any grade can play
    'max_rounds': 20,
    # Dummy overflow slots: not attached to a real time/venue, eases solver burden.
    # Adjust count as needed. Penalty for using them is set in PENALTY_WEIGHTS['dummy_slots'].
    'num_dummy_timeslots': 0,
    
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

    # Preferred / avoided away-ground weekends (spec-006)
    'preferred_weekends': PREFERRED_WEEKENDS,

    # Preferred games — soft weighted FORCED analogue (spec-020)
    'preferred_games': PREFERRED_GAMES,
    
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

    # Objective lower bound: prune any branch with objective below this value.
    # Speeds up proving optimality by cutting off clearly bad search space.
    # Set to None to disable. Typical value: -400000 (based on observed solver runs).
    'objective_lower_bound': -500_000,
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
