# config/defaults.py
"""
Perennial default configuration.

Contains settings that carry over year to year and do NOT change between seasons.
Season-specific configs should import from here and extend as needed.

Things that belong here:
  - Field definitions (venues don't change)
  - Game times by venue/day (standard timeslots)
  - Perennial BLOCKED_GAMES (e.g., rounds 1-2 at Broadmeadow only)
  - Constraint defaults that are standing policy
  - Home field mappings
  - Grade order

Things that do NOT belong here:
  - Field unavailabilities (change every year)
  - Club days (change every year)
  - Season dates
  - Forced games
  - Team-specific blocked games
  - PHL preferences / preferred dates
"""

from datetime import time as tm

# ============== Playing Fields ==============
# Standard venues - same every year unless a new venue is added

FIELDS = [
    {'location': 'Newcastle International Hockey Centre', 'name': 'SF'},  # South Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'EF'},  # East Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'WF'},  # West Field
    {'location': 'Maitland Park', 'name': 'Maitland Main Field'},
    {'location': 'Central Coast Hockey Park', 'name': 'Wyong Main Field'},
]

# ============== Game Times by Venue/Day ==============
# Standard game times - these stay year to year

DAY_TIME_MAP = {
    'Newcastle International Hockey Centre': {
        'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    },
    'Central Coast Hockey Park': {
        'Sunday': [tm(12, 0), tm(13, 30)],
    }
}

# ============== Home Field Mappings ==============
# Clubs not listed default to Newcastle International Hockey Centre

HOME_FIELD_MAP = {
    'Maitland': 'Maitland Park',
    'Gosford': 'Central Coast Hockey Park',
}

# ============== Grade Order ==============

GRADE_ORDER = ['PHL', '2nd', '3rd', '4th', '5th', '6th']

# ============== Perennial Blocked Games ==============
# Standing rules that apply every season.
# Season configs should include these in their BLOCKED_GAMES list.
#
# See docs/PERENNIAL_RULES.md for rationale.

PERENNIAL_BLOCKED_GAMES = [
    # === Rounds 1-2: All games at Broadmeadow only ===
    # No games at Maitland Park or Central Coast in the first two playing rounds.
    {'round_no': 1, 'field_location': 'Maitland Park',
     'description': 'Rounds 1-2 at Broadmeadow only (perennial rule)',
     'reason': 'All games at central venue for opening rounds'},
    {'round_no': 2, 'field_location': 'Maitland Park',
     'description': 'Rounds 1-2 at Broadmeadow only (perennial rule)',
     'reason': 'All games at central venue for opening rounds'},
    {'round_no': 1, 'field_location': 'Central Coast Hockey Park',
     'description': 'Rounds 1-2 at Broadmeadow only (perennial rule)',
     'reason': 'All games at central venue for opening rounds'},
    {'round_no': 2, 'field_location': 'Central Coast Hockey Park',
     'description': 'Rounds 1-2 at Broadmeadow only (perennial rule)',
     'reason': 'All games at central venue for opening rounds'},
]
