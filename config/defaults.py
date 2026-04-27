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

# ============== Per-Club Away-Venue Rules ==============
# Per-club tuning for the generic "non-default-home" constraints (Phase 6).
# Adding/removing a club here is the only change needed to scope all
# Maitland-style and Gosford-style constraints to a new club. Keys not set
# fall back to perennial CONSTRAINT_DEFAULTS values.
#
#   max_consecutive_home — max consecutive home weeks (NonDefaultHomeGrouping)
#   friday_games         — exact PHL Friday games at this venue per season
#   max_away_clubs       — max distinct away clubs at this venue per week
#                          (AwayAtNonDefaultGrouping)

AWAY_VENUE_RULES = {
    'Maitland': {
        'max_consecutive_home': 1,
        'friday_games': 2,
        'max_away_clubs': 3,
    },
    'Gosford': {
        'max_consecutive_home': 2,
        'friday_games': 8,
        'max_away_clubs': None,  # no per-week away-clubs cap at Gosford
    },
}

# ============== Grade Order ==============

GRADE_ORDER = ['PHL', '2nd', '3rd', '4th', '5th', '6th']

# ============== Perennial Blocked Games ==============
# Standing rules that apply every season.
# Season configs should include these in their BLOCKED_GAMES list.
#
# See docs/PERENNIAL_RULES.md for rationale.

# ============== Default CONSTRAINT_DEFAULTS ==============
# These ship with every season. Season configs may override or extend.
# All numeric thresholds and parameter constants used by constraints
# should resolve through this dict so atoms have one source of truth.

CONSTRAINT_DEFAULTS = {
    # Spacing
    'spacing_base_slack': 0,
    # Friday-night game counts
    'max_friday_broadmeadow': 3,
    'gosford_friday_games': 8,
    'maitland_friday_games': 2,
    'gosford_friday_rounds': [2, 4, 5, 9, 10],
    # Maitland home-game grouping
    'maitland_max_consecutive_home': 1,
    'away_maitland_max_clubs': 3,
    # Broadmeadow field counts
    'max_clubs_per_field': 5,
    # Club game spread
    'club_game_spread_max_gap': 2,
    'club_game_spread_max_overlap': 0,
    # Club-vs-club alignment
    'club_vs_club_alignment_base_slack': 0,
    # PHL/2nd adjacency time window
    'phl_adjacency_window_minutes': 180,
    # Worst timeslot (penalised by EnsureBestTimeslotChoices)
    'worst_timeslot_time': '19:00',
}


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
