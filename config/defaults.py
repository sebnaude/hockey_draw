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
        # No explicit overrides — Maitland falls back to CONSTRAINT_DEFAULTS
        # (`maitland_max_consecutive_home`, `away_maitland_max_clubs`,
        # `maitland_friday_games`). Season configs may override here per-club
        # without touching the global defaults.
    },
    'Gosford': {
        'max_consecutive_home': 2,    # Gosford allows 2 consecutive home weekends
        'friday_games': 8,
        'max_away_clubs': None,       # no per-week away-clubs cap at Gosford
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
    # spec-007: TeamPairNoConcurrency convenor list. Each entry is
    # (team_a, team_b) or (team_a, team_b, weight_multiplier). Empty by
    # default — season configs append entries when there's a real-world
    # conflict (e.g. siblings in non-adjacent grades).
    'TEAM_PAIR_NO_CONCURRENCY': [],
}


# ============== Solver Stages (Phase 7b) ==============
# Default ordered list of solver stages, each a dict of:
#   name: short identifier (used for --stage-only / --skip-stage CLI)
#   description: human-readable summary
#   atoms: list of canonical constraint names from constraints/registry.py
# Optional fields:
#   time_limit_seconds, use_prior_solution_as_hint, soft_only, requires_complete_solution
#
# Season configs may override `SOLVER_STAGES` to reorder, add, or remove stages.

DEFAULT_STAGES = [
    {
        'name': 'critical_feasibility',
        'description': 'Hard feasibility — every constraint that must hold for a valid draw',
        'atoms': [
            'NoDoubleBookingTeams', 'NoDoubleBookingFields',
            'EqualGamesAndBalanceMatchUps',
            'PHLConcurrencyAtBroadmeadow', 'PHLAnd2ndConcurrencyAtBroadmeadow',
            'GosfordFridayRoundsForced', 'PHLRoundOnePlay',
            # spec-007: hard same-grade-same-club rule (was the hard portion
            # of the obsolete `ClubGradeAdjacency` cluster).
            'SameGradeSameClubNoConcurrency',
            # spec-003: NIHC field-fill ordering — WF before EF, EF before SF.
            # The two implications transitively imply SF -> WF (no third atom).
            'NIHCFillWFBeforeEF', 'NIHCFillEFBeforeSF',
        ],
    },
    {
        'name': 'home_away_balance',
        'description': 'Per-pair home/away + non-default-home grouping',
        'atoms': [
            # spec-004: replaces the obsolete `FiftyFiftyHomeandAway`. Two atoms
            # in cooperation — `AwayClubHomeWeekendsCount` pins per-club Friday
            # / Sunday / total home-weekend counts (FORCED-Friday aware);
            # `AwayClubPerOpponentAndAggregateHomeBalance` enforces per-pair +
            # per-team aggregate home/away balance.
            'AwayClubHomeWeekendsCount',
            'AwayClubPerOpponentAndAggregateHomeBalance',
            'NonDefaultHomeGrouping', 'AwayAtNonDefaultGrouping',
        ],
    },
    {
        'name': 'club_alignment',
        'description': 'Cross-grade coincidence + field limits',
        'atoms': [
            # spec-007: `ClubGradeAdjacency` removed entirely. Hard portion is
            # `SameGradeSameClubNoConcurrency` (in `critical_feasibility`); the
            # soft adjacent-grade portion is gone.
            'ClubVsClubCoincidence', 'ClubVsClubFieldLimit',
            'PHLAnd2ndBackToBackSameField',
        ],
    },
    {
        'name': 'club_day',
        'description': 'Per-club day-of-week constraints',
        'atoms': [
            'ClubDayParticipation', 'ClubDayIntraClubMatchup',
            'ClubDayOpponentMatchup', 'ClubDaySameField', 'ClubDayContiguousSlots',
        ],
    },
    {
        'name': 'soft_optimisation',
        'description': 'Soft penalties and optimisation',
        'soft_only': True,
        'atoms': [
            'EqualMatchUpSpacing', 'ClubGameSpread',
            'ClubVsClubDeficitPenalty', 'PreferredDates',
            'EnsureBestTimeslotChoices', 'PreferredTimes',
            'MaximiseClubsPerTimeslotBroadmeadow', 'MinimiseClubsOnAFieldBroadmeadow',
            # spec-002: predictable alphabetical matchup tie-break.
            'SoftLexMatchupOrdering',
            # spec-007: convenor-supplied per-team-pair no-concurrency soft.
            'TeamPairNoConcurrency',
            # spec-006: preferred / avoided away-ground weekends (e.g. NRL clash dates).
            'PreferredWeekendsAwayGround',
        ],
    },
]


# Each PERENNIAL entry carries `'perennial': True`. This flag is read by
# generate_X (utils.py) as a permission-to-be-overridden marker: a variable
# that matches BOTH a perennial BLOCKED scope AND any FORCED_GAMES scope is
# kept (FORCED wins). Non-perennial BLOCKED entries always eliminate the
# variable, even when a FORCED scope also matches. See spec-001.
PERENNIAL_BLOCKED_GAMES = [
    # === Rounds 1-2: All games at Broadmeadow only ===
    # No games at Maitland Park or Central Coast in the first two playing rounds.
    {'round_no': 1, 'field_location': 'Maitland Park',
     'description': 'Rounds 1-2 at Broadmeadow only (perennial rule)',
     'reason': 'All games at central venue for opening rounds',
     'perennial': True},
    {'round_no': 2, 'field_location': 'Maitland Park',
     'description': 'Rounds 1-2 at Broadmeadow only (perennial rule)',
     'reason': 'All games at central venue for opening rounds',
     'perennial': True},
    {'round_no': 1, 'field_location': 'Central Coast Hockey Park',
     'description': 'Rounds 1-2 at Broadmeadow only (perennial rule)',
     'reason': 'All games at central venue for opening rounds',
     'perennial': True},
    {'round_no': 2, 'field_location': 'Central Coast Hockey Park',
     'description': 'Rounds 1-2 at Broadmeadow only (perennial rule)',
     'reason': 'All games at central venue for opening rounds',
     'perennial': True},
]
