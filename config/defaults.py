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
# Per-club tuning for away-ground (non-default-home) venues. Adding/removing
# a club here scopes the venue-specific Friday-game counts to a new club. Keys
# not set fall back to perennial CONSTRAINT_DEFAULTS values.
#
#   friday_games — exact PHL Friday games at this venue per season
#
# spec-018: `max_consecutive_home` / `max_away_clubs` removed — the
# venue-sequencing rules (`NonDefaultHomeGrouping` / `AwayAtNonDefaultGrouping`)
# that read them were deleted.

AWAY_VENUE_RULES = {
    'Maitland': {
        # No explicit overrides — Maitland falls back to CONSTRAINT_DEFAULTS
        # (`maitland_friday_games`). Season configs may override here per-club
        # without touching the global defaults.
    },
    'Gosford': {
        'friday_games': 8,
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
    # spec-008 Part B: bye-spacing base slack (own knob, separate from
    # matchup `spacing_base_slack`). Loosens the per-team bye spread.
    'bye_spacing_base_slack': 0,
    # Friday-night game counts
    'max_friday_broadmeadow': 3,
    'gosford_friday_games': 8,
    'maitland_friday_games': 2,
    # spec-015: 'gosford_friday_rounds' removed — it only fed the deleted
    # GosfordFridayRoundsForced atom. Gosford Friday rounds are now FORCED_GAMES
    # count entries in the season config (see FORCED_GAMES_AS_COUNT_RULES.md).
    # spec-018: `maitland_max_consecutive_home` / `away_maitland_max_clubs`
    # removed — the venue-sequencing rules they tuned were deleted.
    # Broadmeadow field counts
    'max_clubs_per_field': 5,
    # Club game spread
    'club_game_spread_max_gap': 2,
    'club_game_spread_max_overlap': 0,
    # Club-vs-club alignment
    'club_vs_club_alignment_base_slack': 0,
    # spec-014: PHL/2nd same-club adjacency. The same-venue rule is a
    # slot-adjacency rule (no minute threshold); the cross-venue rule requires
    # this minimum start-time gap in REAL minutes (game length + warm-down +
    # travel + warm-up, measured start-to-start). 180 = 3 h.
    'phl_2nd_cross_venue_min_minutes': 180,
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
            # spec-014: PHL/2nd same-club adjacency — same-venue games must be
            # back-to-back on one field; cross-venue games >= 180-min start gap.
            # Non-engine atom (dispatched via the stages.py fallback). NOTE:
            # under locked-week runs this can over-constrain Gosford PHL (zero
            # margin) — exclude it via --exclude PHLAnd2ndAdjacency for locked
            # re-solves.
            'PHLAnd2ndAdjacency',
            # spec-010: PHLRoundOnePlay removed (atom + registry entry deleted)
            # — convenor uses FORCED_GAMES to express "team X plays round 1".
            # spec-015: GosfordFridayRoundsForced removed — Gosford Friday rounds
            # are FORCED_GAMES count entries (see FORCED_GAMES_AS_COUNT_RULES.md).
            # spec-007: hard same-grade-same-club rule (was the hard portion
            # of the obsolete `ClubGradeAdjacency` cluster).
            'SameGradeSameClubNoConcurrency',
            # spec-016: NIHC field-fill ordering (WF→EF→SF) moved to
            # soft_optimisation as a SOFT symmetry-breaker (was hard here).
            # spec-008 Part B: byes spread evenly across the season for
            # every team in every grade (HARD, own slack key).
            'BalancedByeSpacing',
            # spec-017: matchup-spacing promoted to HARD here (was only in the
            # soft_only stage, so its hard part never applied in production).
            # Sits right after BalancedByeSpacing — same "spread it out" intent.
            # Both its hard (forbidden gaps) and soft (density penalty) parts
            # now apply, since apply_solver_stage always runs apply_stage_2_soft.
            # Slack via --slack EqualMatchUpSpacingConstraint N. Byes stay a
            # separate atom with their own slack key (deliberate, not merged).
            'EqualMatchUpSpacing',
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
            # spec-018: `NonDefaultHomeGrouping` / `AwayAtNonDefaultGrouping`
            # removed — the convenor no longer wants venue-sequencing enforced.
            # Per-club home-weekend counts + per-pair/aggregate 50/50 balance
            # (the two atoms above) are KEPT.
            'AwayClubHomeWeekendsCount',
            'AwayClubPerOpponentAndAggregateHomeBalance',
        ],
    },
    {
        'name': 'club_alignment',
        'description': 'Cross-grade stacked weekends + co-location',
        'atoms': [
            # spec-007: `ClubGradeAdjacency` removed entirely. Hard portion is
            # `SameGradeSameClubNoConcurrency` (in `critical_feasibility`); the
            # soft adjacent-grade portion is gone.
            #
            # spec-005: replaces the four obsolete Phase-3c atoms
            # (`ClubVsClubCoincidence`, `ClubVsClubFieldLimit`,
            # `ClubVsClubDeficitPenalty`, `PHLAnd2ndBackToBackSameField`) with
            # the precise stacking-and-co-location pair. The old atoms remain
            # in the registry + on disk as parity reference; they are NOT in
            # any production stage. ORDER MATTERS — Weekends must run before
            # CoLocation so the `cvc_stack_play` indicators exist when the
            # co-location atom looks them up.
            'ClubVsClubStackedWeekends', 'ClubVsClubStackedCoLocation',
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
            # spec-017: 'EqualMatchUpSpacing' moved to critical_feasibility so
            # its HARD part actually applies (this stage is soft_only, which
            # skips apply_stage_1_hard). Its soft part still runs there.
            'ClubGameSpread',
            # spec-005: `ClubVsClubDeficitPenalty` removed — the soft deficit
            # penalty is unnecessary because the stacking atom enforces the
            # exact Sunday-meeting count via a HARD `sum == budget`
            # constraint (no deficit possible at solve time).
            # spec-020: `PreferredDates` replaced by the generic `PreferredGames`
            # soft atom (penalty-on-deviation over the full FORCED grammar).
            'PreferredGames',
            'EnsureBestTimeslotChoices', 'PreferredTimes',
            'MaximiseClubsPerTimeslotBroadmeadow', 'MinimiseClubsOnAFieldBroadmeadow',
            # spec-002: predictable alphabetical matchup tie-break.
            'SoftLexMatchupOrdering',
            # spec-016: NIHC field-fill ordering (WF→EF→SF) as a soft
            # symmetry-breaker — penalises out-of-order field fill.
            'NIHCFillWFBeforeEF', 'NIHCFillEFBeforeSF',
            # spec-007: convenor-supplied per-team-pair no-concurrency soft.
            'TeamPairNoConcurrency',
            # spec-006: preferred / avoided away-ground weekends (e.g. NRL clash dates).
            'PreferredWeekendsAwayGround',
            # spec-018: `MaitlandAlternateHomeAway` (spec-012) removed — its
            # purpose was to push an H-A-H-A sequence, the venue-sequencing the
            # convenor is discarding.
        ],
    },
]


# spec-020: empty default for the soft, weighted FORCED_GAMES analogue. Season
# configs may override with a list of preference entries (same grammar as
# FORCED_GAMES + optional `weight`). Read as `data['preferred_games']` via
# `build_season_data`. Penalty weight lives in each season's PENALTY_WEIGHTS
# dict (`preferred_games`), NOT here — defaults.py has no PENALTY_WEIGHTS.
PREFERRED_GAMES = []


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
