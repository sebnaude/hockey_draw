# constraints/unified.py
"""
Unified Constraint Engine for hockey draw scheduling.

Replaces the 19 individual AI constraint classes with a single engine that:
1. Makes ONE pass over X to build all grouping dicts
2. Creates shared indicator variables (consumed by multiple constraints)
3. Splits constraints into 2 stages: hard (feasibility) → soft (penalties + optimization)

Usage:
    engine = UnifiedConstraintEngine(model, X, data)
    engine.build_groupings()
    engine.apply_stage_1_hard()  # Hard constraints (feasibility)
    engine.apply_stage_2_soft()  # Soft penalties + optimization
"""

from ortools.sat.python import cp_model
from collections import defaultdict
from itertools import combinations

from utils import (
    get_teams_from_club, get_club_from_clubname, get_nearest_week_by_date,
    get_duplicated_graded_teams, normalize_club_day,
)
from constraints.helper_vars import HelperVarRegistry, SharedVariablePool
from constraints.registry import run_count_adjusters
from constraints.atoms import (
    PHLConcurrencyAtBroadmeadow,
    PHLAnd2ndConcurrencyAtBroadmeadow,
    PreferredDates,
    ClubDayParticipation,
    ClubDayIntraClubMatchup,
    ClubDayOpponentMatchup,
    ClubDaySameField,
    ClubDayContiguousSlots,
)

# Venue constants
BROADMEADOW = 'Newcastle International Hockey Centre'
GOSFORD = 'Central Coast Hockey Park'
MAITLAND = 'Maitland Park'

AWAY_VENUES = {
    'Maitland': MAITLAND,
    'Gosford': GOSFORD,
}

GRADE_ORDER = ["PHL", "2nd", "3rd", "4th", "5th", "6th"]


class UnifiedConstraintEngine:
    """Single-pass constraint engine with shared groupings and indicators."""

    # PHL_ADJACENCY_MINUTES, MAITLAND_AWAY_HARD_LIMIT, CLUBS_ON_FIELD_HARD_LIMIT
    # and CLUB_GAME_SPREAD_HARD_LIMIT used to live here as class constants.
    # PHL/2nd adjacency moved out entirely (spec-014): it's now the
    # `PHLAnd2ndAdjacency` atom reading constraint_defaults
    # ['phl_2nd_cross_venue_min_minutes']. The others were unused (dead config).
    # BROADMEADOW_MAX_SLOTS stays here as a tuning param (the threshold above
    # which Broadmeadow slot caps relax).
    BROADMEADOW_MAX_SLOTS = 6

    def __init__(self, model: cp_model.CpModel, X: dict, data: dict, skip_constraints=None):
        self.model = model
        self.X = X
        self.data = data
        self.skip_constraints = skip_constraints or set()
        self.locked_weeks = data.get('locked_weeks', set())
        self.teams = data['teams']
        self.clubs = data['clubs']
        self.grades = data['grades']
        self.games = data.get('games', [])
        self.timeslots = data.get('timeslots', [])
        self.fields = data.get('fields', [])
        self.num_rounds = data.get('num_rounds', {})
        self.constraints_added = 0

        # Ensure penalties dict
        if 'penalties' not in data:
            data['penalties'] = {}

        # Slack config
        self.slack = data.get('constraint_slack', {})

        # Config-driven constraint defaults
        defaults = data.get('constraint_defaults', {})

        # O(1) lookups (replacing O(N) get_club calls)
        self.team_club = {}
        self.club_teams_map = defaultdict(list)
        self.team_grade = {}
        self.club_home_field = {}
        self.club_2nd_teams = defaultdict(list)
        self.club_dup_grades = defaultdict(lambda: defaultdict(list))
        self._build_lookups()

        # Grouping dicts (populated by build_groupings)
        self._groupings_built = False

        # Helper-var registry (declarative API for atoms; pool-style methods for legacy engine).
        # `self.pool` retained as alias so existing internal methods keep working.
        self.registry = HelperVarRegistry(self.model)
        self.pool = self.registry

        # CGS iteration keys (populated by _club_game_spread_hard for soft phase)
        self._cgs_keys = []

        # Best-timeslot per-field indicators (populated by hard, used by soft)
        self._loc_fields = None

    # ================================================================
    # LOOKUPS
    # ================================================================

    def _build_lookups(self):
        """Pre-compute O(1) lookup dicts from team/club/grade objects."""
        for team in self.teams:
            self.team_club[team.name] = team.club.name
            self.club_teams_map[team.club.name].append(team.name)
            self.team_grade[team.name] = team.grade
            if team.grade == '2nd':
                self.club_2nd_teams[team.club.name].append(team.name)

        for club in self.clubs:
            self.club_home_field[club.name] = club.home_field

        # Duplicate-grade teams per club
        for club in self.clubs:
            for grade in self.grades:
                dup = get_duplicated_graded_teams(club.name, grade.name, self.teams)
                if dup:
                    self.club_dup_grades[club.name][grade.name] = dup

    def _get_club(self, team_name):
        """O(1) club lookup."""
        return self.team_club.get(team_name)

    # ================================================================
    # SINGLE-PASS GROUPINGS
    # ================================================================

    def build_groupings(self):
        """Single pass over X to populate all grouping dicts.

        Phase 4: also runs every registered FORCED/BLOCKED count adjuster
        once, populating `data['count_adjustments']`. Atoms read their
        adjustment by canonical name during `apply()`.
        """
        if self._groupings_built:
            return

        # Phase 4: FORCED/BLOCKED count adjusters run before any atom apply().
        run_count_adjusters(self.data)

        # --- Core groupings ---
        self.by_week_team = defaultdict(list)
        self.by_slot_field = defaultdict(list)
        self.by_week_day_slot_team = defaultdict(list)

        # --- Matchup spacing ---
        self.by_grade_pair_round = defaultdict(lambda: defaultdict(list))

        # --- Club-grade adjacency REMOVED in spec-007.
        # The legacy hard + soft cluster has been replaced by the
        # `SameGradeSameClubNoConcurrency` atom (hard, dispatched outside the
        # engine) and `TeamPairNoConcurrency` (soft). No groupings needed.

        # --- Club vs Club alignment ---
        self.by_grade_clubpair_round = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.by_sunday_clubpair_round_field = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # --- Venue groupings ---
        # spec-018: the `non_default_*` per-club home/away-week maps (and their
        # `maitland_*` aliases) plus the `by_week_location` map were all deleted
        # along with the venue-sequencing rules (`NonDefaultHomeGrouping` /
        # `AwayAtNonDefaultGrouping`). Nothing consumes them anymore.

        # --- Broadmeadow groupings ---
        self.bm_slot_club = defaultdict(lambda: defaultdict(list))
        self.bm_field_club = defaultdict(lambda: defaultdict(list))

        # --- Club game spread ---
        self.by_club_week_day_slot = defaultdict(list)

        # --- PHL/2nd grade ---
        # spec-014: `phl_club_week_day` / `second_club_week_day` engine maps
        # removed — the legacy `_phl_adjacency_hard` they fed is gone, replaced
        # by the self-contained `PHLAnd2ndAdjacency` atom (reads X directly).
        self.phl_slot_vars = defaultdict(list)
        self.club_phl_vars = defaultdict(list)
        self.club_2nd_vars = defaultdict(list)
        self.phl_friday_broadmeadow = []
        self.phl_friday_gosford = []
        self.phl_friday_gosford_round = defaultdict(list)
        self.phl_round1_vars = defaultdict(list)
        self.preferred_date_vars = defaultdict(list)

        # --- Home/away ---
        self.home_away_venue = defaultdict(lambda: {'home': [], 'away': []})

        # --- Timeslot optimization ---
        self.slot_vars_by_location = defaultdict(list)
        self.slot_field_vars = defaultdict(list)
        self.games_per_location = defaultdict(lambda: defaultdict(list))

        # --- Club day ---
        self.club_day_game_vars = defaultdict(dict)

        # --- EnsureEqualGames (uses games x timeslots pattern) ---
        self.grade_team_vars = defaultdict(lambda: defaultdict(list))
        self.grade_pair_vars = defaultdict(lambda: defaultdict(list))

        # Get PHL preferred dates
        phl_prefs = self.data.get('phl_preferences', {})
        pref_dates = set(
            d.date().strftime('%Y-%m-%d')
            for d in phl_prefs.get('preferred_dates', [])
        )

        # Get club day dates (normalize_club_day handles the dict form
        # `{'date': dt, 'opponent': 'X'}` used when an opponent is set).
        club_days = self.data.get('club_days', {})
        club_day_dates = {}
        for club_name, value in club_days.items():
            dt, _opponent = normalize_club_day(value)
            club_day_dates[dt.date().strftime('%Y-%m-%d')] = club_name

        # === SINGLE PASS OVER X ===
        for key, var in self.X.items():
            if not key[3]:
                continue

            t1, t2, grade = key[0], key[1], key[2]
            day, day_slot = key[3], key[4]
            week, date_str, round_no = key[6], key[7], key[8]
            field_name, location = key[9], key[10]

            # Skip locked weeks for constraint groupings
            is_locked = week in self.locked_weeks

            t1_club = self._get_club(t1)
            t2_club = self._get_club(t2)

            # --- EnsureEqualGames (needs all vars including locked) ---
            self.grade_team_vars[grade][t1].append(var)
            self.grade_team_vars[grade][t2].append(var)
            self.grade_pair_vars[grade][tuple(sorted((t1, t2)))].append(var)

            if is_locked:
                continue

            # --- NoDoubleBookingTeams ---
            self.by_week_team[(week, t1)].append(var)
            self.by_week_team[(week, t2)].append(var)

            # --- NoDoubleBookingFields ---
            self.by_slot_field[(week, day, day_slot, field_name)].append(var)

            # --- TeamConflict ---
            self.by_week_day_slot_team[(week, day_slot, t1)].append(var)
            self.by_week_day_slot_team[(week, day_slot, t2)].append(var)

            # --- EqualMatchUpSpacing ---
            self.by_grade_pair_round[(t1, t2, grade)][round_no].append(var)

            # --- ClubGradeAdjacency REMOVED (spec-007) ---
            # The legacy combined cluster has been replaced by
            # `SameGradeSameClubNoConcurrency` (hard atom, dispatched
            # outside the engine) and `TeamPairNoConcurrency` (soft atom).
            # No groupings need to be built here.

            # --- ClubVsClubAlignment (3rd-6th only) ---
            if grade not in ['PHL', '2nd'] and t1_club and t2_club:
                club_pair = tuple(sorted((t1_club, t2_club)))
                self.by_grade_clubpair_round[grade][club_pair][round_no].append(var)
                if day == 'Sunday':
                    self.by_sunday_clubpair_round_field[club_pair][round_no][field_name].append(var)

            # spec-018: venue groupings for the deleted MaxMaitlandHomeWeekends /
            # NonDefaultHomeGrouping / AwayAtNonDefaultGrouping rules removed.

            # --- Broadmeadow slot/field club groupings ---
            if location == BROADMEADOW and day in ['Saturday', 'Sunday']:
                if t1_club:
                    self.bm_slot_club[(week, day_slot)][t1_club].append(var)
                if t2_club:
                    self.bm_slot_club[(week, day_slot)][t2_club].append(var)
                if t1_club:
                    self.bm_field_club[(week, date_str, field_name)][t1_club].append(var)
                if t2_club:
                    self.bm_field_club[(week, date_str, field_name)][t2_club].append(var)

            # --- ClubGameSpread ---
            if t1_club:
                self.by_club_week_day_slot[(t1_club, week, day, day_slot)].append(var)
            if t2_club and t2_club != t1_club:
                self.by_club_week_day_slot[(t2_club, week, day, day_slot)].append(var)

            # --- PHL/2nd grade specific ---
            if grade == 'PHL':
                slot_id = (week, day, day_slot, location)
                self.phl_slot_vars[slot_id].append(var)

                for team in [t1, t2]:
                    club = self._get_club(team)
                    if club:
                        self.club_phl_vars[(*slot_id, club)].append(var)

                if day == 'Friday' and location == BROADMEADOW:
                    self.phl_friday_broadmeadow.append(var)
                if day == 'Friday' and location == GOSFORD:
                    self.phl_friday_gosford.append(var)
                    self.phl_friday_gosford_round[round_no].append(var)
                if round_no == 1:
                    self.phl_round1_vars[t1].append(var)
                    self.phl_round1_vars[t2].append(var)
                if date_str in pref_dates:
                    self.preferred_date_vars[date_str].append(var)

            elif grade == '2nd':
                slot_id = (week, day, day_slot, location)
                for team in [t1, t2]:
                    club = self._get_club(team)
                    if club:
                        self.club_2nd_vars[(*slot_id, club)].append(var)

            # --- FiftyFiftyHomeAway ---
            for venue_name, venue_location in AWAY_VENUES.items():
                for team, other in [(t1, t2), (t2, t1)]:
                    if venue_name in team and venue_name not in other:
                        pair_key = (team, other)
                        if location == venue_location:
                            self.home_away_venue[pair_key]['home'].append(var)
                        else:
                            self.home_away_venue[pair_key]['away'].append(var)

            # --- EnsureBestTimeslotChoices ---
            slot_key = (week, day, location, day_slot)
            self.slot_vars_by_location[slot_key].append(var)
            self.slot_field_vars[(week, day, location, field_name, day_slot)].append(var)
            self.games_per_location[(week, day)][location].append(var)

            # --- ClubDay ---
            if date_str in club_day_dates:
                club_name = club_day_dates[date_str]
                club_team_names = self.club_teams_map.get(club_name, [])
                if t1 in club_team_names or t2 in club_team_names:
                    self.club_day_game_vars[club_name][key] = var

        self._groupings_built = True

    # ================================================================
    # PENALTY WEIGHT HELPER
    # ================================================================

    def _get_penalty_weight(self, name, default):
        """Get penalty weight from config, falling back to default."""
        return self.data.get('penalty_weights', {}).get(name, default)

    # ================================================================
    # STAGE 1: HARD CONSTRAINTS (feasibility)
    # ================================================================

    def apply_stage_1_hard(self):
        """Apply all hard constraints (feasibility). Respects skip_constraints."""
        assert self._groupings_built, "Call build_groupings() first"
        c = 0
        _skip = self.skip_constraints
        if 'NoDoubleBookingTeams' not in _skip:
            c += self._no_double_booking_teams()
        if 'NoDoubleBookingFields' not in _skip:
            c += self._no_double_booking_fields()
        if 'EqualGamesAndBalanceMatchUps' not in _skip:
            c += self._equal_games_balanced_matchups()
        if 'FiftyFiftyHomeandAway' not in _skip:
            c += self._fifty_fifty_home_away()
        if 'TeamConflict' not in _skip:
            c += self._team_conflict()
        # spec-018: `MaxMaitlandHomeWeekends` (`_max_venue_weekends`) deleted —
        # superseded by the spec-004 `AwayClubHomeWeekendsCount` atom.
        # spec-014: PHL/2nd adjacency is now the `PHLAnd2ndAdjacency` atom,
        # dispatched via the non-engine fallback in constraints/stages.py — no
        # engine skip-key block here anymore.
        if 'PHLAndSecondGradeTimes' not in _skip:
            c += self._phl_times_atoms_hard()
        if 'EqualMatchUpSpacing' not in _skip:
            c += self._matchup_spacing_hard()
        # ClubGradeAdjacency REMOVED (spec-007): the legacy hard rule is now
        # the `SameGradeSameClubNoConcurrency` atom dispatched via the
        # non-engine fallback in `constraints/stages.py`. Nothing to do here.
        if 'ClubVsClubAlignment' not in _skip:
            c += self._club_alignment_hard()
        # spec-018: `MaitlandHomeGrouping` / `AwayAtMaitlandGrouping` hard
        # dispatch removed — venue-sequencing rules deleted.
        if 'ClubDay' not in _skip:
            c += self._club_day_atoms_hard()
        if 'ClubGameSpread' not in _skip:
            c += self._club_game_spread_hard()
        if 'EnsureBestTimeslotChoices' not in _skip:
            c += self._best_timeslot_choices_hard()
        self.constraints_added += c
        return c

    # Backward-compatible alias
    def apply_phase_a(self):
        """Alias for apply_stage_1_hard()."""
        return self.apply_stage_1_hard()

    def _no_double_booking_teams(self):
        n = 0
        for (week, team), vars_list in self.by_week_team.items():
            if len(vars_list) > 1:
                self.model.Add(sum(vars_list) <= 1)
                n += 1
        return n

    def _no_double_booking_fields(self):
        n = 0
        for slot, vars_list in self.by_slot_field.items():
            if len(vars_list) > 1:
                self.model.Add(sum(vars_list) <= 1)
                n += 1
        return n

    def _equal_games_balanced_matchups(self):
        n = 0
        for grade, teams_in_grade in self.grade_team_vars.items():
            T = len(teams_in_grade)
            R = self.num_rounds.get(grade, 0)
            if T < 2 or R == 0:
                continue
            for team, vars_list in teams_in_grade.items():
                self.model.Add(sum(vars_list) == R)
                n += 1
            base = R // (T - 1) if T % 2 == 0 else R // T
            for pair, vars_list in self.grade_pair_vars[grade].items():
                self.model.Add(sum(vars_list) >= base)
                self.model.Add(sum(vars_list) <= base + 1)
                n += 2
        return n

    def _fifty_fifty_home_away(self):
        n = 0
        for pair_key, ha in self.home_away_venue.items():
            home_vars, away_vars = ha['home'], ha['away']
            if not home_vars or not away_vars:
                continue
            hc = self.model.NewIntVar(0, len(home_vars), f'u_home_{pair_key[0]}_{pair_key[1]}')
            tc = self.model.NewIntVar(0, len(home_vars) + len(away_vars), f'u_total_{pair_key[0]}_{pair_key[1]}')
            self.model.Add(hc == sum(home_vars))
            self.model.Add(tc == sum(home_vars) + sum(away_vars))
            self.model.Add(hc * 2 >= tc - 1)
            self.model.Add(hc * 2 <= tc + 1)
            n += 2
        return n

    def _team_conflict(self):
        conflicts = self.data.get('team_conflicts', [])
        if not conflicts:
            return 0
        n = 0
        # Regroup by (week, day_slot)
        slot_teams = defaultdict(lambda: defaultdict(list))
        for (week, day_slot, team), vars_list in self.by_week_day_slot_team.items():
            slot_teams[(week, day_slot)][team].extend(vars_list)

        for team1, team2 in conflicts:
            for slot, team_vars in slot_teams.items():
                v1 = team_vars.get(team1, [])
                v2 = team_vars.get(team2, [])
                if v1 and v2:
                    self.model.Add(sum(v1) + sum(v2) <= 1)
                    n += 1
        return n

    # spec-018: `_max_venue_weekends` (the `MaxMaitlandHomeWeekends` rule)
    # deleted — superseded by the spec-004 `AwayClubHomeWeekendsCount` atom.

    # spec-014: `_phl_adjacency_hard` removed. The PHL/2nd adjacency rule is
    # now the self-contained `PHLAnd2ndAdjacency` atom
    # (constraints/atoms/phl_2nd_adjacency.py), which forces same-club
    # back-to-back at one venue or a >= 180-min cross-venue start gap — the
    # legacy method only *forbade* two patterns inside a +/-180-min window and
    # never forced adjacency.

    # ----------------------------------------------------------------
    # PHL/2nd-grade times — atom dispatch (Phase 3, replaces _phl_times_hard /
    # _phl_times_soft). Legacy methods retained below for parity reference.
    # ----------------------------------------------------------------

    # spec-010: PHLRoundOnePlay removed from hard atoms. Convenor uses
    # FORCED_GAMES to express round-1 intent when needed. File kept on disk
    # for parity reference; import retained so parity tests still work.
    # spec-015: GosfordFridayRoundsForced removed — Gosford Friday rounds are
    # now FORCED_GAMES count entries in the season config, not a code atom.
    _PHL_HARD_ATOMS = (
        PHLConcurrencyAtBroadmeadow,
        PHLAnd2ndConcurrencyAtBroadmeadow,
    )
    _PHL_SOFT_ATOMS = (PreferredDates,)

    def _phl_times_atoms_hard(self):
        n = 0
        for atom_cls in self._PHL_HARD_ATOMS:
            n += atom_cls().apply(self.model, self.X, self.data, self.registry)
        return n

    def _phl_times_atoms_soft(self):
        n = 0
        for atom_cls in self._PHL_SOFT_ATOMS:
            n += atom_cls().apply(self.model, self.X, self.data, self.registry)
        return n

    def _phl_times_hard(self):
        """Legacy single-method PHL times — retained for parity reference. Not called."""
        n = 0
        # No concurrent PHL at Broadmeadow
        for slot_id, vars_list in self.phl_slot_vars.items():
            if slot_id[3] == BROADMEADOW and len(vars_list) > 1:
                self.model.Add(sum(vars_list) <= 1)
                n += 1

        # No concurrent 2nd grade and PHL from same club at Broadmeadow
        for club_slot, phl_vars in self.club_phl_vars.items():
            week, day, day_slot, location, club = club_slot
            if location != BROADMEADOW:
                continue
            second_vars = self.club_2nd_vars.get(club_slot, [])
            if phl_vars and second_vars:
                self.model.Add(sum(phl_vars) + sum(second_vars) <= 1)
                n += 1

        # Read configurable limits
        defaults = self.data.get('constraint_defaults', {})
        max_friday_broadmeadow = defaults.get('max_friday_broadmeadow', 3)
        gosford_friday_games = defaults.get('gosford_friday_games', 8)

        # Max Friday night games at Broadmeadow
        if self.phl_friday_broadmeadow:
            self.model.Add(sum(self.phl_friday_broadmeadow) <= max_friday_broadmeadow)
            n += 1

        # Exactly N Friday night games at Gosford
        if self.phl_friday_gosford:
            self.model.Add(sum(self.phl_friday_gosford) == gosford_friday_games)
            n += 1

        # Gosford Friday games in specific rounds
        gosford_friday_rounds = set(
            self.data.get('constraint_defaults', {}).get(
                'gosford_friday_rounds', [2, 4, 5, 9, 10]
            )
        )
        for round_no, round_vars in self.phl_friday_gosford_round.items():
            if round_no in gosford_friday_rounds:
                self.model.Add(sum(round_vars) == 1)
                n += 1

        # Every PHL team must play in round 1
        phl_teams = [t.name for t in self.teams if t.grade == 'PHL']
        for phl_team in phl_teams:
            if phl_team in self.phl_round1_vars:
                self.model.Add(sum(self.phl_round1_vars[phl_team]) >= 1)
                n += 1

        return n

    def _matchup_spacing_hard(self):
        """Pairwise forbidden gaps only (no sliding window penalty).

        spec-008 Part A: uses the intuitive "S played rounds between
        meetings" semantics via `ideal_gap(T)` from
        `constraints.atoms._spacing`. A pair's gap `r2 - r1` is forbidden
        when `gap <= S` where S = effective_spacing(T, base_slack, config_slack +
        forced_rounds_consumed). This is the spec-008 off-by-one fix:
        old code forbade `gap < min_gap`; the new rule forbids `gap <= S`
        with `S = min_gap - 1` so the physical schedule is unchanged at
        default slack while the input number's meaning is now "rounds in
        between" rather than "calendar distance."
        """
        from constraints.atoms._spacing import effective_spacing

        n = 0
        grade_num_teams = {g.name: g.num_teams for g in self.grades}
        defaults = self.data.get('constraint_defaults', {})
        base_slack_config = int(defaults.get('spacing_base_slack', 0) or 0)
        config_slack = int(self.slack.get('EqualMatchUpSpacingConstraint', 0) or 0)
        # Phase 4 adjuster: per-pair forced rounds reduce flexibility.
        forced_rounds_per_pair = self.data.get('count_adjustments', {}).get(
            'EqualMatchUpSpacing', {}
        ) or {}

        for (t1, t2, grade), round_map in self.by_grade_pair_round.items():
            T = grade_num_teams.get(grade, 0)
            if T < 3:
                # ideal_gap is 0 for T<3 — no meaningful spacing.
                continue
            forced_rounds = forced_rounds_per_pair.get(
                (t1, t2, grade)
            ) or forced_rounds_per_pair.get((t2, t1, grade)) or set()
            # Each FORCED round eats one "free" round of spacing flexibility:
            # the remaining unpinned meetings have less room to spread out,
            # so the constraint tightens. Subtracting the forced count from
            # total slack achieves this — `effective_spacing` accepts
            # negative slack, in which case S grows above ideal_gap(T).
            forced_tighten = len(forced_rounds)
            net_slack = base_slack_config + config_slack - forced_tighten
            S = effective_spacing(T, base_slack=net_slack, config_slack=0)
            if S <= 0:
                # Slack has fully relaxed the constraint for this pair.
                continue

            active_rounds = sorted(r for r in round_map if round_map[r])
            for i, r1 in enumerate(active_rounds):
                vars_r1 = round_map[r1]
                for r2 in active_rounds[i + 1:]:
                    gap = r2 - r1
                    if gap > S:
                        break
                    self.model.Add(sum(vars_r1) + sum(round_map[r2]) <= 1)
                    n += 1
        return n

    # _grade_adjacency_hard REMOVED (spec-007): hard same-grade-same-club
    # rule moved to the `SameGradeSameClubNoConcurrency` atom. The legacy
    # combined `ClubGradeAdjacencyConstraint` lives only in
    # `constraints/archived/original.py` and is no longer invoked by the
    # engine.

    # ----------------------------------------------------------------
    # ClubVsClub — legacy in-engine implementation. The Phase-3c atoms that
    # used to wrap these methods (Coincidence / FieldLimit / DeficitPenalty /
    # PHLAnd2ndBackToBackSameField) were deleted (spec-005); production now
    # uses the `ClubVsClubStackedAlignment` cluster dispatched outside the
    # engine. These methods remain as the parity-reference behaviour for the
    # `ClubVsClubAlignment` engine key.
    # ----------------------------------------------------------------

    def _club_alignment_hard(self):
        """Hard: min coincidences + max 2 fields when coinciding."""
        n = 0
        per_team_games = {
            g.name: (self.num_rounds['max'] // (g.num_teams - 1)) if g.num_teams > 1 and g.num_teams % 2 == 0
                    else (self.num_rounds['max'] // g.num_teams) if g.num_teams > 0
                    else 0
            for g in self.grades
        }
        ordered_grades = sorted(per_team_games.items(), key=lambda x: x[1])
        config_slack = self.slack.get('ClubVsClubAlignment', 0)

        processed = []
        prev_num = 0
        for grade, num_games in ordered_grades:
            if grade in ['PHL', '2nd']:
                continue
            processed.append(grade)
            if num_games <= prev_num:
                continue
            prev_num = num_games

            for other_grade in [g for g, _ in ordered_grades if g not in processed]:
                for club_pair, rounds in self.by_grade_clubpair_round[grade].items():
                    other_rounds = self.by_grade_clubpair_round[other_grade].get(club_pair, {})
                    coincide_vars = []
                    for round_no, vars_list in rounds.items():
                        if round_no not in other_rounds:
                            continue
                        ind1 = self.pool.get_or_create_bool(
                            ('align_g1', grade, club_pair, round_no), vars_list,
                            f"u_g1_{grade}_{club_pair}_{round_no}")
                        ind2 = self.pool.get_or_create_bool(
                            ('align_g2', other_grade, club_pair, round_no), other_rounds[round_no],
                            f"u_g2_{other_grade}_{club_pair}_{round_no}")
                        coincide = self.model.NewBoolVar(f"u_coin_{club_pair}_{round_no}")
                        self.model.Add(coincide <= ind1)
                        self.model.Add(coincide <= ind2)
                        self.model.Add(coincide >= ind1 + ind2 - 1)
                        self.pool.register(('coin', grade, other_grade, club_pair, round_no), coincide)
                        coincide_vars.append(coincide)

                        # Max 2 fields when coinciding
                        field_round = self.by_sunday_clubpair_round_field[club_pair].get(round_no, {})
                        if field_round:
                            fi_list = []
                            for fn, gvars in field_round.items():
                                fi = self.pool.get_or_create_bool(
                                    ('align_fld', club_pair, round_no, fn), gvars,
                                    f"u_fld_{club_pair}_{round_no}_{fn}")
                                fi_list.append(fi)
                            if fi_list:
                                nf = self.model.NewIntVar(0, len(fi_list), f"u_nflds_{club_pair}_{round_no}")
                                self.model.Add(nf == sum(fi_list))
                                self.model.Add(nf <= 2).OnlyEnforceIf(coincide)
                                n += 1

                    if coincide_vars:
                        min_req = max(0, num_games - config_slack)
                        self.model.Add(sum(coincide_vars) >= min_req)
                        n += 1
        return n

    # spec-018: `_maitland_grouping_hard` (NonDefaultHomeGrouping) and
    # `_away_maitland_hard` (AwayAtNonDefaultGrouping) deleted — the convenor
    # no longer wants venue-sequencing enforced. Per-club home-weekend counts
    # remain via the spec-004 `AwayClubHomeWeekendsCount` atom.

    # ----------------------------------------------------------------
    # ClubDay — atom dispatch (Phase 3b, replaces _club_day_scheduling /
    # _club_day_field_contiguity). Legacy methods retained below for parity
    # reference. The atom set additionally enforces opponent-matchup
    # semantics from `original.py`, which the legacy unified methods omitted.
    # ----------------------------------------------------------------

    _CLUB_DAY_HARD_ATOMS = (
        ClubDayParticipation,
        ClubDayIntraClubMatchup,
        ClubDayOpponentMatchup,
        ClubDaySameField,
        ClubDayContiguousSlots,
    )

    def _club_day_atoms_hard(self):
        n = 0
        for atom_cls in self._CLUB_DAY_HARD_ATOMS:
            n += atom_cls().apply(self.model, self.X, self.data, self.registry)
        return n

    def _club_day_scheduling(self):
        """Inter-week: every club team plays on club day + intra-club matchups."""
        n = 0
        club_days = self.data.get('club_days', {})
        if not club_days:
            return 0

        for club_name, desired_date in club_days.items():
            date_str = desired_date.date().strftime('%Y-%m-%d')
            closest_week = get_nearest_week_by_date(date_str, self.timeslots)
            if closest_week in self.locked_weeks:
                continue

            game_vars = self.club_day_game_vars.get(club_name, {})
            if not game_vars:
                continue
            club_team_names = self.club_teams_map.get(club_name, [])

            # Every club team must play
            for team in club_team_names:
                team_vars = [v for k, v in game_vars.items() if team in (k[0], k[1])]
                if team_vars:
                    self.model.Add(sum(team_vars) >= 1)
                    n += 1

            # Intra-club matchups for same-grade teams
            teams_by_grade = defaultdict(list)
            for team in club_team_names:
                grade = team.rsplit(' ', 1)[1]
                teams_by_grade[grade].append(team)

            for grade, grade_teams in teams_by_grade.items():
                if len(grade_teams) > 1:
                    intra_pairs = list(combinations(grade_teams, 2))
                    intra_vars = []
                    for key, var in game_vars.items():
                        pair = (key[0], key[1])
                        if pair in intra_pairs or (pair[1], pair[0]) in intra_pairs:
                            intra_vars.append(var)
                    if intra_vars:
                        expected = len(grade_teams) // 2
                        self.model.Add(sum(intra_vars) >= expected)
                        n += 1
        return n

    # ================================================================
    # STAGE 2: SOFT CONSTRAINTS (penalties + optimization)
    # ================================================================

    def apply_stage_2_soft(self):
        """Add soft penalties + optimization constraints. Respects skip_constraints."""
        assert self._groupings_built, "Call build_groupings() first"
        c = 0
        _skip = self.skip_constraints
        if 'EqualMatchUpSpacing' not in _skip:
            c += self._matchup_spacing_soft()
        # ClubGradeAdjacency soft REMOVED ENTIRELY (spec-007). The convenor-
        # facing adjacent-grade penalty was over-restrictive in practice.
        # If a specific team-pair conflict matters, declare it via
        # `TEAM_PAIR_NO_CONCURRENCY` (handled by the `TeamPairNoConcurrency`
        # atom outside the engine).
        if 'ClubVsClubAlignment' not in _skip:
            c += self._club_alignment_soft()
        # spec-018: `MaitlandHomeGrouping` / `AwayAtMaitlandGrouping` soft
        # dispatch removed — venue-sequencing rules deleted.
        if 'PHLAndSecondGradeTimes' not in _skip:
            c += self._phl_times_atoms_soft()
        if 'PreferredTimesConstraint' not in _skip:
            c += self._preferred_times()
        if 'EnsureBestTimeslotChoices' not in _skip:
            c += self._best_timeslot_choices_soft()
        if 'ClubGameSpread' not in _skip:
            c += self._club_game_spread_soft()
        self.constraints_added += c
        return c

    # Backward-compatible aliases
    def apply_phase_b(self):
        """Alias for apply_stage_2_soft()."""
        return self.apply_stage_2_soft()

    def apply_phase_c(self):
        """Phase C placeholder — intra-day constraints are in Stage 1."""
        return 0

    def _matchup_spacing_soft(self):
        """Sliding window density penalties."""
        n = 0
        weight = self._get_penalty_weight('EqualMatchUpSpacing', 5000)
        self.data['penalties']['EqualMatchUpSpacing'] = {'weight': weight, 'penalties': []}
        R = self.num_rounds.get('max', 0)
        grade_num_teams = {g.name: g.num_teams for g in self.grades}

        for (t1, t2, grade), round_map in self.by_grade_pair_round.items():
            T = grade_num_teams.get(grade, 0)
            if T < 2:
                continue
            space = T - 1
            if space >= R:
                continue
            for r_start in range(1, R - space + 2):
                r_end = r_start + space - 1
                window_vars = []
                for r in range(r_start, r_end + 1):
                    if r in round_map:
                        window_vars.extend(round_map[r])
                if len(window_vars) < 2:
                    continue
                pen = self.model.NewIntVar(0, len(window_vars),
                    f"u_eqsp_wpen_{t1}_{t2}_{grade}_w{r_start}")
                self.model.Add(pen >= sum(window_vars) - 1)
                self.data['penalties']['EqualMatchUpSpacing']['penalties'].append(pen)
                n += 1
        return n

    # _grade_adjacency_soft REMOVED ENTIRELY (spec-007). The convenor-facing
    # adjacent-grade-same-timeslot penalty was over-restrictive and is no
    # longer enforced anywhere. For per-team-pair conflicts, use the
    # `TeamPairNoConcurrency` atom + `TEAM_PAIR_NO_CONCURRENCY` config.

    def _club_alignment_soft(self):
        """Soft penalties for coincidence deficit and field excess."""
        n = 0
        coincide_weight = self._get_penalty_weight('ClubVsClubAlignment', 100000)
        field_weight = self._get_penalty_weight('ClubVsClubAlignmentField', 50000)
        self.data['penalties']['ClubVsClubAlignment'] = {'weight': coincide_weight, 'penalties': []}
        self.data['penalties']['ClubVsClubAlignmentField'] = {'weight': field_weight, 'penalties': []}

        per_team_games = {
            g.name: (self.num_rounds['max'] // (g.num_teams - 1)) if g.num_teams > 1 and g.num_teams % 2 == 0
                    else (self.num_rounds['max'] // g.num_teams) if g.num_teams > 0
                    else 0
            for g in self.grades
        }
        ordered_grades = sorted(per_team_games.items(), key=lambda x: x[1])
        config_slack = self.slack.get('ClubVsClubAlignment', 0)
        fidx = 0

        processed = []
        prev_num = 0
        for grade, num_games in ordered_grades:
            if grade in ['PHL', '2nd']:
                continue
            processed.append(grade)
            if num_games <= prev_num:
                continue
            prev_num = num_games

            for other_grade in [g for g, _ in ordered_grades if g not in processed]:
                for club_pair, rounds in self.by_grade_clubpair_round[grade].items():
                    other_rounds = self.by_grade_clubpair_round[other_grade].get(club_pair, {})
                    coincide_vars = []
                    for round_no, vars_list in rounds.items():
                        if round_no not in other_rounds:
                            continue
                        # Reuse indicators from Phase A if available
                        ind1_key = f"u_g1_{grade}_{club_pair}_{round_no}"
                        ind2_key = f"u_g2_{other_grade}_{club_pair}_{round_no}"
                        coin_key = f"u_coin_{club_pair}_{round_no}"

                        # These were created in Phase A, look them up
                        coincide = self.pool.get(('coin', grade, other_grade, club_pair, round_no))
                        if coincide is None:
                            # Create if Phase A wasn't run (fallback via pool)
                            ind1 = self.pool.get_or_create_bool(
                                ('align_g1', grade, club_pair, round_no), vars_list, ind1_key)
                            ind2 = self.pool.get_or_create_bool(
                                ('align_g2', other_grade, club_pair, round_no),
                                other_rounds[round_no], ind2_key)
                            coincide = self.model.NewBoolVar(coin_key)
                            self.model.Add(coincide <= ind1)
                            self.model.Add(coincide <= ind2)
                            self.model.Add(coincide >= ind1 + ind2 - 1)
                            self.pool.register(('coin', grade, other_grade, club_pair, round_no), coincide)

                        coincide_vars.append(coincide)

                        # Field excess penalty
                        field_round = self.by_sunday_clubpair_round_field[club_pair].get(round_no, {})
                        if field_round:
                            fi_list = []
                            for fn, gvars in field_round.items():
                                fi = self.model.NewBoolVar(f"u_sfld_{club_pair}_{round_no}_{fn}_{fidx}")
                                self.model.AddMaxEquality(fi, gvars)
                                fi_list.append(fi)
                            if fi_list:
                                nf = self.model.NewIntVar(0, len(fi_list),
                                    f"u_sfexcess_{club_pair}_{round_no}_{fidx}")
                                self.model.Add(nf == sum(fi_list))
                                fex = self.model.NewIntVar(0, len(fi_list),
                                    f"u_fex_{club_pair}_{round_no}_{fidx}")
                                self.model.Add(fex >= nf - 1).OnlyEnforceIf(coincide)
                                self.model.Add(fex == 0).OnlyEnforceIf(coincide.Not())
                                self.data['penalties']['ClubVsClubAlignmentField']['penalties'].append(fex)
                                fidx += 1

                    if coincide_vars:
                        actual = self.model.NewIntVar(0, len(coincide_vars),
                            f"u_actual_coin_{club_pair}_{grade}_{other_grade}")
                        self.model.Add(actual == sum(coincide_vars))
                        deficit = self.model.NewIntVar(0, num_games,
                            f"u_coin_def_{club_pair}_{grade}_{other_grade}")
                        self.model.Add(deficit >= num_games - actual)
                        self.data['penalties']['ClubVsClubAlignment']['penalties'].append(deficit)
                        n += 1
        return n

    # spec-018: `_maitland_grouping_soft` (NonDefaultHomeGrouping soft
    # imbalance penalty) and `_away_maitland_soft` (AwayAtNonDefaultGrouping
    # soft) deleted — venue-sequencing rules removed.

    def _phl_times_soft(self):
        """Legacy preferred-date penalties — retained for parity reference. Not called."""
        n = 0
        weight = self._get_penalty_weight('phl_preferences', 10000)
        self.data['penalties']['phl_preferences'] = {'weight': weight, 'penalties': []}
        for date_str, vars_list in self.preferred_date_vars.items():
            if vars_list:
                week_no = get_nearest_week_by_date(date_str, self.timeslots)
                if week_no not in self.locked_weeks:
                    pen = self.model.NewIntVar(0, len(vars_list), f"u_pref_date_{date_str}")
                    self.model.AddAbsEquality(pen, sum(vars_list) - 1)
                    self.data['penalties']['phl_preferences']['penalties'].append(pen)
                    n += 1
        return n

    def _preferred_times(self):
        """Soft: no-play date penalties (2025 + 2026 format)."""
        n = 0
        weight = self._get_penalty_weight('PreferredTimesConstraint', 10000000)
        self.data['penalties']['PreferredTimesConstraint'] = {'weight': weight, 'penalties': []}
        noplay = self.data.get('preference_no_play', {})
        if not noplay:
            return 0

        from utils import normalize_preference_no_play as _normalize_preference_no_play
        allowed_keys = ['team_name', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']
        allowed_keys2 = ['team1', 'team_name', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']

        normalized = _normalize_preference_no_play(noplay, self.teams, self.clubs)
        # spec-012: support time-only (date-less) preference entries.
        # Previously this method silently skipped any entry without 'date',
        # making a `{'club': 'Maitland', 'time': '08:30'}` entry a no-op.
        # Now: only the locked-week short-circuit requires 'date'. Without a
        # date the entry penalises matching X-vars across every week. At
        # least one matchable filter is required (otherwise the entry would
        # penalise every game for the club and gridlock the objective).
        _matchable = ('time', 'date', 'day', 'day_slot', 'week',
                      'field_name', 'field_location', 'grade',
                      'team_name', 'team1', 'team2')
        for entry_key, club_name, club_teams, constraint in normalized:
            if not any(k in constraint for k in _matchable):
                continue
            if 'date' in constraint and (
                get_nearest_week_by_date(constraint['date'], self.timeslots)
                in self.locked_weeks
            ):
                continue
            for i, game_key in enumerate(self.X):
                if len(game_key) < 11:
                    continue
                if game_key[0] not in club_teams and game_key[1] not in club_teams:
                    continue
                gd = dict(zip(allowed_keys, game_key))
                gd2 = dict(zip(allowed_keys2, game_key))
                matches = all(gd.get(k) == v for k, v in constraint.items())
                matches2 = all(gd2.get(k) == v for k, v in constraint.items())
                if matches or matches2:
                    pv = self.model.NewIntVar(0, 1, f"u_noplay_{entry_key}_{i}")
                    self.model.Add(pv == self.X[game_key])
                    self.data['penalties']['PreferredTimesConstraint']['penalties'].append(pv)
                    n += 1
        return n

    def _best_timeslot_choices_hard(self):
        """Hard: no-gap constraint, slot-number bounding, stacking."""
        n = 0
        location_day_slots = defaultdict(dict)
        slot_number_vars = defaultdict(dict)

        for slot_key, vars_list in self.slot_vars_by_location.items():
            week, day, location, day_slot = slot_key
            if len(vars_list) > 1:
                ind = self.pool.get_or_create_bool(
                    ('slot_used', slot_key), vars_list,
                    f'u_slot_used_{week}_{day}_{location}_{day_slot}')
                location_day_slots[(week, day, location)][day_slot] = ind
                nv = self.model.NewIntVar(0, 100, f'u_slot_num_{week}_{location}_{day_slot}')
                self.model.Add(nv == int(day_slot))
                slot_number_vars[(week, day, location)][day_slot] = nv

        # No gaps
        for (week, day, location), day_slots in location_day_slots.items():
            sorted_slots = sorted(day_slots.keys())
            for i in range(1, len(sorted_slots) - 1):
                prev_s = sorted_slots[i - 1]
                curr_s = sorted_slots[i]
                next_s = sorted_slots[i + 1]
                if prev_s in day_slots and curr_s in day_slots and next_s in day_slots:
                    self.model.Add(
                        day_slots[prev_s] + day_slots[next_s] <= 1
                    ).OnlyEnforceIf(day_slots[curr_s].Not())
                    n += 1

        # Slot number bounding
        for (week, day), locations in self.games_per_location.items():
            for location, location_vars in locations.items():
                fields_at_loc = [f for f in self.fields if f.location == location]
                num_fields = len(fields_at_loc)
                if num_fields == 0:
                    continue
                nlg = self.model.NewIntVar(0, len(self.games), f'u_nlg_{week}_{location}')
                self.model.Add(nlg == sum(location_vars))
                quot = self.model.NewIntVar(0, len(self.timeslots), f'u_quot_{week}_{location}')
                self.model.AddDivisionEquality(quot, nlg, num_fields)
                nts = self.model.NewIntVar(0, len(self.timeslots), f'u_nts_{week}_{location}')
                self.model.Add(nts == quot + 1)

                nvars = slot_number_vars.get((week, day, location), {})
                for ds, nv in nvars.items():
                    ind = location_day_slots[(week, day, location)][ds]
                    if location == BROADMEADOW:
                        eq = self.model.NewIntVar(0, 200, f'u_eq_{week}_{location}_{ds}')
                        self.model.Add(eq >= self.BROADMEADOW_MAX_SLOTS)
                        self.model.Add(eq >= nts)
                        ci = self.model.NewBoolVar(f'u_ci_{week}_{location}_{ds}')
                        self.model.Add(nts <= self.BROADMEADOW_MAX_SLOTS).OnlyEnforceIf(ci)
                        self.model.Add(nts > self.BROADMEADOW_MAX_SLOTS).OnlyEnforceIf(ci.Not())
                        self.model.Add(eq <= self.BROADMEADOW_MAX_SLOTS).OnlyEnforceIf(ci)
                        self.model.Add(eq <= nts).OnlyEnforceIf(ci.Not())
                        self.model.Add(nv <= eq).OnlyEnforceIf(ind)
                    else:
                        self.model.Add(nv <= nts).OnlyEnforceIf(ind)
                    n += 1

        # Stacking: if any field uses slot N+1, all fields must use slot N
        # Build per-field indicators at each location (stored for soft phase)
        self._loc_fields = defaultdict(lambda: defaultdict(dict))
        for sfk, vars_list in self.slot_field_vars.items():
            week, day, location, field_name, day_slot = sfk
            if vars_list:
                ind = self.pool.get_or_create_bool(
                    ('slot_field_used', week, day, location, field_name, day_slot),
                    vars_list,
                    f'u_sf_used_{week}_{day}_{location}_{field_name}_{day_slot}')
                self._loc_fields[(week, day, location)][field_name][day_slot] = ind

        for (week, day, location), fields_dict in self._loc_fields.items():
            all_field_names = sorted(fields_dict.keys())
            all_slots = set()
            for field_slots in fields_dict.values():
                all_slots.update(field_slots.keys())
            if not all_slots:
                continue
            sorted_all_slots = sorted(all_slots)

            # Stacking: if any field uses slot s, all other fields must use slot s-1
            for i, s in enumerate(sorted_all_slots):
                if i == 0:
                    continue
                prev_s = sorted_all_slots[i - 1]
                for fn in all_field_names:
                    if s in fields_dict.get(fn, {}) and prev_s in fields_dict.get(fn, {}):
                        for other_fn in all_field_names:
                            if other_fn == fn:
                                continue
                            if s in fields_dict.get(other_fn, {}) and prev_s in fields_dict.get(other_fn, {}):
                                self.model.AddImplication(
                                    fields_dict[other_fn][s],
                                    fields_dict[fn][prev_s])
                                n += 1

        return n

    def _best_timeslot_choices_soft(self):
        """Soft: prefer West Field for last-slot-only games at Broadmeadow."""
        n = 0
        if self._loc_fields is None:
            return 0
        weight = self._get_penalty_weight('BestTimeslotWF', 50000)
        penalties = []

        for (week, day, location), fields_dict in self._loc_fields.items():
            if location != BROADMEADOW:
                continue
            all_slots = set()
            for field_slots in fields_dict.values():
                all_slots.update(field_slots.keys())
            if not all_slots:
                continue
            max_slot = max(all_slots)

            # Count fields active on the last slot
            last_slot_fields = []
            for fn, slots in fields_dict.items():
                if max_slot in slots:
                    last_slot_fields.append((fn, slots[max_slot]))

            if len(last_slot_fields) >= 2:
                field_inds = {fn: ind for fn, ind in last_slot_fields}
                total_active = self.model.NewIntVar(0, len(last_slot_fields),
                    f'u_ls_total_{week}_{day}_{max_slot}')
                self.model.Add(total_active == sum(ind for _, ind in last_slot_fields))

                single_field = self.model.NewBoolVar(f'u_ls_single_{week}_{day}_{max_slot}')
                self.model.Add(total_active == 1).OnlyEnforceIf(single_field)
                self.model.Add(total_active != 1).OnlyEnforceIf(single_field.Not())

                if 'WF' in field_inds:
                    wf_ind = field_inds['WF']
                    # Penalty: single field on last slot but NOT West Field
                    violation = self.model.NewBoolVar(f'u_wf_viol_{week}_{day}_{max_slot}')
                    self.model.AddBoolAnd([single_field, wf_ind.Not()]).OnlyEnforceIf(violation)
                    self.model.AddBoolOr([single_field.Not(), wf_ind]).OnlyEnforceIf(violation.Not())
                    penalties.append(violation)
                    n += 1

        if penalties:
            self.data['penalties']['BestTimeslotWF'] = {'weight': weight, 'penalties': penalties}
        return n

    def _club_day_field_contiguity(self):
        """Intra-day: club day games on same field + contiguous timeslots."""
        n = 0
        club_days = self.data.get('club_days', {})
        if not club_days:
            return 0

        for club_name in club_days:
            game_vars = self.club_day_game_vars.get(club_name, {})
            if not game_vars:
                continue

            # Same field constraint
            field_vars = defaultdict(list)
            for key, var in game_vars.items():
                field_vars[key[9]].append(var)
            if len(field_vars) > 1:
                fi_list = []
                for fn, vl in field_vars.items():
                    fi = self.pool.get_or_create_bool(
                        ('cd_field', club_name, fn), vl,
                        f'u_cd_field_{club_name}_{fn}')
                    fi_list.append(fi)
                self.model.Add(sum(fi_list) == 1)
                n += 1

            # Contiguous timeslots
            slot_vars = defaultdict(list)
            for key, var in game_vars.items():
                slot_vars[key[4]].append(var)  # day_slot
            slot_inds = {}
            for ds, vl in slot_vars.items():
                si = self.pool.get_or_create_bool(
                    ('cd_slot', club_name, ds), vl,
                    f'u_cd_slot_{club_name}_{ds}')
                slot_inds[ds] = si
            sorted_slots = sorted(slot_inds.keys())
            for i in range(1, len(sorted_slots) - 1):
                ps, cs, ns = sorted_slots[i - 1], sorted_slots[i], sorted_slots[i + 1]
                self.model.Add(
                    slot_inds[ps] + slot_inds[ns] <= 1
                ).OnlyEnforceIf(slot_inds[cs].Not())
                n += 1
        return n

    def _club_game_spread_hard(self):
        """Hard: limit gap and double-ups between a club's games on a given day."""
        n = 0
        defaults = self.data.get('constraint_defaults', {})
        config_slack = self.slack.get('ClubGameSpread', 0)
        hard_limit = defaults.get('club_game_spread_max_gap', 2) + config_slack
        max_overlap = defaults.get('club_game_spread_max_overlap', 0)

        # Regroup by (club, week, day)
        club_week_day_groups = defaultdict(dict)
        for (club, week, day, day_slot), vars_list in self.by_club_week_day_slot.items():
            club_week_day_groups[(club, week, day)][day_slot] = vars_list

        self._cgs_keys = []

        for (club, week, day), slots_dict in club_week_day_groups.items():
            unique_slots = sorted(slots_dict.keys())
            if len(unique_slots) <= 1:
                continue

            min_slot, max_slot = unique_slots[0], unique_slots[-1]
            is_active = {}
            for s in unique_slots:
                ia = self.pool.get_or_create_bool(
                    ('cgs_active', club, week, day, s), slots_dict[s],
                    f'u_cgs_act_{club}_w{week}_{day}_s{s}')
                is_active[s] = ia

            num_used = self.model.NewIntVar(0, len(unique_slots), f'u_cgs_used_{club}_w{week}_{day}')
            self.model.Add(num_used == sum(is_active[s] for s in unique_slots))

            # num_games counts total games (can be > num_used for double-ups)
            all_vars_flat = []
            for s in unique_slots:
                all_vars_flat.extend(slots_dict[s])
            num_games = self.model.NewIntVar(0, len(all_vars_flat), f'u_cgs_ngames_{club}_w{week}_{day}')
            self.model.Add(num_games == sum(all_vars_flat))

            min_active = self.model.NewIntVar(min_slot, max_slot, f'u_cgs_min_{club}_w{week}_{day}')
            max_active = self.model.NewIntVar(min_slot, max_slot, f'u_cgs_max_{club}_w{week}_{day}')
            for s in unique_slots:
                self.model.Add(min_active <= s).OnlyEnforceIf(is_active[s])
                self.model.Add(max_active >= s).OnlyEnforceIf(is_active[s])

            range_size = self.model.NewIntVar(1, max_slot - min_slot + 1, f'u_cgs_range_{club}_w{week}_{day}')
            self.model.Add(range_size == max_active - min_active + 1)

            # gap = range_size - num_games (can be negative for double-ups)
            max_gap_val = max_slot - min_slot
            gap = self.model.NewIntVar(-(max_slot - min_slot + 1), max_gap_val, f'u_cgs_gap_{club}_w{week}_{day}')
            self.model.Add(gap == range_size - num_games)

            has_multi = self.model.NewBoolVar(f'u_cgs_multi_{club}_w{week}_{day}')
            self.model.Add(num_games >= 2).OnlyEnforceIf(has_multi)
            self.model.Add(num_games <= 1).OnlyEnforceIf(has_multi.Not())
            self.pool.register(('cgs_multi', club, week, day), has_multi)

            # UPPER: spread (using num_used for gap, not num_games) <= max_gap + slack
            spread = self.model.NewIntVar(0, max_gap_val, f'u_cgs_spread_{club}_w{week}_{day}')
            self.model.Add(spread == range_size - num_used)
            self.model.Add(spread <= hard_limit).OnlyEnforceIf(has_multi)

            # LOWER: double-up prevention (num_games - num_used <= max_overlap + slack)
            overlap = self.model.NewIntVar(0, len(all_vars_flat), f'u_cgs_overlap_{club}_w{week}_{day}')
            self.model.Add(overlap == num_games - num_used)
            self.model.Add(overlap <= max_overlap + config_slack).OnlyEnforceIf(has_multi)

            n += 1

            # Register for soft phase reuse via pool
            self.pool.register(('cgs_gap', club, week, day), gap)
            self.pool.register(('cgs_max_gap_val', club, week, day), max_gap_val)
            self._cgs_keys.append((club, week, day))
        return n

    def _club_game_spread_soft(self):
        """Soft: penalize gap between a club's games on a given day."""
        n = 0
        weight = self._get_penalty_weight('ClubGameSpread', 5000)
        self.data['penalties']['ClubGameSpread'] = {'weight': weight, 'penalties': []}

        for (club, week, day) in self._cgs_keys:
            gap = self.pool.get(('cgs_gap', club, week, day))
            has_multi = self.pool.get(('cgs_multi', club, week, day))
            max_gap_val = self.pool.get(('cgs_max_gap_val', club, week, day))
            if gap is None or has_multi is None:
                continue

            # Penalty on absolute gap (spread)
            abs_gap = self.model.NewIntVar(0, max_gap_val, f'u_cgs_absgap_{club}_w{week}_{day}')
            self.model.AddAbsEquality(abs_gap, gap)
            pen = self.model.NewIntVar(0, max_gap_val, f'u_cgs_pen_{club}_w{week}_{day}')
            self.model.Add(pen >= abs_gap).OnlyEnforceIf(has_multi)
            self.model.Add(pen == 0).OnlyEnforceIf(has_multi.Not())
            self.data['penalties']['ClubGameSpread']['penalties'].append(pen)
            n += 1
        return n

    # ================================================================
    # CONVENIENCE: APPLY ALL
    # ================================================================

    def apply_all(self):
        """Apply all constraints (equivalent to all 19 AI constraints)."""
        self.build_groupings()
        a = self.apply_stage_1_hard()
        b = self.apply_stage_2_soft()
        return a + b
