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
    # spec-021: BROADMEADOW_MAX_SLOTS removed — the EnsureBestTimeslotChoices
    # slot-cap IntVars it gated are gone (replaced by the VenueEarliestSlotFill
    # atom's monotone-fill chain, which needs no slot cap).

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

        # Helper-var registry (pool-style API: shared by atoms and the engine).
        # `self.pool` retained as alias so existing internal methods keep working.
        self.registry = HelperVarRegistry(self.model)
        self.pool = self.registry

        # CGS iteration keys + per-group hole indicators (populated by
        # _club_game_spread_hard, consumed by _club_game_spread_soft).
        self._cgs_keys = []
        self._cgs_hole_vars = {}

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

        # spec-036 Unit B: `by_grade_clubpair_round` / `by_sunday_clubpair_round_field`
        # removed — they fed only the legacy `_club_alignment_hard/_soft` engine
        # methods (deleted). Club-vs-club alignment is now the spec-005
        # `ClubVsClubStackedWeekends` / `ClubVsClubStackedCoLocation` atoms,
        # dispatched outside the engine; they build their own groupings from X.

        # --- Venue groupings ---
        # spec-018: the `non_default_*` per-club home/away-week maps (and their
        # `maitland_*` aliases) plus the `by_week_location` map were all deleted
        # along with the venue-sequencing rules (`NonDefaultHomeGrouping` /
        # `AwayAtNonDefaultGrouping`). Nothing consumes them anymore.

        # spec-024: bm_slot_club / bm_field_club removed -- they fed only the
        # deleted Maximise/MinimiseClubsBroadmeadow rules (never read by the engine).

        # --- Club game spread (spec-024: per-field) ---
        self.by_club_week_day_slot = defaultdict(list)
        # spec-024: per (club, week, day, field) -> {day_slot: [vars]} for the
        # field-aware contiguity rule; by_club_week_day_slot is kept for the
        # off-primary-field soft penalty's per-(club, week, day) totals.
        self.by_club_week_day_field_slot = defaultdict(lambda: defaultdict(list))

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
        # spec-020: `preferred_date_vars` removed — PreferredDates deleted; the
        # generic `PreferredGames` atom scans X directly (no engine groupings).

        # --- Home/away ---
        self.home_away_venue = defaultdict(lambda: {'home': [], 'away': []})

        # spec-021: slot_vars_by_location / slot_field_vars / games_per_location
        # removed — they fed only the deleted _best_timeslot_choices_* methods.
        # Earliest-slot fill is now the VenueEarliestSlotFill atom, which builds
        # its own per-(week, date, location) slot grouping from X.

        # --- Club day ---
        self.club_day_game_vars = defaultdict(dict)

        # --- EnsureEqualGames (uses games x timeslots pattern) ---
        self.grade_team_vars = defaultdict(lambda: defaultdict(list))
        self.grade_pair_vars = defaultdict(lambda: defaultdict(list))

        # spec-020: PHL preferred-date grouping removed (PreferredDates deleted).

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

            # spec-036 Unit B: the ClubVsClubAlignment grouping build (3rd-6th
            # club-pair/round + Sunday field) is removed along with the legacy
            # `_club_alignment_*` engine methods. The stacked atoms scan X directly.

            # spec-018: venue groupings for the deleted MaxMaitlandHomeWeekends /
            # NonDefaultHomeGrouping / AwayAtNonDefaultGrouping rules removed.

            # spec-024: Broadmeadow slot/field club groupings removed -- they fed
            # only the deleted Maximise/MinimiseClubsBroadmeadow rules.

            # --- ClubGameSpread (spec-024: day-level + per-field) ---
            if t1_club:
                self.by_club_week_day_slot[(t1_club, week, day, day_slot)].append(var)
                self.by_club_week_day_field_slot[(t1_club, week, day, field_name)][day_slot].append(var)
            if t2_club and t2_club != t1_club:
                self.by_club_week_day_slot[(t2_club, week, day, day_slot)].append(var)
                self.by_club_week_day_field_slot[(t2_club, week, day, field_name)][day_slot].append(var)

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

            # spec-021: EnsureBestTimeslotChoices groupings removed (see above).

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
        # spec-033 Unit C: `TeamConflict` is no longer a hard feasibility rule.
        # Its soft analogue (`_team_conflict_soft`) is dispatched in
        # `apply_stage_2_soft()`. No hard `<= 1` constraint is added here.
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
        # spec-036 Unit B: the legacy `_club_alignment_hard` dispatch is removed.
        # Club-vs-club alignment is enforced by the spec-005
        # `ClubVsClubStackedWeekends` / `ClubVsClubStackedCoLocation` atoms,
        # dispatched outside the engine; the `ClubVsClubAlignment` engine key has
        # no `groups=` so it was never reached by any group selection anyway.
        # spec-018: `MaitlandHomeGrouping` / `AwayAtMaitlandGrouping` hard
        # dispatch removed — venue-sequencing rules deleted.
        if 'ClubDay' not in _skip:
            c += self._club_day_atoms_hard()
        if 'ClubGameSpread' not in _skip:
            c += self._club_game_spread_hard()
        # spec-021: EnsureBestTimeslotChoices hard removed — VenueEarliestSlotFill atom.
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

    # spec-033 Unit C: the hard `_team_conflict` (which forbade two named
    # conflicting teams from sharing a `(week, day_slot)` via
    # `Add(sum(v1) + sum(v2) <= 1)`) has been REMOVED. TeamConflict is now a
    # soft preference — see `_team_conflict_soft` in apply_stage_2_soft below.

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
    # spec-030: PHLAnd2ndConcurrencyAtBroadmeadow removed (subsumed by
    # PHLAnd2ndAdjacency's same-venue branch).
    _PHL_HARD_ATOMS = (
        PHLConcurrencyAtBroadmeadow,
    )
    # spec-020: `_PHL_SOFT_ATOMS` / `_phl_times_atoms_soft` removed — the only
    # member was PreferredDates, now deleted. The generic `PreferredGames`
    # soft atom has no atom_group and dispatches via the non-engine fallback
    # in `apply_solver_stage` (constraints/stages.py), not through the engine.

    def _phl_times_atoms_hard(self):
        n = 0
        for atom_cls in self._PHL_HARD_ATOMS:
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

    # spec-036 Unit B: the legacy in-engine ClubVsClub alignment methods
    # `_club_alignment_hard` / `_club_alignment_soft` were DELETED. They were a
    # `--simple`-only (`apply_phase_*`) parity-reference path; after spec-036
    # Unit A rerouted the single-solve path through the stage dispatcher, no
    # solve path dispatched them (the `ClubVsClubAlignment` engine key has no
    # `groups=`). Club-vs-club alignment is fully covered in every solve path by
    # the spec-005 `ClubVsClubStackedWeekends` / `ClubVsClubStackedCoLocation`
    # atoms (dispatched outside the engine). The `ClubVsClubAlignment` registry
    # entry + tester check (`_check_club_vs_club_alignment`) are retained.

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
            # spec-029: normalize the dict form (optional 'note') to a datetime.
            desired_date, _ = normalize_club_day(desired_date)
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
        # spec-033 Unit C: TeamConflict soft penalty (no hard component).
        if 'TeamConflict' not in _skip:
            c += self._team_conflict_soft()
        # ClubGradeAdjacency soft REMOVED ENTIRELY (spec-007). The convenor-
        # facing adjacent-grade penalty was over-restrictive in practice.
        # If a specific team-pair conflict matters, declare it via
        # `TEAM_PAIR_NO_CONCURRENCY` (handled by the `TeamPairNoConcurrency`
        # atom outside the engine).
        # spec-036 Unit B: the legacy `_club_alignment_soft` dispatch is removed
        # (superseded by the spec-005 stacked atoms dispatched outside the engine).
        # spec-018: `MaitlandHomeGrouping` / `AwayAtMaitlandGrouping` soft
        # dispatch removed — venue-sequencing rules deleted.
        # spec-020: PHL/2nd soft dispatch removed — its only member
        # (PreferredDates) is deleted; PHL/2nd times soft has no remaining
        # penalties. `PreferredGames` dispatches via the non-engine fallback.
        if 'PreferredTimesConstraint' not in _skip:
            c += self._preferred_times()
        # spec-021: EnsureBestTimeslotChoices soft removed (BestTimeslotWF deleted).
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

    def _team_conflict_soft(self):
        """spec-033 Unit C: soft penalty for conflicting team pairs sharing a slot.

        For each declared `(team1, team2)` conflict pair and each `(week, day_slot)`
        where BOTH teams could appear, add a penalty BoolVar `p` forced to 1 only
        when both teams actually play that slot:

            p >= sum(v1) + sum(v2) - 1

        (When at most one of the two plays the slot, `sum(v1)+sum(v2) <= 1`, so the
        lower bound is <= 0 and `p` is free to be 0; when both play,
        `sum(v1)+sum(v2) == 2`, forcing `p >= 1`.) No hard constraint is added, so
        a tolerated clash never blocks feasibility. Empty `team_conflicts` ⇒ the
        bucket is created with zero penalty vars.
        """
        weight = self._get_penalty_weight('TeamConflict', 200000)
        self.data['penalties']['TeamConflict'] = {'weight': weight, 'penalties': []}
        conflicts = self.data.get('team_conflicts', [])
        if not conflicts:
            return 0
        n = 0
        # Regroup by (week, day_slot) -> {team: [vars]}
        slot_teams = defaultdict(lambda: defaultdict(list))
        for (week, day_slot, team), vars_list in self.by_week_day_slot_team.items():
            slot_teams[(week, day_slot)][team].extend(vars_list)

        for team1, team2 in conflicts:
            for slot, team_vars in slot_teams.items():
                v1 = team_vars.get(team1, [])
                v2 = team_vars.get(team2, [])
                if v1 and v2:
                    p = self.model.NewBoolVar(
                        f"u_tconf_{team1}_{team2}_w{slot[0]}_s{slot[1]}")
                    self.model.Add(p >= sum(v1) + sum(v2) - 1)
                    self.data['penalties']['TeamConflict']['penalties'].append(p)
                    n += 1
        return n

    # _grade_adjacency_soft REMOVED ENTIRELY (spec-007). The convenor-facing
    # adjacent-grade-same-timeslot penalty was over-restrictive and is no
    # longer enforced anywhere. For per-team-pair conflicts, use the
    # `TeamPairNoConcurrency` atom + `TEAM_PAIR_NO_CONCURRENCY` config.

    # spec-036 Unit B: `_club_alignment_soft` was DELETED here (see the matching
    # note where `_club_alignment_hard` lived). It registered the
    # `ClubVsClubAlignment` / `ClubVsClubAlignmentField` penalty buckets via
    # `_get_penalty_weight(..., default)`; both weights were removed from
    # `PENALTY_WEIGHTS` (no remaining live reader). Soft alignment is now the
    # spec-005 stacked-atom cluster dispatched outside the engine.

    # spec-018: `_maitland_grouping_soft` (NonDefaultHomeGrouping soft
    # imbalance penalty) and `_away_maitland_soft` (AwayAtNonDefaultGrouping
    # soft) deleted — venue-sequencing rules removed.

    # spec-020: `_phl_times_soft` (legacy preferred-date parity reference)
    # removed — it read the now-deleted `self.preferred_date_vars` grouping and
    # the removed `phl_preferences` config key. The generic `PreferredGames`
    # atom replaces it.

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
        """Hard: a club's games on a day form a near-contiguous block PER FIELD.

        spec-024: re-scoped from (club, week, day) to (club, week, day, field).
        spec-021 (resolved decision A): the allowed hole count is games-derived,
        ``gap_cap = max(0, min(1, n_field - 3))`` (+ ``--slack ClubGameSpread``),
        where ``n_field`` is the club's games ON THAT FIELD that day. A club with
        <=3 games on a field must have NO interior holes on that field; with >=4
        at most ONE (a soft penalty drives it to zero). Encoded with ``slot_used``
        indicators (shared ``_contiguity`` primitive) + per-position
        "used-before/after" channels + a per-slot hole indicator. The off-primary
        (multi-field) soft penalty lives in ``_club_game_spread_soft``. The lower
        no-double-up bound moved to the ``ClubNoConcurrentSlot`` atom.
        """
        from constraints.atoms._contiguity import slot_used_indicators
        n = 0
        config_slack = self.slack.get('ClubGameSpread', 0)

        # spec-024: per (club, week, day, field). by_club_week_day_field_slot is
        # already keyed that way -> {day_slot: [vars]} per field group.
        self._cgs_keys = []
        self._cgs_hole_vars = {}

        for (club, week, day, field), slots_dict in self.by_club_week_day_field_slot.items():
            sorted_slots = sorted(slots_dict.keys())
            if len(sorted_slots) < 2:
                continue

            slot_inds = slot_used_indicators(
                self.registry, slots_dict, 'club_spread_slot_used',
                club, week, day, field)

            # Channel "a used slot exists before / after each position".
            m = len(sorted_slots)
            pref = [slot_inds[sorted_slots[0]]] + [None] * (m - 1)
            for i in range(1, m):
                p = self.model.NewBoolVar(f'u_cgs_pref_{club}_w{week}_{day}_{field}_{i}')
                self.model.AddMaxEquality(p, [pref[i - 1], slot_inds[sorted_slots[i]]])
                pref[i] = p
            suf = [None] * (m - 1) + [slot_inds[sorted_slots[m - 1]]]
            for i in range(m - 2, -1, -1):
                s = self.model.NewBoolVar(f'u_cgs_suf_{club}_w{week}_{day}_{field}_{i}')
                self.model.AddMaxEquality(s, [suf[i + 1], slot_inds[sorted_slots[i]]])
                suf[i] = s

            # hole[i] = (used before) AND (used after) AND (this slot empty).
            hole_vars = []
            for i in range(1, m - 1):
                used_before, used_after = pref[i - 1], suf[i + 1]
                cur = slot_inds[sorted_slots[i]]
                h = self.model.NewBoolVar(
                    f'u_cgs_hole_{club}_w{week}_{day}_{field}_s{sorted_slots[i]}')
                self.model.AddBoolAnd(
                    [used_before, used_after, cur.Not()]).OnlyEnforceIf(h)
                self.model.AddBoolOr(
                    [used_before.Not(), used_after.Not(), cur]).OnlyEnforceIf(h.Not())
                hole_vars.append(h)

            # gap_cap = max(0, min(1, n_field - 3)) = (n_field >= 4 ? 1 : 0), + slack,
            # where n_field = the club's games ON THIS FIELD that day.
            all_vars_flat = [v for s in sorted_slots for v in slots_dict[s]]
            ge4 = self.model.NewBoolVar(f'u_cgs_ge4_{club}_w{week}_{day}_{field}')
            self.model.Add(sum(all_vars_flat) >= 4).OnlyEnforceIf(ge4)
            self.model.Add(sum(all_vars_flat) <= 3).OnlyEnforceIf(ge4.Not())

            if hole_vars:
                self.model.Add(sum(hole_vars) <= config_slack).OnlyEnforceIf(ge4.Not())
                self.model.Add(sum(hole_vars) <= 1 + config_slack).OnlyEnforceIf(ge4)
                n += 1

            self._cgs_hole_vars[(club, week, day, field)] = hole_vars
            self._cgs_keys.append((club, week, day, field))

        # spec-033 Unit D: field-concentration HARD cap — the second of
        # ClubGameSpread's two interlocked structures. The contiguity hole-cap
        # above bounds holes WITHIN a field; this bounds the number of distinct
        # fields a club may spread across in one (club, week, day). Together with
        # the off-primary soft (which pushes the field count -> 1, see
        # _club_game_spread_soft) it forms: contiguity (<=1 hole/field, push->0
        # holes) + field concentration (<=max_fields, push->1 field).
        defaults = self.data.get('constraint_defaults', {})
        max_fields = defaults.get('club_game_spread_max_fields', 2)

        # Re-key the per-(club,week,day,field) grouping that is ALREADY iterated
        # above into per-(club,week,day) -> {field: [all game vars on that field]}.
        cwd_field_vars = defaultdict(dict)  # (club,week,day) -> {field: [vars]}
        for (club, week, day, field), slots_dict in self.by_club_week_day_field_slot.items():
            flat = [v for vs in slots_dict.values() for v in vs]
            if flat:
                cwd_field_vars[(club, week, day)][field] = flat

        for (club, week, day), fields in cwd_field_vars.items():
            if len(fields) <= max_fields + config_slack:
                # Fewer candidate fields than the cap allows -> cap can never
                # bite, skip to avoid pointless vars/constraints.
                continue
            field_used = []
            for field, vlist in fields.items():
                fu = self.model.NewBoolVar(
                    f'u_cgs_fieldused_{club}_w{week}_{day}_{field}')
                # Channel: if ANY game var on field f is 1, field_used[f] is
                # forced to 1. Each constraint field_used >= v alone forces the
                # var up whenever its v is set. Because scheduling a game on f
                # requires that f's field_used be 1, the sum-cap below bounds how
                # many distinct fields can carry a game -> the cap bites.
                for v in vlist:
                    self.model.Add(fu >= v)
                field_used.append(fu)
            self.model.Add(sum(field_used) <= max_fields + config_slack)
            n += 1
        return n

    def _club_game_spread_soft(self):
        """Soft: per-field residual holes + the off-primary-field game count.

        Two ClubGameSpread soft pressures share one penalty bucket (spec-024):
        1. Every residual interior hole on a field (drives the per-field hard cap
           toward zero holes even when it permits one — spec-021).
        2. ``off_primary`` per (club, week, day): the number of the club's games
           NOT on its most-used field that day = ``total_games - max_field_count``.
           Zero iff all the club's games that day sit on a single field. This
           discourages splitting a club's day across multiple fields.

        spec-033 Unit D: this off-primary term is the SOFT half of ClubGameSpread's
        field-concentration structure — it pushes the distinct-field count -> 1.
        The matching HARD half lives in ``_club_game_spread_hard`` as a per-(club,
        week, day) ``sum(field_used) <= club_game_spread_max_fields + slack`` cap
        (default 2). Together: hard ceiling of <=2 fields + soft push toward 1.
        (Mirrors the contiguity structure: hard <=1 hole/field + soft push to 0.)
        """
        n = 0
        weight = self._get_penalty_weight('ClubGameSpread', 5000)
        self.data['penalties']['ClubGameSpread'] = {'weight': weight, 'penalties': []}

        # (1) per-field hole indicators (keys are 4-tuples after spec-024).
        hole_map = getattr(self, '_cgs_hole_vars', {})
        for key in self._cgs_keys:
            for h in hole_map.get(key, []):
                self.data['penalties']['ClubGameSpread']['penalties'].append(h)
                n += 1

        # (2) off-primary-field penalty per (club, week, day).
        # Aggregate all of a club's day vars by field from the per-field grouping.
        club_day_fields = defaultdict(dict)  # (club,week,day) -> {field: [vars]}
        for (club, week, day, field), slots_dict in self.by_club_week_day_field_slot.items():
            flat = [v for vs in slots_dict.values() for v in vs]
            if flat:
                club_day_fields[(club, week, day)][field] = flat

        for (club, week, day), fields in club_day_fields.items():
            if len(fields) < 2:
                continue  # single field -> off_primary is structurally 0.
            all_vars = [v for fl in fields.values() for v in fl]
            total = len(all_vars)  # upper bound on games that day
            field_counts = []
            for field, fl in fields.items():
                c = self.model.NewIntVar(0, len(fl),
                                         f'u_cgs_fcount_{club}_w{week}_{day}_{field}')
                self.model.Add(c == sum(fl))
                field_counts.append(c)
            max_count = self.model.NewIntVar(0, total,
                                             f'u_cgs_fmax_{club}_w{week}_{day}')
            self.model.AddMaxEquality(max_count, field_counts)
            off_primary = self.model.NewIntVar(0, total,
                                               f'u_cgs_offprimary_{club}_w{week}_{day}')
            self.model.Add(off_primary == sum(all_vars) - max_count)
            self.data['penalties']['ClubGameSpread']['penalties'].append(off_primary)
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
