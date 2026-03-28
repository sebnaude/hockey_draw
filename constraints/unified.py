# constraints/unified.py
"""
Unified Constraint Engine for hockey draw scheduling.

Replaces the 19 individual AI constraint classes with a single engine that:
1. Makes ONE pass over X to build all grouping dicts
2. Creates shared indicator variables (consumed by multiple constraints)
3. Splits constraints into 3 phases: hard inter-week → soft inter-week → intra-day

Usage:
    engine = UnifiedConstraintEngine(model, X, data)
    engine.build_groupings()
    engine.apply_phase_a()  # Hard inter-week (feasibility)
    engine.apply_phase_b()  # Soft inter-week penalties
    engine.apply_phase_c()  # Intra-day optimization
"""

from ortools.sat.python import cp_model
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

from utils import (
    get_teams_from_club, get_club_from_clubname, get_nearest_week_by_date,
    get_duplicated_graded_teams
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

    # Configurable limits
    PHL_ADJACENCY_MINUTES = 180
    BROADMEADOW_MAX_SLOTS = 6
    MAITLAND_AWAY_HARD_LIMIT = 3
    CLUBS_ON_FIELD_HARD_LIMIT = 5
    CLUB_GAME_SPREAD_HARD_LIMIT = 2

    def __init__(self, model: cp_model.CpModel, X: dict, data: dict):
        self.model = model
        self.X = X
        self.data = data
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

        # Shared indicators (populated lazily during constraint application)
        self._shared_indicators = {}

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
        """Single pass over X to populate all grouping dicts."""
        if self._groupings_built:
            return

        # --- Core groupings ---
        self.by_week_team = defaultdict(list)
        self.by_slot_field = defaultdict(list)
        self.by_week_day_slot_team = defaultdict(list)

        # --- Matchup spacing ---
        self.by_grade_pair_round = defaultdict(lambda: defaultdict(list))

        # --- Club-grade adjacency ---
        self.by_slot_club_grade = defaultdict(list)
        self.club_dup_games = defaultdict(list)

        # --- Club vs Club alignment ---
        self.by_grade_clubpair_round = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.by_sunday_clubpair_round_field = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # --- Venue groupings ---
        self.by_week_location = defaultdict(list)
        self.maitland_home_week = defaultdict(list)
        self.maitland_all_week = defaultdict(list)
        self.maitland_away_club_week = defaultdict(lambda: defaultdict(list))

        # --- Broadmeadow groupings ---
        self.bm_slot_club = defaultdict(lambda: defaultdict(list))
        self.bm_field_club = defaultdict(lambda: defaultdict(list))

        # --- Club game spread ---
        self.by_club_week_day_slot = defaultdict(list)
        self.by_club_week_day_field = defaultdict(list)

        # --- PHL/2nd grade ---
        self.phl_club_week_day = defaultdict(lambda: defaultdict(list))
        self.second_club_week_day = defaultdict(lambda: defaultdict(list))
        self.phl_slot_vars = defaultdict(list)
        self.club_phl_vars = defaultdict(list)
        self.club_2nd_vars = defaultdict(list)
        self.phl_friday_broadmeadow = []
        self.phl_friday_gosford = []
        self.phl_friday_maitland = []
        self.phl_friday_gosford_round = defaultdict(list)
        self.phl_round1_vars = defaultdict(list)
        self.preferred_date_vars = defaultdict(list)

        # --- Home/away ---
        self.home_away_venue = defaultdict(lambda: {'home': [], 'away': []})

        # --- Timeslot optimization ---
        self.slot_vars_by_location = defaultdict(list)
        self.games_per_location = defaultdict(lambda: defaultdict(list))
        self.worst_time_vars = []  # 7pm (19:00) non-Friday vars

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

        # Get club day dates
        club_days = self.data.get('club_days', {})
        club_day_dates = {}
        for club_name, dt in club_days.items():
            club_day_dates[dt.date().strftime('%Y-%m-%d')] = club_name

        # === SINGLE PASS OVER X ===
        for key, var in self.X.items():
            if len(key) < 11:
                # Dummy timeslot - handle for EnsureEqualGames
                if len(key) == 4:
                    t1, t2, grade, dummy_idx = key
                    self.grade_team_vars[grade][t1].append(var)
                    self.grade_team_vars[grade][t2].append(var)
                    self.grade_pair_vars[grade][tuple(sorted((t1, t2)))].append(var)
                continue

            t1, t2, grade = key[0], key[1], key[2]
            day, day_slot, time_val = key[3], key[4], key[5]
            week, date_str, round_no = key[6], key[7], key[8]
            field_name, location = key[9], key[10]

            # Skip dummy timeslots
            if not day:
                continue

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

            # --- ClubGradeAdjacency ---
            slot = (week, day_slot)
            if t1_club and t2_club:
                if t1_club != t2_club:
                    self.by_slot_club_grade[(slot, t1_club, grade)].append(var)
                    self.by_slot_club_grade[(slot, t2_club, grade)].append(var)
                    # Duplicate-grade tracking
                    if t1 in self.club_dup_grades.get(t1_club, {}).get(grade, []):
                        self.club_dup_games[(t1_club, slot, grade)].append(var)
                    if t2 in self.club_dup_grades.get(t2_club, {}).get(grade, []):
                        self.club_dup_games[(t2_club, slot, grade)].append(var)
                else:
                    self.by_slot_club_grade[(slot, t1_club, grade)].append(var)

            # --- ClubVsClubAlignment (3rd-6th only) ---
            if grade not in ['PHL', '2nd'] and t1_club and t2_club:
                club_pair = tuple(sorted((t1_club, t2_club)))
                self.by_grade_clubpair_round[grade][club_pair][round_no].append(var)
                if day == 'Sunday':
                    self.by_sunday_clubpair_round_field[club_pair][round_no][field_name].append(var)

            # --- MaxMaitlandHomeWeekends (non-Broadmeadow) ---
            if location != BROADMEADOW:
                self.by_week_location[(week, location)].append(var)

            # --- MaitlandHomeGrouping ---
            if 'Maitland' in t1 or 'Maitland' in t2:
                self.maitland_all_week[week].append(var)
                if location == MAITLAND:
                    self.maitland_home_week[week].append(var)

            # --- AwayAtMaitlandGrouping ---
            if location == MAITLAND:
                for team in [t1, t2]:
                    if 'Maitland' not in team:
                        club = self._get_club(team)
                        if club:
                            self.maitland_away_club_week[week][club].append(var)

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
                self.by_club_week_day_field[(t1_club, week, day, (field_name, location))].append(var)
            if t2_club and t2_club != t1_club:
                self.by_club_week_day_slot[(t2_club, week, day, day_slot)].append(var)
                self.by_club_week_day_field[(t2_club, week, day, (field_name, location))].append(var)

            # --- PHL/2nd grade specific ---
            if grade == 'PHL':
                slot_id = (week, day, day_slot, location)
                self.phl_slot_vars[slot_id].append(var)

                for team in [t1, t2]:
                    club = self._get_club(team)
                    if club:
                        self.club_phl_vars[(*slot_id, club)].append(var)
                        if club in self.club_2nd_teams:
                            self.phl_club_week_day[(club, week, day)][(time_val, location)].append(var)

                if day == 'Friday' and location == BROADMEADOW:
                    self.phl_friday_broadmeadow.append(var)
                if day == 'Friday' and location == GOSFORD:
                    self.phl_friday_gosford.append(var)
                    self.phl_friday_gosford_round[round_no].append(var)
                if day == 'Friday' and location == MAITLAND:
                    self.phl_friday_maitland.append(var)
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
                        self.second_club_week_day[(club, week, day)][(time_val, location, team)].append(var)

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
            self.games_per_location[(week, day)][location].append(var)
            if time_val == '19:00' and day != 'Friday':
                self.worst_time_vars.append((key, var))

            # --- ClubDay ---
            if date_str in club_day_dates:
                club_name = club_day_dates[date_str]
                club_team_names = self.club_teams_map.get(club_name, [])
                if t1 in club_team_names or t2 in club_team_names:
                    self.club_day_game_vars[club_name][key] = var

        self._groupings_built = True

    # ================================================================
    # SHARED INDICATOR HELPERS
    # ================================================================

    def _get_or_create_bool(self, cache_key, vars_list, label):
        """Get or create a shared BoolVar indicator via AddMaxEquality."""
        if cache_key in self._shared_indicators:
            return self._shared_indicators[cache_key]
        indicator = self.model.NewBoolVar(label)
        if vars_list:
            self.model.AddMaxEquality(indicator, vars_list)
        else:
            self.model.Add(indicator == 0)
        self._shared_indicators[cache_key] = indicator
        return indicator

    def _get_or_create_presence(self, cache_key, vars_list, label):
        """Get or create a club-presence BoolVar (BoolOr/BoolAnd channeling)."""
        if cache_key in self._shared_indicators:
            return self._shared_indicators[cache_key]
        indicator = self.model.NewBoolVar(label)
        self.model.AddBoolOr(vars_list).OnlyEnforceIf(indicator)
        self.model.AddBoolAnd([v.Not() for v in vars_list]).OnlyEnforceIf(indicator.Not())
        self._shared_indicators[cache_key] = indicator
        return indicator

    # ================================================================
    # PHASE A: HARD INTER-WEEK CONSTRAINTS
    # ================================================================

    def apply_stage_1_hard(self):
        """Stage 1: All hard constraints (feasibility).

        These are binary constraints that MUST be satisfied for a valid draw.
        """
        assert self._groupings_built, "Call build_groupings() first"
        c = 0
        c += self._no_double_booking_teams()
        c += self._no_double_booking_fields()
        c += self._equal_games_balanced_matchups()
        c += self._fifty_fifty_home_away()
        c += self._team_conflict()
        c += self._max_venue_weekends()
        c += self._phl_adjacency_hard()
        c += self._phl_times_hard()
        c += self._matchup_spacing_hard()
        c += self._grade_adjacency_hard()
        c += self._club_alignment_hard()
        c += self._maitland_grouping_hard()
        c += self._away_maitland_hard()
        c += self._club_day_scheduling()
        c += self._best_timeslot_choices()
        c += self._club_day_field_contiguity()
        c += self._club_game_spread_hard()
        self.constraints_added += c
        return c

    # Keep old name as alias for backward compatibility
    apply_phase_a = apply_stage_1_hard

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
        num_dummy = self.data.get('num_dummy_timeslots', 0)
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

    def _max_venue_weekends(self):
        n = 0
        # Calculate max games per home field
        grade_games = {g.name: g.num_games for g in self.grades}
        max_games_per_field = defaultdict(int)
        for club_name, home_field in self.club_home_field.items():
            for team_name in self.club_teams_map[club_name]:
                tg = self.team_grade.get(team_name)
                if tg:
                    max_games_per_field[home_field] = max(
                        max_games_per_field[home_field],
                        grade_games.get(tg, 0)
                    )

        location_indicators = defaultdict(list)
        for (week, location), vars_list in self.by_week_location.items():
            ind = self._get_or_create_bool(
                ('weekend_used', week, location), vars_list,
                f"u_weekend_{week}_{location}")
            location_indicators[location].append(ind)

        for location, indicators in location_indicators.items():
            max_weekends = max_games_per_field.get(location, 10) // 2 + 1
            self.model.Add(sum(indicators) <= max_weekends)
            n += 1
        return n

    def _phl_adjacency_hard(self):
        n = 0
        for club_week_day, phl_slots in self.phl_club_week_day.items():
            club, week, day = club_week_day
            second_slots = self.second_club_week_day.get(club_week_day, {})

            for (phl_time, phl_loc), phl_vars in phl_slots.items():
                if not phl_time:
                    continue
                phl_time_dt = datetime.strptime(phl_time, '%H:%M')
                min_time = (phl_time_dt - timedelta(minutes=self.PHL_ADJACENCY_MINUTES)).time()
                max_time = (phl_time_dt + timedelta(minutes=self.PHL_ADJACENCY_MINUTES)).time()

                for (sec_time, sec_loc, sec_team), sec_vars in second_slots.items():
                    if not sec_time:
                        continue
                    sec_time_t = datetime.strptime(sec_time, '%H:%M').time()
                    if min_time <= sec_time_t <= max_time and sec_loc != phl_loc:
                        self.model.Add(sum(phl_vars) + sum(sec_vars) <= 1)
                        n += 1
                    elif (sec_time_t >= max_time or sec_time_t <= min_time) and sec_loc == phl_loc:
                        self.model.Add(sum(phl_vars) + sum(sec_vars) <= 1)
                        n += 1
        return n

    def _phl_times_hard(self):
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

        # Exactly N Friday night games at Maitland
        maitland_friday_games = defaults.get('maitland_friday_games', 2)
        if self.phl_friday_maitland:
            self.model.Add(sum(self.phl_friday_maitland) == maitland_friday_games)
            n += 1

        # Gosford Friday games in specific rounds
        for round_no, round_vars in self.phl_friday_gosford_round.items():
            if round_no in [2, 4, 5, 9, 10]:
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

        Formula matches ai.py EqualMatchUpSpacingConstraintAI exactly:
            ideal = T - 2
            floor = min(T // 2, T - 2)
            min_gap = max(floor, ideal - base_slack - config_slack)
        """
        n = 0
        grade_num_teams = {g.name: g.num_teams for g in self.grades}
        config_slack = self.slack.get('EqualMatchUpSpacingConstraint', 0)
        defaults = self.data.get('constraint_defaults', {})
        base_slack = defaults.get('spacing_base_slack', 0)

        for (t1, t2, grade), round_map in self.by_grade_pair_round.items():
            T = grade_num_teams.get(grade, 0)
            if T < 2:
                continue
            ideal = T - 2
            floor = min(T // 2, T - 2)
            min_gap = max(floor, ideal - base_slack - config_slack)

            active_rounds = sorted(r for r in round_map if round_map[r])
            for i, r1 in enumerate(active_rounds):
                vars_r1 = round_map[r1]
                for r2 in active_rounds[i + 1:]:
                    gap = r2 - r1
                    if gap >= min_gap:
                        break
                    self.model.Add(sum(vars_r1) + sum(round_map[r2]) <= 1)
                    n += 1
        return n

    def _grade_adjacency_hard(self):
        """Hard: duplicate teams in same grade can't play simultaneously."""
        n = 0
        for (club, slot, grade), vars_ in self.club_dup_games.items():
            if vars_:
                self.model.Add(sum(vars_) <= 1)
                n += 1
        return n

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
                        ind1 = self.model.NewBoolVar(f"u_g1_{grade}_{club_pair}_{round_no}")
                        self.model.AddMaxEquality(ind1, vars_list)
                        ind2 = self.model.NewBoolVar(f"u_g2_{other_grade}_{club_pair}_{round_no}")
                        self.model.AddMaxEquality(ind2, other_rounds[round_no])
                        coincide = self.model.NewBoolVar(f"u_coin_{club_pair}_{round_no}")
                        self.model.Add(coincide <= ind1)
                        self.model.Add(coincide <= ind2)
                        self.model.Add(coincide >= ind1 + ind2 - 1)
                        coincide_vars.append(coincide)

                    if coincide_vars:
                        min_req = max(0, num_games - config_slack)
                        self.model.Add(sum(coincide_vars) >= min_req)
                        n += 1
        return n

    def _maitland_grouping_hard(self):
        """Hard: max N consecutive home weekends (sliding window)."""
        n = 0
        defaults = self.data.get('constraint_defaults', {})
        base_max = defaults.get('maitland_max_consecutive_home', 1)
        slack = self.slack.get('MaitlandHomeGrouping', 0)
        max_consecutive = base_max + slack  # already 1 + slack

        home_indicators = {}
        for week in sorted(self.maitland_all_week.keys()):
            home_vars = self.maitland_home_week.get(week, [])
            ind = self._get_or_create_bool(
                ('maitland_home_ind', week), home_vars,
                f'u_mait_home_{week}')
            home_indicators[week] = ind

        sorted_weeks = sorted(home_indicators.keys())
        window_size = max_consecutive + 1
        for i in range(len(sorted_weeks) - window_size + 1):
            window_indicators = [home_indicators[sorted_weeks[j]] for j in range(i, i + window_size)]
            self.model.Add(sum(window_indicators) <= max_consecutive)
            n += 1
        return n

    def _away_maitland_hard(self):
        """Hard: max 3 away clubs per weekend at Maitland."""
        n = 0
        slack = self.slack.get('AwayAtMaitlandGrouping', 0)
        hard_limit = self.MAITLAND_AWAY_HARD_LIMIT + slack

        for week, club_vars in self.maitland_away_club_week.items():
            club_indicators = []
            for club, vars_list in club_vars.items():
                ind = self._get_or_create_bool(
                    ('maitland_away_club', club, week), vars_list,
                    f'u_away_{club}_{week}')
                club_indicators.append(ind)
            if club_indicators:
                nc = self.model.NewIntVar(0, len(club_indicators), f'u_naway_{week}')
                self.model.Add(nc == sum(club_indicators))
                self.model.Add(nc <= hard_limit)
                n += 1
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

    def _clubs_per_timeslot_hard(self):
        """Hard: dynamic minimum clubs per timeslot at Broadmeadow."""
        n = 0
        slack = self.slack.get('MaximiseClubsPerTimeslotBroadmeadow', 0)
        hard_limit_offset = -slack

        for slot, club_vars in self.bm_slot_club.items():
            presence_vars = []
            all_game_vars = []
            for club, vars_list in club_vars.items():
                ind = self._get_or_create_presence(
                    ('bm_club_presence', club, slot), vars_list,
                    f'u_bm_club_{club}_{slot}')
                presence_vars.append(ind)
                all_game_vars.extend(vars_list)

            if not presence_vars:
                continue

            nc = self.model.NewIntVar(0, len(presence_vars), f'u_bm_nclubs_{slot}')
            self.model.Add(nc == sum(presence_vars))

            tt = self.model.NewIntVar(0, len(all_game_vars), f'u_bm_nteams_{slot}')
            self.model.Add(tt == sum(all_game_vars))

            hmin_base = self.model.NewIntVar(0, len(presence_vars), f'u_bm_hmin_base_{slot}')
            self.model.AddDivisionEquality(hmin_base, tt, 2)

            raw_min = self.model.NewIntVar(-10, len(presence_vars), f'u_bm_raw_min_{slot}')
            self.model.Add(raw_min == hmin_base + hard_limit_offset)
            hard_min = self.model.NewIntVar(0, len(presence_vars), f'u_bm_hmin_{slot}')
            self.model.AddMaxEquality(hard_min, [raw_min, self.model.NewConstant(0)])

            self.model.Add(nc >= hard_min)
            n += 1
        return n

    def _clubs_on_field_hard(self):
        """Hard: max 5 clubs per field per day at Broadmeadow."""
        n = 0
        slack = self.slack.get('MinimiseClubsOnAFieldBroadmeadow', 0)
        hard_limit = self.CLUBS_ON_FIELD_HARD_LIMIT + slack

        for field_key, club_vars in self.bm_field_club.items():
            presence_vars = []
            for club, vars_list in club_vars.items():
                ind = self._get_or_create_presence(
                    ('bm_field_club', club, field_key), vars_list,
                    f'u_fld_club_{club}_{field_key}')
                presence_vars.append(ind)
            if presence_vars:
                nc = self.model.NewIntVar(0, len(presence_vars), f'u_fld_nclubs_{field_key}')
                self.model.Add(nc == sum(presence_vars))
                self.model.Add(nc <= hard_limit)
                n += 1
        return n

    # ================================================================
    # PHASE B: SOFT INTER-WEEK PENALTIES
    # ================================================================

    def apply_stage_2_soft(self):
        """Stage 2: All soft constraints (optimization).

        These create penalty variables that the objective minimizes.
        They further optimize a feasible solution from Stage 1.
        """
        assert self._groupings_built, "Call build_groupings() first"
        c = 0
        c += self._matchup_spacing_soft()
        c += self._grade_adjacency_soft()
        c += self._club_alignment_soft()
        c += self._maitland_grouping_soft()
        c += self._away_maitland_soft()
        c += self._phl_times_soft()
        c += self._preferred_times()
        c += self._club_game_spread_soft()
        self.constraints_added += c
        return c

    # Keep old name as alias for backward compatibility
    apply_phase_b = apply_stage_2_soft

    def _matchup_spacing_soft(self):
        """Sliding window density penalties."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['EqualMatchUpSpacing'] = {'weight': weights.get('EqualMatchUpSpacing', 5000), 'penalties': []}
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

    def _grade_adjacency_soft(self):
        """Soft penalty for adjacent grades playing same timeslot."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['ClubGradeAdjacencyConstraint'] = {'weight': weights.get('ClubGradeAdjacencyConstraint', 50000), 'penalties': []}
        adj_pairs = [(GRADE_ORDER[i], GRADE_ORDER[i + 1]) for i in range(len(GRADE_ORDER) - 1)]
        idx = 0

        for club in [c.name for c in self.clubs]:
            club_slots = {k[0] for k in self.by_slot_club_grade if k[1] == club}
            for slot in club_slots:
                for g1, g2 in adj_pairs:
                    v1 = self.by_slot_club_grade.get((slot, club, g1), [])
                    v2 = self.by_slot_club_grade.get((slot, club, g2), [])
                    if v1 and v2:
                        mx = len(v1) + len(v2)
                        combined = self.model.NewIntVar(0, mx, f'u_adj_comb_{idx}')
                        self.model.Add(combined == sum(v1) + sum(v2))
                        penalty = self.model.NewIntVar(0, mx, f'u_adj_pen_{idx}')
                        self.model.AddMaxEquality(penalty, [combined - 1, self.model.NewConstant(0)])
                        self.data['penalties']['ClubGradeAdjacencyConstraint']['penalties'].append(penalty)
                        idx += 1
                        n += 1
        return n

    def _club_alignment_soft(self):
        """Soft penalties for coincidence deficit and field excess."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['ClubVsClubAlignment'] = {'weight': weights.get('ClubVsClubAlignment', 100000), 'penalties': []}

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
                        # Reuse indicators from Stage 1 if available
                        coincide = self._shared_indicators.get(('coin', grade, other_grade, club_pair, round_no))
                        if coincide is None:
                            # Create if Stage 1 wasn't run (fallback)
                            ind1 = self.model.NewBoolVar(f"u_g1_{grade}_{club_pair}_{round_no}")
                            self.model.AddMaxEquality(ind1, vars_list)
                            ind2 = self.model.NewBoolVar(f"u_g2_{other_grade}_{club_pair}_{round_no}")
                            self.model.AddMaxEquality(ind2, other_rounds[round_no])
                            coincide = self.model.NewBoolVar(f"u_coin_{club_pair}_{round_no}")
                            self.model.Add(coincide <= ind1)
                            self.model.Add(coincide <= ind2)
                            self.model.Add(coincide >= ind1 + ind2 - 1)

                        coincide_vars.append(coincide)

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

    def _maitland_grouping_soft(self):
        """Soft: min(home, away) imbalance penalty per week."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['MaitlandHomeGrouping'] = {'weight': weights.get('MaitlandHomeGrouping', 1000000), 'penalties': []}

        for week in sorted(self.maitland_all_week.keys()):
            all_vars = self.maitland_all_week[week]
            home_vars = self.maitland_home_week.get(week, [])
            if not all_vars:
                continue
            mx = len(all_vars)
            hc = self.model.NewIntVar(0, mx, f'u_mait_hc_{week}')
            if home_vars:
                self.model.Add(hc == sum(home_vars))
            else:
                self.model.Add(hc == 0)
            ac = self.model.NewIntVar(0, mx, f'u_mait_ac_{week}')
            self.model.Add(ac == sum(all_vars) - hc)
            pen = self.model.NewIntVar(0, mx, f'u_mait_pen_{week}')
            self.model.AddMinEquality(pen, [hc, ac])
            self.data['penalties']['MaitlandHomeGrouping']['penalties'].append(pen)
            n += 1
        return n

    def _away_maitland_soft(self):
        """Soft: penalize multiple away clubs per weekend."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['AwayAtMaitlandGrouping'] = {'weight': weights.get('AwayAtMaitlandGrouping', 100000), 'penalties': []}

        for week, club_vars in self.maitland_away_club_week.items():
            club_indicators = []
            for club, vars_list in club_vars.items():
                ind = self._get_or_create_bool(
                    ('maitland_away_club', club, week), vars_list,
                    f'u_away_{club}_{week}')
                club_indicators.append(ind)
            if not club_indicators:
                continue
            nc = self.model.NewIntVar(0, len(club_indicators), f'u_saway_nc_{week}')
            self.model.Add(nc == sum(club_indicators))
            gt1 = self.model.NewBoolVar(f'u_saway_gt1_{week}')
            self.model.Add(nc > 1).OnlyEnforceIf(gt1)
            self.model.Add(nc <= 1).OnlyEnforceIf(gt1.Not())
            pen = self.model.NewIntVar(0, len(club_indicators), f'u_saway_pen_{week}')
            self.model.Add(pen == nc - 1).OnlyEnforceIf(gt1)
            self.model.Add(pen == 0).OnlyEnforceIf(gt1.Not())
            self.data['penalties']['AwayAtMaitlandGrouping']['penalties'].append(pen)
            n += 1
        return n

    def _phl_times_soft(self):
        """Soft: preferred date penalties."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['phl_preferences'] = {'weight': weights.get('phl_preferences', 10000), 'penalties': []}
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
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['PreferredTimesConstraint'] = {'weight': weights.get('PreferredTimesConstraint', 10000000), 'penalties': []}
        noplay = self.data.get('preference_no_play', {})
        if not noplay:
            return 0

        from constraints.original import _normalize_preference_no_play
        allowed_keys = ['team_name', 'team2', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']
        allowed_keys2 = ['team1', 'team_name', 'grade', 'day', 'day_slot', 'time', 'week', 'date', 'field_name', 'field_location']

        normalized = _normalize_preference_no_play(noplay, self.teams, self.clubs)
        for entry_key, club_name, club_teams, constraint in normalized:
            if 'date' not in constraint:
                continue
            if get_nearest_week_by_date(constraint['date'], self.timeslots) in self.locked_weeks:
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

    # ================================================================
    # PHASE C: INTRA-DAY OPTIMIZATION
    # ================================================================

    def apply_phase_c(self):
        """DEPRECATED: Use apply_stage_1_hard() + apply_stage_2_soft() instead.
        Kept for backward compatibility only."""
        return 0

    def _best_timeslot_choices(self):
        """Per-field stacking (cross-field implication) + 7pm penalty.

        Matches ai.py EnsureBestTimeslotChoicesAI:
        - Build per-field-slot indicators: (week, day, location) -> {field: {slot: indicator}}
        - For consecutive slots, if f2 uses next_slot then f must use curr_slot
          (same-field gives contiguity; cross-field gives stacking)
        - 7pm (19:00) non-Friday games get soft penalty
        """
        n = 0
        weights = self.data.get('penalty_weights', {})

        # Build per-field-slot indicators from slot_vars_by_location
        # We need to regroup by (week, day, location, field_name, day_slot)
        field_slot_vars = defaultdict(list)
        for key, var in self.X.items():
            if len(key) < 11 or not key[3]:
                continue
            if key[6] in self.locked_weeks:
                continue
            fs_key = (key[6], key[3], key[10], key[9], key[4])  # week, day, location, field_name, day_slot
            field_slot_vars[fs_key].append(var)

        # Build indicators: (week, day, location) -> {field_name: {day_slot: indicator}}
        loc_fields = defaultdict(lambda: defaultdict(dict))
        for fs_key, vars_list in field_slot_vars.items():
            week, day, location, field_name, day_slot = fs_key
            if len(vars_list) == 1:
                indicator = vars_list[0]
            else:
                indicator = self._get_or_create_bool(
                    ('fs_ind', fs_key), vars_list,
                    f'u_fs_{week}_{field_name}_{day_slot}')
            loc_fields[(week, day, location)][field_name][day_slot] = indicator

        # Stacking constraint: for consecutive slots,
        # if f2 uses next_slot then f must use curr_slot
        for (week, day, location), fields_dict in loc_fields.items():
            field_names = list(fields_dict.keys())
            all_slots = set()
            for field_slots in fields_dict.values():
                all_slots.update(field_slots.keys())
            sorted_slots = sorted(all_slots)

            if len(sorted_slots) < 2:
                continue

            for i in range(len(sorted_slots) - 1):
                curr_slot = sorted_slots[i]
                next_slot = sorted_slots[i + 1]

                for f in field_names:
                    curr_ind = fields_dict[f].get(curr_slot)
                    if curr_ind is None:
                        continue

                    for f2 in field_names:
                        next_ind = fields_dict[f2].get(next_slot)
                        if next_ind is None:
                            continue

                        # If f2 uses next_slot, then f must use curr_slot
                        self.model.AddImplication(next_ind, curr_ind)
                        n += 1

        # 7pm (19:00) worst timeslot penalty
        penalty_key = 'EnsureBestTimeslotChoices_7pm'
        self.data['penalties'][penalty_key] = {
            'weight': weights.get(penalty_key, 100_000),
            'penalties': []
        }
        for key, var in self.worst_time_vars:
            pv = self.model.NewIntVar(0, 1, f'u_7pm_pen_{key[6]}_{key[0]}_{key[1]}')
            self.model.Add(pv == var)
            self.data['penalties'][penalty_key]['penalties'].append(pv)
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
                    fi = self.model.NewBoolVar(f'u_cd_field_{club_name}_{fn}')
                    self.model.AddMaxEquality(fi, vl)
                    fi_list.append(fi)
                self.model.Add(sum(fi_list) == 1)
                n += 1

            # Contiguous timeslots
            slot_vars = defaultdict(list)
            for key, var in game_vars.items():
                slot_vars[key[4]].append(var)  # day_slot
            slot_inds = {}
            for ds, vl in slot_vars.items():
                si = self.model.NewBoolVar(f'u_cd_slot_{club_name}_{ds}')
                self.model.AddMaxEquality(si, vl)
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
        """Hard: limit gaps between a club's games on a given day + field concentration cap.

        For each (club, week, day):
        - gaps = range_size - num_used_slots; must be <= hard_limit
        - field_spread = num_games - max_field; hard cap: 2*field_spread <= num_games - 2 + 2*slack
        """
        n = 0
        config_slack = self.slack.get('ClubGameSpread', 0)
        hard_limit = self.CLUB_GAME_SPREAD_HARD_LIMIT + config_slack

        # Build shared groupings for hard+soft
        self._cgs_groups = defaultdict(dict)
        for (club, week, day, day_slot), vars_list in self.by_club_week_day_slot.items():
            self._cgs_groups[(club, week, day)][day_slot] = vars_list

        self._cgs_field_groups = defaultdict(dict)
        for (club, week, day, field_key), vars_list in self.by_club_week_day_field.items():
            self._cgs_field_groups[(club, week, day)][field_key] = vars_list

        # Store shared vars for soft phase reuse
        self._cgs_shared = {}

        for (club, week, day), slots_dict in self._cgs_groups.items():
            unique_slots = sorted(slots_dict.keys())
            if len(unique_slots) <= 1:
                continue

            min_slot, max_slot = unique_slots[0], unique_slots[-1]
            all_vars_for_day = []
            for s in unique_slots:
                all_vars_for_day.extend(slots_dict[s])

            is_active = {}
            for s in unique_slots:
                ia = self._get_or_create_bool(
                    ('cgs_active', club, week, day, s), slots_dict[s],
                    f'u_cgs_act_{club}_w{week}_{day}_s{s}')
                is_active[s] = ia

            num_games = self.model.NewIntVar(0, len(all_vars_for_day),
                                              f'u_cgs_ng_{club}_w{week}_{day}')
            self.model.Add(num_games == sum(all_vars_for_day))

            num_used = self.model.NewIntVar(0, len(unique_slots), f'u_cgs_used_{club}_w{week}_{day}')
            self.model.Add(num_used == sum(is_active[s] for s in unique_slots))

            min_active = self.model.NewIntVar(min_slot, max_slot, f'u_cgs_min_{club}_w{week}_{day}')
            max_active = self.model.NewIntVar(min_slot, max_slot, f'u_cgs_max_{club}_w{week}_{day}')
            for s in unique_slots:
                self.model.Add(min_active <= s).OnlyEnforceIf(is_active[s])
                self.model.Add(max_active >= s).OnlyEnforceIf(is_active[s])

            range_size = self.model.NewIntVar(1, max_slot - min_slot + 1, f'u_cgs_range_{club}_w{week}_{day}')
            self.model.Add(range_size == max_active - min_active + 1)

            max_gaps = max_slot - min_slot
            gaps = self.model.NewIntVar(0, max_gaps, f'u_cgs_gaps_{club}_w{week}_{day}')
            self.model.Add(gaps == range_size - num_used)

            has_multi = self.model.NewBoolVar(f'u_cgs_multi_{club}_w{week}_{day}')
            self.model.Add(num_games >= 2).OnlyEnforceIf(has_multi)
            self.model.Add(num_games <= 1).OnlyEnforceIf(has_multi.Not())

            # Hard: gap limit
            self.model.Add(gaps <= hard_limit).OnlyEnforceIf(has_multi)
            n += 1

            # Hard: field concentration cap
            field_dict = self._cgs_field_groups.get((club, week, day), {})
            if field_dict:
                field_game_counts = []
                n_all = len(all_vars_for_day)
                for f_key, f_vars in field_dict.items():
                    fcount = self.model.NewIntVar(0, len(f_vars),
                                                   f'u_cgs_fc_{club}_w{week}_{day}_{f_key[0]}')
                    self.model.Add(fcount == sum(f_vars))
                    field_game_counts.append(fcount)

                max_field = self.model.NewIntVar(0, n_all,
                                                  f'u_cgs_maxf_{club}_w{week}_{day}')
                self.model.AddMaxEquality(max_field, field_game_counts)

                field_spread = self.model.NewIntVar(0, n_all,
                                                     f'u_cgs_fspread_{club}_w{week}_{day}')
                self.model.Add(field_spread == num_games - max_field)

                self.model.Add(2 * field_spread <= num_games - 2 + 2 * config_slack).OnlyEnforceIf(has_multi)
                n += 1

                # Store for soft phase
                self._cgs_shared[(club, week, day)] = {
                    'gaps': gaps, 'has_multi': has_multi,
                    'field_spread': field_spread, 'max_gaps': max_gaps,
                    'n_all': n_all,
                }
            else:
                self._cgs_shared[(club, week, day)] = {
                    'gaps': gaps, 'has_multi': has_multi,
                    'max_gaps': max_gaps,
                }

        return n

    def _club_game_spread_soft(self):
        """Soft: penalize gaps and field spread."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['ClubGameSpread'] = {
            'weight': weights.get('ClubGameSpread', 5000), 'penalties': []
        }
        self.data['penalties']['ClubFieldConcentration'] = {
            'weight': weights.get('ClubFieldConcentration', 5000), 'penalties': []
        }

        for (club, week, day), shared in self._cgs_shared.items():
            gaps = shared['gaps']
            has_multi = shared['has_multi']
            max_gaps = shared['max_gaps']

            pen = self.model.NewIntVar(0, max_gaps, f'u_cgs_pen_{club}_w{week}_{day}')
            self.model.Add(pen >= gaps).OnlyEnforceIf(has_multi)
            self.model.Add(pen == 0).OnlyEnforceIf(has_multi.Not())
            self.data['penalties']['ClubGameSpread']['penalties'].append(pen)
            n += 1

            if 'field_spread' in shared:
                field_spread = shared['field_spread']
                n_all = shared['n_all']
                f_penalty = self.model.NewIntVar(0, n_all,
                                                  f'u_cgs_fpen_{club}_w{week}_{day}')
                self.model.Add(f_penalty >= field_spread).OnlyEnforceIf(has_multi)
                self.model.Add(f_penalty == 0).OnlyEnforceIf(has_multi.Not())
                self.data['penalties']['ClubFieldConcentration']['penalties'].append(f_penalty)
                n += 1

        return n

    # ================================================================
    # ARCHIVED: Removed constraints (superseded by ClubGameSpread)
    # MaximiseClubsPerTimeslotBroadmeadow and MinimiseClubsOnAFieldBroadmeadow
    # are redundant with ClubGameSpread's field concentration + contiguity.
    # ================================================================

    def _clubs_per_timeslot_soft(self):
        """Soft: diversity penalty (total_teams - num_clubs)."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['MaximiseClubsPerTimeslotBroadmeadow'] = {'weight': weights.get('MaximiseClubsPerTimeslotBroadmeadow', 5000), 'penalties': []}

        for slot, club_vars in self.bm_slot_club.items():
            presence_vars = []
            all_game_vars = []
            for club, vars_list in club_vars.items():
                ind = self._get_or_create_presence(
                    ('bm_club_presence', club, slot), vars_list,
                    f'u_bm_club_{club}_{slot}')
                presence_vars.append(ind)
                all_game_vars.extend(vars_list)
            if not presence_vars:
                continue

            nc = self.model.NewIntVar(0, len(presence_vars), f'u_sbm_nc_{slot}')
            self.model.Add(nc == sum(presence_vars))
            tt = self.model.NewIntVar(0, len(all_game_vars), f'u_sbm_tt_{slot}')
            self.model.Add(tt == sum(all_game_vars))
            su = self.model.NewBoolVar(f'u_sbm_su_{slot}')
            self.model.Add(tt >= 1).OnlyEnforceIf(su)
            self.model.Add(tt == 0).OnlyEnforceIf(su.Not())

            pen = self.model.NewIntVar(0, len(all_game_vars), f'u_sbm_pen_{slot}')
            self.model.Add(pen >= tt - nc).OnlyEnforceIf(su)
            self.model.Add(pen == 0).OnlyEnforceIf(su.Not())
            self.data['penalties']['MaximiseClubsPerTimeslotBroadmeadow']['penalties'].append(pen)
            n += 1
        return n

    def _clubs_on_field_soft(self):
        """Soft: |num_clubs - 2| penalty per field per day."""
        n = 0
        weights = self.data.get('penalty_weights', {})
        self.data['penalties']['MinimiseClubsOnAFieldBroadmeadow'] = {'weight': weights.get('MinimiseClubsOnAFieldBroadmeadow', 5000), 'penalties': []}

        for field_key, club_vars in self.bm_field_club.items():
            presence_vars = []
            for club, vars_list in club_vars.items():
                ind = self._get_or_create_presence(
                    ('bm_field_club', club, field_key), vars_list,
                    f'u_fld_club_{club}_{field_key}')
                presence_vars.append(ind)
            if not presence_vars:
                continue
            nc = self.model.NewIntVar(0, len(presence_vars), f'u_sfld_nc_{field_key}')
            self.model.Add(nc == sum(presence_vars))
            pen = self.model.NewIntVar(0, len(presence_vars), f'u_sfld_pen_{field_key}')
            self.model.AddAbsEquality(pen, nc - 2)
            self.data['penalties']['MinimiseClubsOnAFieldBroadmeadow']['penalties'].append(pen)
            n += 1
        return n

    # ================================================================
    # CONVENIENCE: APPLY ALL PHASES
    # ================================================================

    def apply_all(self):
        """Apply all constraints: Stage 1 (hard) + Stage 2 (soft)."""
        self.build_groupings()
        h = self.apply_stage_1_hard()
        s = self.apply_stage_2_soft()
        return h + s
