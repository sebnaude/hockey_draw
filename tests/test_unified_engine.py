"""
Tests for constraints/unified.py - UnifiedConstraintEngine.

Uses real CP-SAT models, real config data, and real constraint application.
No mocks, no patches, no MagicMock.
"""

import os
import sys
import pytest
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraints.unified import UnifiedConstraintEngine, SharedVariablePool, BROADMEADOW, MAITLAND, GOSFORD
from models import PlayingField, Team, Club, Grade, Timeslot
from tests.conftest import create_model_and_vars, solve_with_timeout


# ============== Fixtures ==============

@pytest.fixture
def mini_unified_data():
    """
    Build a small but realistic data dict with 4 clubs (incl. Maitland),
    2 grades, 4 weeks, and Broadmeadow + Maitland fields.
    Designed to exercise most grouping paths in the engine.
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    mf = PlayingField(location=MAITLAND, name='Maitland Main Field')
    fields = [ef, wf, mf]

    clubs = [
        Club(name='Tigers', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Maitland', home_field=MAITLAND),
    ]

    teams = []
    for club in clubs:
        for grade in ['3rd', '4th']:
            teams.append(Team(name=f'{club.name} 3rd' if grade == '3rd' else f'{club.name} 4th',
                              club=club, grade=grade))

    grades = [
        Grade(name='3rd', teams=[t.name for t in teams if t.grade == '3rd']),
        Grade(name='4th', teams=[t.name for t in teams if t.grade == '4th']),
    ]

    # Build timeslots: 4 weeks, Sunday only, 2 fields at Broadmeadow + 1 Maitland
    timeslots = []
    base_date = datetime(2025, 3, 23)
    for week in range(1, 5):
        week_date = base_date + timedelta(weeks=week - 1)
        date_str = week_date.strftime('%Y-%m-%d')
        for field in [ef, wf]:
            for slot, time in enumerate(['10:00', '11:30', '13:00'], 1):
                timeslots.append(Timeslot(
                    date=date_str, day='Sunday', time=time, week=week,
                    day_slot=slot, field=field, round_no=week,
                ))
        # Maitland field - 2 slots per week
        for slot, time in enumerate(['10:00', '11:30'], 1):
            timeslots.append(Timeslot(
                date=date_str, day='Sunday', time=time, week=week,
                day_slot=slot, field=mf, round_no=week,
            ))

    # Generate round-robin games
    games = []
    for grade_obj in grades:
        for t1, t2 in combinations(grade_obj.teams, 2):
            games.append((t1, t2, grade_obj.name))

    data = {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grades,
        'clubs': clubs,
        'fields': fields,
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {'3rd': 4, '4th': 4, 'max': 4},
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND},
        'constraint_defaults': {},
    }
    return data


@pytest.fixture
def mini_engine(mini_unified_data):
    """Create a UnifiedConstraintEngine with mini data and real model/vars."""
    model, X = create_model_and_vars(
        mini_unified_data['games'],
        mini_unified_data['timeslots'],
    )
    engine = UnifiedConstraintEngine(model, X, mini_unified_data)
    return engine




# ============== TestUnifiedEngineInit ==============

class TestUnifiedEngineInit:

    def test_instantiation_with_mini_data(self, mini_engine):
        """Engine initializes with correct attributes from mini data."""
        engine = mini_engine
        assert engine.model is not None
        assert len(engine.X) > 0
        assert engine.constraints_added == 0
        assert engine._groupings_built is False

    def test_team_club_lookup_populated(self, mini_engine):
        """team_club dict maps every team to its club."""
        engine = mini_engine
        assert len(engine.team_club) == len(engine.data['teams'])
        assert engine.team_club['Tigers 3rd'] == 'Tigers'
        assert engine.team_club['Maitland 4th'] == 'Maitland'

    def test_club_teams_map_populated(self, mini_engine):
        """club_teams_map lists all teams per club."""
        engine = mini_engine
        assert 'Tigers' in engine.club_teams_map
        tiger_teams = engine.club_teams_map['Tigers']
        assert 'Tigers 3rd' in tiger_teams
        assert 'Tigers 4th' in tiger_teams

    def test_team_grade_populated(self, mini_engine):
        """team_grade maps each team to its grade string."""
        engine = mini_engine
        assert engine.team_grade['Tigers 3rd'] == '3rd'
        assert engine.team_grade['Wests 4th'] == '4th'

    def test_club_home_field_populated(self, mini_engine):
        """club_home_field maps clubs to their home venue."""
        engine = mini_engine
        assert engine.club_home_field['Tigers'] == BROADMEADOW
        assert engine.club_home_field['Maitland'] == MAITLAND

    def test_locked_weeks_default_empty(self, mini_engine):
        """With no locked_weeks in data, engine has empty set."""
        assert mini_engine.locked_weeks == set()

    def test_penalties_dict_created(self, mini_engine):
        """Engine ensures penalties dict exists in data."""
        assert 'penalties' in mini_engine.data

    def test_get_club_lookup(self, mini_engine):
        """_get_club returns correct club for known teams."""
        assert mini_engine._get_club('Norths 3rd') == 'Norths'
        assert mini_engine._get_club('nonexistent') is None

    def test_skip_constraints_default_empty(self, mini_engine):
        """Default skip_constraints is empty set."""
        assert mini_engine.skip_constraints == set()

    def test_skip_constraints_parameter(self, mini_unified_data):
        """skip_constraints parameter is stored correctly."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data,
                                         skip_constraints={'NoDoubleBookingTeams'})
        assert 'NoDoubleBookingTeams' in engine.skip_constraints


# ============== TestBuildGroupings ==============

class TestBuildGroupings:

    def test_groupings_built_flag(self, mini_engine):
        """build_groupings sets _groupings_built to True."""
        mini_engine.build_groupings()
        assert mini_engine._groupings_built is True

    def test_idempotent(self, mini_engine):
        """Calling build_groupings twice does not re-process."""
        mini_engine.build_groupings()
        first_count = len(mini_engine.by_week_team)
        mini_engine.build_groupings()  # should early-return
        assert len(mini_engine.by_week_team) == first_count

    def test_by_week_team_populated(self, mini_engine):
        """by_week_team has entries for every (week, team) combo."""
        mini_engine.build_groupings()
        assert len(mini_engine.by_week_team) > 0
        # Tigers 3rd should appear in week 1
        assert any(k[1] == 'Tigers 3rd' for k in mini_engine.by_week_team)

    def test_by_slot_field_populated(self, mini_engine):
        """by_slot_field has entries for field/timeslot combos."""
        mini_engine.build_groupings()
        assert len(mini_engine.by_slot_field) > 0

    def test_grade_team_vars_populated(self, mini_engine):
        """grade_team_vars has entries for both grades."""
        mini_engine.build_groupings()
        assert '3rd' in mini_engine.grade_team_vars
        assert '4th' in mini_engine.grade_team_vars
        assert 'Tigers 3rd' in mini_engine.grade_team_vars['3rd']

    def test_grade_pair_vars_populated(self, mini_engine):
        """grade_pair_vars has matchup pair entries."""
        mini_engine.build_groupings()
        assert len(mini_engine.grade_pair_vars['3rd']) > 0

    # spec-018: test_maitland_groupings_populated removed — the
    # `maitland_all_week` / `non_default_*` grouping maps were deleted along
    # with the venue-sequencing rules that consumed them.

    def test_broadmeadow_slot_club_populated(self, mini_engine):
        """bm_slot_club has entries for Broadmeadow timeslots."""
        mini_engine.build_groupings()
        assert len(mini_engine.bm_slot_club) > 0

    def test_home_away_venue_populated(self, mini_engine):
        """home_away_venue has entries for Maitland team pairs."""
        mini_engine.build_groupings()
        # There should be home/away entries for Maitland vs other teams
        assert len(mini_engine.home_away_venue) > 0

    def test_locked_weeks_skipped(self, mini_unified_data):
        """Variables in locked weeks are excluded from constraint groupings."""
        mini_unified_data['locked_weeks'] = {1, 2}
        model, X = create_model_and_vars(
            mini_unified_data['games'],
            mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.build_groupings()

        # by_week_team should NOT have week 1 or 2
        for (week, team) in engine.by_week_team:
            assert week not in {1, 2}, f"Locked week {week} found in by_week_team"

        # But grade_team_vars SHOULD still include locked weeks (for equal games)
        assert len(engine.grade_team_vars['3rd']) > 0

    def test_club_alignment_groupings_for_lower_grades(self, mini_engine):
        """by_grade_clubpair_round populated for 3rd/4th (non-PHL/2nd)."""
        mini_engine.build_groupings()
        # Both 3rd and 4th are lower grades, should have entries
        assert len(mini_engine.by_grade_clubpair_round) > 0

    def test_slot_vars_by_location_populated(self, mini_engine):
        """slot_vars_by_location has entries for timeslot choices."""
        mini_engine.build_groupings()
        assert len(mini_engine.slot_vars_by_location) > 0


# ============== TestStage1Hard ==============

class TestStage1Hard:

    def test_stage_1_returns_positive_count(self, mini_engine):
        """Stage 1 adds constraints and returns count > 0."""
        mini_engine.build_groupings()
        count = mini_engine.apply_stage_1_hard()
        assert count > 0

    def test_stage_1_requires_groupings(self, mini_engine):
        """Stage 1 asserts groupings are built."""
        with pytest.raises(AssertionError, match="build_groupings"):
            mini_engine.apply_stage_1_hard()

    def test_stage_1_model_not_invalid(self, mini_engine):
        """Model with stage 1 constraints is not MODEL_INVALID."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        status, solver = solve_with_timeout(mini_engine.model, timeout_seconds=3.0)
        assert status != cp_model.MODEL_INVALID, "Stage 1 produced an invalid model"

    def test_stage_1_updates_constraints_added(self, mini_engine):
        """constraints_added field is updated by stage 1."""
        mini_engine.build_groupings()
        count = mini_engine.apply_stage_1_hard()
        assert mini_engine.constraints_added == count
        assert mini_engine.constraints_added > 0

    def test_phase_a_alias_works(self, mini_engine):
        """apply_phase_a() is an alias for apply_stage_1_hard()."""
        mini_engine.build_groupings()
        count = mini_engine.apply_phase_a()
        assert count > 0

    def test_no_double_booking_teams(self, mini_unified_data):
        """No team plays more than once per week after stage 1."""
        model, X = create_model_and_vars(
            mini_unified_data['games'],
            mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.build_groupings()
        engine.apply_stage_1_hard()

        # Add objective to schedule as many games as possible
        real_vars = [v for k, v in X.items() if len(k) >= 11]
        model.Maximize(sum(real_vars))

        status, solver = solve_with_timeout(model, timeout_seconds=3.0)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            # Check no team plays twice in same week
            team_week_count = defaultdict(int)
            for key, var in X.items():
                if len(key) >= 11 and solver.Value(var) == 1:
                    t1, t2, grade = key[0], key[1], key[2]
                    week = key[6]
                    team_week_count[(t1, week)] += 1
                    team_week_count[(t2, week)] += 1
            for (team, week), cnt in team_week_count.items():
                assert cnt <= 1, f"{team} plays {cnt} games in week {week}"

    def test_no_double_booking_fields(self, mini_unified_data):
        """No field has more than one game per timeslot after stage 1."""
        model, X = create_model_and_vars(
            mini_unified_data['games'],
            mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.build_groupings()
        engine.apply_stage_1_hard()

        real_vars = [v for k, v in X.items() if len(k) >= 11]
        model.Maximize(sum(real_vars))

        status, solver = solve_with_timeout(model, timeout_seconds=3.0)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            field_slot_count = defaultdict(int)
            for key, var in X.items():
                if len(key) >= 11 and solver.Value(var) == 1:
                    slot_key = (key[6], key[3], key[4], key[9])  # week, day, day_slot, field
                    field_slot_count[slot_key] += 1
            for slot_key, cnt in field_slot_count.items():
                assert cnt <= 1, f"Field slot {slot_key} has {cnt} games"


# ============== TestStage2Soft ==============

class TestStage2Soft:

    def test_stage_2_returns_count(self, mini_engine):
        """Stage 2 adds penalty constraints."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        count = mini_engine.apply_stage_2_soft()
        assert count >= 0

    def test_stage_2_requires_groupings(self, mini_engine):
        """Stage 2 asserts groupings are built."""
        with pytest.raises(AssertionError, match="build_groupings"):
            mini_engine.apply_stage_2_soft()

    def test_penalty_dicts_created(self, mini_engine):
        """Stage 2 creates penalty entries in data['penalties']."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        mini_engine.apply_stage_2_soft()

        penalties = mini_engine.data['penalties']
        expected_keys = [
            'EqualMatchUpSpacing',
            # spec-007: 'ClubGradeAdjacencyConstraint' bucket removed (the
            # legacy adjacent-grade soft was deleted; the surviving hard
            # `SameGradeSameClubNoConcurrency` atom has no penalty bucket).
            # spec-018: 'MaitlandHomeGrouping' / 'AwayAtMaitlandGrouping'
            # penalty buckets removed (venue-sequencing soft penalties deleted).
            'ClubGameSpread',
        ]
        for key in expected_keys:
            assert key in penalties, f"Missing penalty key: {key}"

    def test_penalty_structures_have_weight(self, mini_engine):
        """Each penalty entry has a 'weight' and 'penalties' list."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        mini_engine.apply_stage_2_soft()

        for key, penalty_info in mini_engine.data['penalties'].items():
            assert 'weight' in penalty_info, f"Penalty {key} missing 'weight'"
            assert 'penalties' in penalty_info, f"Penalty {key} missing 'penalties'"
            assert isinstance(penalty_info['penalties'], list)

    def test_model_still_valid_after_stage_2(self, mini_engine):
        """Model is not invalid after stage 1 + stage 2."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        mini_engine.apply_stage_2_soft()

        status, solver = solve_with_timeout(mini_engine.model, timeout_seconds=3.0)
        assert status != cp_model.MODEL_INVALID

    def test_phase_b_alias_works(self, mini_engine):
        """apply_phase_b() is an alias for apply_stage_2_soft()."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        count = mini_engine.apply_phase_b()
        assert count >= 0

    def test_total_constraints_higher_than_stage_1(self, mini_unified_data):
        """Both stages together produce more constraints than stage 1 alone."""
        # Stage 1 only
        model_a, X_a = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data_a = dict(mini_unified_data)
        data_a['penalties'] = {}
        engine_a = UnifiedConstraintEngine(model_a, X_a, data_a)
        engine_a.build_groupings()
        count_a = engine_a.apply_stage_1_hard()

        # All stages
        model_all, X_all = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data_all = dict(mini_unified_data)
        data_all['penalties'] = {}
        engine_all = UnifiedConstraintEngine(model_all, X_all, data_all)
        engine_all.build_groupings()
        total = engine_all.apply_stage_1_hard() + engine_all.apply_stage_2_soft()

        assert total >= count_a

    def test_model_valid_after_both_stages(self, mini_engine):
        """Model is not invalid after both stages."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        mini_engine.apply_stage_2_soft()

        status, solver = solve_with_timeout(mini_engine.model, timeout_seconds=3.0)
        assert status != cp_model.MODEL_INVALID

    def test_spread_penalties_created(self, mini_engine):
        """Stage 2 creates ClubGameSpread penalty entries."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        mini_engine.apply_stage_2_soft()

        penalties = mini_engine.data['penalties']
        assert 'ClubGameSpread' in penalties


# ============== TestApplyAll ==============

class TestApplyAll:

    def test_apply_all_returns_total(self, mini_unified_data):
        """apply_all returns total constraint count across all stages."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        total = engine.apply_all()
        assert total > 0
        assert engine.constraints_added == total

    def test_apply_all_builds_groupings(self, mini_unified_data):
        """apply_all calls build_groupings internally."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        assert engine._groupings_built is False
        engine.apply_all()
        assert engine._groupings_built is True

    def test_apply_all_feasible(self, mini_unified_data):
        """Model is feasible after apply_all with short timeout."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.apply_all()

        # Add objective
        real_vars = [v for k, v in X.items() if len(k) >= 11]
        model.Maximize(sum(real_vars))

        status, solver = solve_with_timeout(model, timeout_seconds=5.0)
        assert status in (cp_model.FEASIBLE, cp_model.OPTIMAL), \
            f"Expected FEASIBLE/OPTIMAL, got {solver.status_name(status)}"


# ============== TestSharedIndicators ==============

class TestSharedIndicators:

    def test_shared_indicators_populated_after_stage_1(self, mini_engine):
        """Stage 1 creates shared indicator variables."""
        mini_engine.build_groupings()
        mini_engine.apply_stage_1_hard()
        assert len(mini_engine.pool._cache) > 0

    def test_indicators_reused_across_stages(self, mini_unified_data):
        """Indicators created in stage 1 are reused by stage 2 (not recreated)."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.build_groupings()
        engine.apply_stage_1_hard()

        count_after_1 = len(engine.pool._cache)
        assert count_after_1 > 0

        engine.apply_stage_2_soft()
        count_after_2 = len(engine.pool._cache)

        # Stage 2 may add some new indicators but should also reuse existing ones.
        assert count_after_2 >= count_after_1

    def test_get_or_create_bool_caching(self, mini_engine):
        """pool.get_or_create_bool returns same var for same cache_key."""
        mini_engine.build_groupings()
        # Create a test indicator
        test_var = mini_engine.model.NewBoolVar('test_var')
        ind1 = mini_engine.pool.get_or_create_bool('test_key', [test_var], 'label1')
        ind2 = mini_engine.pool.get_or_create_bool('test_key', [test_var], 'label2')
        assert ind1 is ind2

    def test_get_or_create_bool_empty_list(self, mini_engine):
        """pool.get_or_create_bool with empty list creates indicator == 0."""
        mini_engine.build_groupings()
        ind = mini_engine.pool.get_or_create_bool('empty_test', [], 'empty_label')
        assert ind is not None
        # Solve to verify it's forced to 0
        mini_engine.model.Maximize(ind)
        status, solver = solve_with_timeout(mini_engine.model, timeout_seconds=1.0)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            assert solver.Value(ind) == 0

    def test_get_or_create_presence_caching(self, mini_engine):
        """pool.get_or_create_presence returns same var for same cache_key."""
        mini_engine.build_groupings()
        test_var = mini_engine.model.NewBoolVar('pres_test')
        ind1 = mini_engine.pool.get_or_create_presence('pres_key', [test_var], 'label1')
        ind2 = mini_engine.pool.get_or_create_presence('pres_key', [test_var], 'label2')
        assert ind1 is ind2

    def test_coincidence_indicators_stored(self, mini_unified_data):
        """Coincidence indicators from _club_alignment_hard are stored in pool._cache."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.build_groupings()
        engine._club_alignment_hard()
        # Check that at least one coincidence indicator was stored
        coin_keys = [k for k in engine.pool._cache if isinstance(k, tuple) and k[0] == 'coin']
        assert len(coin_keys) > 0, "No coincidence indicators stored in _shared_indicators"


# ============== TestConstraintCoverage ==============

class TestConstraintCoverage:

    def test_penalty_keys_after_apply_all(self, mini_unified_data):
        """apply_all populates penalty entries for all soft constraint types."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.apply_all()

        penalties = engine.data['penalties']
        expected_penalty_keys = [
            'EqualMatchUpSpacing',
            # spec-007: 'ClubGradeAdjacencyConstraint' bucket removed.
            # spec-018: 'MaitlandHomeGrouping' / 'AwayAtMaitlandGrouping' buckets
            # removed (venue-sequencing soft penalties deleted).
            'ClubGameSpread',
            'ClubVsClubAlignment',
            'ClubVsClubAlignmentField',
        ]
        for key in expected_penalty_keys:
            assert key in penalties, f"Missing penalty key after apply_all: {key}"

    def test_stage_1_covers_hard_constraints(self, mini_unified_data):
        """Stage 1 adds constraints for core hard constraint types."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.build_groupings()

        counts = {
            'no_double_booking_teams': engine._no_double_booking_teams(),
            'no_double_booking_fields': engine._no_double_booking_fields(),
            'equal_games': engine._equal_games_balanced_matchups(),
            'fifty_fifty': engine._fifty_fifty_home_away(),
            # spec-018: maitland_grouping / away_maitland hard methods deleted.
        }

        # These should definitely have constraints with our mini data
        assert counts['no_double_booking_teams'] > 0
        assert counts['no_double_booking_fields'] > 0
        assert counts['equal_games'] > 0

    def test_constraint_methods_exist(self, mini_unified_data):
        """
        The unified engine has all expected constraint methods.
        """
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)

        # Stage 1 methods (hard constraints)
        # spec-007: `_grade_adjacency_hard` and `_grade_adjacency_soft` were
        # removed from the engine when ClubGradeAdjacency was split. The
        # surviving hard rule lives in the `SameGradeSameClubNoConcurrency`
        # atom (dispatched outside the engine).
        stage_1_methods = [
            '_no_double_booking_teams',
            '_no_double_booking_fields',
            '_equal_games_balanced_matchups',
            '_fifty_fifty_home_away',
            '_team_conflict',
            # spec-018: `_max_venue_weekends` (MaxMaitlandHomeWeekends),
            # `_maitland_grouping_hard` and `_away_maitland_hard` removed —
            # venue-sequencing rules deleted.
            # spec-014: `_phl_adjacency_hard` removed — PHL/2nd adjacency is now
            # the `PHLAnd2ndAdjacency` atom (dispatched outside the engine).
            '_phl_times_hard',
            '_matchup_spacing_hard',
            '_club_alignment_hard',
            '_club_day_scheduling',
            '_club_game_spread_hard',
            '_best_timeslot_choices_hard',
        ]

        # Stage 2 methods (soft penalties + optimization)
        stage_2_methods = [
            '_matchup_spacing_soft',
            '_club_alignment_soft',
            # spec-018: `_maitland_grouping_soft` / `_away_maitland_soft`
            # removed — venue-sequencing soft penalties deleted.
            # spec-020: `_phl_times_soft` removed — PreferredDates deleted.
            '_preferred_times',
            '_best_timeslot_choices_soft',
            '_club_game_spread_soft',
        ]

        # Methods that moved to stage 1 (still exist, just not in stage 2)
        stage_1_extras = ['_club_day_field_contiguity']

        all_methods = stage_1_methods + stage_2_methods + stage_1_extras
        for method_name in all_methods:
            assert hasattr(engine, method_name), f"Missing method: {method_name}"
            assert callable(getattr(engine, method_name)), f"Not callable: {method_name}"


# ============== TestSlackHandling ==============

class TestSlackHandling:

    def test_slack_affects_matchup_spacing(self, mini_unified_data):
        """Higher slack should produce fewer matchup spacing constraints."""
        # No slack
        model1, X1 = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data1 = dict(mini_unified_data)
        data1['constraint_slack'] = {}
        data1['penalties'] = {}
        engine1 = UnifiedConstraintEngine(model1, X1, data1)
        engine1.build_groupings()
        count1 = engine1._matchup_spacing_hard()

        # High slack
        model2, X2 = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data2 = dict(mini_unified_data)
        data2['constraint_slack'] = {'EqualMatchUpSpacingConstraint': 10}
        data2['penalties'] = {}
        engine2 = UnifiedConstraintEngine(model2, X2, data2)
        engine2.build_groupings()
        count2 = engine2._matchup_spacing_hard()

        # More slack should mean fewer or equal hard constraints
        assert count2 <= count1

    # spec-018: test_slack_affects_maitland_grouping /
    # test_slack_affects_away_maitland removed — the MaitlandHomeGrouping /
    # AwayAtMaitlandGrouping rules and their slack keys were deleted.


# ============== TestEquivalenceWithAI ==============

class TestEquivalenceWithAI:

    def test_both_engines_feasible(self, mini_unified_data):
        """Unified engine produces feasible model."""
        model_u, X_u = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data_u = dict(mini_unified_data)
        data_u['penalties'] = {}
        engine = UnifiedConstraintEngine(model_u, X_u, data_u)
        total_u = engine.apply_all()

        real_vars_u = [v for k, v in X_u.items() if len(k) >= 11]
        model_u.Maximize(sum(real_vars_u))

        status_u, solver_u = solve_with_timeout(model_u, timeout_seconds=5.0)

        assert status_u != cp_model.MODEL_INVALID, \
            f"Unified model invalid: {solver_u.status_name(status_u)}"

    def test_unified_constraint_count_reasonable(self, mini_unified_data):
        """Unified engine produces a reasonable number of constraints (not zero, not absurd)."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        total = engine.apply_all()

        assert total >= 20, f"Too few constraints: {total}"
        assert total < 100000, f"Suspiciously many constraints: {total}"


# ============== TestClubDayConstraint ==============

class TestClubDayConstraint:

    def test_club_day_without_config(self, mini_unified_data):
        """No club days configured: _club_day_scheduling returns 0."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.build_groupings()
        count = engine._club_day_scheduling()
        assert count == 0

    def test_club_day_with_config(self, mini_unified_data):
        """With club day configured, constraints are added."""
        mini_unified_data['club_days'] = {
            'Tigers': datetime(2025, 3, 23),  # Week 1
        }
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine = UnifiedConstraintEngine(model, X, mini_unified_data)
        engine.build_groupings()
        count = engine._club_day_scheduling()
        assert count > 0


# ============== TestVenueConstants ==============

class TestVenueConstants:

    def test_venue_constants_correct(self):
        """Verify venue constant strings match expected values."""
        assert BROADMEADOW == 'Newcastle International Hockey Centre'
        assert MAITLAND == 'Maitland Park'
        assert GOSFORD == 'Central Coast Hockey Park'

    def test_grade_order(self):
        """GRADE_ORDER matches expected sequence."""
        from constraints.unified import GRADE_ORDER
        assert GRADE_ORDER == ["PHL", "2nd", "3rd", "4th", "5th", "6th"]


# ============== TestSkipConstraints ==============

class TestSkipConstraints:

    def test_skip_double_booking_teams(self, mini_unified_data):
        """skip_constraints={'NoDoubleBookingTeams'} skips that constraint."""
        # With constraint: team can only play once per week
        model1, X1 = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        engine1 = UnifiedConstraintEngine(model1, X1, dict(mini_unified_data))
        engine1.build_groupings()
        count1 = engine1.apply_stage_1_hard()

        # Without NoDoubleBookingTeams
        model2, X2 = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data2 = dict(mini_unified_data)
        data2['penalties'] = {}
        engine2 = UnifiedConstraintEngine(model2, X2, data2,
                                          skip_constraints={'NoDoubleBookingTeams'})
        engine2.build_groupings()
        count2 = engine2.apply_stage_1_hard()

        # Skipping should produce fewer constraints
        assert count2 < count1

    def test_skip_does_not_affect_other_constraints(self, mini_unified_data):
        """Skipping one constraint does not affect others."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data,
                                         skip_constraints={'NoDoubleBookingTeams'})
        engine.build_groupings()
        engine.apply_stage_1_hard()

        # NoDoubleBookingFields should still work (team can have double bookings
        # but fields should still be single-booked)
        real_vars = [v for k, v in X.items() if len(k) >= 11]
        model.Maximize(sum(real_vars))

        status, solver = solve_with_timeout(model, timeout_seconds=3.0)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            field_slot_count = defaultdict(int)
            for key, var in X.items():
                if len(key) >= 11 and solver.Value(var) == 1:
                    slot_key = (key[6], key[3], key[4], key[9])
                    field_slot_count[slot_key] += 1
            for slot_key, cnt in field_slot_count.items():
                assert cnt <= 1, f"Field slot {slot_key} has {cnt} games"

    def test_skip_soft_constraint(self, mini_unified_data):
        """Skipping a soft constraint removes its penalty key."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data,
                                         skip_constraints={'ClubGameSpread'})
        engine.build_groupings()
        engine.apply_stage_1_hard()
        engine.apply_stage_2_soft()

        # ClubGameSpread should NOT be in penalties since it was skipped
        assert 'ClubGameSpread' not in engine.data['penalties']


# ============== TestHardConstraintEnforcement ==============

class TestHardConstraintEnforcement:
    """Tests that verify hard constraints actually PREVENT violations."""

    def test_club_game_spread_hard_prevents_double_ups(self, mini_unified_data):
        """ClubGameSpread hard with max_overlap=0 prevents double-ups."""
        mini_unified_data['constraint_defaults'] = {
            'club_game_spread_max_gap': 10,    # permissive gap
            'club_game_spread_max_overlap': 0,  # no double-ups
        }

        # Without spread constraint: can have two club games at same slot
        model1, X1 = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data1 = dict(mini_unified_data)
        data1['penalties'] = {}
        engine1 = UnifiedConstraintEngine(model1, X1, data1,
                                          skip_constraints={'ClubGameSpread'})
        engine1.build_groupings()
        engine1.apply_stage_1_hard()

        real_vars = [v for k, v in X1.items() if len(k) >= 11]
        model1.Maximize(sum(real_vars))

        status1, solver1 = solve_with_timeout(model1, timeout_seconds=5.0)

        # With spread constraint
        model2, X2 = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data2 = dict(mini_unified_data)
        data2['penalties'] = {}
        engine2 = UnifiedConstraintEngine(model2, X2, data2)
        engine2.build_groupings()
        engine2.apply_stage_1_hard()

        real_vars2 = [v for k, v in X2.items() if len(k) >= 11]
        model2.Maximize(sum(real_vars2))

        status2, solver2 = solve_with_timeout(model2, timeout_seconds=5.0)

        # Both should be feasible
        assert status1 in (cp_model.FEASIBLE, cp_model.OPTIMAL)
        assert status2 in (cp_model.FEASIBLE, cp_model.OPTIMAL)

        # The constraint should either reduce total games or maintain feasibility
        # Verify no negative gap (no double-ups) in the constrained solution
        if status2 in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            club_week_day_slot = defaultdict(int)
            for key, var in X2.items():
                if len(key) >= 11 and solver2.Value(var) == 1:
                    t1, t2 = key[0], key[1]
                    week, day, day_slot = key[6], key[3], key[4]
                    for team in [t1, t2]:
                        club = team.rsplit(' ', 1)[0]
                        club_week_day_slot[(club, week, day, day_slot)] += 1
            # With max_overlap=0, no club should have >1 game at the same slot
            for key, cnt in club_week_day_slot.items():
                assert cnt <= 1, f"Double-up found: {key} has {cnt} games"

    # spec-018: test_maitland_grouping_hard_sliding_window and
    # test_away_maitland_hard_reads_config removed — the MaitlandHomeGrouping /
    # AwayAtMaitlandGrouping rules they exercised were deleted.

    def test_best_timeslot_stacking(self, mini_unified_data):
        """Best timeslot choices stacking: if slot 3 used on WF, slot 2 must be used on EF."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        count = engine._best_timeslot_choices_hard()

        # The constraint adds stacking logic (WF preference moved to soft)
        assert count > 0

        # Solve and check no gaps in slot usage at Broadmeadow
        real_vars = [v for k, v in X.items() if len(k) >= 11]
        model.Maximize(sum(real_vars))

        status, solver = solve_with_timeout(model, timeout_seconds=5.0)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            # For each (week, location), check slots used are contiguous
            loc_slots = defaultdict(set)
            for key, var in X.items():
                if len(key) >= 11 and solver.Value(var) == 1:
                    loc_key = (key[6], key[10])  # week, location
                    loc_slots[loc_key].add(key[4])  # day_slot
            for loc_key, slots in loc_slots.items():
                if len(slots) >= 3:
                    sorted_slots = sorted(slots)
                    for i in range(len(sorted_slots) - 2):
                        # No gap: if slot i and slot i+2 exist, slot i+1 must exist
                        if sorted_slots[i+2] - sorted_slots[i] == 2:
                            assert sorted_slots[i] + 1 in slots, \
                                f"Gap found at {loc_key}: slots {sorted_slots}"

    def test_best_timeslot_last_slot_wf(self, mini_unified_data):
        """Last-slot-WF: if only 1 field active on last slot at Broadmeadow, prefer WF (soft)."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        engine._best_timeslot_choices_hard()
        engine._best_timeslot_choices_soft()

        # Include penalty in objective so solver optimizes for WF preference
        real_vars = [v for k, v in X.items() if len(k) >= 11]
        penalty_terms = []
        for name, info in data['penalties'].items():
            w = info['weight']
            for pv in info['penalties']:
                penalty_terms.append(w * pv)
        model.Maximize(sum(real_vars) * 1000 - sum(penalty_terms))

        status, solver = solve_with_timeout(model, timeout_seconds=5.0)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            # For each week at Broadmeadow, check last slot
            week_field_slots = defaultdict(lambda: defaultdict(set))
            for key, var in X.items():
                if len(key) >= 11 and solver.Value(var) == 1:
                    if key[10] == BROADMEADOW:
                        week = key[6]
                        field = key[9]
                        day_slot = key[4]
                        week_field_slots[week][field].add(day_slot)

            for week, field_slots in week_field_slots.items():
                all_slots = set()
                for slots in field_slots.values():
                    all_slots.update(slots)
                if not all_slots:
                    continue
                max_slot = max(all_slots)
                fields_on_last = [fn for fn, slots in field_slots.items() if max_slot in slots]
                if len(fields_on_last) == 1:
                    assert fields_on_last[0] == 'WF', \
                        f"Week {week}: single field on last slot {max_slot} is {fields_on_last[0]}, expected WF"


# ============== TestSharedVariablePool ==============

class TestSharedVariablePool:
    """Tests for the SharedVariablePool class."""

    def test_pool_bool_caching(self):
        """Same key returns same BoolVar."""
        model = cp_model.CpModel()
        pool = SharedVariablePool(model)
        v = model.NewBoolVar('x')
        ind1 = pool.get_or_create_bool('k1', [v], 'label1')
        ind2 = pool.get_or_create_bool('k1', [v], 'label2')
        assert ind1 is ind2

    def test_pool_bool_different_keys(self):
        """Different keys return different BoolVars."""
        model = cp_model.CpModel()
        pool = SharedVariablePool(model)
        v = model.NewBoolVar('x')
        ind1 = pool.get_or_create_bool('k1', [v], 'label1')
        ind2 = pool.get_or_create_bool('k2', [v], 'label2')
        assert ind1 is not ind2

    def test_pool_register(self):
        """Manually registered vars are retrievable via get()."""
        model = cp_model.CpModel()
        pool = SharedVariablePool(model)
        v = model.NewBoolVar('manual')
        pool.register('manual_key', v)
        assert pool.get('manual_key') is v

    def test_pool_get_returns_none(self):
        """Missing key returns None."""
        model = cp_model.CpModel()
        pool = SharedVariablePool(model)
        assert pool.get('nonexistent') is None

    def test_pool_diagnostics(self):
        """Diagnostics reports correct creation and hit counts."""
        model = cp_model.CpModel()
        pool = SharedVariablePool(model)
        v = model.NewBoolVar('x')
        pool.get_or_create_bool('k1', [v], 'label1')  # create
        pool.get_or_create_bool('k1', [v], 'label2')  # hit
        pool.get_or_create_bool('k2', [v], 'label3')  # create
        d = pool.diagnostics()
        assert d['created'] == 2
        assert d['hits'] == 1
        assert d['pool_size'] == 2

    def test_pool_empty_vars_list(self):
        """BoolVar with empty list is forced to 0."""
        model = cp_model.CpModel()
        pool = SharedVariablePool(model)
        ind = pool.get_or_create_bool('empty', [], 'empty_ind')
        model.Maximize(ind)
        status, solver = solve_with_timeout(model, timeout_seconds=1.0)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            assert solver.Value(ind) == 0

    def test_pool_presence_caching(self):
        """Presence indicators are cached by key."""
        model = cp_model.CpModel()
        pool = SharedVariablePool(model)
        v = model.NewBoolVar('x')
        ind1 = pool.get_or_create_presence('pk', [v], 'pres1')
        ind2 = pool.get_or_create_presence('pk', [v], 'pres2')
        assert ind1 is ind2


# ============== TestStageCorrectness ==============

class TestStageCorrectness:
    """Verify hard stage has no penalties, soft stage populates penalties."""

    def test_hard_stage_no_penalties(self, mini_unified_data):
        """Stage 1 does not populate data['penalties'] except for atoms whose
        single idea spans both hard and soft components.

        After Phase 3c, `ClubVsClubFieldLimit` (HARD ≤2 fields + SOFT field
        excess) and `PHLAnd2ndBackToBackSameField` (HARD back-to-back + SOFT
        coincidence deficit) legitimately register their penalty buckets in
        stage 1 because the atom is the unit, not the stage. Every other
        constraint should still be clean.
        """
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        engine.apply_stage_1_hard()

        allowed_in_stage_1 = {
            'ClubVsClubAlignment', 'ClubVsClubAlignmentField',
        }
        unexpected = set(data['penalties']) - allowed_in_stage_1
        assert not unexpected, \
            f"unexpected penalty entries from hard stage: {sorted(unexpected)}"

    def test_soft_stage_populates_penalties(self, mini_unified_data):
        """After both stages, penalties dict has entries with weight and penalties keys."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        engine.apply_stage_1_hard()
        engine.apply_stage_2_soft()

        # Soft stage should have created at least some penalty entries
        assert len(data['penalties']) > 0, "Soft stage created no penalty entries"
        for name, info in data['penalties'].items():
            assert 'weight' in info, f"Penalty '{name}' missing 'weight' key"
            assert 'penalties' in info, f"Penalty '{name}' missing 'penalties' key"
            assert isinstance(info['weight'], int), f"Penalty '{name}' weight not int"

    def test_club_day_contiguity_in_stage_1(self, mini_unified_data):
        """ClubDay hard constraints (incl. contiguity) are dispatched from stage 1.

        Phase 3b replaced `_club_day_scheduling` + `_club_day_field_contiguity`
        with a single `_club_day_atoms_hard` that runs all 5 ClubDay atoms.
        """
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()

        import inspect
        stage2_source = inspect.getsource(engine.apply_stage_2_soft)
        assert '_club_day_atoms_hard' not in stage2_source, \
            "ClubDay atoms should not be called from stage 2"
        assert '_club_day_field_contiguity' not in stage2_source

        stage1_source = inspect.getsource(engine.apply_stage_1_hard)
        assert '_club_day_atoms_hard' in stage1_source, \
            "ClubDay atoms should be called from stage 1"

    def test_phase_c_exists(self, mini_engine):
        """apply_phase_c exists and returns 0."""
        mini_engine.build_groupings()
        result = mini_engine.apply_phase_c()
        assert result == 0


# ============== TestPenaltyWeightsConfig ==============

class TestPenaltyWeightsConfig:
    """Verify penalty weights are read from config when available."""

    def test_default_weights_without_config(self, mini_unified_data):
        """Without penalty_weights in data, defaults are used."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        # Ensure no penalty_weights key
        data.pop('penalty_weights', None)
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        engine.apply_stage_1_hard()
        engine.apply_stage_2_soft()

        # EqualMatchUpSpacing default is 5000
        if 'EqualMatchUpSpacing' in data['penalties']:
            assert data['penalties']['EqualMatchUpSpacing']['weight'] == 5000

    def test_config_overrides_defaults(self, mini_unified_data):
        """penalty_weights in data override hard-coded defaults."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        data['penalty_weights'] = {'EqualMatchUpSpacing': 99999}
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        engine.apply_stage_1_hard()
        engine.apply_stage_2_soft()

        if 'EqualMatchUpSpacing' in data['penalties']:
            assert data['penalties']['EqualMatchUpSpacing']['weight'] == 99999

    def test_missing_config_key_uses_default(self, mini_unified_data):
        """Partial config — unspecified keys use defaults."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        # spec-018: MaitlandHomeGrouping penalty removed — use ClubGameSpread
        # as the overridden bucket instead.
        data['penalty_weights'] = {'ClubGameSpread': 42}
        engine = UnifiedConstraintEngine(model, X, data)
        engine.build_groupings()
        engine.apply_stage_1_hard()
        engine.apply_stage_2_soft()

        # ClubGameSpread overridden
        if 'ClubGameSpread' in data['penalties']:
            assert data['penalties']['ClubGameSpread']['weight'] == 42
        # EqualMatchUpSpacing should be default
        if 'EqualMatchUpSpacing' in data['penalties']:
            assert data['penalties']['EqualMatchUpSpacing']['weight'] == 5000


# ============== TestNoOrphanVars ==============

class TestNoOrphanVars:
    """Verify skipping a constraint doesn't create its helper vars."""

    def test_skip_alignment_no_coin_vars(self, mini_unified_data):
        """Skipping ClubVsClubAlignment creates no coincide indicators."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data,
            skip_constraints={'ClubVsClubAlignment'})
        engine.build_groupings()
        engine.apply_stage_1_hard()
        engine.apply_stage_2_soft()

        coin_keys = [k for k in engine.pool._cache
                     if isinstance(k, tuple) and k[0] == 'coin']
        assert len(coin_keys) == 0, \
            f"Skipped ClubVsClubAlignment but found coin keys: {coin_keys}"

    def test_skip_cgs_no_cgs_vars(self, mini_unified_data):
        """Skipping ClubGameSpread creates no CGS variables."""
        model, X = create_model_and_vars(
            mini_unified_data['games'], mini_unified_data['timeslots'],
        )
        data = dict(mini_unified_data)
        data['penalties'] = {}
        engine = UnifiedConstraintEngine(model, X, data,
            skip_constraints={'ClubGameSpread'})
        engine.build_groupings()
        engine.apply_stage_1_hard()
        engine.apply_stage_2_soft()

        cgs_keys = [k for k in engine.pool._cache
                    if isinstance(k, tuple) and len(k) >= 1 and
                    isinstance(k[0], str) and k[0].startswith('cgs_')]
        assert len(cgs_keys) == 0, \
            f"Skipped ClubGameSpread but found CGS keys: {cgs_keys}"
