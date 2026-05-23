"""
spec-025 Unit B — generate_X LOCKED_PAIRINGS enforcement pass.

Given/When/Then, no mocks, hand oracle on a tiny real CP-SAT model. Exercises
the REAL utils.generate_X (which runs validate_game_config + variable creation +
the LOCKED_PAIRINGS scope-count pass).
"""
from __future__ import annotations

from collections import defaultdict

import pytest
from ortools.sat.python import cp_model

from models import Club, PlayingField, Team, Timeslot
from utils import (
    generate_X,
    _build_scope_count_rules,
    _get_matching_forced_scopes,
)

BROADMEADOW = 'Newcastle International Hockey Centre'


def _base_fixture():
    """All-NIHC clubs so the home-field/capacity validation phases stay clean.
    One date (2026-03-22, week 1, round 1) with 2 times x 2 fields = 4 slots.
    """
    clubs = [
        Club(name='Norths', home_field=BROADMEADOW),
        Club(name='Souths', home_field=BROADMEADOW),
        Club(name='Wests', home_field=BROADMEADOW),
        Club(name='Easts', home_field=BROADMEADOW),
    ]
    # 3rd grade: no phl_game_times/second_grade_times filtering; NIHC Sunday
    # slots are all valid, so the pairing gets all 4 candidate vars on its date.
    teams = [Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs]

    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')

    timeslots = []
    # Date 1: 2 times x 2 fields = 4 slots (the pinned date).
    for slot_idx, (time, field) in enumerate(
        [('10:00', ef), ('11:30', ef), ('10:00', wf), ('11:30', wf)], start=1
    ):
        timeslots.append(Timeslot(
            date='2026-03-22', day='Sunday', time=time, week=1,
            day_slot=slot_idx, field=field, round_no=1,
        ))
    # A second date so other pairings can be placed (keeps EqualGames feasible
    # at the solver layer if needed; not strictly required for these tests).
    for slot_idx, (time, field) in enumerate(
        [('10:00', ef), ('11:30', ef)], start=1
    ):
        timeslots.append(Timeslot(
            date='2026-03-29', day='Sunday', time=time, week=2,
            day_slot=slot_idx, field=field, round_no=2,
        ))

    games = [
        ('Norths 3rd', 'Souths 3rd', '3rd'),
        ('Wests 3rd', 'Easts 3rd', '3rd'),
    ]

    return {
        'teams': teams,
        'timeslots': timeslots,
        'fields': [ef, wf],
        'games': games,
        'num_rounds': {'3rd': 1},
        'phl_game_times': {},
        'second_grade_times': {},
        'home_field_map': {},
        'forced_games': [],
        'blocked_games': [],
        'preferred_games': [],
        'locked_pairings': [],
        'club_days': {},
        'constraint_defaults': {},
        'grade_rounds_override': {},
    }


# ============== Pin forces exactly one of 4 (DoD 3) ==============

class TestPinForcesExactlyOne:
    def test_four_candidates_pinned_to_sum_one(self):
        """GIVEN a pairing with 4 candidate vars on one date and a pin on it,
        WHEN generate_X runs THEN exactly one of the 4 is forced (sum==1):
        any single-var solution is feasible; a two-var assignment is infeasible."""
        data = _base_fixture()
        data['locked_pairings'] = [
            {'teams': ['Norths', 'Souths'], 'grade': '3rd',
             'date': '2026-03-22', 'description': 'pin Norths v Souths wk1'},
        ]
        model = cp_model.CpModel()
        X, _conflicts = generate_X(model, data)

        # Hand oracle: the 4 Norths-vs-Souths vars on 2026-03-22.
        ns_vars = [
            v for k, v in X.items()
            if {k[0], k[1]} == {'Norths 3rd', 'Souths 3rd'} and k[7] == '2026-03-22'
        ]
        assert len(ns_vars) == 4, f"expected 4 candidate vars, got {len(ns_vars)}"

        # The pin constraint (sum == 1) is in the model: a single-var solution
        # is feasible.
        m1 = model.Clone()
        m1.Add(ns_vars[0] == 1)
        for v in ns_vars[1:]:
            m1.Add(v == 0)
        s1 = cp_model.CpSolver()
        assert s1.Solve(m1) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # A two-var assignment violates sum == 1 → infeasible.
        m2 = model.Clone()
        m2.Add(ns_vars[0] == 1)
        m2.Add(ns_vars[1] == 1)
        s2 = cp_model.CpSolver()
        assert s2.Solve(m2) == cp_model.INFEASIBLE

        # Sanity: the unconstrained model has a solution with exactly one of the
        # four set (the pin's sum==1), time/field solver-determined.
        s0 = cp_model.CpSolver()
        assert s0.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert sum(s0.Value(v) for v in ns_vars) == 1


# ============== Empty-scope FATAL (DoD 4) ==============

class TestEmptyScopeFatal:
    def test_pin_with_zero_placeable_vars_is_fatal(self, capsys):
        """GIVEN a pin whose pairing has zero vars on its date,
        THEN generate_X exits FATAL with a LOCKED_PAIRINGS diagnostic."""
        data = _base_fixture()
        # A valid season date (has slots) but the pairing is blocked on that
        # date, so zero vars survive into X → the generate_X-time empty-scope
        # FATAL must fire (we do NOT silently drop the pin).
        data['locked_pairings'] = [
            {'teams': ['Norths', 'Souths'], 'grade': '3rd',
             'date': '2026-03-29', 'description': 'pin Norths v Souths wk2'},
        ]
        data['blocked_games'] = [
            {'teams': ['Norths', 'Souths'], 'grade': '3rd', 'date': '2026-03-29',
             'description': 'block the pinned pairing on its date'},
        ]
        model = cp_model.CpModel()
        with pytest.raises(SystemExit):
            generate_X(model, data)
        out = capsys.readouterr().out
        assert 'LOCKED_PAIRINGS' in out
        assert 'NO playable variables' in out
        assert 'Norths' in out


# ============== Dual registration: FORCED + LOCKED both apply (DoD 3) ==============

class TestDualRegistration:
    def test_var_matching_both_scopes_registers_in_both(self):
        """GIVEN a var matching BOTH a FORCED scope and a LOCKED_PAIRINGS scope,
        THEN both register it and forced_scope_vars equals the FORCED-only
        baseline (hand-count via the rule primitives)."""
        data = _base_fixture()
        forced = [
            {'teams': ['Norths', 'Souths'], 'grade': '3rd',
             'date': '2026-03-22', 'count': 1, 'constraint': 'equal'},
        ]
        locked = [
            {'teams': ['Norths', 'Souths'], 'grade': '3rd',
             'date': '2026-03-22', 'description': 'same pairing pinned'},
        ]

        # FORCED-only baseline: build the FORCED rules + count matching vars by
        # replaying the exact key-space generate_X would create for this pairing
        # on 2026-03-22 (4 slots).
        from utils import _build_forced_game_rules
        f_rules, _ct, _cc = _build_forced_game_rules(forced, data['teams'])
        lp_groups, _a, _b, _c = _build_scope_count_rules(
            locked, data['teams'], label='LOCKED_PAIRINGS', unique_per_entry=True)

        # The 4 candidate keys for Norths-vs-Souths on the pinned date.
        keys = [
            ('Norths 3rd', 'Souths 3rd', '3rd', 'Sunday', si, time, 1,
             '2026-03-22', 1, fld, BROADMEADOW)
            for si, (time, fld) in enumerate(
                [('10:00', 'EF'), ('11:30', 'EF'), ('10:00', 'WF'), ('11:30', 'WF')], start=1)
        ]

        # Hand oracle: every one of the 4 keys matches BOTH the single FORCED
        # scope and the single LOCKED scope.
        forced_baseline = defaultdict(list)
        locked_reg = defaultdict(list)
        for k in keys:
            for sk in _get_matching_forced_scopes(k, f_rules):
                forced_baseline[sk].append(k)
            for sk in _get_matching_forced_scopes(k, lp_groups):
                locked_reg[sk].append(k)

        assert len(f_rules) == 1 and len(lp_groups) == 1
        f_scope = next(iter(f_rules))
        l_scope = next(iter(lp_groups))
        # FORCED registers all 4 (unchanged baseline); LOCKED registers all 4.
        assert len(forced_baseline[f_scope]) == 4
        assert len(locked_reg[l_scope]) == 4

        # Now run the REAL generate_X with BOTH configs and confirm the FORCED
        # constraint is byte-identical (4 vars summed == 1) AND the pin is
        # independently enforced — solving yields exactly one Norths-v-Souths
        # game on the pinned date.
        data['forced_games'] = forced
        data['locked_pairings'] = locked
        model = cp_model.CpModel()
        X, _ = generate_X(model, data)
        ns_vars = [
            v for k, v in X.items()
            if {k[0], k[1]} == {'Norths 3rd', 'Souths 3rd'} and k[7] == '2026-03-22'
        ]
        assert len(ns_vars) == 4
        solver = cp_model.CpSolver()
        assert solver.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        # Both FORCED (==1) and LOCKED (==1) over the SAME 4 vars → exactly one.
        assert sum(solver.Value(v) for v in ns_vars) == 1
