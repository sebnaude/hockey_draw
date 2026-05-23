"""Tests for the `BalancedByeSpacing` atom (spec-008 Part B).

Real CP-SAT models, no mocks. Each scenario builds a tiny grade fixture
end-to-end (teams, grade, decision-variable dict) and hand-computes the
expected outcome in the test docstring.

Scenarios
---------

1. **Two-byes per team, must NOT be adjacent.**
   Given 6 teams, R=20 playable rounds, games_per_team=16 → byes=4 per
   team. `ideal_bye_gap(20, 4)` = 20//4 - 1 = 4 — so any pair of bye
   rounds with gap <= 4 is forbidden. The test forces a team into 16
   games arranged so two byes land at adjacent rounds and verifies the
   model is INFEASIBLE.

2. **Five-team grade, one bye per round structurally — byes must rotate.**
   With 5 teams in each round you have one team byed every round; over
   R=10 rounds each team byes exactly twice. `ideal_bye_gap(10, 2)` =
   10//2 - 1 = 4. The test pins one team's first bye at round 1 and a
   second bye at round 3 (gap=2 <= 4) and asserts INFEASIBLE; then
   moves the second bye to round 6 (gap=5 > 4) and asserts FEASIBLE.

3. **Locked weeks: byes inside locked window are ignored.**
   With every round locked the atom must emit zero constraints (every
   pair is locked-locked). With locked_weeks empty the same fixture emits
   a positive count. The locked-to-unlocked variant verifies that a
   bye in a locked round paired with a bye in an unlocked round IS still
   constrained (only locked-locked pairs are skipped, matching the
   matchup-spacing convention).

4. **Slack disables the atom.**
   With `BalancedByeSpacing` slack = ideal_bye_gap, S clamps to 0 and
   the atom adds zero constraints. Verified by counting `model.Proto().constraints`.

5. **Atom is a no-op for grades with 0 or 1 byes per team.**
   When games_per_team == R the atom adds no constraints for that grade.

The tests use `max_games_per_grade` indirectly via a tiny synthetic data
dict and grade with `num_games` per team set via `GRADE_ROUNDS_OVERRIDE`,
matching production behaviour.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import pytest
from ortools.sat.python import cp_model

from constraints.atoms.balanced_bye_spacing import BalancedByeSpacing
from constraints.atoms._spacing import ideal_bye_gap
from constraints.helper_vars import HelperVarRegistry
from constraints.atoms.base import BROADMEADOW
from models import Club, Grade, PlayingField, Team, Timeslot


# ----------------------------------------------------------------------
# Helpers — minimal grade fixture for bye tests
# ----------------------------------------------------------------------


def _make_grade_fixture(
    *,
    grade_name: str,
    num_teams: int,
    num_rounds: int,
    games_per_team: int,
) -> Dict:
    """Build a tiny one-grade fixture sufficient to exercise BalancedByeSpacing.

    Creates `num_teams` teams (one per dummy club for clean t1<t2 ordering),
    one timeslot per round, and the cross-product of pair-game variables.
    `GRADE_ROUNDS_OVERRIDE` is set so `max_games_per_grade` returns
    `games_per_team` exactly (no formula path).
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')

    clubs = [Club(name=f'C{i}', home_field=BROADMEADOW) for i in range(num_teams)]
    teams = [
        Team(name=f'{clubs[i].name} {grade_name}', club=clubs[i], grade=grade_name)
        for i in range(num_teams)
    ]
    grade = Grade(name=grade_name, teams=[t.name for t in teams])

    timeslots = []
    for r in range(1, num_rounds + 1):
        # Single Sunday slot per round — enough vars for any pair to play r.
        timeslots.append(Timeslot(
            date=f'2026-03-{r:02d}', day='Sunday', time='11:30',
            week=r, day_slot=1, field=ef, round_no=r,
        ))

    games = [(t1.name, t2.name, grade_name) for t1, t2 in combinations(teams, 2)]

    data = {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': [grade],
        'clubs': clubs,
        'fields': [ef],
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': {grade_name: games_per_team, 'max': num_rounds},
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {},
        'constraint_defaults': {},
        'GRADE_ROUNDS_OVERRIDE': {grade_name: games_per_team},
    }
    return data


def _build_X(model: cp_model.CpModel, data: Dict) -> Dict:
    """Build BoolVars for every (game, timeslot) combination."""
    X = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            key = (
                t1, t2, grade, ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no, ts.field.name,
                ts.field.location,
            )
            X[key] = model.NewBoolVar(
                f'x_{t1}_{t2}_w{ts.week}'
            )
    return X


def _enforce_one_game_per_team_per_round(
    model: cp_model.CpModel, X: Dict, data: Dict
):
    """Add NoDoubleBookingTeams so bye indicators are well-defined."""
    from collections import defaultdict
    per_team_round = defaultdict(list)
    for key, var in X.items():
        t1, t2, _g, _d, _s, _ti, _w, _d2, r, _fn, _fl = key
        per_team_round[(t1, r)].append(var)
        per_team_round[(t2, r)].append(var)
    for vs in per_team_round.values():
        model.Add(sum(vs) <= 1)


def _pin_team_to_rounds(
    model: cp_model.CpModel,
    X: Dict,
    team_name: str,
    grade: str,
    play_rounds: List[int],
):
    """Force `team_name` to play in exactly the rounds in `play_rounds` (and not others).

    Sums of team's vars per round are pinned to 1 / 0 accordingly.
    """
    from collections import defaultdict
    per_round = defaultdict(list)
    for key, var in X.items():
        t1, t2, g, _d, _s, _ti, _w, _d2, r, _fn, _fl = key
        if g != grade:
            continue
        if team_name in (t1, t2):
            per_round[r].append(var)
    for r, vs in per_round.items():
        if r in play_rounds:
            model.Add(sum(vs) == 1)
        else:
            model.Add(sum(vs) == 0)


def _registry(model):
    r = HelperVarRegistry(model)
    return r


def _solve(model, seconds: float = 8.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    return solver.Solve(model), solver


# ----------------------------------------------------------------------
# Scenario 1 — two byes per team must not land adjacent
# ----------------------------------------------------------------------


class TestTwoByesNotAdjacent:
    """Six-team grade, R=20, games_per_team=16 → 4 byes/team.

    Hand calc: ideal_bye_gap(20, 4) = 20//4 - 1 = 4. So byes must be
    > 4 rounds apart. Any team with two of its 4 byes at gap <= 4 makes
    the model INFEASIBLE.
    """

    def test_byes_in_consecutive_rounds_infeasible(self):
        """Pin one team to play rounds {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}
        — leaves byes at rounds 1, 18, 19, 20. Byes 18-19 have gap 1 (and
        19-20 has gap 1). Hand: S=4 → gap 1 <= 4 → INFEASIBLE.
        """
        data = _make_grade_fixture(
            grade_name='3rd', num_teams=6, num_rounds=20, games_per_team=16
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)

        target_team = data['teams'][0].name  # 'C0 3rd'
        play_rounds = list(range(2, 18))  # 16 plays, byes at {1, 18, 19, 20}
        _pin_team_to_rounds(model, X, target_team, '3rd', play_rounds)

        n = BalancedByeSpacing().apply(model, X, data, _registry(model))
        assert n > 0, "atom should emit constraints when byes-per-team>=2"

        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            f"Byes at rounds 18, 19, 20 (gap 1) must be INFEASIBLE under "
            f"ideal_bye_gap(20, 4)={ideal_bye_gap(20, 4)}; got "
            f"{cp_model.CpSolver().status_name(status)}"
        )

    def test_byes_evenly_spread_feasible(self):
        """Pin the team to play 16 rounds with byes at {5, 10, 15, 20} —
        every gap is 5, which is > ideal_bye_gap(20,4)=4. FEASIBLE.
        """
        data = _make_grade_fixture(
            grade_name='3rd', num_teams=6, num_rounds=20, games_per_team=16
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)

        target_team = data['teams'][0].name
        bye_rounds = {5, 10, 15, 20}
        play_rounds = [r for r in range(1, 21) if r not in bye_rounds]
        _pin_team_to_rounds(model, X, target_team, '3rd', play_rounds)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"Byes at {{5,10,15,20}} (gap 5) must be FEASIBLE; "
            f"got {cp_model.CpSolver().status_name(status)}"
        )


# ----------------------------------------------------------------------
# Scenario 2 — five-team grade, byes must rotate
# ----------------------------------------------------------------------


class TestFiveTeamGradeByeRotation:
    """Five-team grade, R=10, games_per_team=8 → 2 byes/team.

    Hand calc: ideal_bye_gap(10, 2) = 10//2 - 1 = 4. Pairs with gap <= 4
    forbidden.
    """

    def test_close_byes_infeasible(self):
        """Bye at rounds 1 and 3 (gap=2) — must be INFEASIBLE."""
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=8
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)

        target = data['teams'][0].name  # 'C0 5th'
        play_rounds = [r for r in range(1, 11) if r not in (1, 3)]  # byes at 1,3
        _pin_team_to_rounds(model, X, target, '5th', play_rounds)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            f"Byes at rounds 1, 3 (gap 2) with S=ideal_bye_gap(10,2)=4 "
            f"must be INFEASIBLE; got {cp_model.CpSolver().status_name(status)}"
        )

    def test_well_spaced_byes_feasible(self):
        """Byes at rounds 1 and 6 (gap=5 > 4) — FEASIBLE."""
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=8
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)

        target = data['teams'][0].name
        play_rounds = [r for r in range(1, 11) if r not in (1, 6)]
        _pin_team_to_rounds(model, X, target, '5th', play_rounds)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            f"Byes at rounds 1, 6 (gap 5) must be FEASIBLE; got "
            f"{cp_model.CpSolver().status_name(status)}"
        )


# ----------------------------------------------------------------------
# Scenario 3 — locked weeks
# ----------------------------------------------------------------------


class TestLockedWeeksByesIgnored:
    """Byes inside locked-week window do not trigger constraints.

    Lock rounds 1 and 2. Force a team to bye at rounds 1 and 2 (the
    locked window). Hand: the atom must skip the (1,2) pair because
    both are locked, so the model stays FEASIBLE even though gap=1
    would otherwise violate ideal_bye_gap(10,2)=4.
    """

    def test_locked_pair_skipped_no_constraint_emitted(self):
        """Count the constraints the atom emits with all rounds locked.

        Hand: with locked_weeks = {1..10} every (r1, r2) pair has both
        locked, so the atom must emit ZERO constraints. With locked
        weeks empty (baseline) the atom emits a positive count for the
        same fixture. The diff isolates the "locked-locked skipped"
        behaviour without depending on overall model feasibility (which
        is constrained by structural arithmetic for the *other* teams).
        """
        baseline_data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=8
        )
        model_base = cp_model.CpModel()
        X_base = _build_X(model_base, baseline_data)
        n_base = BalancedByeSpacing().apply(
            model_base, X_base, baseline_data, _registry(model_base)
        )
        assert n_base > 0, "sanity: baseline should emit some constraints"

        locked_data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=8
        )
        locked_data['locked_weeks'] = set(range(1, 11))
        model_locked = cp_model.CpModel()
        X_locked = _build_X(model_locked, locked_data)
        n_locked = BalancedByeSpacing().apply(
            model_locked, X_locked, locked_data, _registry(model_locked)
        )
        assert n_locked == 0, (
            f"All-locked must skip every pair, got {n_locked} constraints"
        )

    def test_locked_to_unlocked_pair_still_enforced(self):
        """Bye in locked round 1 + bye in unlocked round 3 (gap=2 <= 4)
        — the pair is (locked, unlocked) so the constraint IS emitted.
        Forcing both leads to INFEASIBLE.
        """
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=8
        )
        data['locked_weeks'] = {1, 2}
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)

        target = data['teams'][0].name
        play_rounds = [r for r in range(1, 11) if r not in (1, 3)]
        _pin_team_to_rounds(model, X, target, '5th', play_rounds)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            f"Locked-to-unlocked bye pair (1, 3) gap=2 must still be "
            f"forbidden; got {cp_model.CpSolver().status_name(status)}"
        )


# ----------------------------------------------------------------------
# Scenario 4 — slack disables the atom
# ----------------------------------------------------------------------


class TestSlackDisables:
    def test_high_slack_zeros_S_no_constraints(self):
        """With slack >= ideal_bye_gap(R, byes), S clamps to 0 and the
        atom emits zero constraints. Hand: R=10, byes=2 → S0=4. Slack 4
        ⇒ S=0 ⇒ no constraints.
        """
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=8
        )
        data['constraint_slack'] = {'BalancedByeSpacing': 4}
        model = cp_model.CpModel()
        X = _build_X(model, data)
        n = BalancedByeSpacing().apply(model, X, data, _registry(model))
        assert n == 0, f"slack=4 must disable bye spacing; got {n} constraints"


# ----------------------------------------------------------------------
# Scenario 5 — no byes per team
# ----------------------------------------------------------------------


class TestNoByesNoOp:
    def test_zero_byes_no_constraints(self):
        """When games_per_team == R every team plays every round — no
        byes. Atom must emit zero constraints for that grade.
        """
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=10
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        n = BalancedByeSpacing().apply(model, X, data, _registry(model))
        assert n == 0, f"zero byes should yield zero constraints; got {n}"

    def test_one_bye_per_team_no_constraints(self):
        """With one bye per team, no pairwise check is meaningful → no
        constraints emitted.
        """
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=9
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        n = BalancedByeSpacing().apply(model, X, data, _registry(model))
        assert n == 0, f"one-bye-per-team should yield zero constraints; got {n}"


# ----------------------------------------------------------------------
# Sanity — ideal_bye_gap formula is what the atom uses
# ----------------------------------------------------------------------


class TestIdealByeGapFormula:
    @pytest.mark.parametrize('R,byes,expected', [
        (20, 4, 4),  # 20//4 - 1
        (18, 2, 8),  # 18//2 - 1
        (10, 2, 4),  # 10//2 - 1
        (22, 1, 0),  # <2 byes -> 0
        (22, 0, 0),  # 0 byes -> 0
        (0, 5, 0),   # R=0 -> 0
        (6, 3, 1),   # 6//3 - 1
        (5, 2, 1),   # 5//2 = 2 - 1 = 1
    ])
    def test_formula_matches_spec(self, R, byes, expected):
        assert ideal_bye_gap(R, byes) == expected
