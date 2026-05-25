"""Tests for the spec-033 Unit B NORMAL-MODE soft analogue of
`BalancedByeSpacing`.

Real CP-SAT models, no mocks/patches/monkeypatch. Every expected value is
hand-computed in the docstring/comments (the oracle is the hand calc, never
the code's own output).

Design recap (see `constraints/atoms/balanced_bye_spacing.py`):
  - Hard floor forbids bye pairs with gap <= S where
        S = max(0, ideal_bye_gap(R, byes) - bye_spacing_base_slack - config_slack).
  - With `bye_spacing_base_slack = 2` (the 2026 default after spec-033) the
    hard floor sits 2 rounds BELOW the raw ideal `S_base = ideal_bye_gap(R,b)`.
  - The SOFT term penalises (does NOT forbid) bye pairs in the band
        (S, S_base]   — closer than ideal, but hard-tolerated.
  - The soft term NEVER relaxes the hard floor; it only adds sub-threshold
    pressure on gaps the hard floor already permits.
  - Penalties land in `data['penalties']['BalancedByeSpacing']['penalties']`
    (the NORMAL bucket key, NOT the regen atom's `regen_balanced_bye_spacing`).
  - One penalty BoolVar fires iff BOTH rounds of a (r1, r2) band pair are byes.

Primary oracle (DoD 21):
  9-team grade, R=18, byes_per_team=2 (games_per_team=16).
    ideal_bye_gap(18, 2): avg = 18 // 2 = 9; S_base = 9 - 1 = 8.
    bye_spacing_base_slack = 2, config_slack = 0
      => hard floor S = max(0, 8 - 2 - 0) = 6.
    => gap 7 is FEASIBLE (7 > 6); gap 6 is HARD-FORBIDDEN at slack 0.
    => soft band is (6, 8] = {7, 8}.

  Soft-penalty delta (hand): a team with its two byes
    - 7 rounds apart  -> the single (r1, r2) pair has gap 7 in (6,8]
                          -> exactly 1 penalty BoolVar, and it FIRES (both
                          rounds are byes) => 1 penalty unit.
    - 9 rounds apart  -> gap 9 > S_base=8 -> NOT in the band, no penalty var
                          enumerated for that pair => 0 penalty units.
    So moving the two byes from 7-apart to 9-apart REMOVES one penalty unit:
    the soft term strictly prefers the larger gap. delta = 1.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

from ortools.sat.python import cp_model

from constraints.atoms.balanced_bye_spacing import BalancedByeSpacing
from constraints.atoms._spacing import ideal_bye_gap
from constraints.helper_vars import HelperVarRegistry
from constraints.atoms.base import BROADMEADOW
from models import Club, Grade, PlayingField, Team, Timeslot


# ----------------------------------------------------------------------
# Fixture helpers (mirror the existing hard-atom test fixture, but allow
# overriding constraint_defaults / penalty_weights and skipping no-play rounds).
# ----------------------------------------------------------------------


def _make_grade_fixture(
    *,
    grade_name: str,
    num_teams: int,
    num_rounds: int,
    games_per_team: int,
    base_slack: int = 2,
    soft_weight: int = 100_000,
    no_play_week: int | None = None,
) -> Dict:
    """One-grade fixture.

    Round numbering is DENSE (1..num_rounds) — exactly how production behaves:
    `generate_timeslots` only increments `round_no` on PLAYABLE weekends, so a
    no-play weekend consumes a calendar week but NO round_no. We model this by
    optionally inserting an extra calendar week (`no_play_week`) that carries a
    HIGHER `week` value but NO timeslot and NO round_no — the round sequence the
    atom sees stays contiguous 1..num_rounds. `num_rounds['max']` is the
    playable-round count, so the missing calendar week is invisible to the atom.
    """
    ef = PlayingField(location=BROADMEADOW, name='EF')

    clubs = [Club(name=f'C{i}', home_field=BROADMEADOW) for i in range(num_teams)]
    teams = [
        Team(name=f'{clubs[i].name} {grade_name}', club=clubs[i], grade=grade_name)
        for i in range(num_teams)
    ]
    grade = Grade(name=grade_name, teams=[t.name for t in teams])

    # Dense round_nos 1..num_rounds. `week` may differ from round_no when a
    # no-play calendar week is inserted, but the atom keys off round_no (key[8]).
    timeslots = []
    week = 1
    for r in range(1, num_rounds + 1):
        if no_play_week is not None and r == no_play_week:
            # A no-play calendar week sits BEFORE this round: bump the week
            # counter but emit NO timeslot for it (no round_no consumed).
            week += 1
        timeslots.append(Timeslot(
            date=f'2026-03-{r:02d}', day='Sunday', time='11:30',
            week=week, day_slot=1, field=ef, round_no=r,
        ))
        week += 1

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
        'penalty_weights': {'BalancedByeSpacing': soft_weight},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {},
        'constraint_defaults': {'bye_spacing_base_slack': base_slack},
        'GRADE_ROUNDS_OVERRIDE': {grade_name: games_per_team},
    }
    return data


def _build_X(model: cp_model.CpModel, data: Dict) -> Dict:
    X = {}
    for (t1, t2, grade) in data['games']:
        for ts in data['timeslots']:
            key = (
                t1, t2, grade, ts.day, ts.day_slot, ts.time,
                ts.week, ts.date, ts.round_no, ts.field.name,
                ts.field.location,
            )
            X[key] = model.NewBoolVar(f'x_{t1}_{t2}_w{ts.week}')
    return X


def _enforce_one_game_per_team_per_round(model, X, data):
    from collections import defaultdict
    per = defaultdict(list)
    for key, var in X.items():
        t1, t2, _g, _d, _s, _ti, _w, _d2, r, _fn, _fl = key
        per[(t1, r)].append(var)
        per[(t2, r)].append(var)
    for vs in per.values():
        model.Add(sum(vs) <= 1)


def _pin_team_to_rounds(model, X, team_name, grade, play_rounds):
    from collections import defaultdict
    per = defaultdict(list)
    for key, var in X.items():
        t1, t2, g, _d, _s, _ti, _w, _d2, r, _fn, _fl = key
        if g != grade:
            continue
        if team_name in (t1, t2):
            per[r].append(var)
    play = set(play_rounds)
    for r, vs in per.items():
        model.Add(sum(vs) == (1 if r in play else 0))


def _registry(model):
    return HelperVarRegistry(model)


def _solve(model, seconds: float = 10.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    return solver.Solve(model), solver


def _bucket(data):
    return data.get('penalties', {}).get('BalancedByeSpacing')


# ----------------------------------------------------------------------
# Scenario A — hard floor sits BELOW the raw ideal (base slack 2).
#   R=18, byes=2 => S_base = ideal_bye_gap(18,2) = 8; S = 8-2-0 = 6.
#   gap 7 feasible; gap 6 forbidden at slack 0.
# ----------------------------------------------------------------------


class TestHardFloorBelowIdeal:
    def test_ideal_gap_oracle(self):
        # Hand: avg = 18 // 2 = 9; S_base = 9 - 1 = 8.
        assert ideal_bye_gap(18, 2) == 8

    def test_gap_7_feasible_at_base_slack_2(self):
        """Two byes 7 rounds apart: gap 7 > hard floor S=6 => FEASIBLE.
        (At base slack 0 the floor would be 8 and 7 would be forbidden;
        the base-slack-2 change is what makes this solvable.)"""
        data = _make_grade_fixture(
            grade_name='3rd', num_teams=9, num_rounds=18, games_per_team=16,
            base_slack=2,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)
        target = data['teams'][0].name
        # byes at rounds 1 and 8 -> gap 7. Plays the other 16 rounds.
        bye_rounds = {1, 8}
        play = [r for r in range(1, 19) if r not in bye_rounds]
        _pin_team_to_rounds(model, X, target, '3rd', play)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
            "gap 7 with hard floor S=6 must be FEASIBLE; "
            f"got {cp_model.CpSolver().status_name(status)}"
        )

    def test_gap_6_forbidden_at_slack_0(self):
        """Two byes 6 rounds apart: gap 6 <= hard floor S=6 => HARD-FORBIDDEN
        => INFEASIBLE at config slack 0."""
        data = _make_grade_fixture(
            grade_name='3rd', num_teams=9, num_rounds=18, games_per_team=16,
            base_slack=2,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)
        target = data['teams'][0].name
        # byes at rounds 1 and 7 -> gap 6.
        bye_rounds = {1, 7}
        play = [r for r in range(1, 19) if r not in bye_rounds]
        _pin_team_to_rounds(model, X, target, '3rd', play)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            "gap 6 <= hard floor S=6 must be INFEASIBLE; "
            f"got {cp_model.CpSolver().status_name(status)}"
        )


# ----------------------------------------------------------------------
# Scenario B — soft term prefers larger gaps (the penalty delta).
#   Soft band (S, S_base] = (6, 8] = {7, 8}. A bye pair at gap 7 fires one
#   penalty; the SAME team byes 9 apart fire zero (gap 9 not in band).
# ----------------------------------------------------------------------


class TestSoftPenaltyDelta:
    def test_gap_7_incurs_exactly_one_penalty(self):
        """Byes at rounds 1 and 8 (gap 7, in band (6,8]) => exactly one
        penalty BoolVar, and it must take value 1 (both rounds are byes)."""
        data = _make_grade_fixture(
            grade_name='3rd', num_teams=9, num_rounds=18, games_per_team=16,
            base_slack=2,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)
        target = data['teams'][0].name
        bye_rounds = {1, 8}
        play = [r for r in range(1, 19) if r not in bye_rounds]
        _pin_team_to_rounds(model, X, target, '3rd', play)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        bucket = _bucket(data)
        assert bucket is not None, "soft bucket 'BalancedByeSpacing' must exist"
        assert bucket['weight'] == 100_000

        # Force the model to a solution, then verify exactly one of THIS team's
        # band penalty vars fires (its 1-8 pair). Other teams may also fire band
        # penalties, so we minimise total penalties to isolate the pinned team.
        model.Minimize(sum(bucket['penalties']))
        status, solver = _solve(model)
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        # The pinned team's only in-band bye pair is (1, 8). Find its penalty
        # var by name and assert it fired.
        fired_for_target = [
            v for v in bucket['penalties']
            if v.Name().startswith(f'u_bye_soft_pen_{target}_3rd_r1_r8')
        ]
        assert len(fired_for_target) == 1, (
            "exactly one band penalty var for the pinned (1,8) pair"
        )
        assert solver.Value(fired_for_target[0]) == 1, (
            "the (1,8) gap-7 bye pair must fire its soft penalty"
        )

    def test_gap_9_incurs_zero_penalty_for_that_pair(self):
        """Byes at rounds 1 and 10 (gap 9 > S_base=8) => that pair is NOT in
        the soft band, so NO penalty var is enumerated for it. Confirms the
        soft term prefers the larger gap (delta vs gap-7 = 1 penalty unit)."""
        data = _make_grade_fixture(
            grade_name='3rd', num_teams=9, num_rounds=18, games_per_team=16,
            base_slack=2,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)
        target = data['teams'][0].name
        bye_rounds = {1, 10}
        play = [r for r in range(1, 19) if r not in bye_rounds]
        _pin_team_to_rounds(model, X, target, '3rd', play)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        bucket = _bucket(data)
        assert bucket is not None
        # No penalty var should name the (1,10) pair — it is outside the band.
        named_1_10 = [
            v for v in bucket['penalties']
            if v.Name().startswith(f'u_bye_soft_pen_{target}_3rd_r1_r10')
        ]
        assert named_1_10 == [], (
            "gap 9 (> S_base=8) must NOT enumerate a soft penalty var"
        )


# ----------------------------------------------------------------------
# Scenario C — soft term never relaxes the hard floor.
#   Even with the soft term active (weight>0), a sub-floor bye pair (gap 6)
#   is still HARD-infeasible at slack 0.
# ----------------------------------------------------------------------


class TestSoftNeverRelaxesHard:
    def test_soft_active_but_hard_floor_still_binds(self):
        data = _make_grade_fixture(
            grade_name='3rd', num_teams=9, num_rounds=18, games_per_team=16,
            base_slack=2, soft_weight=100_000,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)
        target = data['teams'][0].name
        bye_rounds = {1, 7}  # gap 6 <= S=6
        play = [r for r in range(1, 19) if r not in bye_rounds]
        _pin_team_to_rounds(model, X, target, '3rd', play)

        n_hard = BalancedByeSpacing().apply(model, X, data, _registry(model))
        assert n_hard > 0, "hard clauses must still be emitted"
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            "soft term must NOT relax the hard floor; gap 6 stays INFEASIBLE"
        )


# ----------------------------------------------------------------------
# Scenario D — single-bye grade emits zero penalties.
#   byes_per_team < 2 => no pairwise check => no soft penalties.
# ----------------------------------------------------------------------


class TestSingleByeNoPenalty:
    def test_one_bye_zero_penalties(self):
        """games_per_team = R-1 => 1 bye/team => no pairwise check => the
        soft bucket has zero penalty vars (and ideally is never created)."""
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=9,
            base_slack=2,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        BalancedByeSpacing().apply(model, X, data, _registry(model))
        bucket = _bucket(data)
        n = 0 if bucket is None else len(bucket['penalties'])
        assert n == 0, f"single-bye grade must emit 0 soft penalties; got {n}"

    def test_zero_bye_zero_penalties(self):
        """games_per_team == R => 0 byes => 0 soft penalties."""
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=10,
            base_slack=2,
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        BalancedByeSpacing().apply(model, X, data, _registry(model))
        bucket = _bucket(data)
        n = 0 if bucket is None else len(bucket['penalties'])
        assert n == 0, f"zero-bye grade must emit 0 soft penalties; got {n}"


# ----------------------------------------------------------------------
# Scenario E — a no-play CALENDAR week is not counted as a bye (DoD 19).
#   Production: no-play weekends consume a calendar week but NO round_no, so
#   round_no stays DENSE (1..max) and byes are measured in PLAYABLE-round terms.
#   We insert a no-play calendar week (higher `week`, no timeslot, no round_no)
#   and confirm: (a) every decision var still keys off a dense round_no, (b)
#   the bye count is unaffected by the missing calendar week, (c) gaps are
#   measured in round_no terms, not calendar weeks.
# ----------------------------------------------------------------------


class TestNoPlayWeekNotCountedAsBye:
    def test_round_nos_stay_dense_across_no_play_week(self):
        """Insert a no-play calendar week before round 5. The atom must see
        contiguous round_nos 1..10 (no phantom bye round), and the no-play
        week must NOT appear as a round_no on any variable.

        Hand: byes_per_team = max(10) - games(8) = 2; ideal_bye_gap(10,2)=4;
        S = max(0, 4-2-0) = 2; band (2,4] = {3,4}. The structural setup is
        unchanged by the calendar-week insertion."""
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=8,
            base_slack=2, no_play_week=5,
        )
        round_nos = sorted({ts.round_no for ts in data['timeslots']})
        assert round_nos == list(range(1, 11)), "round_nos must stay dense 1..10"
        # The no-play week bumped `week` so weeks are NOT contiguous, proving a
        # calendar week was skipped without consuming a round_no.
        weeks = sorted({ts.week for ts in data['timeslots']})
        assert max(weeks) > 10, "a no-play calendar week must have been inserted"

        model = cp_model.CpModel()
        X = _build_X(model, data)
        # Every var keys off a dense round_no in 1..10 — no phantom round.
        assert all(key[8] in range(1, 11) for key in X)

    def test_byes_measured_in_round_terms_not_calendar_weeks(self):
        """With a no-play calendar week inserted, two byes at PLAYABLE rounds
        3 and 5 have round-gap = 5 - 3 = 2 = S => HARD-forbidden => INFEASIBLE.
        If the atom (wrongly) measured the calendar-week gap it would see a
        larger separation (the inserted no-play week pads weeks 3->5 to 4
        calendar weeks) and might allow it. INFEASIBLE proves round_no
        arithmetic is used.

        Hand: max=10, games=8 => byes=2; ideal_bye_gap(10,2)=4; S=4-2-0=2;
        gap 5-3 = 2 <= S => forbidden."""
        data = _make_grade_fixture(
            grade_name='5th', num_teams=5, num_rounds=10, games_per_team=8,
            base_slack=2, no_play_week=4,  # no-play week sits between r3 and r4
        )
        model = cp_model.CpModel()
        X = _build_X(model, data)
        _enforce_one_game_per_team_per_round(model, X, data)
        target = data['teams'][0].name
        bye_rounds = {3, 5}
        play = [r for r in range(1, 11) if r not in bye_rounds]
        _pin_team_to_rounds(model, X, target, '5th', play)

        BalancedByeSpacing().apply(model, X, data, _registry(model))
        status, _ = _solve(model)
        assert status == cp_model.INFEASIBLE, (
            "byes at round_nos 3 and 5 (round-gap 2 = S) must be INFEASIBLE "
            "regardless of the inserted no-play calendar week; "
            f"got {cp_model.CpSolver().status_name(status)}"
        )
