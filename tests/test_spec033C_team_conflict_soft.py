"""spec-033 Unit C — TeamConflict is now SOFT-ONLY (no hard component).

The convenor wants a declared conflict pair sharing a `(week, day_slot)` to be a
PREFERENCE to avoid, not a feasibility blocker. So `_team_conflict` (the old hard
`Add(sum(v1)+sum(v2) <= 1)`) is gone, replaced by `_team_conflict_soft` dispatched
in `apply_stage_2_soft`: per conflict pair × `(week, day_slot)` where both teams
could appear, a penalty BoolVar `p` with `p >= sum(v1) + sum(v2) - 1`, appended to
`data['penalties']['TeamConflict']`.

NO mocks / patches / monkeypatch. We use the real `UnifiedConstraintEngine` and a
real CP-SAT model (same pattern as `tests/test_spec033A_kept_soft.py`). We touch
the real `load_season_data(2026)` to confirm the live config carries the
`'TeamConflict'` penalty weight (so the bucket is not just a test artefact). Every
expected value is hand-computed in a comment.

--------------------------------------------------------------------------------
Fixture & oracle
--------------------------------------------------------------------------------
A single 4-team grade '3rd': teams T0, T1, A, B. Two games:
    G1 = (T0, A)   G2 = (B, T1)
Conflict pair = (T0, T1). T0 and T1 are in DIFFERENT games, so they CAN share a
`(week, day_slot)` (they are never the same game). Two timeslots exist in week 1
on one NIHC field: day_slot 1 and day_slot 2.

The soft method builds, per conflict pair × slot where BOTH teams could appear, a
penalty `p >= sum(v1) + sum(v2) - 1`:

  * SAME slot:  pin G1 and G2 both into (week 1, day_slot 1). At that slot
    sum(v1)=sum(v2)=1, so p >= 1 — the bucket sums to EXACTLY 1 penalty unit.
    Crucially the model is FEASIBLE (no hard `<= 1`), proving the rule no longer
    blocks feasibility — the old hard rule would have made this INFEASIBLE.
  * DIFFERENT slots:  pin G1 into day_slot 1, G2 into day_slot 2. No slot has
    both teams, so every `p >= sum(v1)+sum(v2)-1` has RHS <= 0 and the solver is
    free to set every p = 0 — the bucket sums to EXACTLY 0.

  * EMPTY team_conflicts:  the bucket is created (`{'weight':..,'penalties':[]}`)
    with ZERO penalty vars.
"""
from __future__ import annotations

from typing import List, Tuple

from ortools.sat.python import cp_model

from config import load_season_data
from constraints.atoms.base import BROADMEADOW
from constraints.unified import UnifiedConstraintEngine
from models import Club, Grade, PlayingField, Team, Timeslot

from tests.conftest import create_model_and_vars


# Conflict pair and their (distinct-game) opponents.
T0, T1, A, B = 'T0', 'T1', 'A', 'B'
GAMES: List[Tuple[str, str, str]] = [(T0, A, '3rd'), (B, T1, '3rd')]
CONFLICT_PAIR = (T0, T1)


def _fixture(conflicts, num_fields: int = 1):
    """4-team grade, `num_fields` NIHC field(s), two day_slots in week 1.
    `conflicts` is the `team_conflicts` list passed straight into data (no mocks).

    `num_rounds` is 1 so the hard EqualGames rule (each team plays exactly 1 game)
    is satisfiable in the stage-1 test below. With 2 fields, both games can share a
    day_slot without tripping NoDoubleBookingFields (one game per field+slot)."""
    fields = [PlayingField(location=BROADMEADOW, name=fn)
              for fn in (['EF', 'WF'][:num_fields])]
    club = Club(name='C', home_field=BROADMEADOW)
    teams = [Team(name=n, club=club, grade='3rd') for n in (T0, T1, A, B)]
    grade_objs = [Grade(name='3rd', teams=[T0, T1, A, B])]
    timeslots = [
        Timeslot(date='2026-03-22', day='Sunday', time=f'{9 + s}:00',
                 week=1, day_slot=s, field=field, round_no=1)
        for s in (1, 2) for field in fields
    ]
    model, X = create_model_and_vars(GAMES, timeslots)
    data = {
        'games': GAMES, 'timeslots': timeslots, 'teams': teams,
        'grades': grade_objs, 'clubs': [club], 'fields': fields,
        'current_week': 0, 'locked_weeks': set(),
        'num_rounds': {'3rd': 1, 'max': 1},
        'constraint_slack': {}, 'penalty_weights': {}, 'penalties': {},
        'forced_games': [], 'blocked_games': [], 'team_conflicts': conflicts,
        'phl_preferences': {}, 'club_days': {}, 'preference_no_play': {},
        'home_field_map': {}, 'constraint_defaults': {},
    }
    return model, X, data


def _engine(model, X, data):
    eng = UnifiedConstraintEngine(model, X, data)
    eng.build_groupings()
    return eng


def _pin_slot(model, X, t1, t2, slot):
    """Force matchup (t1,t2) to play in day_slot `slot` (any field). Vars outside
    that slot are forced to 0; exactly one var at that slot must be 1 (the solver
    picks the field when more than one exists)."""
    at_slot = [v for k, v in X.items() if {k[0], k[1]} == {t1, t2} and k[4] == slot]
    for k, v in X.items():
        if {k[0], k[1]} == {t1, t2} and k[4] != slot:
            model.Add(v == 0)
    model.Add(sum(at_slot) == 1)


def _solve(model):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    return solver, solver.Solve(model)


# --------------------------------------------------------------------------
# (0) The live 2026 config carries the new penalty weight.
# --------------------------------------------------------------------------

def test_2026_config_declares_team_conflict_penalty_weight():
    """The real config gives TeamConflict a high weight (spec-033 DoD 4 = 200_000).
    This is what the engine reads via `_get_penalty_weight('TeamConflict', ...)`."""
    data = load_season_data(2026)
    weight = data['penalty_weights']['TeamConflict']
    assert weight == 200_000, f"expected TeamConflict weight 200_000, got {weight}"


# --------------------------------------------------------------------------
# (1) No hard component: same-slot conflict is FEASIBLE + exactly 1 penalty.
# --------------------------------------------------------------------------

def test_same_slot_conflict_is_feasible_and_one_penalty():
    """Both games pinned into (week 1, day_slot 1). The old HARD rule would make
    this INFEASIBLE; the soft rule keeps it FEASIBLE and records exactly 1 penalty
    unit (p >= 1+1-1 = 1)."""
    model, X, data = _fixture([CONFLICT_PAIR])
    eng = _engine(model, X, data)
    n = eng._team_conflict_soft()
    # One conflict pair; BOTH day_slots (1 and 2) are slots where both teams could
    # appear (each game has a var at each slot), so the method emits ONE penalty
    # var per slot => 2 penalty vars total.
    assert n == 2, f"expected exactly 2 penalty vars emitted (one per slot), got {n}"
    assert 'TeamConflict' in data['penalties']

    _pin_slot(model, X, T0, A, 1)   # G1 -> slot 1
    _pin_slot(model, X, B, T1, 1)   # G2 -> slot 1  (same slot as G1)

    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
        "same-slot conflict must remain FEASIBLE — TeamConflict is no longer hard"
    )
    penalties = data['penalties']['TeamConflict']['penalties']
    total = sum(solver.Value(p) for p in penalties)
    assert total == 1, f"expected exactly 1 TeamConflict penalty unit, got {total}"


# --------------------------------------------------------------------------
# (2) Different slots: feasible + ZERO penalty.
# --------------------------------------------------------------------------

def test_different_slots_conflict_incurs_zero_penalty():
    """G1 in slot 1, G2 in slot 2. No slot holds both conflicting teams, so every
    penalty `p >= sum(v1)+sum(v2)-1` has RHS <= 0 and the bucket sums to 0."""
    model, X, data = _fixture([CONFLICT_PAIR])
    eng = _engine(model, X, data)
    eng._team_conflict_soft()

    _pin_slot(model, X, T0, A, 1)   # G1 -> slot 1
    _pin_slot(model, X, B, T1, 2)   # G2 -> slot 2  (different slot)

    solver, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    penalties = data['penalties']['TeamConflict']['penalties']
    total = sum(solver.Value(p) for p in penalties)
    assert total == 0, f"expected 0 TeamConflict penalty units, got {total}"


# --------------------------------------------------------------------------
# (3) Empty team_conflicts => zero penalty vars (bucket still created).
# --------------------------------------------------------------------------

def test_empty_team_conflicts_emits_zero_penalty_vars():
    """With no declared conflicts the soft method returns 0 and the bucket is
    created with an empty penalties list."""
    model, X, data = _fixture([])
    eng = _engine(model, X, data)
    n = eng._team_conflict_soft()
    assert n == 0, f"expected 0 penalty vars for empty conflicts, got {n}"
    assert data['penalties']['TeamConflict']['penalties'] == []


# --------------------------------------------------------------------------
# (4) Stage 1 (hard) adds NO team-conflict constraint.
# --------------------------------------------------------------------------

def test_stage_1_hard_does_not_constrain_team_conflict():
    """apply_stage_1_hard must NOT add any `<= 1` team-conflict constraint: even
    with a conflict declared and both games forced onto the SAME (week, day_slot),
    applying ONLY the hard stage leaves the model FEASIBLE. We use TWO fields so
    the two games can share a day_slot on different fields without tripping the
    (unrelated) NoDoubleBookingFields rule; num_rounds=1 so EqualGames is satisfied
    with one game per team. The only feasibility loss for this layout would have
    been the now-removed hard team-conflict rule."""
    model, X, data = _fixture([CONFLICT_PAIR], num_fields=2)
    eng = _engine(model, X, data)
    eng.apply_stage_1_hard()
    # No TeamConflict bucket is created by the hard stage.
    assert 'TeamConflict' not in data['penalties']
    _pin_slot(model, X, T0, A, 1)
    _pin_slot(model, X, B, T1, 1)
    _, status = _solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE), (
        "hard stage must not block a same-slot conflict (rule is now soft)"
    )
