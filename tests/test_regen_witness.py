"""spec-027 DoD-7 — infeasibility→feasibility witness.

A synthetic fixture that is INFEASIBLE under the full HARD rule (a frozen game
boxes a freed game's only feasible time, violating hard PHL/2nd adjacency)
becomes FEASIBLE under the regen treatment, with the adjacency penalty recorded
> 0. No mocks; real CP-SAT; the violation is hand-constructed so it is provable.

The witness isolates the ONE softened rule that drives the flip (PHL/2nd
adjacency): the same pinned arrangement is (a) rejected by the hard atom
`PHLAnd2ndAdjacency` (INFEASIBLE) and (b) accepted by its regen-soft analogue
`PHLAnd2ndAdjacencyRegenSoft` (FEASIBLE, penalty == 1). The full 32-constraint
`regen` group solving feasibly end-to-end is the post-merge DoD-10 verification
(a real season solve, too large for pytest).
"""
from ortools.sat.python import cp_model

from constraints.atoms.phl_2nd_adjacency import PHLAnd2ndAdjacency
from constraints.atoms.phl_2nd_adjacency_regen_soft import PHLAnd2ndAdjacencyRegenSoft
from constraints.helper_vars import HelperVarRegistry
import tests.atoms.conftest as cf


def _find_pair_key(X, team_a, team_b, grade, day_slot):
    """Return the week-1 Sunday EF key for the EXACT pairing {team_a, team_b} in
    `grade` at `day_slot`. We pin a specific pairing (not just 'any pairing with
    team X') because the adjacency atom buckets each game under BOTH teams'
    clubs — so to get exactly ONE violating club-weekend the two pinned games
    must share only ONE common club (Tigers) and have DISTINCT opponents."""
    want = {team_a, team_b}
    for key in X:
        t1, t2, g, day, slot, _time, week, _date, _rno, fname, _loc = key
        if (g == grade and week == 1 and day == 'Sunday' and slot == day_slot
                and fname == 'EF' and {t1, t2} == want):
            return key
    raise AssertionError(f"no key for {want} {grade} slot {day_slot}")


def _pin_violating_arrangement(model, X):
    """Pin Tigers-vs-Wests PHL @ EF slot 1 (11:30) and Tigers-vs-Norths 2nd @ EF
    slot 3 (14:30), week 1 — same venue (Broadmeadow), |1 - 3| = 2 slots apart →
    NOT adjacent, and same-venue so the >=180-min cross-venue escape never
    applies: exactly what the hard PHL/2nd adjacency rule forbids. Opponents are
    DISTINCT (Wests fields only PHL here, Norths only 2nd), so ONLY the Tigers
    club fields both grades → exactly one violating (club, week, day) weekend."""
    phl_key = _find_pair_key(X, 'Tigers PHL', 'Wests PHL', 'PHL', 1)
    second_key = _find_pair_key(X, 'Tigers 2nd', 'Norths 2nd', '2nd', 3)
    # Fully determine the assignment: these two games ON, every other game OFF.
    # That leaves EXACTLY one (club, week, day) weekend in play — Tigers/week-1
    # Sunday — so the adjacency penalty has a single, hand-provable value and the
    # solver has no freedom to introduce additional violations elsewhere.
    for key, var in X.items():
        if key in (phl_key, second_key):
            model.Add(var == 1)
        else:
            model.Add(var == 0)
    return phl_key, second_key


def test_hard_adjacency_makes_pinned_arrangement_infeasible():
    """GIVEN the pinned violating PHL/2nd arrangement,
    WHEN the HARD PHLAnd2ndAdjacency atom is applied,
    THEN the model is INFEASIBLE (the hard rule forbids the pinned games)."""
    data = cf._build_phl_fixture()
    model, X = cf.build_model_X(data)
    _pin_violating_arrangement(model, X)
    PHLAnd2ndAdjacency().apply(model, X, data, HelperVarRegistry(model))
    status, _ = cf.solve_with_timeout(model)
    assert status == cp_model.INFEASIBLE


def test_regen_soft_adjacency_makes_same_arrangement_feasible_with_penalty():
    """GIVEN the SAME pinned violating arrangement,
    WHEN the regen-soft PHLAnd2ndAdjacencyRegenSoft atom is applied instead,
    THEN the model is FEASIBLE and the adjacency penalty is exactly 1 (one
    violating (club, week, day) weekend) — the infeasible→feasible flip."""
    data = cf._build_phl_fixture()
    model, X = cf.build_model_X(data)
    _pin_violating_arrangement(model, X)
    PHLAnd2ndAdjacencyRegenSoft().apply(model, X, data, HelperVarRegistry(model))

    # No objective needed: the penalty BoolVar v for the Tigers/week-1 weekend is
    # PINNED to 1 by its defining constraints (v >= p + q - 1, with p = q = 1
    # forced), so it equals 1 in EVERY feasible assignment. A pure feasibility
    # solve is enough (and far faster than minimising over the whole model).
    bucket = data['penalties']['regen_phl_2nd_adjacency']
    status, solver = cf.solve_with_timeout(model, seconds=30.0)

    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    total = sum(solver.Value(v) for v in bucket['penalties'])
    # Hand oracle: exactly one (club, week, day) weekend violates adjacency
    # (Tigers, week 1, Sunday) → penalty == 1. > 0 proves the rule was breached
    # but honoured softly rather than blocking the solve (the INFEASIBLE→FEASIBLE
    # flip vs the hard atom in the test above).
    assert total == 1
