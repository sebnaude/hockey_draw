"""spec-033 Unit E — ClubNoConcurrentSlot soft+slack, REAL-DATA build/dispatch.

No mocks/patches/monkeypatch: real `load_season_data(2026)`, a real CP-SAT model,
real `generate_X`, and a direct atom `apply()`. Confirms:
  - the atom populates `data['penalties']['ClubNoConcurrentSlot']` (the soft bucket
    that the staged objective reads — the atom is non-engine so `--simple` does NOT
    dispatch it; this direct apply is the build/dispatch smoke), and
  - the hard cap constraints are added to the model.

The unit-level pigeonhole/penalty hand-oracles (2 games/1 slot, 3 games/2 slots)
are in `test_club_no_concurrent_slot.py`.
"""
from __future__ import annotations

from ortools.sat.python import cp_model

from config import load_season_data
from constraints.atoms import ClubNoConcurrentSlot
from constraints.helper_vars import HelperVarRegistry
from utils import generate_X


def test_real_2026_atom_populates_soft_bucket_and_adds_hard_caps():
    data = load_season_data(2026)
    # Ensure the slack key + penalty weight are present as the production path
    # would supply them (PENALTY_WEIGHTS already carries the weight; slack default 0).
    data.setdefault('constraint_slack', {})
    data['constraint_slack'].setdefault('ClubNoConcurrentSlot', 0)
    data.setdefault('penalties', {})

    model = cp_model.CpModel()
    X, _Y = generate_X(model, data)
    assert X, "generate_X produced no variables on real 2026 data"

    n_before = len(model.Proto().constraints)
    n_caps = ClubNoConcurrentSlot().apply(model, X, data, HelperVarRegistry(model))
    n_after = len(model.Proto().constraints)

    # The atom added hard cap constraints (and channelling for the over IntVars).
    assert n_caps > 0, "expected at least one (club, slot) overlap cap on real data"
    assert n_after > n_before, "atom added no constraints to the model"

    # The soft bucket reaches the objective via data['penalties'] — confirm it is
    # populated with the configured weight and at least one `over` penalty var.
    bucket = data['penalties'].get('ClubNoConcurrentSlot')
    assert bucket is not None, "atom did not write data['penalties']['ClubNoConcurrentSlot']"
    assert bucket['weight'] == 200_000, f"unexpected weight {bucket['weight']}"
    assert len(bucket['penalties']) > 0, "no soft `over` penalty vars emitted"
    # One penalty var per capped slot (each cap with >=2 vars also emits one over).
    assert len(bucket['penalties']) == n_caps
