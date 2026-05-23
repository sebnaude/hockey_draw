"""Tests for HelperVarRegistry — the single pool-style pathway + single-pathway guard.

spec-022 removed the vestigial declarative API (declare/freeze/get_declared/
HelperVar). These tests cover the surviving pool-style API, assert the
single-pathway invariant so a second store can't silently reappear, and lightly
guard the `(kind, *discriminators)` pool-key convention.
"""
import pytest
from ortools.sat.python import cp_model

from constraints.helper_vars import HelperVarRegistry, SharedVariablePool
from constraints.registry import CONSTRAINT_REGISTRY, HELPER_VAR_CATALOG
from constraints.atoms.base import Atom


@pytest.fixture
def model():
    return cp_model.CpModel()


@pytest.fixture
def registry(model):
    return HelperVarRegistry(model)


# ---------- Pool-style API ----------

def test_pool_get_or_create_bool_dedups(model, registry):
    # Given one underlying var and two requests with the SAME pool key,
    # When both call get_or_create_bool,
    # Then the second is a cache hit returning the identical BoolVar.
    v = model.NewBoolVar('underlying')
    ind1 = registry.get_or_create_bool(('k', 1), [v], 'lbl1')
    ind2 = registry.get_or_create_bool(('k', 1), [v], 'lbl2')  # cached
    assert ind1 is ind2
    diag = registry.diagnostics()
    # Hand oracle: exactly one var created, exactly one cache hit.
    assert diag['pool_created'] == 1
    assert diag['pool_hits'] == 1


def test_pool_get_or_create_bool_with_empty_list_forces_zero(model, registry):
    # Given an empty vars_list,
    # When get_or_create_bool builds the indicator,
    # Then it is hard-pinned to 0 (max of nothing is 0).
    ind = registry.get_or_create_bool(('k', 'empty'), [], 'lbl')
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(ind) == 0


def test_pool_get_or_create_bool_channels_max(model, registry):
    # Given two underlying vars, one forced 1 and one forced 0,
    # When the indicator is channelled as their max,
    # Then the indicator solves to 1 (max(1, 0) == 1).
    a = model.NewBoolVar('a')
    b = model.NewBoolVar('b')
    model.Add(a == 1)
    model.Add(b == 0)
    ind = registry.get_or_create_bool(('m', 1), [a, b], 'ind')
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(ind) == 1


def test_pool_get_returns_none_for_missing(registry):
    assert registry.get(('nope',)) is None
    assert registry.lookup(('nope',)) is None


def test_pool_register_and_lookup(model, registry):
    v = model.NewBoolVar('v')
    registry.register(('k',), v)
    assert registry.lookup(('k',)) is v
    assert registry.get(('k',)) is v


def test_pool_get_or_create_presence_dedups(model, registry):
    v = model.NewBoolVar('underlying')
    ind1 = registry.get_or_create_presence(('p', 1), [v], 'lbl')
    ind2 = registry.get_or_create_presence(('p', 1), [v], 'lbl2')
    assert ind1 is ind2


def test_pool_get_or_create_presence_channels_or(model, registry):
    # Given two underlying vars,
    # When the presence indicator is channelled (OR / AND-of-Nots),
    # Then forcing the indicator to 1 forces at least one underlying to 1.
    a = model.NewBoolVar('a')
    b = model.NewBoolVar('b')
    ind = registry.get_or_create_presence(('q', 1), [a, b], 'ind')
    model.Add(ind == 1)
    model.Add(a == 0)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    # a is pinned 0, so b must be 1 for the OR to hold under ind==1.
    assert solver.Value(b) == 1


# ---------- Backward-compat alias ----------

def test_shared_variable_pool_alias_is_registry():
    assert SharedVariablePool is HelperVarRegistry


# ---------- Diagnostics ----------

def test_diagnostics_shape_is_pool_only(model, registry):
    # Given one created helper (a hit follows on the same key),
    # When we read diagnostics,
    # Then only pool keys are present — no declarative keys survive.
    registry.get_or_create_bool(('p', 1), [], 'p1')
    registry.get_or_create_bool(('p', 1), [], 'p1again')  # hit
    diag = registry.diagnostics()
    assert set(diag.keys()) == {
        'pool_created', 'pool_hits', 'created', 'hits', 'pool_size',
    }
    # Hand oracle: 1 created, 1 hit, cache holds 1 entry.
    assert diag['pool_created'] == 1
    assert diag['pool_hits'] == 1
    assert diag['created'] == 1
    assert diag['hits'] == 1
    assert diag['pool_size'] == 1
    # Declarative diagnostics must be gone.
    for dead in ('declared', 'declared_total', 'declared_kinds',
                 'redeclared_same_kind', 'frozen'):
        assert dead not in diag


# ---------- Single-pathway guard (spec-022 DoD 4) ----------

def test_registry_has_no_declarative_api():
    # A future contributor reintroducing the declarative API trips this.
    for attr in ('declare', 'freeze', 'get_declared', 'declared_kinds',
                 'declared_count', '_declared', '_frozen'):
        assert not hasattr(HelperVarRegistry, attr), (
            f"declarative API attribute {attr!r} reappeared on HelperVarRegistry"
        )


def test_helper_var_dataclass_removed():
    import constraints.helper_vars as hv
    assert not hasattr(hv, 'HelperVar'), "HelperVar dataclass should be deleted"


def test_atom_has_no_declare_helpers():
    assert not hasattr(Atom, 'declare_helpers'), (
        "Atom.declare_helpers reappeared — atoms must use the pool API in apply()"
    )


# ---------- Key-convention guard (spec-022 DoD 6) ----------

def test_required_helper_kinds_are_nonempty_catalog_strings():
    # Every kind a constraint declares it requires must be a non-empty string
    # that exists in HELPER_VAR_CATALOG — keeps the (kind, *discriminators)
    # pool-key convention meaningful.
    for name, info in CONSTRAINT_REGISTRY.items():
        for kind in info.required_helpers:
            assert isinstance(kind, str) and kind, (
                f"{name}: required_helpers kind {kind!r} is not a non-empty string"
            )
            assert kind in HELPER_VAR_CATALOG, (
                f"{name}: required_helpers kind {kind!r} not in HELPER_VAR_CATALOG"
            )
