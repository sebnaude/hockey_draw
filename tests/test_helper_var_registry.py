"""Tests for HelperVarRegistry — both declarative and pool-style APIs."""
import pytest
from ortools.sat.python import cp_model

from constraints.helper_vars import HelperVarRegistry, SharedVariablePool, HelperVar


@pytest.fixture
def model():
    return cp_model.CpModel()


@pytest.fixture
def registry(model):
    return HelperVarRegistry(model)


# ---------- Declarative API ----------

def test_declare_and_get_returns_built_var(model, registry):
    sentinel = model.NewBoolVar('sentinel')

    def builder(m, X, data):
        return sentinel

    registry.declare('is_slot_used', (1, 'Sunday', 'NIHC', 1), builder, 'a slot')
    registry.freeze(X={}, data={})
    assert registry.get_declared('is_slot_used', (1, 'Sunday', 'NIHC', 1)) is sentinel


def test_redeclare_same_kind_and_key_is_noop(model, registry):
    builder1 = lambda m, X, d: model.NewBoolVar('first')
    builder2 = lambda m, X, d: model.NewBoolVar('second')

    registry.declare('k', (1,), builder1)
    registry.declare('k', (1,), builder2)  # ignored
    assert registry.declared_count() == 1
    assert registry.diagnostics()['redeclared_same_kind'] == 1


def test_declare_after_freeze_raises(model, registry):
    registry.declare('k', (1,), lambda m, X, d: model.NewBoolVar('a'))
    registry.freeze(X={}, data={})
    with pytest.raises(RuntimeError, match='frozen'):
        registry.declare('k', (2,), lambda m, X, d: model.NewBoolVar('b'))


def test_get_declared_for_undeclared_raises(registry):
    with pytest.raises(KeyError, match='not declared'):
        registry.get_declared('nonexistent_kind', (1, 2))


def test_freeze_is_idempotent(model, registry):
    calls = []

    def builder(m, X, d):
        v = m.NewBoolVar('once')
        calls.append(v)
        return v

    registry.declare('k', (1,), builder)
    registry.freeze(X={}, data={})
    registry.freeze(X={}, data={})
    assert len(calls) == 1


def test_declared_kinds_listing(model, registry):
    registry.declare('alpha', (1,), lambda m, X, d: model.NewBoolVar('a'))
    registry.declare('alpha', (2,), lambda m, X, d: model.NewBoolVar('a2'))
    registry.declare('beta', (1,), lambda m, X, d: model.NewBoolVar('b'))
    assert registry.declared_kinds() == ['alpha', 'beta']
    assert registry.declared_count('alpha') == 2
    assert registry.declared_count('beta') == 1


def test_distinct_kinds_with_same_key_get_separate_helpers(model, registry):
    a = model.NewBoolVar('a')
    b = model.NewBoolVar('b')
    registry.declare('alpha', (1,), lambda m, X, d: a)
    registry.declare('beta', (1,), lambda m, X, d: b)
    registry.freeze(X={}, data={})
    assert registry.get_declared('alpha', (1,)) is a
    assert registry.get_declared('beta', (1,)) is b


# ---------- Pool-style API (legacy compat) ----------

def test_pool_get_or_create_bool_dedups(model, registry):
    v = model.NewBoolVar('underlying')
    ind1 = registry.get_or_create_bool(('k', 1), [v], 'lbl1')
    ind2 = registry.get_or_create_bool(('k', 1), [v], 'lbl2')  # cached
    assert ind1 is ind2
    diag = registry.diagnostics()
    assert diag['pool_created'] == 1
    assert diag['pool_hits'] == 1


def test_pool_get_or_create_bool_with_empty_list_forces_zero(model, registry):
    ind = registry.get_or_create_bool(('k', 'empty'), [], 'lbl')
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert solver.Value(ind) == 0


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


# ---------- Backward-compat alias ----------

def test_shared_variable_pool_alias_is_registry():
    assert SharedVariablePool is HelperVarRegistry


# ---------- Diagnostics ----------

def test_diagnostics_shape(model, registry):
    registry.declare('k', (1,), lambda m, X, d: model.NewBoolVar('a'))
    registry.get_or_create_bool(('p', 1), [], 'p1')
    diag = registry.diagnostics()
    for key in ('declared', 'pool_created', 'pool_hits', 'redeclared_same_kind',
                'pool_size', 'declared_total', 'declared_kinds', 'frozen'):
        assert key in diag
    assert diag['declared'] == 1
    assert diag['pool_created'] == 1
    assert diag['frozen'] is False


def test_helper_var_dataclass_fields():
    spec = HelperVar(kind='k', key=(1,), builder=lambda m, X, d: None, description='d')
    assert spec.kind == 'k'
    assert spec.key == (1,)
    assert spec.description == 'd'
    assert spec._built is False
