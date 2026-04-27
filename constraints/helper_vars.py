"""
Helper-variable registry for the constraint engine.

Atoms declare the helper variables they need (BoolVars, IntVars, indicators built
from MaxEquality / BoolOr / BoolAnd) via the registry instead of hand-rolling them.
The registry deduplicates by `(kind, key)` so multiple atoms asking for the same
helper share one variable.

Two usage modes are supported:

1. Declarative (atoms in Phase 3+):
    registry.declare('is_slot_used', (week, day, location, slot),
                     builder=lambda m, x, d: <build>, description='...')
    ...all atoms declare()...
    registry.freeze(model, X, data)   # builds every declared helper once
    var = registry.get('is_slot_used', key)

2. Lazy / pool-style (legacy `unified.py` engine):
    var = registry.get_or_create_bool(key, vars_list, label)
    var = registry.get_or_create_presence(key, vars_list, label)
    registry.register(key, var)
    registry.lookup(key)   # alias for the pool's `get(key)`

The pool-style API exists so the existing `UnifiedConstraintEngine`, which
already centralised all helpers via `SharedVariablePool`, continues to work
without rewriting its internals — atoms get a cleaner API without forcing a
big-bang migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


HelperKey = Tuple[Any, ...]
Builder = Callable[[Any, Dict, Dict], Any]


@dataclass
class HelperVar:
    """Declarative spec for one helper variable.

    `kind` is a stable string discriminator (e.g. 'is_slot_used', 'pair_plays_in_week').
    `key` is the tuple uniquely identifying this instance within the kind.
    `builder` is called once at freeze: builder(model, X, data) -> Var.
    """
    kind: str
    key: HelperKey
    builder: Builder
    description: str = ''
    _var: Any = field(default=None, repr=False)
    _built: bool = field(default=False, repr=False)


class HelperVarRegistry:
    """Registry + cache for shared helper variables.

    Atoms can declare what they need before solving; the engine builds each
    declared helper once at freeze time. Lazy / pool-style methods coexist for
    legacy code paths.
    """

    def __init__(self, model):
        self.model = model
        self._declared: Dict[Tuple[str, HelperKey], HelperVar] = {}
        self._cache: Dict[HelperKey, Any] = {}
        self._frozen = False
        self._stats = {
            'declared': 0,
            'redeclared_same_kind': 0,
            'pool_created': 0,
            'pool_hits': 0,
        }

    # ------------------------------------------------------------------
    # Declarative API (used by atoms in Phase 3)
    # ------------------------------------------------------------------

    def declare(
        self,
        kind: str,
        key: HelperKey,
        builder: Builder,
        description: str = '',
    ) -> None:
        """Register intent to build a helper var. Idempotent for the same (kind, key).

        Re-declaring with the same (kind, key) is a no-op (the first builder wins).
        Declaring with a different `kind` for the same `key` shape is allowed
        because (kind, key) is the registry key.
        """
        if self._frozen:
            raise RuntimeError(
                f"HelperVarRegistry is frozen — cannot declare {kind!r} key={key!r}. "
                "Call registry.declare() in atom.declare_helpers(), not apply()."
            )
        registry_key = (kind, key)
        if registry_key in self._declared:
            self._stats['redeclared_same_kind'] += 1
            return
        self._declared[registry_key] = HelperVar(
            kind=kind, key=key, builder=builder, description=description
        )
        self._stats['declared'] += 1

    def freeze(self, X: Dict, data: Dict) -> None:
        """Build every declared helper exactly once. Idempotent."""
        if self._frozen:
            return
        for spec in self._declared.values():
            if not spec._built:
                spec._var = spec.builder(self.model, X, data)
                spec._built = True
        self._frozen = True

    def get_declared(self, kind: str, key: HelperKey) -> Any:
        """Look up a built helper by (kind, key). Raises if not declared."""
        registry_key = (kind, key)
        spec = self._declared.get(registry_key)
        if spec is None:
            raise KeyError(
                f"HelperVarRegistry.get_declared({kind!r}, {key!r}): not declared. "
                "Make sure the atom declared this helper in declare_helpers()."
            )
        if not spec._built:
            if self._frozen:
                raise RuntimeError(
                    f"HelperVar({kind!r}, {key!r}) not built — registry frozen "
                    "without building it. This is a registry bug."
                )
            spec._var = spec.builder(self.model, X=None, data=None)
            spec._built = True
        return spec._var

    def declared_kinds(self) -> List[str]:
        """List every distinct kind currently declared (for introspection)."""
        return sorted({spec.kind for spec in self._declared.values()})

    def declared_count(self, kind: Optional[str] = None) -> int:
        if kind is None:
            return len(self._declared)
        return sum(1 for spec in self._declared.values() if spec.kind == kind)

    # ------------------------------------------------------------------
    # Lazy / pool-style API (used by legacy UnifiedConstraintEngine)
    # ------------------------------------------------------------------

    def get_or_create_bool(self, key: HelperKey, vars_list: List, label: str):
        """Cached BoolVar with AddMaxEquality channeling.

        If `vars_list` is non-empty, channels `indicator = max(vars_list)`.
        If empty, forces `indicator == 0`. Cache key is just the raw `key` —
        the pool path does not use kind discriminators.
        """
        if key in self._cache:
            self._stats['pool_hits'] += 1
            return self._cache[key]
        self._stats['pool_created'] += 1
        ind = self.model.NewBoolVar(label)
        if vars_list:
            self.model.AddMaxEquality(ind, vars_list)
        else:
            self.model.Add(ind == 0)
        self._cache[key] = ind
        return ind

    def get_or_create_presence(self, key: HelperKey, vars_list: List, label: str):
        """Cached BoolVar with BoolOr/BoolAnd channeling (bidirectional)."""
        if key in self._cache:
            self._stats['pool_hits'] += 1
            return self._cache[key]
        self._stats['pool_created'] += 1
        ind = self.model.NewBoolVar(label)
        self.model.AddBoolOr(vars_list).OnlyEnforceIf(ind)
        self.model.AddBoolAnd([v.Not() for v in vars_list]).OnlyEnforceIf(ind.Not())
        self._cache[key] = ind
        return ind

    def register(self, key: HelperKey, var):
        """Cache a manually-channeled variable (caller adds own constraints)."""
        self._cache[key] = var
        return var

    def lookup(self, key: HelperKey):
        """Pool-style lookup. Returns None if not cached. Does not touch declarative API."""
        return self._cache.get(key)

    def get(self, key: HelperKey):
        """Pool-style lookup (back-compat alias for `lookup`). Returns None if not cached."""
        return self._cache.get(key)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> Dict[str, Any]:
        return {
            **self._stats,
            # Back-compat aliases for the old SharedVariablePool diagnostics shape
            'created': self._stats['pool_created'],
            'hits': self._stats['pool_hits'],
            'pool_size': len(self._cache),
            'declared_total': len(self._declared),
            'declared_kinds': self.declared_kinds(),
            'frozen': self._frozen,
        }


# ----------------------------------------------------------------------
# Backwards-compatible alias for code that still imports SharedVariablePool
# ----------------------------------------------------------------------

SharedVariablePool = HelperVarRegistry
