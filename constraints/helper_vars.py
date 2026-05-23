"""
Helper-variable registry for the constraint engine.

Atoms and the engine create shared helper variables (BoolVars built from
MaxEquality / BoolOr / BoolAnd) via the registry instead of hand-rolling them.
The registry deduplicates by pool key so multiple callers asking for the same
helper share one CP-SAT variable (no model bloat).

Usage (pool-style — the single pathway):

    var = registry.get_or_create_bool(key, vars_list, label)
    var = registry.get_or_create_presence(key, vars_list, label)
    registry.register(key, var)
    registry.lookup(key)   # alias for the pool's `get(key)`

Key convention: a shared helper's pool key is `(kind, *discriminators)` where
`kind` is one of the strings in `HELPER_VAR_CATALOG` (`constraints/registry.py`).
Keeping `kind` as the first element keeps the catalog meaningful and lets two
callers asking for the same logical helper land on the same cache entry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


HelperKey = Tuple[Any, ...]


class HelperVarRegistry:
    """Registry + cache for shared helper variables.

    A single pool-style cache keyed by the raw pool key. Callers ask for a
    helper via `get_or_create_*`; the first caller builds it and channels it,
    later callers with the same key get the cached variable back.
    """

    def __init__(self, model):
        self.model = model
        self._cache: Dict[HelperKey, Any] = {}
        self._stats = {
            'pool_created': 0,
            'pool_hits': 0,
        }

    # ------------------------------------------------------------------
    # Pool-style API
    # ------------------------------------------------------------------

    def get_or_create_bool(self, key: HelperKey, vars_list: List, label: str):
        """Cached BoolVar with AddMaxEquality channeling.

        If `vars_list` is non-empty, channels `indicator = max(vars_list)`.
        If empty, forces `indicator == 0`. Cache key is the raw `key`, by
        convention `(kind, *discriminators)`.
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
        """Pool-style lookup. Returns None if not cached."""
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
        }


# ----------------------------------------------------------------------
# Backwards-compatible alias for code that still imports SharedVariablePool
# ----------------------------------------------------------------------

SharedVariablePool = HelperVarRegistry
