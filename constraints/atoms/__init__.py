"""Atomic constraints — one idea per class.

Each atom subclasses `Atom` (constraints/atoms/base.py) and:
- `declare_helpers(registry, data)` declares helper-vars it needs (default: none).
- `apply(model, X, data, registry)` adds CP-SAT constraints, returns count added.

Atoms are instantiated and dispatched by `UnifiedConstraintEngine`. Their
`canonical_name` matches a `ConstraintInfo` entry in `constraints/registry.py`
(with `atom_group` set to the legacy combined-constraint they were split from).
"""
from constraints.atoms.base import Atom
from constraints.atoms.phl_concurrency import PHLConcurrencyAtBroadmeadow
from constraints.atoms.phl_2nd_concurrency import PHLAnd2ndConcurrencyAtBroadmeadow
from constraints.atoms.gosford_friday_rounds import GosfordFridayRoundsForced
from constraints.atoms.phl_round_one_play import PHLRoundOnePlay
from constraints.atoms.preferred_dates import PreferredDates


PHL_TIMES_ATOMS = [
    PHLConcurrencyAtBroadmeadow,
    PHLAnd2ndConcurrencyAtBroadmeadow,
    GosfordFridayRoundsForced,
    PHLRoundOnePlay,
    PreferredDates,
]


__all__ = [
    'Atom',
    'PHLConcurrencyAtBroadmeadow',
    'PHLAnd2ndConcurrencyAtBroadmeadow',
    'GosfordFridayRoundsForced',
    'PHLRoundOnePlay',
    'PreferredDates',
    'PHL_TIMES_ATOMS',
]
