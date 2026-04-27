"""Base class for atomic constraints."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional


# Venue-name constants. Atoms compare against these instead of hardcoding strings.
BROADMEADOW = 'Newcastle International Hockey Centre'
GOSFORD = 'Central Coast Hockey Park'
MAITLAND = 'Maitland Park'


class Atom(ABC):
    """One idea, one constraint atom.

    Subclasses set `canonical_name` (matches a `ConstraintInfo` entry) and
    `atom_group` (the legacy combined-constraint name they were split from).

    Lifecycle (driven by `UnifiedConstraintEngine`):
      1. `declare_helpers(registry, data)` — declare helper-vars needed.
      2. `registry.freeze(X, data)` — engine builds every helper once.
      3. `apply(model, X, data, registry)` — add CP-SAT constraints to model.
    """

    canonical_name: ClassVar[str] = ''
    atom_group: ClassVar[str] = ''

    def declare_helpers(self, registry, data: Dict) -> None:
        """Override to declare helpers via `registry.declare(...)`. Default: no helpers."""
        return

    @abstractmethod
    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        """Add constraints to `model`. Return count of constraints added."""
        ...


# ----------------------------------------------------------------------
# Shared helpers used across PHL atoms.
# ----------------------------------------------------------------------


def get_team_club_map(data: Dict) -> Dict[str, str]:
    """Cached `team_name -> club_name` map. Stored at `data['_team_club_map']`."""
    cache = data.get('_team_club_map')
    if cache is None:
        cache = {team.name: team.club.name for team in data['teams']}
        data['_team_club_map'] = cache
    return cache


def iter_phl_keys(X: Dict, data: Dict, *, include_locked: bool = False):
    """Yield (key, var) for every PHL real-game variable in X.

    Skips dummy keys (len < 11) and variables with no day. By default skips
    locked-week vars to match `UnifiedConstraintEngine` build_groupings, which
    excludes locked weeks from constraint groupings.
    """
    locked_weeks = set() if include_locked else set(data.get('locked_weeks', set()))
    for key, var in X.items():
        if len(key) < 11:
            continue
        if key[2] != 'PHL':
            continue
        if not key[3]:
            continue
        if locked_weeks and key[6] in locked_weeks:
            continue
        yield key, var


def iter_grade_keys(X: Dict, data: Dict, grade: str, *, include_locked: bool = False):
    """Yield (key, var) for every real-game variable in X matching `grade`."""
    locked_weeks = set() if include_locked else set(data.get('locked_weeks', set()))
    for key, var in X.items():
        if len(key) < 11:
            continue
        if key[2] != grade:
            continue
        if not key[3]:
            continue
        if locked_weeks and key[6] in locked_weeks:
            continue
        yield key, var
