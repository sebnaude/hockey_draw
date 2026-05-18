"""Soft penalty for scheduling (or not scheduling) games at an away ground on specific dates.

## Design decision (spec-006)

**Why a new `PREFERRED_WEEKENDS` config list and NOT an extension of FORCED_GAMES:**

`FORCED_GAMES` entries are processed in `generate_X()` (utils.py) to *eliminate decision
variables* from the model.  Every entry there participates in variable-elimination logic
and the `validate_game_config` pre-solver check.  Adding a `soft: true` flag to that
structure would require branching the variable-elimination pipeline and all its validators
to skip soft entries — turning a structural list into a mixed-mode list.

A separate `PREFERRED_WEEKENDS` list keeps concerns cleanly separated:
  - `FORCED_GAMES` / `BLOCKED_GAMES` → variable elimination at model-build time.
  - `PREFERRED_WEEKENDS` → soft objective penalties at constraint-application time.

This atom reads only from `data['preferred_weekends']` (set by the season config).
Nothing in `generate_X()` changes.

## Entry format

Each entry in `PREFERRED_WEEKENDS` is a dict:

    {
        'date':           '2026-04-05',    # single date string YYYY-MM-DD
        'dates':          [...],           # OR a list of date strings (exclusive with 'date')
        'field_location': 'Maitland Park', # venue string (must match field_location in X keys)
        'field_name':     'Maitland Main Field',  # optional — venue-level if omitted
        'mode':           'prefer' | 'avoid',     # required
        'weight':         1000,           # optional — falls back to PENALTY_WEIGHTS default;
                                          # used as a per-entry multiplier on top of the
                                          # bucket-level default weight (multiplier =
                                          # max(1, entry_weight // default_weight)).
        'target_count':   1,              # optional — prefer-mode target. Default 1.
        'description':    '...',          # human-readable, ignored by solver
    }

## Penalty semantics

The atom stores **unscaled raw penalty IntVars** in the
`preferred_weekends_away_ground` penalty bucket. The bucket weight is the
default from `PENALTY_WEIGHTS`. `main_staged._build_normalized_penalty`
normalises by var count: per-var coefficient = `bucket_weight // n_vars`.

- **avoid** entry: raw_penalty = games_at_venue_on_date
  → solver discourages scheduling ANY games at that venue on that date.
  → If `field_name` is set, only games on that specific field count.

- **prefer** entry: raw_penalty = max(0, target_count - games_at_venue_on_date)
  → solver is penalised for NOT having games there. `target_count` defaults to 1.

A per-entry `weight` different from the default is encoded as a positive integer
multiplier applied on top of the raw IntVar (matching the pattern used by
`TeamPairNoConcurrency`), so a higher per-entry weight gives that entry's
violations proportionally more pull on the objective.

Conflicting entries (prefer + avoid on same date/venue): no crash — penalties
accumulate independently from both directions. The solver finds the
least-total-penalty assignment.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from constraints.atoms.base import Atom


class PreferredWeekendsAwayGround(Atom):
    """Soft penalty: prefer / avoid scheduling at a specific away ground on specific dates."""

    canonical_name = 'PreferredWeekendsAwayGround'
    atom_group = ''

    # Default target count for 'prefer' entries when not explicitly set.
    _DEFAULT_PREFER_TARGET = 1

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        entries: List[Dict] = data.get('preferred_weekends', [])
        if not entries:
            return 0

        default_weight = data.get('penalty_weights', {}).get(
            'preferred_weekends_away_ground', 1000
        )
        if default_weight <= 0:
            # Disabled — don't create any penalty vars or bucket.
            return 0

        locked_weeks = set(data.get('locked_weeks', set()))

        # Normalise entries: expand 'dates' lists, resolve weight, validate mode.
        normalised = _normalise_entries(entries, default_weight)
        if not normalised:
            return 0

        # Build index: (date_str, field_location, field_name_or_None) -> list[var]
        # field_name_or_None: None = venue-level (any field at location).
        # We populate BOTH the venue-level key (None) and the field-level key
        # for every var so lookups by either granularity hit the same vars.
        venue_date_vars: Dict[tuple, list] = defaultdict(list)
        for key, var in X.items():
            if len(key) < 11:
                continue  # dummy key
            if not key[3]:
                continue  # no day
            week = key[6]
            if locked_weeks and week in locked_weeks:
                continue
            date_str: str = key[7]
            field_name: str = key[9]
            field_location: str = key[10]
            venue_date_vars[(date_str, field_location, None)].append(var)
            venue_date_vars[(date_str, field_location, field_name)].append(var)

        data.setdefault('penalties', {})
        bucket = data['penalties'].setdefault(
            'preferred_weekends_away_ground',
            {'weight': default_weight, 'penalties': []},
        )

        n = 0
        for entry_idx, norm in enumerate(normalised):
            date_str: str = norm['date']
            field_location: str = norm['field_location']
            field_name: Optional[str] = norm.get('field_name')  # None = venue-level
            mode: str = norm['mode']
            entry_weight: int = norm['weight']
            target: int = norm.get('target_count', self._DEFAULT_PREFER_TARGET)

            # Per-entry weight multiplier on top of the bucket's default weight.
            # Same pattern as TeamPairNoConcurrency.
            multiplier = max(1, entry_weight // default_weight)

            lookup_key = (date_str, field_location, field_name)
            vars_on_date = venue_date_vars.get(lookup_key, [])

            if not vars_on_date:
                if mode == 'prefer':
                    # No vars at all means we can never satisfy the preference.
                    # Represent the unsatisfiable shortage as a constant penalty so
                    # the operator sees a non-zero penalty in the report.
                    const_pen = model.NewConstant(target * multiplier)
                    bucket['penalties'].append(const_pen)
                    n += 1
                # avoid mode + no vars: nothing to penalise.
                continue

            actual_sum = sum(vars_on_date)
            max_games = len(vars_on_date)

            if mode == 'avoid':
                # raw = sum of game vars at venue on date.
                raw = model.NewIntVar(
                    0, max_games,
                    f'u_pref_avoid_{entry_idx}_{date_str}',
                )
                model.Add(raw == actual_sum)
                _append_scaled(model, bucket, raw, multiplier, entry_idx,
                               date_str, 'avoid', max_games)
                n += 1

            elif mode == 'prefer':
                # raw = max(0, target - actual_sum) — shortage clamped at 0.
                shortage = model.NewIntVar(
                    0, target,
                    f'u_pref_shortage_{entry_idx}_{date_str}',
                )
                model.AddMaxEquality(
                    shortage,
                    [target - actual_sum, model.NewConstant(0)],
                )
                _append_scaled(model, bucket, shortage, multiplier, entry_idx,
                               date_str, 'prefer', target)
                n += 1

        return n


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_entries(entries: List[Dict], default_weight: int) -> List[Dict]:
    """Expand multi-date entries; validate mode; resolve weight. Returns list of dicts."""
    normalised = []
    for entry in entries:
        mode = entry.get('mode', '')
        if mode not in ('prefer', 'avoid'):
            raise ValueError(
                f"PreferredWeekendsAwayGround: entry has invalid mode={mode!r}. "
                f"Must be 'prefer' or 'avoid'. Entry: {entry!r}"
            )
        if 'field_location' not in entry:
            raise ValueError(
                f"PreferredWeekendsAwayGround: entry missing 'field_location'. "
                f"Entry: {entry!r}"
            )
        # Resolve date(s)
        if 'date' in entry and 'dates' in entry:
            raise ValueError(
                f"PreferredWeekendsAwayGround: entry has both 'date' and 'dates'. "
                f"Use one or the other. Entry: {entry!r}"
            )
        if 'date' in entry:
            date_list = [entry['date']]
        elif 'dates' in entry:
            date_list = list(entry['dates'])
        else:
            raise ValueError(
                f"PreferredWeekendsAwayGround: entry missing 'date' or 'dates'. "
                f"Entry: {entry!r}"
            )
        weight = int(entry.get('weight', default_weight))
        if weight <= 0:
            raise ValueError(
                f"PreferredWeekendsAwayGround: entry weight must be positive; "
                f"got {weight!r}. Entry: {entry!r}"
            )
        for date_str in date_list:
            norm = {
                'date': date_str,
                'field_location': entry['field_location'],
                'mode': mode,
                'weight': weight,
            }
            if 'field_name' in entry:
                norm['field_name'] = entry['field_name']
            if 'target_count' in entry:
                tc = int(entry['target_count'])
                if tc <= 0:
                    raise ValueError(
                        f"PreferredWeekendsAwayGround: target_count must be positive; "
                        f"got {tc!r}. Entry: {entry!r}"
                    )
                norm['target_count'] = tc
            normalised.append(norm)
    return normalised


def _append_scaled(model, bucket: Dict, raw_var, multiplier: int,
                   entry_idx: int, date_str: str, mode: str, max_raw: int) -> None:
    """Append raw_var (or multiplier * raw_var) to bucket['penalties']."""
    if multiplier == 1:
        bucket['penalties'].append(raw_var)
        return
    scaled = model.NewIntVar(
        0, max_raw * multiplier,
        f'u_pref_{mode}_scaled_{entry_idx}_{date_str}',
    )
    model.Add(scaled == multiplier * raw_var)
    bucket['penalties'].append(scaled)
