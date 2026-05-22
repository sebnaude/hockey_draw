"""Soft, weighted analogue of the whole FORCED_GAMES grammar (spec-020).

## Design decision (spec-020)

`FORCED_GAMES` entries are processed in `generate_X()` (utils.py) to *eliminate
decision variables* and add a HARD `sum(vars) <op> count` constraint per scope.
`PREFERRED_GAMES` reuses the **exact same scope/team/club grammar** but applies a
penalty-on-deviation instead of a hard constraint — so a preference that can't be
met costs the solver objective rather than making the model infeasible.

This generalises the bespoke `PreferredDates` (PHL-only, `|sum − 1|` on a date)
and `PreferredWeekendsAwayGround` (venue/date prefer/avoid) special cases into ONE
mechanism covering the full FORCED grammar.

This atom reads only from `data['preferred_games']` (set by the season config).
Nothing in `generate_X()` changes — preferences touch zero variables.

## Entry format

Same as FORCED_GAMES, **plus** an optional `weight`:

    {
        'grade':  'PHL',                # any FORCED scope field (grade/grades, day,
        'date':   '2026-04-19',         #   day_slot, time, week, date, round_no,
        ...                             #   field_name, field_location)
        'teams':  ['Maitland', 'Norths'],   # optional team/club matchers (FORCED grammar)
        'constraint': 'equal',          # equal | lesse | less | greater | greatere
        'count':  1,                    # target N (default 1)
        'weight': 10000,                # optional per-entry multiplier (see below)
        'description': '...',           # human-readable, ignored by solver
    }

## Penalty semantics — deviation from `count` (= N) per constraint type

    equal     (== N)   penalty = |sum − N|                (two-sided; AddAbsEquality)
    lesse     (<= N)   penalty = max(0, sum − N)
    greatere  (>= N)   penalty = max(0, N − sum)
    greater   (>  N)   penalty = max(0, (N+1) − sum)      (CP-SAT: sum > N == sum >= N+1)
    less      (<  N)   penalty = max(0, sum − (N−1))      (CP-SAT: sum < N == sum <= N−1)

The penalty IntVar upper bound is computed per type so CP-SAT never rejects the
model when `count` exceeds the candidate count (review C2):
    equal             -> max(N, len(candidates))
    greatere/greater  -> max(0, N+1)
    lesse/less        -> len(candidates)

## Weighting model — a SINGLE shared bucket

All preferred entries penalise into ONE `data['penalties']['preferred_games']`
bucket carrying the single default weight `PENALTY_WEIGHTS['preferred_games']`.
With no per-entry `weight`, every preferred entry has equal pull. An optional
per-entry `weight` is a multiplier on top of that default
(`multiplier = max(1, entry_weight // default_weight)`, same pattern as
`PreferredWeekendsAwayGround` / `TeamPairNoConcurrency`), applied to that entry's
raw penalty IntVar so only flagged entries get scaled.

## Non-fatal contract

An empty / zero-candidate scope produces NO penalty and a logged warning — never
`sys.exit` (that FORCED behaviour is deliberately NOT copied). Locked-week vars
are skipped per-variable (review M4).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from constraints.atoms.base import Atom

logger = logging.getLogger(__name__)


class PreferredGames(Atom):
    """Soft, weighted penalty on deviation from a FORCED_GAMES-style target."""

    canonical_name = 'PreferredGames'
    atom_group = ''

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        entries: List[Dict] = data.get('preferred_games', [])
        if not entries:
            return 0

        default_weight = data.get('penalty_weights', {}).get('preferred_games', 10_000)
        if default_weight <= 0:
            # Disabled — don't create any penalty vars or bucket.
            return 0

        # Build scope/team/club rules via the shared parser (unique per entry so
        # each preference gets its own bucket entry + weight, no merging).
        from utils import _build_scope_count_rules, _get_matching_forced_scopes
        scope_groups, constraint_types, constraint_counts, constraint_weights = (
            _build_scope_count_rules(
                entries, data['teams'], label='PREFERRED_GAMES',
                unique_per_entry=True,
            )
        )
        if not scope_groups:
            return 0

        locked_weeks = set(data.get('locked_weeks', set()))

        # Bucket vars per scope_key in a single pass over X. Locked-week vars are
        # skipped per-variable (review M4): X includes locked-week vars but they
        # are fixed, so penalising on them is meaningless.
        scope_vars: Dict = defaultdict(list)
        for key, var in X.items():
            if len(key) < 11:
                continue  # dummy key (short tuple)
            if not key[3]:
                continue  # no day (dummy slot)
            if locked_weeks and key[6] in locked_weeks:
                continue
            for scope_key in _get_matching_forced_scopes(key, scope_groups):
                scope_vars[scope_key].append(var)

        data.setdefault('penalties', {})
        bucket = data['penalties'].setdefault(
            'preferred_games',
            {'weight': default_weight, 'penalties': []},
        )

        n = 0
        for scope_idx, scope_key in enumerate(scope_groups):
            ctype = constraint_types.get(scope_key, 'equal')
            count = constraint_counts.get(scope_key, 1)
            entry_weight = constraint_weights.get(scope_key, default_weight)
            multiplier = max(1, int(entry_weight) // default_weight)

            vars_list = scope_vars.get(scope_key, [])
            if not vars_list:
                logger.warning(
                    "PreferredGames: scope %r matched no candidate variables "
                    "(constraint=%s count=%s) — no penalty applied.",
                    dict(scope_key), ctype, count,
                )
                continue

            raw = self._deviation_penalty(
                model, vars_list, ctype, count, scope_idx,
            )
            self._append_scaled(model, bucket, raw, multiplier, scope_idx)
            n += 1

        return n

    @staticmethod
    def _deviation_penalty(model, vars_list, ctype, count, scope_idx):
        """Return an IntVar equal to the deviation penalty for this scope.

        Upper bounds computed per type so CP-SAT never rejects the model when
        `count` > candidate count (review C2).
        """
        total = sum(vars_list)
        n_cand = len(vars_list)
        N = count
        zero = model.NewConstant(0)

        if ctype == 'equal':
            # penalty = |sum − N|, two-sided.
            ub = max(N, n_cand)
            pen = model.NewIntVar(0, ub, f'u_pref_eq_{scope_idx}')
            model.AddAbsEquality(pen, total - N)
            return pen
        if ctype == 'lesse':
            # penalty = max(0, sum − N).
            ub = n_cand
            pen = model.NewIntVar(0, ub, f'u_pref_lesse_{scope_idx}')
            model.AddMaxEquality(pen, [total - N, zero])
            return pen
        if ctype == 'less':
            # CP-SAT: sum < N  ==  sum <= N−1.  penalty = max(0, sum − (N−1)).
            ub = n_cand
            pen = model.NewIntVar(0, ub, f'u_pref_less_{scope_idx}')
            model.AddMaxEquality(pen, [total - (N - 1), zero])
            return pen
        if ctype == 'greatere':
            # penalty = max(0, N − sum).
            ub = max(0, N + 1)
            pen = model.NewIntVar(0, ub, f'u_pref_greatere_{scope_idx}')
            model.AddMaxEquality(pen, [N - total, zero])
            return pen
        if ctype == 'greater':
            # CP-SAT: sum > N  ==  sum >= N+1.  penalty = max(0, (N+1) − sum).
            ub = max(0, N + 1)
            pen = model.NewIntVar(0, ub, f'u_pref_greater_{scope_idx}')
            model.AddMaxEquality(pen, [(N + 1) - total, zero])
            return pen
        # Defensive: validation warns on unknown ctype; treat as equal.
        ub = max(N, n_cand)
        pen = model.NewIntVar(0, ub, f'u_pref_default_{scope_idx}')
        model.AddAbsEquality(pen, total - N)
        return pen

    @staticmethod
    def _append_scaled(model, bucket, raw_var, multiplier, scope_idx):
        """Append raw_var (or multiplier * raw_var) to the shared bucket."""
        if multiplier == 1:
            bucket['penalties'].append(raw_var)
            return
        # raw_var upper bound is unknown here; reuse a generous bound by
        # multiplying CP-SAT's domain max would require introspection. Instead
        # bound the scaled var by multiplier * raw_var's max via a fresh IntVar
        # constrained to equal the product (CP-SAT infers the domain).
        scaled = model.NewIntVar(0, 2_000_000_000, f'u_pref_scaled_{scope_idx}')
        model.Add(scaled == multiplier * raw_var)
        bucket['penalties'].append(scaled)
