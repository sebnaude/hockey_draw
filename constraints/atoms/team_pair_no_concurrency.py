"""Soft penalty for specified team pairs playing in the same (week, day_slot).

Reads `TEAM_PAIR_NO_CONCURRENCY` from `data['constraint_defaults']` (or, if
absent, from `data` itself for legacy callers). Each entry is a tuple of:

    (team_a, team_b)
    (team_a, team_b, weight_multiplier)

`team_a`, `team_b` are exact team names. `weight_multiplier` is a per-pair
multiplier applied on top of the bucket-level base weight (defaults to 1).
The bucket-level base weight is read from
`PENALTY_WEIGHTS['TeamPairNoConcurrency']` (or 1000 if unset).

For each (week, day_slot) where both teams could appear, the penalty is:

    raw = max(0, sum(vars_team_a) + sum(vars_team_b) - 1)
    scaled = weight_multiplier * raw

`scaled` is added to the `TeamPairNoConcurrency` penalty bucket. `raw` cannot
exceed 1 in practice because NoDoubleBookingTeams caps each team's slot
participation at 1 — but the formula stays clamped at 0 either way.

Severity 3 (MEDIUM): convenor-supplied preference, soft only. Solver will
avoid co-occurrence when feasibly avoidable.
"""
from collections import defaultdict

from constraints.atoms.base import Atom


class TeamPairNoConcurrency(Atom):
    canonical_name = 'TeamPairNoConcurrency'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        defaults = data.get('constraint_defaults', {}) or {}
        pairs_raw = defaults.get('TEAM_PAIR_NO_CONCURRENCY')
        if pairs_raw is None:
            pairs_raw = data.get('TEAM_PAIR_NO_CONCURRENCY', [])
        if not pairs_raw:
            return 0

        base_weight = data.get('penalty_weights', {}).get(
            'TeamPairNoConcurrency', 1000
        )

        # Normalise entries: list of (team_a, team_b, weight_multiplier),
        # with team names sorted to collapse duplicates.
        normalised = []
        for entry in pairs_raw:
            if len(entry) == 2:
                a, b = entry
                w = 1
            elif len(entry) == 3:
                a, b, w = entry
            else:
                raise ValueError(
                    f'TEAM_PAIR_NO_CONCURRENCY entry must be (team_a, team_b) '
                    f'or (team_a, team_b, weight); got {entry!r}'
                )
            if a == b:
                raise ValueError(
                    f'TEAM_PAIR_NO_CONCURRENCY entry has identical teams: {entry!r}'
                )
            key = tuple(sorted((a, b)))
            normalised.append((key[0], key[1], int(w)))

        # Index decision variables per (team, week, day_slot).
        team_slot_vars = defaultdict(list)
        locked_weeks = set(data.get('locked_weeks', set()))
        for key, var in X.items():
            if len(key) < 11:
                continue
            if not key[3]:
                continue
            t1, t2, _grade, _day, day_slot, _time, week, _date, _round_no, _fname, _floc = key
            if locked_weeks and week in locked_weeks:
                continue
            team_slot_vars[(t1, week, day_slot)].append(var)
            team_slot_vars[(t2, week, day_slot)].append(var)

        data.setdefault('penalties', {})
        bucket = data['penalties'].setdefault(
            'TeamPairNoConcurrency',
            {'weight': base_weight, 'penalties': []},
        )

        n = 0
        for idx, (team_a, team_b, multiplier) in enumerate(normalised):
            slots_a = {(w, s) for (t, w, s) in team_slot_vars if t == team_a}
            slots_b = {(w, s) for (t, w, s) in team_slot_vars if t == team_b}
            shared = slots_a & slots_b
            if not shared:
                continue
            for w, s in shared:
                vars_a = team_slot_vars.get((team_a, w, s), [])
                vars_b = team_slot_vars.get((team_b, w, s), [])
                if not vars_a or not vars_b:
                    continue
                total = sum(vars_a) + sum(vars_b)
                upper = len(vars_a) + len(vars_b)
                raw = model.NewIntVar(
                    0, upper, f'u_team_pair_raw_{idx}_w{w}_s{s}'
                )
                # raw = max(0, total - 1)
                model.AddMaxEquality(
                    raw, [total - 1, model.NewConstant(0)]
                )
                if multiplier == 1:
                    bucket['penalties'].append(raw)
                else:
                    scaled = model.NewIntVar(
                        0, upper * multiplier,
                        f'u_team_pair_scaled_{idx}_w{w}_s{s}',
                    )
                    model.Add(scaled == multiplier * raw)
                    bucket['penalties'].append(scaled)
                n += 1
        return n
