"""Soft penalty that encourages alphabetically-earlier matchups to play in earlier rounds.

For each grade, pairs are sorted alphabetically by their canonical key ``(team1, team2)``.
Each pair is assigned a rank ``r`` (0-indexed from 0 for the alphabetically-first pair).
For every scheduled game, the penalty contribution is ``weight * r * X[key]``.

Because earlier-alphabetical pairs have a lower rank, the solver incurs less penalty
by scheduling them in *any* round.  Higher-ranked (later-alphabetical) pairs cost more
per game, so the solver pushes them toward later rounds where they compete against
fewer good slots.  The net effect: alphabetically-earlier matchups tend to appear in
earlier rounds — a deterministic tie-break, never a hard constraint.

Weight: ``PENALTY_WEIGHTS['soft_lex_ordering']``, defaults to ``1``.  Kept tiny
intentionally; this must never block feasibility or compete with real constraints.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

from constraints.atoms.base import Atom


class SoftLexMatchupOrdering(Atom):
    """Soft lex-ordering atom: alphabetically-earlier matchups preferred in earlier rounds."""

    canonical_name = 'SoftLexMatchupOrdering'
    atom_group = ''

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        weight = data.get('penalty_weights', {}).get('soft_lex_ordering', 1)
        if weight == 0:
            return 0

        locked_weeks = set(data.get('locked_weeks', set()))

        # Collect all (grade, pair) combinations present in X, and the vars per pair.
        # pair = (team1, team2) with team1 <= team2 alphabetically (enforced by generate_X).
        # key indices: 0=team1, 1=team2, 2=grade, 6=week
        grade_pairs: Dict[str, Dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
        for key, var in X.items():
            if len(key) < 11:
                continue  # dummy key
            if not key[3]:
                continue  # no day
            if locked_weeks and key[6] in locked_weeks:
                continue
            grade = key[2]
            pair = (key[0], key[1])
            grade_pairs[grade][pair].append(var)

        if not grade_pairs:
            return 0

        data.setdefault('penalties', {})
        bucket = data['penalties'].setdefault(
            'soft_lex_ordering',
            {'weight': weight, 'penalties': []},
        )

        n = 0
        for grade, pairs_vars in grade_pairs.items():
            # Sort pairs alphabetically by (team1, team2).
            sorted_pairs = sorted(pairs_vars.keys())
            for rank, pair in enumerate(sorted_pairs):
                if rank == 0:
                    # Rank-0 pair contributes zero penalty regardless; skip to save vars.
                    continue
                vars_list = pairs_vars[pair]
                # For each variable in this pair, add `rank * var` to penalties.
                # The solver objective accumulates: weight * sum(rank * var) over all pairs.
                # Use a single IntVar per (grade, pair) = rank * sum(vars).
                pair_sum = model.NewIntVar(
                    0, len(vars_list) * rank,
                    f'lex_pen_{grade}_{pair[0]}_{pair[1]}',
                )
                model.Add(pair_sum == rank * sum(vars_list))
                bucket['penalties'].append(pair_sum)
                n += 1

        return n
