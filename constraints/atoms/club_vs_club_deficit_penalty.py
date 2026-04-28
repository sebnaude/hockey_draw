"""SOFT penalty for missing club-pair coincidences across a grade pair.

For each (grade, other_grade, club_pair) the penalty is
`num_games_for_grade - sum(coincide_vars)` (clamped at 0). Reads `coincide`
vars registered by `ClubVsClubCoincidence`.
"""
from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_shared import (
    COINCIDE_KEY_PREFIX,
    collect_grade_pair_round_vars,
    lower_grade_pairs_to_compare,
)


class ClubVsClubDeficitPenalty(Atom):
    canonical_name = 'ClubVsClubDeficitPenalty'
    atom_group = 'ClubVsClubAlignment'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        weight = data.get('penalty_weights', {}).get(
            'ClubVsClubAlignment', 100000,
        )
        if 'penalties' not in data:
            data['penalties'] = {}
        # setdefault not assignment: PHLAnd2ndBackToBackSameField in stage 1
        # may have already populated this bucket with PHL deficit penalties.
        bucket = data['penalties'].setdefault(
            'ClubVsClubAlignment',
            {'weight': weight, 'penalties': []},
        )

        per_grade_vars = {}
        for grade, other_grade, num_games in lower_grade_pairs_to_compare(data):
            if grade not in per_grade_vars:
                per_grade_vars[grade] = collect_grade_pair_round_vars(
                    X, data, grade,
                )
            if other_grade not in per_grade_vars:
                per_grade_vars[other_grade] = collect_grade_pair_round_vars(
                    X, data, other_grade,
                )
            grade_pairs = per_grade_vars[grade]
            other_pairs = per_grade_vars[other_grade]

            for club_pair, rounds in grade_pairs.items():
                if club_pair not in other_pairs:
                    continue
                other_rounds = other_pairs[club_pair]
                coincide_vars = []
                for round_no in rounds:
                    if round_no not in other_rounds:
                        continue
                    coincide = registry.get(
                        (COINCIDE_KEY_PREFIX, grade, other_grade, club_pair, round_no)
                    )
                    if coincide is None:
                        continue
                    coincide_vars.append(coincide)

                if not coincide_vars:
                    continue
                actual = model.NewIntVar(
                    0, len(coincide_vars),
                    f'actual_coin_{grade}_{other_grade}_{club_pair}',
                )
                model.Add(actual == sum(coincide_vars))
                deficit = model.NewIntVar(
                    0, num_games,
                    f'coin_def_{grade}_{other_grade}_{club_pair}',
                )
                model.Add(deficit >= num_games - actual)
                bucket['penalties'].append(deficit)
                n += 1
        return n
