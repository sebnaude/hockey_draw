"""3rd-6th club-pair meetings should coincide on the same round across grades.

For each pair of grades (X, Y) where Y has strictly more per-team games than X
(see `per_team_games` walk in `_club_vs_club_shared.py`), and each club_pair
seen in both grades, this atom:

1. Builds per-round indicators (`g1`, `g2`) using the helper-var pool.
2. Channels a `coincide` BoolVar = `g1 AND g2` per round.
3. Registers the `coincide` var in the helper-var pool keyed by
   `('cvc_coincide', grade, other_grade, club_pair, round_no)` so that
   `ClubVsClubFieldLimit` and `ClubVsClubDeficitPenalty` can read it back.
4. Adds the HARD requirement `sum(coincide_vars) >= num_games - slack`.
"""
from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_shared import (
    COINCIDE_KEY_PREFIX,
    collect_grade_pair_round_vars,
    lower_grade_pairs_to_compare,
    per_team_games,
)


class ClubVsClubCoincidence(Atom):
    canonical_name = 'ClubVsClubCoincidence'
    atom_group = 'ClubVsClubAlignment'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        slack = data.get('constraint_slack', {}).get('ClubVsClubAlignment', 0)
        games = per_team_games(data)

        per_grade_vars = {}
        for grade, _other, _num in lower_grade_pairs_to_compare(data):
            if grade not in per_grade_vars:
                per_grade_vars[grade] = collect_grade_pair_round_vars(
                    X, data, grade,
                )

        for grade, other_grade, num_games in lower_grade_pairs_to_compare(data):
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
                for round_no, vars_list in rounds.items():
                    if round_no not in other_rounds:
                        continue
                    ind1 = registry.get_or_create_bool(
                        ('align_g1', grade, club_pair, round_no), vars_list,
                        f'g1_{grade}_{club_pair}_{round_no}',
                    )
                    ind2 = registry.get_or_create_bool(
                        ('align_g2', other_grade, club_pair, round_no),
                        other_rounds[round_no],
                        f'g2_{other_grade}_{club_pair}_{round_no}',
                    )
                    coincide = model.NewBoolVar(
                        f'coin_{grade}_{other_grade}_{club_pair}_{round_no}'
                    )
                    model.Add(coincide <= ind1)
                    model.Add(coincide <= ind2)
                    model.Add(coincide >= ind1 + ind2 - 1)
                    registry.register(
                        (COINCIDE_KEY_PREFIX, grade, other_grade, club_pair, round_no),
                        coincide,
                    )
                    coincide_vars.append(coincide)

                if coincide_vars:
                    min_required = max(0, num_games - slack)
                    model.Add(sum(coincide_vars) >= min_required)
                    n += 1
        return n
