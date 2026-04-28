"""PHL/2nd Sunday: when a club-pair coincides on a round, at least one
back-to-back same-field pair must exist (HARD), and missing coincidences are
penalised (SOFT, into the `ClubVsClubAlignment` penalty bucket).

Lifted from the second half of `original.py:ClubVsClubAlignment` (lines
1096–1198) — extracted into its own atom because the PHL/2nd alignment is a
structurally different rule (back-to-back same-field) from the lower-grade
≤2-fields rule. See `ATOMIZATION_PLAN.md` Phase 3c.
"""
from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_shared import (
    PHL_BTB_COINCIDE_PREFIX,
    collect_phl_2nd_sunday_field_slot,
    per_team_games,
    phl_2nd_grade_pairs_to_compare,
)


class PHLAnd2ndBackToBackSameField(Atom):
    canonical_name = 'PHLAnd2ndBackToBackSameField'
    atom_group = 'ClubVsClubAlignment'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        defaults = data.get('constraint_defaults', {})
        base_slack = defaults.get('club_vs_club_alignment_base_slack', 0)
        config_slack = data.get('constraint_slack', {}).get(
            'ClubVsClubAlignment', 0
        ) + base_slack

        weight = data.get('penalty_weights', {}).get(
            'ClubVsClubAlignment', 100000,
        )
        if 'penalties' not in data:
            data['penalties'] = {}
        bucket = data['penalties'].setdefault(
            'ClubVsClubAlignment', {'weight': weight, 'penalties': []},
        )

        per_grade = {
            'PHL': collect_phl_2nd_sunday_field_slot(X, data, 'PHL'),
            '2nd': collect_phl_2nd_sunday_field_slot(X, data, '2nd'),
        }

        btb_idx = 0
        for grade, other_grade, num_games in phl_2nd_grade_pairs_to_compare(data):
            grade_pairs = per_grade[grade]
            other_pairs = per_grade[other_grade]

            for club_pair, rounds in grade_pairs.items():
                if club_pair not in other_pairs:
                    continue
                other_rounds = other_pairs[club_pair]
                coincide_vars = []
                for round_no, info_list in rounds.items():
                    if round_no not in other_rounds:
                        continue
                    other_info = other_rounds[round_no]
                    vars1 = [gi[0] for gi in info_list]
                    vars2 = [gi[0] for gi in other_info]
                    if not vars1 or not vars2:
                        continue

                    ind1 = registry.get_or_create_bool(
                        ('phl_btb_g1', grade, club_pair, round_no), vars1,
                        f'phl_g1_{grade}_{club_pair}_{round_no}',
                    )
                    ind2 = registry.get_or_create_bool(
                        ('phl_btb_g2', other_grade, club_pair, round_no), vars2,
                        f'phl_g2_{other_grade}_{club_pair}_{round_no}',
                    )
                    coincide = model.NewBoolVar(
                        f'phl_coin_{grade}_{other_grade}_{club_pair}_{round_no}'
                    )
                    model.Add(coincide <= ind1)
                    model.Add(coincide <= ind2)
                    model.Add(coincide >= ind1 + ind2 - 1)
                    registry.register(
                        (PHL_BTB_COINCIDE_PREFIX, grade, other_grade, club_pair, round_no),
                        coincide,
                    )
                    coincide_vars.append(coincide)

                    btb_pairs = []
                    for var1, field1, slot1 in info_list:
                        for var2, field2, slot2 in other_info:
                            if field1 == field2 and abs(slot1 - slot2) == 1:
                                pair_ind = model.NewBoolVar(
                                    f'btb_{club_pair}_{round_no}_{field1}_'
                                    f'{slot1}_{slot2}_{btb_idx}'
                                )
                                model.Add(pair_ind <= var1)
                                model.Add(pair_ind <= var2)
                                model.Add(pair_ind >= var1 + var2 - 1)
                                btb_pairs.append(pair_ind)
                                btb_idx += 1

                    if btb_pairs:
                        model.Add(sum(btb_pairs) >= 1).OnlyEnforceIf(coincide)
                    else:
                        model.Add(coincide == 0)
                    n += 1

                if coincide_vars:
                    min_required = max(0, num_games - config_slack)
                    model.Add(sum(coincide_vars) >= min_required)
                    n += 1

                    actual = model.NewIntVar(
                        0, len(coincide_vars),
                        f'phl_actual_coin_{grade}_{other_grade}_{club_pair}',
                    )
                    model.Add(actual == sum(coincide_vars))
                    deficit = model.NewIntVar(
                        0, num_games,
                        f'phl_coin_def_{grade}_{other_grade}_{club_pair}',
                    )
                    model.Add(deficit >= num_games - actual)
                    bucket['penalties'].append(deficit)
        return n
