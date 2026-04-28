"""When a club-pair coincides on a round, ≤ 2 fields are used (HARD) and
field-excess (above 1) is penalised (SOFT).

Reads back the `coincide` BoolVar registered by `ClubVsClubCoincidence` from
the helper-var pool. Must run AFTER `ClubVsClubCoincidence` in the engine
dispatch order.
"""
from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_shared import (
    COINCIDE_KEY_PREFIX,
    collect_sunday_clubpair_field_round,
    lower_grade_pairs_to_compare,
)


class ClubVsClubFieldLimit(Atom):
    canonical_name = 'ClubVsClubFieldLimit'
    atom_group = 'ClubVsClubAlignment'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        weight = data.get('penalty_weights', {}).get(
            'ClubVsClubAlignmentField', 50000,
        )
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['ClubVsClubAlignmentField'] = {
            'weight': weight, 'penalties': [],
        }

        per_grade_field_vars = {}
        fidx = 0
        for grade, other_grade, _num in lower_grade_pairs_to_compare(data):
            if grade not in per_grade_field_vars:
                per_grade_field_vars[grade] = (
                    collect_sunday_clubpair_field_round(X, data, grade)
                )
            grade_field_pairs = per_grade_field_vars[grade]

            for club_pair, round_field_map in grade_field_pairs.items():
                for round_no, field_to_vars in round_field_map.items():
                    coincide = registry.get(
                        (COINCIDE_KEY_PREFIX, grade, other_grade, club_pair, round_no)
                    )
                    if coincide is None:
                        continue
                    if not field_to_vars:
                        continue
                    fi_list = []
                    for fname, gvars in field_to_vars.items():
                        fi = model.NewBoolVar(
                            f'fld_{grade}_{other_grade}_{club_pair}_'
                            f'{round_no}_{fname}_{fidx}'
                        )
                        model.AddMaxEquality(fi, gvars)
                        fi_list.append(fi)
                    nf = model.NewIntVar(
                        0, len(fi_list),
                        f'nflds_{grade}_{other_grade}_{club_pair}_{round_no}_{fidx}',
                    )
                    model.Add(nf == sum(fi_list))
                    model.Add(nf <= 2).OnlyEnforceIf(coincide)
                    n += 1

                    field_excess = model.NewIntVar(
                        0, len(fi_list),
                        f'fexcess_{grade}_{other_grade}_{club_pair}_{round_no}_{fidx}',
                    )
                    model.Add(field_excess >= nf - 1).OnlyEnforceIf(coincide)
                    model.Add(field_excess == 0).OnlyEnforceIf(coincide.Not())
                    data['penalties']['ClubVsClubAlignmentField']['penalties'].append(
                        field_excess
                    )
                    fidx += 1
        return n
